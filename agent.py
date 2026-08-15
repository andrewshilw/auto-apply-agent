"""
Terminal agent built as an explicit ReAct loop in LangGraph. Started as the
Week 1 lab (weather only, see the `week1-lab` git branch for that snapshot),
grew a real, visible browser (agent-browser) for LinkedIn in Week 2, and
gained a dedicated JD Evaluation Node in Week 3.

Graph shape:

    START -> input -> classify -+-> reasoning -> [conditional] -> action ---------------> reasoning -> ...
                                 |                              -> evaluate -> [conditional] -> apply_action -> reasoning -> ...
                                 |                                                            -> fallback ----> reasoning -> ...
                                 |                              -> respond -> input -> ...
                                 +-> input (OTHER: ignored, ask again)
                                 +-> END (QUIT)

- `input` reads one line from the terminal and adds it to the conversation.
- `classify` is the dispatch step: instead of hardcoded string checks like
  `if city.lower() in {"quit", "exit"}`, an LLM call reads the message and
  decides ACT / QUIT / OTHER. The whole interactive session lives inside a
  single `graph.invoke()` — the `respond -> input` and `classify -> input`
  edges are what replace Python's `while True`, and `classify -> END` is the
  only way out. ACT is intentionally domain-agnostic (not e.g. WEATHER):
  with unrelated tools bound to the LLM, routing on a fixed set of domain
  names would mean updating the router every time a tool is added. The
  router only decides whether the message needs one of `reasoning`'s tools
  at all; which tool is the reasoning step's job, same as any
  function-calling LLM call.
- `reasoning` is the "Reason" step: the LLM decides whether it has enough
  information to answer, or needs to call a tool.
- `action` is the generic "Act" step for domain-agnostic tools (weather,
  LinkedIn search): it executes whatever tool call the LLM requested and
  always routes back to `reasoning`, never onward. That's what forces an
  Observation after every Action — the tool's result is appended as a
  ToolMessage, and the LLM must look at it before it's allowed to answer.
- `evaluate` is the Week 3 JD Evaluation Node — unlike `action`, this one
  isn't domain-agnostic on purpose: the whole point is that its outcome
  (APPLY vs SKIP, from `evaluation.decide`'s > 80 threshold) has to steer
  the graph itself, not just come back as text for the LLM to interpret.
  `reasoning` still decides *whether* to evaluate a listing (by calling the
  `evaluate_job_listing` tool, same as any other tool), but execution is
  routed here instead of to `action` specifically so its score can drive a
  real conditional edge to `apply_action` or `fallback` — the literal
  "only trigger Apply when score > 80, otherwise auto-fallback" decision
  logic — rather than that branch living only inside a prompt.
- `apply_action` / `fallback` are intentionally thin: neither one clicks
  anything on LinkedIn. They exist so the APPLY/SKIP branch is a real graph
  edge, not a string the LLM has to correctly interpret; the actual
  evaluation + recommendation + logging already happened in `evaluate`.

LangSmith tracing is picked up automatically from the LANGCHAIN_TRACING_V2 /
LANGCHAIN_API_KEY / LANGCHAIN_PROJECT env vars (see .env.example) — no code
here talks to LangSmith directly, LangGraph/LangChain report every node run
to it when those vars are set.
"""

from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import get_weather
from linkedin_tool import linkedin_job_search
from evaluation import Decision, evaluate_job_listing, run_job_evaluation

load_dotenv()

# evaluate_job_listing is bound to the LLM like any other tool (so `reasoning`
# can choose to call it), but its execution is routed to the dedicated
# `evaluate` node below rather than the generic `action` ToolNode — see the
# module docstring for why.
ACTION_TOOLS = [get_weather, linkedin_job_search]
EVALUATE_TOOL_NAME = evaluate_job_listing.name
TOOLS = [*ACTION_TOOLS, evaluate_job_listing]

llm = ChatAnthropic(model="claude-sonnet-5").bind_tools(TOOLS)
router_llm = ChatAnthropic(model="claude-sonnet-5")

ROUTER_PROMPT = (
    "Classify the user's message as exactly one word:\n"
    "ACT - they want you to do something: check weather, search for "
    "LinkedIn jobs, evaluate a job listing against their resume, or "
    "anything else you have a tool for\n"
    "QUIT - they want to exit, stop, or end the session\n"
    "OTHER - anything else (small talk, unrelated questions)\n"
    "Reply with only that one word, nothing else."
)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    intent: str
    pending_decision: Decision | None


def input_node(state: AgentState) -> AgentState:
    text = input("\nYou> ").strip()
    return {"messages": [HumanMessage(content=text)]}


def classify_node(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]
    response = router_llm.invoke([SystemMessage(content=ROUTER_PROMPT), last_message])
    intent = response.content.strip().upper()
    return {"intent": intent if intent in {"ACT", "QUIT", "OTHER"} else "OTHER"}


def route_after_classify(state: AgentState) -> Literal["reasoning", "input", "__end__"]:
    return {"ACT": "reasoning", "QUIT": END, "OTHER": "input"}[state["intent"]]


def reasoning_node(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


action_node = ToolNode(ACTION_TOOLS)


def should_continue(state: AgentState) -> Literal["action", "evaluate", "respond"]:
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None)
    if not tool_calls:
        return "respond"
    if any(tc["name"] == EVALUATE_TOOL_NAME for tc in tool_calls):
        return "evaluate"
    return "action"


def evaluate_node(state: AgentState) -> AgentState:
    """The Evaluation Node: pulls the pending `evaluate_job_listing` tool
    call off the last AIMessage, runs the full JD-vs-resume pipeline
    (`evaluation.run_job_evaluation` — extract JD text, retrieve resume
    context, score, threshold-decide, log), and replies with the matching
    ToolMessage so the message history stays valid. The decision itself is
    stashed in state for `route_after_evaluate` to branch on."""
    last_message = state["messages"][-1]
    tool_call = next(tc for tc in last_message.tool_calls if tc["name"] == EVALUATE_TOOL_NAME)
    text, decision = run_job_evaluation(**tool_call["args"])
    tool_message = ToolMessage(content=text, tool_call_id=tool_call["id"])
    return {"messages": [tool_message], "pending_decision": decision}


def route_after_evaluate(state: AgentState) -> Literal["apply_action", "fallback"]:
    return "apply_action" if state["pending_decision"].recommendation == "APPLY" else "fallback"


def apply_action_node(state: AgentState) -> AgentState:
    """Score cleared the threshold. This is deliberately a no-op beyond
    printing — actually clicking Apply on LinkedIn is a separate, explicit
    decision left to a human, never taken automatically here."""
    decision = state["pending_decision"]
    print(f"[Apply] score {decision.evaluation.match_score}/100 clears the threshold — recommended, not submitted.")
    return {}


def fallback_node(state: AgentState) -> AgentState:
    """Score didn't clear the threshold: auto-fallback, skip this listing."""
    decision = state["pending_decision"]
    print(f"[Skip] score {decision.evaluation.match_score}/100 — below threshold, falling back.")
    return {}


def respond_node(state: AgentState) -> AgentState:
    print(state["messages"][-1].content)
    return {}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("input", input_node)
    graph.add_node("classify", classify_node)
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("action", action_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("apply_action", apply_action_node)
    graph.add_node("fallback", fallback_node)
    graph.add_node("respond", respond_node)

    graph.add_edge(START, "input")
    graph.add_edge("input", "classify")
    graph.add_conditional_edges(
        "classify",
        route_after_classify,
        {"reasoning": "reasoning", "input": "input", END: END},
    )
    graph.add_conditional_edges(
        "reasoning", should_continue, {"action": "action", "evaluate": "evaluate", "respond": "respond"}
    )
    graph.add_edge("action", "reasoning")  # force observation before the next decision
    graph.add_conditional_edges(
        "evaluate", route_after_evaluate, {"apply_action": "apply_action", "fallback": "fallback"}
    )
    graph.add_edge("apply_action", "reasoning")  # observation, same as action -> reasoning
    graph.add_edge("fallback", "reasoning")
    graph.add_edge("respond", "input")  # next turn, driven by the graph instead of a Python loop

    return graph.compile()
