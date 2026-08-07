"""
Terminal agent built as an explicit ReAct loop in LangGraph. Started as the
Week 1 lab (weather only, see the `week1-lab` git branch for that snapshot)
and grew a browser (Playwright) and a job search (JobSpy) in Week 2.

Graph shape:

    START -> input -> classify -+-> reasoning -> [conditional] -> action -> reasoning -> ... -> respond -> input -> ...
                                 +-> input (OTHER: ignored, ask again)
                                 +-> END (QUIT)

- `input` reads one line from the terminal and adds it to the conversation.
- `classify` is the dispatch step: instead of hardcoded string checks like
  `if city.lower() in {"quit", "exit"}`, an LLM call reads the message and
  decides ACT / QUIT / OTHER. The whole interactive session lives inside a
  single `graph.invoke()` — the `respond -> input` and `classify -> input`
  edges are what replace Python's `while True`, and `classify -> END` is the
  only way out. ACT is intentionally domain-agnostic (not e.g. WEATHER):
  with three unrelated tools now bound to the LLM, routing on a fixed set of
  domain names would mean updating the router every time a tool is added.
  The router only decides whether the message needs one of `reasoning`'s
  tools at all; which tool is the reasoning step's job, same as any
  function-calling LLM call.
- `reasoning` is the "Reason" step: the LLM decides whether it has enough
  information to answer, or needs to call a tool.
- `action` is the "Act" step: it executes the tool call the LLM requested.
- The edge from `action` always points back to `reasoning`, never onward.
  This is what forces an Observation after every Action: the tool's result
  is appended to the message list as a ToolMessage, and the LLM is required
  to look at it before it's allowed to produce a final answer. Without this,
  the agent could "act" blindly and never actually use what it found.

LangSmith tracing is picked up automatically from the LANGCHAIN_TRACING_V2 /
LANGCHAIN_API_KEY / LANGCHAIN_PROJECT env vars (see .env.example) — no code
here talks to LangSmith directly, LangGraph/LangChain report every node run
to it when those vars are set.
"""

from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import get_weather
from jobs_tool import search_jobs
from browser_tools import browse_page

load_dotenv()

TOOLS = [get_weather, search_jobs, browse_page]

llm = ChatAnthropic(model="claude-sonnet-5").bind_tools(TOOLS)
router_llm = ChatAnthropic(model="claude-sonnet-5")

ROUTER_PROMPT = (
    "Classify the user's message as exactly one word:\n"
    "ACT - they want you to do something: check weather, search for jobs, "
    "browse/inspect a web page, or anything else you have a tool for\n"
    "QUIT - they want to exit, stop, or end the session\n"
    "OTHER - anything else (small talk, unrelated questions)\n"
    "Reply with only that one word, nothing else."
)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    intent: str


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


action_node = ToolNode(TOOLS)


def should_continue(state: AgentState) -> Literal["action", "respond"]:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "action"
    return "respond"


def respond_node(state: AgentState) -> AgentState:
    print(state["messages"][-1].content)
    return {}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("input", input_node)
    graph.add_node("classify", classify_node)
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("action", action_node)
    graph.add_node("respond", respond_node)

    graph.add_edge(START, "input")
    graph.add_edge("input", "classify")
    graph.add_conditional_edges(
        "classify",
        route_after_classify,
        {"reasoning": "reasoning", "input": "input", END: END},
    )
    graph.add_conditional_edges("reasoning", should_continue, {"action": "action", "respond": "respond"})
    graph.add_edge("action", "reasoning")  # force observation before the next decision
    graph.add_edge("respond", "input")  # next turn, driven by the graph instead of a Python loop

    return graph.compile()
