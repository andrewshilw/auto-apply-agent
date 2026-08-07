"""
Week 1 lab: a terminal Weather Agent built as an explicit ReAct loop in LangGraph.

Graph shape:

    START -> input -> classify -+-> reasoning -> [conditional] -> action -> reasoning -> ... -> respond -> input -> ...
                                 +-> input (OTHER: ignored, ask again)
                                 +-> END (QUIT)

- `input` reads one line from the terminal and adds it to the conversation.
- `classify` is the dispatch step: instead of hardcoded string checks like
  `if city.lower() in {"quit", "exit"}`, an LLM call reads the message and
  decides WEATHER / QUIT / OTHER. The whole interactive session lives inside
  a single `graph.invoke()` — the `respond -> input` and `classify -> input`
  edges are what replace Python's `while True`, and `classify -> END` is the
  only way out.
- `reasoning` is the "Reason" step: the LLM decides whether it has enough
  information to answer, or needs to call a tool.
- `action` is the "Act" step: it executes the tool call the LLM requested.
- The edge from `action` always points back to `reasoning`, never onward.
  This is what forces an Observation after every Action: the tool's result
  is appended to the message list as a ToolMessage, and the LLM is required
  to look at it before it's allowed to produce a final answer. Without this,
  the agent could "act" blindly and never actually use what it found.
"""

from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import get_weather

load_dotenv()

TOOLS = [get_weather]

llm = ChatAnthropic(model="claude-sonnet-5").bind_tools(TOOLS)
router_llm = ChatAnthropic(model="claude-sonnet-5")

ROUTER_PROMPT = (
    "Classify the user's message as exactly one word:\n"
    "WEATHER - they are asking about weather, temperature, or forecast for a place\n"
    "QUIT - they want to exit, stop, or end the session\n"
    "OTHER - anything else\n"
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
    return {"intent": intent if intent in {"WEATHER", "QUIT", "OTHER"} else "OTHER"}


def route_after_classify(state: AgentState) -> Literal["reasoning", "input", "__end__"]:
    return {"WEATHER": "reasoning", "QUIT": END, "OTHER": "input"}[state["intent"]]


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
