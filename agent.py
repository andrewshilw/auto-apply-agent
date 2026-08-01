"""
Week 1 lab: a terminal Weather Agent built as an explicit ReAct loop in LangGraph.

Graph shape:

    START -> reasoning -> [conditional] -> action -> reasoning -> ... -> END

- `reasoning` is the "Reason" step: the LLM decides whether it has enough
  information to answer, or needs to call a tool.
- `action` is the "Act" step: it executes the tool call the LLM requested.
- The edge from `action` always points back to `reasoning`, never to END.
  This is what forces an Observation after every Action: the tool's result
  is appended to the message list as a ToolMessage, and the LLM is required
  to look at it before it's allowed to produce a final answer. Without this,
  the agent could "act" blindly and never actually use what it found.
"""

from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import get_weather

load_dotenv()

TOOLS = [get_weather]

llm = ChatAnthropic(model="claude-sonnet-5").bind_tools(TOOLS)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def reasoning_node(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


action_node = ToolNode(TOOLS)


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "action"
    return END


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("action", action_node)

    graph.add_edge(START, "reasoning")
    graph.add_conditional_edges("reasoning", should_continue, {"action": "action", END: END})
    graph.add_edge("action", "reasoning")  # force observation before the next decision

    return graph.compile()
