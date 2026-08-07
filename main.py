"""Terminal Weather Agent — Week 1 lab."""

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage

from agent import build_graph

SYSTEM_PROMPT = (
    "You are a weather assistant. When the user asks about weather in a city, "
    "use the get_weather tool to look it up before answering. Always base your "
    "answer on the tool's observation rather than guessing."
)

# The interactive loop lives inside the graph itself (input -> ... -> input,
# with classify -> END as the only exit), so a single invoke() runs the whole
# session. LangGraph's default recursion_limit (25 node hops) is meant for
# bounded agent loops, not an open-ended terminal session, so raise it here.
RECURSION_LIMIT = 10_000


def main():
    load_dotenv()
    graph = build_graph()

    print("Weather Agent — ask about the weather, or say you want to quit.")
    graph.invoke(
        {"messages": [SystemMessage(content=SYSTEM_PROMPT)]},
        config={"recursion_limit": RECURSION_LIMIT},
    )


if __name__ == "__main__":
    main()
