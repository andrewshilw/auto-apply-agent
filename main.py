"""Terminal agent — weather and LinkedIn job search tools."""

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage

from agent import build_graph

SYSTEM_PROMPT = (
    "You are a helpful assistant with two tools: get_weather (current "
    "weather for a city) and linkedin_job_search (logs into LinkedIn with a "
    "real, visible browser and searches its job board directly, returning "
    "titles and links). Use whichever tool fits the user's request and "
    "always base your answer on the tool's observation rather than guessing."
)

# The interactive loop lives inside the graph itself (input -> ... -> input,
# with classify -> END as the only exit), so a single invoke() runs the whole
# session. LangGraph's default recursion_limit (25 node hops) is meant for
# bounded agent loops, not an open-ended terminal session, so raise it here.
RECURSION_LIMIT = 10_000


def main():
    load_dotenv()
    graph = build_graph()

    print("Agent — ask about weather, job listings, or a web page, or say you want to quit.")
    graph.invoke(
        {"messages": [SystemMessage(content=SYSTEM_PROMPT)]},
        config={"recursion_limit": RECURSION_LIMIT},
    )


if __name__ == "__main__":
    main()
