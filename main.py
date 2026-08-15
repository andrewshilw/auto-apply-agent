"""Terminal agent — weather and LinkedIn job search tools."""

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage

from agent import build_graph
from resume import build_structured_resume
from vector_store import build_resume_collection, get_resume_collection

SYSTEM_PROMPT = (
    "You are a helpful assistant with three tools: get_weather (current "
    "weather for a city), linkedin_job_search (logs into LinkedIn with a "
    "real, visible browser and searches its job board directly, returning "
    "titles and links), and evaluate_job_listing (opens a LinkedIn job's "
    "detail page, scores it against the candidate's resume, and returns an "
    "APPLY/SKIP recommendation with reasons — never submits an application). "
    "Use whichever tool fits the user's request and always base your answer "
    "on the tool's observation rather than guessing."
)

# The interactive loop lives inside the graph itself (input -> ... -> input,
# with classify -> END as the only exit), so a single invoke() runs the whole
# session. LangGraph's default recursion_limit (25 node hops) is meant for
# bounded agent loops, not an open-ended terminal session, so raise it here.
RECURSION_LIMIT = 10_000


def ensure_resume_indexed() -> None:
    """Parse+structure the resume and index it into ChromaDB on first run;
    later runs reuse both the cached resume JSON and the persisted
    collection instead of redoing either."""
    if get_resume_collection().count() > 0:
        return
    resume = build_structured_resume()
    build_resume_collection(resume)


def main():
    load_dotenv()
    ensure_resume_indexed()
    graph = build_graph()

    print("Agent — ask about weather, job listings, or a web page, or say you want to quit.")
    graph.invoke(
        {"messages": [SystemMessage(content=SYSTEM_PROMPT)]},
        config={"recursion_limit": RECURSION_LIMIT},
    )


if __name__ == "__main__":
    main()
