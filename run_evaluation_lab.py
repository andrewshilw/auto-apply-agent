"""Week 3 lab: the automated evaluate-candidates flow.

Searches LinkedIn for a role, then for each candidate job opens its detail
page, extracts the description, scores it against the resume vector store,
and prints/logs an APPLY/SKIP decision with reasons for each one — no
application is ever submitted, this only produces recommendations.

Run directly so you can watch agent-browser move between job listings:

    python run_evaluation_lab.py

Needs LINKEDIN_EMAIL / LINKEDIN_PASSWORD set in .env, or you'll be prompted
to log in by hand once (see linkedin_tool.py). First run also structures
the resume (sample_data/sample_resume.pdf by default) and builds the
ChromaDB collection if they don't exist yet.
"""

from dotenv import load_dotenv

from evaluation import evaluate_and_decide, format_decision
from linkedin_tool import _ensure_logged_in, _search_jobs, get_job_description_text
from resume import build_structured_resume
from vector_store import build_resume_collection, get_resume_collection

SEARCH_TERM = "Software Engineer Intern"
RESULTS_WANTED = 5


def ensure_resume_indexed() -> None:
    if get_resume_collection().count() > 0:
        return
    resume = build_structured_resume()
    build_resume_collection(resume)


def run(search_term: str = SEARCH_TERM, results_wanted: int = RESULTS_WANTED) -> None:
    ensure_resume_indexed()

    _ensure_logged_in()
    jobs = _search_jobs(search_term, results_wanted)
    if not jobs:
        print(f"No results found for '{search_term}'.")
        return

    for title, url in jobs:
        print(f"\nEvaluating: {title}")
        jd_text = get_job_description_text(url)
        decision = evaluate_and_decide(title, url, jd_text)
        print(format_decision(title, decision))


if __name__ == "__main__":
    load_dotenv()
    run()
