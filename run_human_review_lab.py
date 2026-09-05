"""Week 5 lab: human-in-the-loop exception handling for `fill_application_form`.

Run directly to try it against a local mock ATS page
(`sample_data/mock_human_review_application.html`) that deliberately
includes what the automated form-filler can't handle on its own: a
CAPTCHA-style checkbox and a free-text custom question ("Why do you love
Java?") that isn't in the candidate's profile:

    python run_human_review_lab.py

`form_fill.py`'s `human_review_node` raises a LangGraph `interrupt()` for
each one; `run_form_fill` (called by this script, same as
`run_form_fill_lab.py`) is what actually catches those and prompts for
input in the terminal — watch for it to pause and ask you to (1) solve the
checkbox in the visible browser window, then (2) type an answer to the
Java question, before it resumes and finishes the form.

Pass a real application URL instead to run against a live posting known to
have a custom question or CAPTCHA:

    python run_human_review_lab.py <job_application_url>
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

from form_fill import run_form_fill

DEFAULT_FIXTURE = Path(__file__).parent / "sample_data" / "mock_human_review_application.html"


def run(job_url: str) -> None:
    print(f"Filling application form at: {job_url}\n")
    # Week 6: close_when_done=False — leave the window open afterward for
    # inspection, same as run_form_fill_lab.py.
    print(run_form_fill(job_url, close_when_done=False))


if __name__ == "__main__":
    load_dotenv()
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FIXTURE.resolve().as_uri()
    run(url)
