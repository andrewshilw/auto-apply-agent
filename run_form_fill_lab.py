"""Week 4 lab: end-to-end automated form filling on a real ATS application
page (Greenhouse, Lever, or similar).

Run directly so you can watch agent-browser identify fields, fill them from
the sample applicant profile, and step through the form:

    python run_form_fill_lab.py <job_application_url>

Pass any live Greenhouse/Lever application URL (e.g. one found via
`run_linkedin_lab.py` or `run_evaluation_lab.py`). Never submits — the final
Submit control is only shadow-clicked (highlighted + screenshotted), never
actually clicked. See `form_fill.py` for the identify/fill/click_next graph
this drives.
"""

import sys

from dotenv import load_dotenv

from form_fill import run_form_fill


def run(job_url: str) -> None:
    print(f"Filling application form at: {job_url}\n")
    print(run_form_fill(job_url))


if __name__ == "__main__":
    load_dotenv()
    if len(sys.argv) < 2:
        sys.exit("Usage: python run_form_fill_lab.py <job_application_url>")
    run(sys.argv[1])
