"""Week 4 lab: end-to-end automated form filling on an ATS application page
(Greenhouse, Lever, or similar), including AI agent takeover for dropdown
fields (see `form_fill.py`'s module docstring for the full
identify/fill/fill_dropdowns/click_next graph).

Run directly to try it against a local mock ATS page
(`sample_data/mock_dropdown_application.html`) — built to exercise all three
dropdown shapes `fill_dropdowns_node` has to tell apart: two native
<select>s whose option text doesn't literally contain the profile's
"Yes"/"No" (only reading the option's actual meaning gets it right), and a
custom (non-native) listbox widget ("Gender") the applicant profile has no
field for at all, which should come back skipped rather than guessed at:

    python run_form_fill_lab.py

Pass a real Greenhouse/Lever application URL instead to run the same agent
against a live posting:

    python run_form_fill_lab.py <job_application_url>

Never submits — the final Submit control is only shadow-clicked (highlighted
+ screenshotted), never actually clicked.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

from form_fill import run_form_fill

DEFAULT_FIXTURE = Path(__file__).parent / "sample_data" / "mock_dropdown_application.html"


def run(job_url: str) -> None:
    print(f"Filling application form at: {job_url}\n")
    print(run_form_fill(job_url))


if __name__ == "__main__":
    load_dotenv()
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FIXTURE.resolve().as_uri()
    run(url)
