"""Week 4 lab: automated ATS form filling.

Given a job application page (Greenhouse, Lever, or similar), this drives
the same real, visible agent-browser window as `linkedin_tool.py` through a
LangGraph cycle:

    identify -> fill -> click_next -+-> identify (another page in a
                                     |             multi-step form)
                                     +-> END        (no Next button left, or
                                                      a Submit control was
                                                      found and shadow-clicked)

- `identify` snapshots the current page's interactive elements.
- `fill` maps each textbox/combobox's accessible-name label to a key in the
  candidate's `ApplicantProfile` (semantic mapping, see `map_field_to_key`)
  and fills it if there's a confident match and a non-empty profile value;
  anything else is recorded as skipped rather than guessed at.
- `click_next` looks for a "Next"/"Continue" control among this page's
  buttons and clicks it for real (that's just page navigation, not a
  submission) to loop back to `identify` for the next step. If instead it
  finds a "Submit"/"Apply" control, it performs a **shadow click**: highlight
  + screenshot the exact element (proving the field/button targeting was
  accurate) without ever actually clicking it. Same policy as
  `evaluation.py`'s APPLY recommendation — a real submission stays a
  separate, explicit decision left to a human.
"""

import json
import re
import time
from pathlib import Path
from typing import Literal, TypedDict

from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

import browser

SESSION_NAME = "form-fill-lab"
DEFAULT_PROFILE_PATH = Path(__file__).parent / "sample_data" / "sample_applicant_profile.json"
MAX_FORM_STEPS = 5

NEXT_BUTTON_PATTERNS = ["save and continue", "next", "continue"]
SUBMIT_BUTTON_PATTERNS = ["submit application", "submit", "review and submit", "apply now"]
# Some ATS platforms (Lever in particular) put the job description at the
# given URL and the actual application form behind a separate entry link —
# distinct from NEXT/SUBMIT because it's the *only* case clicked while zero
# fillable fields are present (see `click_next_node`), which is what keeps
# it from ever being confused with a same-labeled in-form submit control.
ENTRY_APPLY_PATTERNS = ["apply for this job", "apply for this position"]

FILLABLE_ROLES = {"textbox", "combobox"}
BUTTON_ROLES = {"button", "link"}

# Semantic mapping: ApplicantProfile key -> label keywords that identify it
# on an ATS form, most-specific phrase first so e.g. "last name" is matched
# before a hypothetical bare "name" pattern would be.
SEMANTIC_FIELD_MAP: list[tuple[str, list[str]]] = [
    ("first_name", ["first name", "given name"]),
    ("last_name", ["last name", "surname", "family name"]),
    ("email", ["email"]),
    ("phone_number", ["phone", "mobile"]),
    ("linkedin_url", ["linkedin"]),
    ("github_url", ["github"]),
    ("portfolio_url", ["portfolio", "personal site", "website"]),
    ("current_company", ["current company", "employer"]),
    ("current_title", ["current title", "job title", "current role"]),
    ("location", ["location", "city"]),
    ("work_authorized", ["authorized to work", "work authorization"]),
    ("requires_sponsorship", ["require sponsorship", "sponsorship"]),
]


class ApplicantProfile(BaseModel):
    first_name: str = ""
    last_name: str = ""
    email: str = ""
    phone_number: str = ""
    location: str = ""
    linkedin_url: str = ""
    github_url: str = ""
    portfolio_url: str = ""
    current_company: str = ""
    current_title: str = ""
    work_authorized: str = ""
    requires_sponsorship: str = ""


def load_profile(path: Path = DEFAULT_PROFILE_PATH) -> ApplicantProfile:
    return ApplicantProfile.model_validate(json.loads(path.read_text()))


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower())


def map_field_to_key(label: str) -> str | None:
    """Semantic mapping: match a form field's accessible-name label to the
    `ApplicantProfile` key it corresponds to, e.g. "Phone Number *" ->
    "phone_number". Returns None rather than guessing when nothing matches,
    so free-text/essay questions and file uploads are left alone."""
    normalized = _normalize(label)
    for key, patterns in SEMANTIC_FIELD_MAP:
        if any(pattern in normalized for pattern in patterns):
            return key
    return None


def _fill_and_verify(session: str, ref: str, value: str, retries: int = 1) -> bool:
    """Fill a field, force a blur, then read its value back to confirm it
    actually stuck. A clean exit code from the CLI `fill` call alone isn't
    proof the value persisted: on a real Greenhouse posting, a plain
    React-hydration race made one field revert immediately, and — more
    subtly — an autocomplete-style "Location" combobox accepted the typed
    value right up until focus moved away, then silently cleared itself
    because no suggestion had been explicitly selected. Checking only
    right after `fill` (no blur) missed that second case entirely; the
    `press Tab` here reproduces the blur that would happen naturally as
    the flow moves to the next field/button, so it surfaces the same
    silent-clear before it's mistaken for a successful fill."""
    for _ in range(retries + 1):
        browser.run(session, "fill", ref, value)
        time.sleep(0.3)
        browser.run(session, "press", "Tab")
        time.sleep(0.3)
        if browser.run(session, "get", "value", ref).strip() == value:
            return True
    return False


def _find_button(elements: list[dict], patterns: list[str]) -> dict | None:
    for el in elements:
        if el["role"] not in BUTTON_ROLES:
            continue
        normalized = _normalize(el["name"])
        if any(pattern in normalized for pattern in patterns):
            return el
    return None


class FormFillState(TypedDict):
    session: str
    profile: ApplicantProfile
    step: int
    max_steps: int
    elements: list[dict]
    filled: list[dict]
    skipped: list[dict]
    shadow_clicks: list[dict]
    advanced: bool


def identify_node(state: FormFillState) -> FormFillState:
    elements = browser.snapshot(state["session"])
    return {"elements": elements, "step": state["step"] + 1}


def fill_node(state: FormFillState) -> FormFillState:
    session = state["session"]
    profile = state["profile"]
    filled, skipped = [], []
    for el in state["elements"]:
        if el["role"] not in FILLABLE_ROLES:
            continue
        key = map_field_to_key(el["name"])
        value = getattr(profile, key) if key else ""
        if key and value:
            browser.focus(session, el["ref"], el["name"])
            if _fill_and_verify(session, el["ref"], value):
                filled.append({"label": el["name"], "key": key, "value": value})
            else:
                skipped.append(
                    {"label": el["name"], "role": el["role"], "reason": "fill did not persist — needs manual review"}
                )
        else:
            reason = "no semantic match" if not key else "no profile value for this field"
            skipped.append({"label": el["name"], "role": el["role"], "reason": reason})
    return {"filled": state["filled"] + filled, "skipped": state["skipped"] + skipped}


def click_next_node(state: FormFillState) -> FormFillState:
    session = state["session"]
    elements = state["elements"]

    next_button = _find_button(elements, NEXT_BUTTON_PATTERNS)
    if next_button:
        browser.focus(session, next_button["ref"], next_button["name"])
        browser.run(session, "click", next_button["ref"])
        return {"advanced": True}

    submit_button = _find_button(elements, SUBMIT_BUTTON_PATTERNS)
    if submit_button:
        # Shadow click: highlight + screenshot the exact Submit control to
        # prove click accuracy, but never actually click it.
        browser.focus(session, submit_button["ref"], f"SHADOW_{submit_button['name']}")
        shadow_clicks = state["shadow_clicks"] + [{"label": submit_button["name"]}]
        return {"advanced": False, "shadow_clicks": shadow_clicks}

    has_fillable_fields = any(el["role"] in FILLABLE_ROLES for el in elements)
    if not has_fillable_fields:
        # No form on this page at all yet (e.g. Lever's job description page)
        # — a real click, same as Next, since it's pure navigation to the
        # actual form rather than a submission of anything.
        entry_button = _find_button(elements, ENTRY_APPLY_PATTERNS)
        if entry_button:
            browser.focus(session, entry_button["ref"], entry_button["name"])
            browser.run(session, "click", entry_button["ref"])
            return {"advanced": True}

    return {"advanced": False}


def route_after_click_next(state: FormFillState) -> Literal["identify", "__end__"]:
    if state["advanced"] and state["step"] < state["max_steps"]:
        return "identify"
    return END


def build_form_fill_graph():
    graph = StateGraph(FormFillState)
    graph.add_node("identify", identify_node)
    graph.add_node("fill", fill_node)
    graph.add_node("click_next", click_next_node)

    graph.add_edge(START, "identify")
    graph.add_edge("identify", "fill")
    graph.add_edge("fill", "click_next")
    graph.add_conditional_edges("click_next", route_after_click_next, {"identify": "identify", END: END})

    return graph.compile()


def format_summary(state: FormFillState) -> str:
    lines = [f"Form-fill run: {state['step']} page(s) processed."]

    lines.append(f"Filled {len(state['filled'])} field(s):")
    lines += [f"  - {f['label']} -> {f['key']} = {f['value']}" for f in state["filled"]]

    if state["skipped"]:
        lines.append(f"Skipped {len(state['skipped'])} field(s):")
        lines += [f"  - {s['label']} ({s['role']}, {s['reason']})" for s in state["skipped"]]

    if state["shadow_clicks"]:
        lines.append("Shadow-clicked (highlighted only, NOT actually clicked):")
        lines += [f"  - {s['label']}" for s in state["shadow_clicks"]]
    else:
        lines.append("No Submit control reached — form may need more steps or manual review.")

    return "\n".join(lines)


def run_form_fill(job_url: str, profile: ApplicantProfile | None = None, max_steps: int = MAX_FORM_STEPS) -> str:
    """Full pipeline: open the application page, then run the
    identify/fill/click_next graph until it either shadow-clicks a Submit
    control or runs out of Next buttons/steps."""
    browser.open_url(SESSION_NAME, job_url)
    browser.run(SESSION_NAME, "wait", "--load", "domcontentloaded")
    time.sleep(1.5)  # let JS-embedded application widgets (e.g. Greenhouse's) finish mounting

    graph = build_form_fill_graph()
    final_state = graph.invoke(
        {
            "session": SESSION_NAME,
            "profile": profile or load_profile(),
            "step": 0,
            "max_steps": max_steps,
            "elements": [],
            "filled": [],
            "skipped": [],
            "shadow_clicks": [],
            "advanced": False,
        }
    )
    return format_summary(final_state)


@tool
def fill_application_form(job_url: str) -> str:
    """Open a job application form (e.g. Greenhouse or Lever) in a real,
    visible browser and fill in the fields it recognizes from the
    candidate's applicant profile via semantic field mapping, following
    "Next"/"Continue" through every step of a multi-page form. Never
    submits — the final Submit control is only highlighted (a "shadow
    click") to prove it was found accurately, never actually clicked."""
    return run_form_fill(job_url)
