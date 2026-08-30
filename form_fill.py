"""Week 4 lab: automated ATS form filling. Week 5 lab: human-in-the-loop
exception handling layered on top of it.

Given a job application page (Greenhouse, Lever, or similar), this drives
the same real, visible agent-browser window as `linkedin_tool.py` through a
LangGraph cycle:

    identify -> fill -> fill_dropdowns -> human_review -> click_next -+-> identify (another page
                                                                       |             in a multi-step
                                                                       |             form)
                                                                       +-> END        (no Next button
                                                                                       left, or a Submit
                                                                                       control was found
                                                                                       and shadow-clicked)

Week 5: exception handling. A field the agent can't confidently handle used
to just be recorded as skipped. Two cases are now escalated to a human
instead: a CAPTCHA on the page, and a free-text custom question the
profile has no answer for (e.g. "Why do you love Java?") — as opposed to,
say, a demographic dropdown question, which is still a deliberate skip
(see `choose_dropdown_option`'s never-guess contract), not something a
human should be asked to answer on the candidate's behalf. `human_review_node`
is where both get raised, via LangGraph's *dynamic* `interrupt()` rather
than the static `interrupt_before` compile-time option: which fields (if
any) need a human is only known after `identify`/`fill` have read *this*
page's actual content, not decidable ahead of time from a fixed node name.
`interrupt()` suspends the whole graph until `run_form_fill`'s CLI loop
resumes it with `Command(resume=...)` — see `human_review_node` and
`run_form_fill` for the mechanics.

- `identify` snapshots the current page's interactive elements, then hands
  every fillable field's label — plus the candidate's `ApplicantProfile`
  keys, nothing else — to an LLM (`map_fields_to_profile`) to decide which
  profile key (if any) each label corresponds to. This is the core Week 4
  task: field identification is a judgment call the agent makes by reading
  the label, *not* a keyword table (`if "phone" in label: ...`) — a
  Greenhouse posting can label the same field "Phone Number", "Mobile", or
  "Best contact number", and nothing here is allowed to special-case any of
  that by name. One batched call per page (not one call per field) since
  the mapping only needs the label text, never page interaction. A label
  the model can't confidently place — a free-text essay question, a resume
  upload, a field the profile has no equivalent for — is left unmapped
  rather than guessed at; the model is never told to invent a key that
  isn't actually one of the profile's fields, and a defensive check in
  `map_fields_to_profile` drops any key it returns that isn't in the
  profile, in case it tries anyway.
- `fill` fills every textbox whose label the AI mapped to a profile key
  that actually has a value; anything else is recorded as skipped.
- `fill_dropdowns` gives every combobox the same agent-takeover treatment,
  one level further: Greenhouse's EEO-style dropdowns (work authorization,
  gender, race, veteran/disability status, ...) render wildly different
  *option* wording per posting — the profile might say `work_authorized:
  "Yes"` while the page's own option reads "I am authorized to work in the
  US without sponsorship" — so knowing the right profile key isn't even
  enough; the agent also has to read this posting's actual option text and
  judge which one it means. For each combobox it reads back whatever
  options the page actually renders (opening it first if that's what
  reveals them — see `browser.options_for`), hands the label + those exact
  option strings + the profile to an LLM (`choose_dropdown_option`), and
  only acts on a confident match. Same never-guess contract as
  `map_fields_to_profile`: for something the profile has no real answer for
  (e.g. gender, disability status), the model is instructed to decline
  rather than pick something, which shows up as an ordinary skip.
- `human_review_node` (Week 5) escalates whatever `identify`/`fill` flagged
  instead of silently skipping it: a CAPTCHA (`_detect_captcha`, a
  name-based heuristic run in `identify_node`) or a free-text custom
  question the AI classified as human-answerable rather than an ordinary
  unmapped field (`custom_question` on `FieldMapping`, from the same
  `map_fields_to_profile` call that does field identification). Every
  `interrupt()` call in this node is resolved — and its answer cached —
  before any side-effecting browser call runs, since a resumed node
  re-executes from its own top; doing the actual `_fill_and_verify` calls
  only after the last interrupt returns keeps each one a one-time effect
  instead of replaying earlier fills on every subsequent resume.
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
import uuid
from pathlib import Path
from typing import Literal, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

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

# Name-based CAPTCHA detection (Week 5): every real CAPTCHA widget (Google
# reCAPTCHA, hCaptcha, a plain security checkbox, ...) gives itself away in
# its accessible name regardless of vendor, so this never needs a
# per-widget selector. A false negative just means the field is attempted
# and fails like anything else the agent can't handle; a false positive
# only costs one confirmation prompt — so this stays a broad heuristic.
CAPTCHA_PATTERNS = [
    "captcha",
    "recaptcha",
    "hcaptcha",
    "not a robot",
    "verify you are human",
    "prove you are human",
    "are you human",
    "security check",
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


class FormFillState(TypedDict):
    session: str
    profile: dict  # plain dict, not ApplicantProfile: passed straight into the AI prompts as JSON
    step: int
    max_steps: int
    elements: list[dict]
    field_mapping: dict[str, str | None]  # label -> ApplicantProfile key, decided by the AI mapper
    custom_questions: list[str]  # labels the AI flagged as human-answerable free-text questions
    captcha_label: str | None  # accessible name of a detected CAPTCHA element, if any
    needs_human: list[dict]  # skip-entries for custom_questions, pending human_review_node
    filled: list[dict]
    skipped: list[dict]
    shadow_clicks: list[dict]
    advanced: bool


# Field identification (Week 4's core task, see the module docstring): an
# LLM reads each on-page label and decides which ApplicantProfile key (if
# any) it corresponds to — no keyword table, no `if "phone" in label`.
FIELD_MAPPING_PROMPT = (
    "You are identifying job application form fields for a candidate. Below "
    "is the full list of profile keys you may use, and every fillable "
    "field's label as it actually reads on the page. For each label, decide "
    "which single profile key it asks for — e.g. a label like \"Phone "
    "Number *\", \"Mobile\", or \"Best contact number\" all ask for "
    "phone_number even though none of them contain that word.\n\n"
    "Leave a label unmapped (key=null) rather than guessing whenever: it's a "
    "free-text/essay question, a file upload, or anything the profile "
    "genuinely has no key for. Never invent a key that isn't in the list "
    "below, and never map two different labels to the same key unless they "
    "really are asking the same thing.\n\n"
    "For every label you leave unmapped, also decide custom_question: true "
    "only if it's a free-text question a human could meaningfully type an "
    "answer into directly (e.g. \"Why do you want to work here?\", \"Why do "
    "you love Java?\", \"Describe a challenging project\"). Set it false for "
    "anything a typed answer wouldn't make sense for (a file upload, a "
    "demographic/EEO question, etc.) — those stay ordinary skips, not "
    "something to hand to a human.\n\n"
    "Profile keys available: {keys}\n\n"
    "Field labels on this page:\n{labels}"
)


class FieldMapping(BaseModel):
    label: str = Field(description="Verbatim field label, copied exactly from the input list")
    key: str | None = Field(default=None, description="The one matching profile key, or null if nothing matches")
    custom_question: bool = Field(
        default=False,
        description=(
            "Only meaningful when key is null: true if this is a free-text question a human "
            "could answer by typing directly into this field, false for anything else"
        ),
    )


class FieldMappingResult(BaseModel):
    mappings: list[FieldMapping]


def map_fields_to_profile(labels: list[str], profile: dict) -> tuple[dict[str, str | None], list[str]]:
    """AI takeover for field identification: one batched call maps every
    fillable label on the current page to an `ApplicantProfile` key (or
    None), *and* — Week 5 — flags which of the unmapped ones are free-text
    custom questions a human should be asked to answer rather than an
    ordinary skip. Batched per page rather than per field since, unlike
    dropdown option text (`choose_dropdown_option`), this never needs to
    interact with the page — every label is already in hand from
    `identify_node`'s snapshot. Returns (field_mapping, custom_questions)."""
    if not labels:
        return {}, []
    llm = ChatAnthropic(model="claude-sonnet-5").with_structured_output(FieldMappingResult, method="json_schema")
    result = llm.invoke(
        FIELD_MAPPING_PROMPT.format(keys=", ".join(profile.keys()), labels="\n".join(f"- {label}" for label in labels))
    )
    assert isinstance(result, FieldMappingResult)
    proposed = {m.label: m.key for m in result.mappings}
    # Defensive, same contract as `choose_dropdown_option`: only trust a key
    # that's actually one of the profile's own fields, in case the model
    # invents one anyway despite the prompt telling it not to.
    field_mapping = {label: (proposed.get(label) if proposed.get(label) in profile else None) for label in labels}
    custom_questions = [m.label for m in result.mappings if field_mapping.get(m.label) is None and m.custom_question]
    return field_mapping, custom_questions


def _detect_captcha(elements: list[dict]) -> str | None:
    """Broad, name-based CAPTCHA heuristic — see `CAPTCHA_PATTERNS`."""
    for el in elements:
        normalized = _normalize(el["name"])
        if any(pattern in normalized for pattern in CAPTCHA_PATTERNS):
            return el["name"]
    return None


def identify_node(state: FormFillState) -> FormFillState:
    """Snapshot the page, then hand every fillable label on it to the AI
    field mapper (`map_fields_to_profile`) — see the module docstring for
    why that's the field-identification step, not `fill_node` itself.
    Covers both textboxes and comboboxes here (not just textboxes) since a
    combobox can turn out to be a free-text field in disguise (see
    `fill_dropdowns_node`'s fallback) and needs the same mapping ready."""
    elements = browser.snapshot(state["session"])
    labels = [el["name"] for el in elements if el["role"] in FILLABLE_ROLES]
    field_mapping, custom_questions = map_fields_to_profile(labels, state["profile"])
    return {
        "elements": elements,
        "field_mapping": field_mapping,
        "custom_questions": custom_questions,
        "captcha_label": _detect_captcha(elements),
        "step": state["step"] + 1,
    }


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


def _fill_text_field(
    session: str, profile: dict, el: dict, field_mapping: dict[str, str | None]
) -> tuple[dict | None, dict | None]:
    """Type-and-verify path for one textbox-like field, using the AI-decided
    `field_mapping` from `identify_node` rather than doing any field
    identification itself. Returns (filled_entry, None) or (None,
    skipped_entry). Shared by `fill_node` (plain textboxes) and
    `fill_dropdowns_node`'s fallback for a combobox that turns out not to be
    a real dropdown at all — e.g. Greenhouse's "Location" field is a
    free-text autocomplete that merely happens to carry combobox role;
    opening it reveals no options (see `browser.options_for`), so it's typed
    into exactly like a textbox."""
    key = field_mapping.get(el["name"])
    value = profile.get(key, "") if key else ""
    if key and value:
        browser.focus(session, el["ref"], el["name"])
        if _fill_and_verify(session, el["ref"], value):
            return {"label": el["name"], "key": key, "value": value}, None
        return None, {
            "label": el["name"],
            "role": el["role"],
            "ref": el["ref"],
            "reason": "fill did not persist — needs manual review",
        }
    reason = "no semantic match" if not key else "no profile value for this field"
    return None, {"label": el["name"], "role": el["role"], "ref": el["ref"], "reason": reason}


def fill_node(state: FormFillState) -> FormFillState:
    session = state["session"]
    profile = state["profile"]
    field_mapping = state["field_mapping"]
    custom_questions = state["custom_questions"]
    filled, skipped, needs_human = [], [], []
    for el in state["elements"]:
        if el["role"] != "textbox":
            continue
        f, s = _fill_text_field(session, profile, el, field_mapping)
        if f:
            filled.append(f)
        elif el["name"] in custom_questions:
            # Week 5: a custom question the AI declined to guess at isn't an
            # ordinary skip — it's handed to `human_review_node` instead.
            needs_human.append(s)
        else:
            skipped.append(s)
    return {
        "filled": state["filled"] + filled,
        "skipped": state["skipped"] + skipped,
        "needs_human": state["needs_human"] + needs_human,
    }


DROPDOWN_PROMPT = (
    "You are filling out one dropdown field on a job application for a candidate. "
    "Below is the field's label, the exact options this specific posting renders "
    "for it, and the candidate's profile. Pick the one option that correctly "
    "reflects the candidate, or decline if you can't.\n\n"
    "Option wording varies a lot between employers — a profile value of "
    '"Yes" might correspond to an option phrased "I am authorized to work in '
    'this country without sponsorship" rather than a bare "Yes" — so read '
    "the options' actual meaning rather than looking for a literal string match.\n\n"
    "Decline (matched=false) rather than guess whenever: nothing in the "
    "options is a confident, unambiguous match for the profile; or the "
    "question asks for information the profile doesn't contain at all — this "
    "includes demographic/EEO questions (gender, race/ethnicity, veteran "
    "status, disability status, and similar) that this profile has no field "
    "for. Never infer a demographic answer from a name, or from any other "
    "field.\n\n"
    "Field label: {label}\n"
    "Options on this page (if you pick one, `option_name` must be copied "
    "verbatim from this list):\n{options}\n\n"
    "Candidate profile:\n{profile}"
)


class DropdownChoice(BaseModel):
    matched: bool = Field(description="Whether a confident, safe match exists among the given options")
    option_name: str = Field(
        default="", description="Verbatim text of the chosen option, copied exactly from the options list"
    )
    reasoning: str = Field(default="", description="One sentence: why this option, or why none matched")


def choose_dropdown_option(label: str, options: list[str], profile: dict) -> DropdownChoice:
    """The AI takeover step for one dropdown: given this posting's actual
    option text (not a hardcoded list — see `browser.options_for`), decide
    which option (if any) the candidate's profile supports."""
    llm = ChatAnthropic(model="claude-sonnet-5").with_structured_output(DropdownChoice, method="json_schema")
    result = llm.invoke(
        DROPDOWN_PROMPT.format(
            label=label, options="\n".join(f"- {o}" for o in options), profile=json.dumps(profile, indent=2)
        )
    )
    assert isinstance(result, DropdownChoice)
    if result.matched and result.option_name not in options:
        # The model must choose verbatim from what's actually on the page —
        # never let a hallucinated option stand in as a "confident match".
        return DropdownChoice(matched=False, reasoning="model chose an option not present on the page")
    return result


def _select_native_option(session: str, select_ref: str, option_ref: str, option_name: str) -> bool:
    """Native <select>: agent-browser's `select` command opens it and picks
    by value-or-label internally — a real click on one of its option refs
    fails outright ("Could not compute box model"), since a native select's
    open popup isn't part of the page's DOM the way a custom widget's is.
    Verify by comparing the resulting `value` attribute against the chosen
    option's own `value` attribute (captured before selecting, since the
    option refs go stale the instant the page's accessibility tree changes)
    rather than reading the field's text back — `get text` on a <select>
    returns *every* option's text concatenated, not just the selected one,
    so it can't confirm which one actually took."""
    target_value = browser.run(session, "get", "attr", option_ref, "value").strip()
    browser.run(session, "select", select_ref, option_name)
    return browser.run(session, "get", "value", select_ref).strip() == target_value


def _click_option(session: str, ref: str, name: str) -> bool:
    """Focus + click one option element, returning False instead of raising
    if its ref has already gone stale by the time the click lands, rather
    than letting that crash the whole form-fill run over a single dropdown.
    Confirmed live on a real Greenhouse posting: a country-code picker's
    ~250-option list can re-render between when its options are read
    (`options_for`) and when one is actually clicked — `choose_dropdown_option`'s
    LLM round-trip sits in between — so a ref that was perfectly valid a
    moment earlier can fail outright with "Could not locate element"."""
    try:
        browser.focus(session, ref, name)
        browser.run(session, "click", ref)
        return True
    except RuntimeError:
        return False


def _select_custom_option(session: str, combobox_name: str, option_ref: str, option_name: str) -> bool:
    """Custom listbox widget (e.g. a styled EEO dropdown): its options are
    ordinary DOM elements once open, so — unlike a native <select> — this
    can click the option directly, same as any other on-page element.

    The click itself is guarded (`_click_option`) with one retry against a
    freshly re-located ref before giving up on it entirely — see there for
    why the original ref can go stale. Once the click actually lands,
    verification is layered three ways, since no single check held up
    against every real widget encountered on a live Greenhouse posting:

    1. Compare the combobox's own displayed text against `option_name` —
       works for most widgets, including this codebase's own mock fixture,
       where the closed state just shows the chosen label.
    2. Reopen and check whether the chosen option now carries ARIA's
       `[selected]` marker — catches widgets whose closed-state display is
       abbreviated instead of showing the full label (Greenhouse's
       react-select-based country-code picker shows a flag + "+1" rather
       than "United States +1", so #1 misses a real, successful pick there).
    3. Reopen and check whether the chosen option is simply *absent* from
       the list instead of marked selected — some widgets (confirmed live:
       Greenhouse's Yes/No visa-sponsorship question) drop the active
       selection from the reopened list entirely rather than flagging it,
       so its total absence — while the list is otherwise non-empty — is
       itself the signal.

    A widget that fails all three (Greenhouse's country picker turned out to
    be one, on top of #1) is a known, disclosed gap rather than a solved
    case; `ok=False` there is a false "needs manual review", not a false
    "filled correctly" — the failure mode stays on the safe side.
    `combobox_name` (not a ref) is what re-finds it across every check: the
    click that closed the dropdown already invalidated every ref from
    before it."""
    if not _click_option(session, option_ref, option_name):
        elements = browser.snapshot(session)
        combobox = next((e for e in elements if e["role"] == "combobox" and e["name"] == combobox_name), None)
        options = browser.options_for(elements, combobox["ref"]) if combobox else []
        retry = next((o for o in options if o["name"] == option_name), None)
        if retry is None or not _click_option(session, retry["ref"], option_name):
            return False
    time.sleep(0.3)

    elements = browser.snapshot(session)
    combobox = next((e for e in elements if e["role"] == "combobox" and e["name"] == combobox_name), None)
    if combobox is None:
        return False
    if browser.run(session, "get", "text", combobox["ref"]).strip() == option_name:
        return True

    browser.run(session, "click", combobox["ref"])
    time.sleep(0.3)
    elements = browser.snapshot(session)
    combobox = next((e for e in elements if e["role"] == "combobox" and e["name"] == combobox_name), None)
    reopened_options = browser.options_for(elements, combobox["ref"]) if combobox else []
    chosen = next((o for o in reopened_options if o["name"] == option_name), None)
    ok = chosen["selected"] if chosen is not None else bool(reopened_options)

    browser.run(session, "press", "Escape")  # leave it closed either way
    return ok


def fill_dropdowns_node(state: FormFillState) -> FormFillState:
    """AI takeover for combobox/dropdown fields — see the module docstring
    for why knowing the right profile key isn't enough here the way it is
    for `fill_node`'s plain textboxes.

    Re-snapshots fresh before handling *each* combobox rather than reusing
    `state["elements"]`: opening one custom dropdown changes the page's
    accessibility tree, which invalidates every ref from any earlier
    snapshot (documented agent-browser behavior, confirmed in practice —
    e.g. a native <select>'s option refs renumber after any unrelated
    click), so acting on a stale ref for the *next* combobox in this loop
    would be unreliable. Comboboxes are matched back up by accessible name
    across snapshots, which assumes distinct labels per page — reasonable
    for an ATS form, same assumption `_find_button` already makes for
    button text elsewhere in this module.
    """
    session = state["session"]
    profile = state["profile"]
    field_mapping = state["field_mapping"]
    filled, skipped = [], []

    combobox_names = [el["name"] for el in state["elements"] if el["role"] == "combobox"]

    for name in combobox_names:
        elements = browser.snapshot(session)
        el = next((e for e in elements if e["role"] == "combobox" and e["name"] == name), None)
        if el is None:
            continue  # page changed since identify(); nothing safe to act on

        options = browser.options_for(elements, el["ref"])
        is_native = bool(options)
        if not is_native:
            # Not present without interacting — open it for real and see what it reveals.
            browser.focus(session, el["ref"], el["name"])
            browser.run(session, "click", el["ref"])
            time.sleep(0.4)
            elements = browser.snapshot(session)
            el = next((e for e in elements if e["role"] == "combobox" and e["name"] == name), None)
            if el is None:
                continue  # opening it changed the page underneath us; nothing safe to act on
            options = browser.options_for(elements, el["ref"])
            if not options:
                # Genuinely not a dropdown — e.g. a free-text "Location" autocomplete
                # that merely has combobox role. Close it and fill it like a textbox.
                browser.run(session, "press", "Escape")
                f, s = _fill_text_field(session, profile, el, field_mapping)
                (filled if f else skipped).append(f or s)
                continue

        option_names = [o["name"] for o in options]
        choice = choose_dropdown_option(el["name"], option_names, profile)
        if not choice.matched:
            if not is_native:
                browser.run(session, "press", "Escape")
            skipped.append(
                {
                    "label": el["name"],
                    "role": el["role"],
                    "ref": el["ref"],
                    "reason": f"AI declined to guess a dropdown option ({choice.reasoning})",
                }
            )
            continue

        chosen = next(o for o in options if o["name"] == choice.option_name)
        if is_native:
            browser.focus(session, el["ref"], el["name"])
            ok = _select_native_option(session, el["ref"], chosen["ref"], chosen["name"])
        else:
            ok = _select_custom_option(session, el["name"], chosen["ref"], chosen["name"])

        if ok:
            filled.append({"label": el["name"], "key": "ai_dropdown", "value": chosen["name"]})
        else:
            skipped.append(
                {
                    "label": el["name"],
                    "role": el["role"],
                    "ref": el["ref"],
                    "reason": "dropdown selection did not persist — needs manual review",
                }
            )

    return {"filled": state["filled"] + filled, "skipped": state["skipped"] + skipped}


def human_review_node(state: FormFillState) -> FormFillState:
    """Week 5's human-in-the-loop pause. Escalates whatever `identify`/`fill`
    flagged instead of letting it fall through as a silent skip:

    1. A CAPTCHA (`captcha_label`, from `_detect_captcha`) — the agent can't
       solve it, so it just asks the human to solve it themselves in the
       visible browser window and confirm.
    2. Free-text custom questions (`needs_human`) — a "Why do you love
       Java?"-style question the profile has no answer for. The human types
       an answer (or leaves it blank to skip), and this node types it into
       the actual field.

    Each `interrupt()` call suspends the *entire graph*, not just this node,
    until `run_form_fill`'s CLI loop resumes it with `Command(resume=...)`.
    All interrupts are resolved — and their answers cached in `answers` —
    before any `_fill_and_verify` call runs: a resumed node re-executes from
    its own top, and every earlier `interrupt()` in that replay just returns
    its cached value instead of pausing again, so a fill that happened
    *before* a later interrupt in this same node would otherwise repeat on
    every subsequent resume. Doing the real fills only after the last
    interrupt has returned keeps each one a one-time effect.
    """
    if state["captcha_label"]:
        interrupt(
            {
                "type": "captcha",
                "label": state["captcha_label"],
                "message": (
                    f'CAPTCHA detected ("{state["captcha_label"]}"). Solve it yourself in the '
                    "visible browser window, then confirm to continue."
                ),
            }
        )

    answers: dict[str, str] = {}
    for item in state["needs_human"]:
        answer = interrupt(
            {
                "type": "custom_question",
                "label": item["label"],
                "message": (
                    f'The application asks: "{item["label"]}" — this isn\'t in the candidate\'s '
                    "profile. Type an answer to fill it in, or leave it blank to skip."
                ),
            }
        )
        answers[item["label"]] = (answer or "").strip()

    session = state["session"]
    filled, skipped = [], []
    for item in state["needs_human"]:
        answer = answers[item["label"]]
        if answer:
            browser.focus(session, item["ref"], item["label"])
        if answer and _fill_and_verify(session, item["ref"], answer):
            filled.append({"label": item["label"], "key": "human", "value": answer})
        else:
            reason = "left blank by human review" if not answer else "fill did not persist — needs manual review"
            skipped.append({**item, "reason": reason})

    return {
        "filled": state["filled"] + filled,
        "skipped": state["skipped"] + skipped,
        "needs_human": [],
        "captcha_label": None,
    }


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower())


def _find_button(elements: list[dict], patterns: list[str]) -> dict | None:
    for el in elements:
        if el["role"] not in BUTTON_ROLES:
            continue
        normalized = _normalize(el["name"])
        if any(pattern in normalized for pattern in patterns):
            return el
    return None


def _wait_for_navigation(session: str) -> None:
    """After a real click that navigates (Next/Continue, or the Lever-style
    entry link), give the new page a moment to actually render before the
    graph loops back to `identify_node` and snapshots it. Confirmed live
    against a real Lever posting: snapshotting immediately after the click
    (no wait at all) caught the page mid-navigation and read back zero
    elements, well before the ~2s it actually took the real form to mount."""
    browser.run(session, "wait", "--load", "domcontentloaded")
    time.sleep(1.5)


def click_next_node(state: FormFillState) -> FormFillState:
    session = state["session"]
    elements = state["elements"]

    next_button = _find_button(elements, NEXT_BUTTON_PATTERNS)
    if next_button:
        browser.focus(session, next_button["ref"], next_button["name"])
        browser.run(session, "click", next_button["ref"])
        _wait_for_navigation(session)
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
            _wait_for_navigation(session)
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
    graph.add_node("fill_dropdowns", fill_dropdowns_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("click_next", click_next_node)

    graph.add_edge(START, "identify")
    graph.add_edge("identify", "fill")
    graph.add_edge("fill", "fill_dropdowns")
    graph.add_edge("fill_dropdowns", "human_review")
    graph.add_edge("human_review", "click_next")
    graph.add_conditional_edges("click_next", route_after_click_next, {"identify": "identify", END: END})

    # A checkpointer is what makes `human_review_node`'s `interrupt()` calls
    # actually pause-and-resume (Week 5) rather than just raise: LangGraph
    # needs somewhere to persist the in-flight state between the `invoke()`
    # that hits the interrupt and the later one that resumes it with
    # `Command(resume=...)`. In-memory is enough here — each `run_form_fill`
    # call gets its own fresh thread id, so nothing needs to survive the
    # process.
    return graph.compile(checkpointer=MemorySaver())


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


def _prompt_human(payload: dict) -> str:
    """The Week 5 CLI human-takeover UI: knows nothing about LangGraph
    itself, only the plain dict `human_review_node` called `interrupt()`
    with. A CAPTCHA needs no typed answer, just a confirmation once it's
    solved by hand in the visible browser window; a custom question gets
    whatever the human types back (blank to skip it)."""
    print(f"\n[Human review needed] {payload['message']}")
    if payload["type"] == "captcha":
        input("Press Enter once you've solved it in the browser window... ")
        return "ok"
    return input("Your answer (blank to skip): ")


def run_form_fill(job_url: str, profile: ApplicantProfile | None = None, max_steps: int = MAX_FORM_STEPS) -> str:
    """Full pipeline: open the application page, then run the
    identify/fill/fill_dropdowns/human_review/click_next graph until it
    either shadow-clicks a Submit control or runs out of Next buttons/steps.
    Also owns the Week 5 human-in-the-loop resume loop: whenever
    `human_review_node` raises an interrupt (a CAPTCHA or a custom
    question), `graph.invoke()` returns with a `__interrupt__` key instead
    of running to completion — this loop prompts for an answer in the
    terminal (`_prompt_human`) and calls `graph.invoke(Command(resume=...))`
    to pick the graph back up exactly where it paused, repeating for as many
    interrupts as the page raises. Always closes the ATS session's browser
    before returning (success or error) — unlike the LinkedIn session, this
    one doesn't need to stay logged in between runs, and leaving it open
    would let a stale tab from this run resurface the next time the same
    session name is reused (see `browser.close_session`)."""
    try:
        browser.open_url(SESSION_NAME, job_url)
        browser.run(SESSION_NAME, "wait", "--load", "domcontentloaded")
        time.sleep(1.5)  # let JS-embedded application widgets (e.g. Greenhouse's) finish mounting

        graph = build_form_fill_graph()
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        state = graph.invoke(
            {
                "session": SESSION_NAME,
                "profile": (profile or load_profile()).model_dump(),
                "step": 0,
                "max_steps": max_steps,
                "elements": [],
                "field_mapping": {},
                "custom_questions": [],
                "captcha_label": None,
                "needs_human": [],
                "filled": [],
                "skipped": [],
                "shadow_clicks": [],
                "advanced": False,
            },
            config=config,
        )
        while "__interrupt__" in state:
            answer = _prompt_human(state["__interrupt__"][0].value)
            state = graph.invoke(Command(resume=answer), config=config)
        return format_summary(state)
    finally:
        browser.close_session(SESSION_NAME)


@tool
def fill_application_form(job_url: str) -> str:
    """Open a job application form (e.g. Greenhouse or Lever) in a real,
    visible browser and fill in the fields it recognizes from the
    candidate's applicant profile via AI-driven field identification,
    following "Next"/"Continue" through every step of a multi-page form. A
    dropdown with no clear match (e.g. a demographic/EEO question) is left
    skipped rather than guessed at. A CAPTCHA or a free-text custom
    question the profile has no answer for (e.g. "Why do you love Java?")
    instead pauses and asks for human input in the terminal — solve the
    CAPTCHA yourself in the visible browser window, or type an answer to
    fill in, then execution resumes automatically. Never submits — the
    final Submit control is only highlighted (a "shadow click") to prove it
    was found accurately, never actually clicked."""
    return run_form_fill(job_url)
