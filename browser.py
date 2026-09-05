"""Shared agent-browser (https://agent-browser.dev) primitives.

Extracted from `linkedin_tool.py` (Week 2) so the Week 4 form-filling tool
can drive a real, visible browser window the same way without duplicating
the subprocess/snapshot-parsing plumbing. Every function takes an explicit
`session` name rather than assuming a single global session, since
different tools use different agent-browser sessions (LinkedIn needs a
persistent logged-in session; ATS application forms generally don't).

Week 6 added `humanize.py`-backed pacing to `focus()` (a curved mouse
approach + jittered pause instead of a straight teleport and a fixed sleep)
and to typing (`type_chunks`/`type_humanized`, a chunked non-uniform typing
cadence instead of one instant `fill`) — realism/robustness improvements,
not an attempt to defeat any platform's bot-detection.
"""

import itertools
import json
import random
import re
import shutil
import subprocess
import time
from pathlib import Path

import humanize

AGENT_BROWSER = shutil.which("agent-browser")
SCREENSHOT_DIR = Path(__file__).parent / "screenshots"
FOCUS_PAUSE_SECONDS = 0.6

ROLE_NAME_RE = re.compile(r'-\s*(?P<role>[A-Za-z]+)\s+"(?P<name>[^"]*)"')
REF_RE = re.compile(r"ref=(e\d+)")
URL_RE = re.compile(r"url=([^\],]+)")

_screenshot_counter = itertools.count(1)

# Week 6: best-effort belief about where the mouse last landed, per session —
# used only to give `_human_approach` a starting point for its curved path.
# Never authoritative (agent-browser exposes no "get current mouse position"
# command), so a wrong guess just means one path starts from a plausible
# nearby point instead of the exact real one; it never affects correctness of
# the click that follows, which always still targets `ref` directly.
_last_mouse_pos: dict[str, tuple[float, float]] = {}


def run(session: str, *args: str, timeout: int = 45) -> str:
    if AGENT_BROWSER is None:
        raise RuntimeError(
            "agent-browser CLI not found. Install it with `npm install -g "
            "agent-browser` then run `agent-browser install` to download Chrome."
        )
    result = subprocess.run(
        [AGENT_BROWSER, *args, "--session", session], capture_output=True, text=True, timeout=timeout
    )
    if result.returncode != 0:
        raise RuntimeError(f"agent-browser {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def open_url(session: str, url: str, headed: bool = True) -> None:
    args = ["open", url, "--restore"]
    if headed:
        args.append("--headed")
    run(session, *args)


def close_session(session: str) -> None:
    """Shut down the session's browser process. Idempotent — safe to call
    even if the session was never opened or was already closed, so callers
    can unconditionally clean up in a `finally` block. Without this, the
    session (and any tabs left open in it, e.g. from a previous run against
    a different job posting) stays alive in the background and resurfaces
    the next time the same session name is reused."""
    run(session, "close")


def current_url(session: str) -> str:
    return run(session, "get", "url").strip()


def parse_snapshot(text: str) -> list[dict]:
    elements = []
    for line in text.splitlines():
        role_match = ROLE_NAME_RE.search(line)
        ref_match = REF_RE.search(line)
        if not role_match or not ref_match:
            continue
        url_match = URL_RE.search(line)
        elements.append(
            {
                "role": role_match.group("role"),
                "name": role_match.group("name"),
                "ref": ref_match.group(1),
                "url": url_match.group(1) if url_match else None,
                # ARIA marks the current pick with "[selected, ...]" on re-open
                # regardless of how a custom widget renders its own closed-state
                # display — see `form_fill._select_custom_option`.
                "selected": "[selected" in line,
            }
        )
    return elements


def snapshot(session: str, urls: bool = False) -> list[dict]:
    args = ["snapshot", "-i"] + (["--urls"] if urls else [])
    return parse_snapshot(run(session, *args))


def find_ref(elements: list[dict], role: str, name_contains: str) -> str:
    for el in elements:
        if el["role"] == role and name_contains.lower() in el["name"].lower():
            return el["ref"]
    raise RuntimeError(f'Could not find a {role} containing "{name_contains}" on the page.')


def options_for(elements: list[dict], combobox_ref: str) -> list[dict]:
    """The option elements belonging to one combobox, read off a single
    `snapshot()` call. agent-browser nests a dropdown's `option` children
    directly under it in the snapshot text (indented), but `parse_snapshot`
    flattens indentation away and keeps only document order — so a
    combobox's own options are exactly the run of consecutive `option`-role
    elements that appear after it, before the next real form field. This is
    true both for a native `<select>` (its options are present in every
    snapshot, no interaction needed) and for a custom listbox widget (only
    true right after the combobox has been clicked open — see
    `form_fill.fill_dropdowns_node`, which is why this always takes a
    freshly-taken `elements` list rather than a cached one).

    Real ATS widgets (confirmed on a live Greenhouse posting) put a chunk of
    the combobox's own chrome between it and its options — e.g. a "Toggle
    flyout" button that's present whether the dropdown is open or not, then
    an (often unlabeled, so nameless in the snapshot) `listbox` wrapper —
    rather than the options sitting immediately adjacent. So this skips over
    anything that isn't itself an option, stopping only once it either (a)
    hits a real option run and then that run ends, or (b) reaches the next
    textbox/combobox — a genuinely different field — without finding any,
    which correctly reports "no options" for a combobox that isn't really a
    dropdown (e.g. a type-to-search location field whose listbox opens empty
    until you type into it)."""
    start = next((i for i, el in enumerate(elements) if el["ref"] == combobox_ref), None)
    if start is None:
        return []
    options = []
    for el in elements[start + 1 :]:
        if el["role"] == "option":
            options.append(el)
        elif options:
            break  # a contiguous run of options just ended
        elif el["role"] in ("textbox", "combobox"):
            break  # reached the next real field without finding any options
        # else: skip this combobox's own chrome (toggle button, listbox
        # wrapper, "locate me"/attach buttons, ...) and keep looking
    return options


def _human_approach(session: str, ref: str) -> None:
    """Move the mouse toward `ref` along a short, curved multi-point path
    (`humanize.mouse_path`) instead of teleporting the cursor straight to it
    — purely cosmetic realism, so any failure here (unsupported `--json`
    flag, an off-screen ref, a box that can't be read) is swallowed rather
    than allowed to block the real action that follows. Falls back silently
    to no movement, same as before this existed."""
    try:
        # `--json` wraps the payload as {"success", "data", "error"} rather
        # than returning the bounding box bare — confirmed live against a
        # real agent-browser session, where reading box["x"] straight off
        # the top level silently KeyError'd into this function's own
        # except-pass on every single call.
        box = json.loads(run(session, "get", "box", ref, "--json"))["data"]
        target_x = box["x"] + box["width"] * random.uniform(0.35, 0.65)
        target_y = box["y"] + box["height"] * random.uniform(0.35, 0.65)
        start_x, start_y = _last_mouse_pos.get(
            session, (target_x + random.uniform(-120, 120), target_y + random.uniform(-90, 90))
        )
        for x, y in humanize.mouse_path(start_x, start_y, target_x, target_y):
            run(session, "mouse", "move", str(x), str(y))
        _last_mouse_pos[session] = (target_x, target_y)
    except Exception:
        pass


def focus(session: str, ref: str, label: str, capture: bool = True) -> None:
    """Highlight the element the agent is about to act on (visible live in
    the headed window), approach it with a human-like mouse path, and — for
    a decision-worthy action — save a numbered screenshot of that moment.
    `capture=False` (Week 6) skips the screenshot for a merely exploratory
    step that doesn't itself represent a filled/chosen/clicked value — e.g.
    opening a dropdown just to read what options it renders — since that
    step isn't what the shadow-click audit trail is meant to prove; halving
    screenshot volume on a dropdown-heavy form without losing the
    audit-worthy shot of the actual selection."""
    run(session, "highlight", ref)
    _human_approach(session, ref)
    time.sleep(humanize.human_delay(FOCUS_PAUSE_SECONDS))
    if not capture:
        return
    SCREENSHOT_DIR.mkdir(exist_ok=True)
    safe_label = re.sub(r"[^\w-]+", "_", label).strip("_")[:60]
    shot_path = SCREENSHOT_DIR / f"{next(_screenshot_counter):03d}_{safe_label}.png"
    run(session, "screenshot", str(shot_path))


def type_chunks(session: str, ref: str, text: str) -> None:
    """Type `text` into `ref` without clearing existing content first (same
    contract as agent-browser's own `type`), but as a burst-and-pause
    sequence of small chunks (`humanize.typing_chunks`) instead of one
    uniform-speed call — a non-linear typing cadence closer to how someone
    actually types. Used by `form_fill.py`'s `_search_autocomplete_options`
    for progressive-search fields."""
    for chunk, delay in humanize.typing_chunks(text):
        run(session, "type", ref, chunk)
        time.sleep(delay)


def type_humanized(session: str, ref: str, text: str) -> None:
    """Clear `ref` then type `text` with a humanized cadence (`type_chunks`)
    — the drop-in replacement for a plain `fill` call. Used by
    `form_fill.py`'s `_fill_and_verify`, which verifies the result
    afterward regardless of how the value got there."""
    run(session, "fill", ref, "")
    type_chunks(session, ref, text)
