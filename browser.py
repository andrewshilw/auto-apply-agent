"""Shared agent-browser (https://agent-browser.dev) primitives.

Extracted from `linkedin_tool.py` (Week 2) so the Week 4 form-filling tool
can drive a real, visible browser window the same way without duplicating
the subprocess/snapshot-parsing plumbing. Every function takes an explicit
`session` name rather than assuming a single global session, since
different tools use different agent-browser sessions (LinkedIn needs a
persistent logged-in session; ATS application forms generally don't).
"""

import itertools
import re
import shutil
import subprocess
import time
from pathlib import Path

AGENT_BROWSER = shutil.which("agent-browser")
SCREENSHOT_DIR = Path(__file__).parent / "screenshots"
FOCUS_PAUSE_SECONDS = 0.6

ROLE_NAME_RE = re.compile(r'-\s*(?P<role>[A-Za-z]+)\s+"(?P<name>[^"]*)"')
REF_RE = re.compile(r"ref=(e\d+)")
URL_RE = re.compile(r"url=([^\],]+)")

_screenshot_counter = itertools.count(1)


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


def focus(session: str, ref: str, label: str) -> None:
    """Highlight the element the agent is about to act on (visible live in
    the headed window) and save a numbered screenshot of that moment."""
    run(session, "highlight", ref)
    time.sleep(FOCUS_PAUSE_SECONDS)
    SCREENSHOT_DIR.mkdir(exist_ok=True)
    safe_label = re.sub(r"[^\w-]+", "_", label).strip("_")[:60]
    shot_path = SCREENSHOT_DIR / f"{next(_screenshot_counter):03d}_{safe_label}.png"
    run(session, "screenshot", str(shot_path))
