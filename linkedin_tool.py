"""LinkedIn job search tool backed by agent-browser (https://agent-browser.dev).

This is the Week 2 lab: an agent that actually opens LinkedIn in a real,
visible browser, logs in, and searches — not a terminal-only API scrape.
LinkedIn's User Agreement prohibits automated login/scraping and actively
bans accounts for it; using this tool is a conscious, informed choice to do
it anyway, and it carries real account-ban risk.

agent-browser is a CLI, not a Python library, so this shells out to it and
parses its text snapshot output (`role "name" [ref=eN, url=...]` per line)
instead of using an SDK. `--headed` keeps the browser window visible so you
can watch the agent operate.

Login: if LINKEDIN_EMAIL/LINKEDIN_PASSWORD are set in .env *and* the login
page shows classic email/password fields, this fills and submits them. Many
accounts (e.g. "Continue with Google") have no password to script, and even
password-based accounts can hit a security checkpoint mid-login — in either
case this pauses and asks you to finish logging in yourself in the visible
browser window, rather than trying to script Google's OAuth flow or get past
a bot check. Either way, login only has to happen once: it runs inside a
named, `--restore`-d agent-browser session, so cookies persist across runs in
`~/.agent-browser/sessions/` and later runs skip straight to being logged in.
"""

import os
import re
import shutil
import subprocess
import time
import urllib.parse

from langchain_core.tools import tool

AGENT_BROWSER = shutil.which("agent-browser")
SESSION_NAME = "linkedin-lab"
LOGIN_URL = "https://www.linkedin.com/login"
FEED_URL = "https://www.linkedin.com/feed/"
JOBS_URL = "https://www.linkedin.com/jobs/search/"

ROLE_NAME_RE = re.compile(r'-\s*(?P<role>[A-Za-z]+)\s+"(?P<name>[^"]*)"')
REF_RE = re.compile(r"ref=(e\d+)")
URL_RE = re.compile(r"url=([^\],]+)")
DISMISS_RE = re.compile(r"Dismiss (.+?) job\b")
JOB_ID_RE = re.compile(r"currentJobId=(\d+)")


def _run(*args: str, timeout: int = 45) -> str:
    if AGENT_BROWSER is None:
        raise RuntimeError(
            "agent-browser CLI not found. Install it with `npm install -g "
            "agent-browser` then run `agent-browser install` to download Chrome."
        )
    result = subprocess.run(
        [AGENT_BROWSER, *args, "--session", SESSION_NAME], capture_output=True, text=True, timeout=timeout
    )
    if result.returncode != 0:
        raise RuntimeError(f"agent-browser {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def _open(url: str, headed: bool = True) -> None:
    args = ["open", url, "--restore"]
    if headed:
        args.append("--headed")
    _run(*args)


def _current_url() -> str:
    return _run("get", "url").strip()


def _parse_snapshot(text: str) -> list[dict]:
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


def _snapshot(urls: bool = False) -> list[dict]:
    args = ["snapshot", "-i"] + (["--urls"] if urls else [])
    return _parse_snapshot(_run(*args))


def _find_ref(elements: list[dict], role: str, name_contains: str) -> str:
    for el in elements:
        if el["role"] == role and name_contains.lower() in el["name"].lower():
            return el["ref"]
    raise RuntimeError(f'Could not find a {role} containing "{name_contains}" on the page.')


def _is_logged_in() -> bool:
    return "/feed" in _current_url()


def _try_password_login(email: str, password: str) -> bool:
    """Fill and submit the classic email/password form if it's present.
    Returns False (without raising) if those fields aren't on the page, e.g.
    an account that only offers "Continue with Google"."""
    elements = _snapshot()
    has_password_field = any(el["role"] == "textbox" and "password" in el["name"].lower() for el in elements)
    if not has_password_field:
        return False

    email_ref = _find_ref(elements, "textbox", "email")
    password_ref = _find_ref(elements, "textbox", "password")
    signin_ref = _find_ref(elements, "button", "sign in")

    _run("fill", email_ref, email)
    _run("fill", password_ref, password)
    _run("click", signin_ref)
    time.sleep(3)
    return True


def _ensure_logged_in() -> None:
    _open(FEED_URL)
    _run("wait", "--load", "domcontentloaded")
    time.sleep(1)
    if _is_logged_in():
        return  # restored from a prior run's saved session

    _open(LOGIN_URL)
    _run("wait", "--load", "domcontentloaded")

    email = os.environ.get("LINKEDIN_EMAIL")
    password = os.environ.get("LINKEDIN_PASSWORD")
    attempted_password_login = bool(email and password) and _try_password_login(email, password)

    if not attempted_password_login or not _is_logged_in():
        input(
            "\nLog into LinkedIn yourself in the visible browser window "
            "(Google sign-in, password, 2FA, security checkpoint — whatever "
            "your account needs), then press Enter here once you're on your "
            "feed..."
        )
        time.sleep(1)

    if not _is_logged_in():
        raise RuntimeError(
            "Still not logged in (current URL: "
            f"{_current_url()}). Run again once you've completed login in the browser window."
        )


def _extract_job_cards(elements: list[dict]) -> list[dict]:
    """Each result is a card rendered as a button whose accessible name is
    the whole card's text, including a "Dismiss <title> job" fragment from a
    nested (do-not-click) dismiss button. Skip the nested button itself
    (name is *exactly* "Dismiss <title> job") and keep only the outer card
    button, using the same fragment to pull out a clean title."""
    cards = []
    for el in elements:
        if el["role"] != "button":
            continue
        match = DISMISS_RE.search(el["name"])
        if not match:
            continue
        title = match.group(1).strip()
        if el["name"].strip() == f"Dismiss {title} job":
            continue
        cards.append({"ref": el["ref"], "title": title})
    return cards


def _search_jobs(search_term: str, results_wanted: int) -> list[tuple[str, str]]:
    query = urllib.parse.urlencode({"keywords": search_term})
    _open(f"{JOBS_URL}?{query}")
    _run("wait", "--load", "domcontentloaded")
    time.sleep(2)

    cards = _extract_job_cards(_snapshot())
    jobs = []
    seen_ids = set()
    for card in cards:
        if len(jobs) >= results_wanted:
            break
        _run("click", card["ref"])
        time.sleep(1)
        current_url = _run("eval", "location.href")
        match = JOB_ID_RE.search(current_url)
        if not match or match.group(1) in seen_ids:
            continue
        seen_ids.add(match.group(1))
        jobs.append((card["title"], f"https://www.linkedin.com/jobs/view/{match.group(1)}/"))
    return jobs


def search_linkedin_jobs(search_term: str, results_wanted: int = 5) -> str:
    """Log into LinkedIn with a real, visible browser and search its job
    board, returning the title and link for each result. LINKEDIN_EMAIL /
    LINKEDIN_PASSWORD in .env are used if the account supports password
    login; otherwise (e.g. "Continue with Google") this pauses for you to
    log in by hand once, then reuses that session on later runs."""
    _ensure_logged_in()
    jobs = _search_jobs(search_term, results_wanted)
    if not jobs:
        return f"No results found for '{search_term}' on LinkedIn (or the page layout changed)."

    return "\n".join(f"{i}. {title} — {url}" for i, (title, url) in enumerate(jobs, start=1))


@tool
def linkedin_job_search(search_term: str, results_wanted: int = 5) -> str:
    """Log into LinkedIn with a real, visible browser (agent-browser) and
    search LinkedIn's job board directly, returning the title and link for
    each result. First run may pause in the terminal asking you to log in by
    hand in the browser window (e.g. Google sign-in); later runs reuse that
    session. This drives an actual LinkedIn login and logged-in session,
    which LinkedIn's User Agreement prohibits automating — prefer
    search_jobs (JobSpy, no login) unless you specifically need this."""
    return search_linkedin_jobs(search_term, results_wanted)
