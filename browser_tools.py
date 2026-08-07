"""Browser control tool backed by Playwright.

This is the Week 2 "AOM Parsing" + "Visual Debugging" study material made
concrete as a tool the agent can call:

- AOM parsing: `Locator.aria_snapshot()` asks Chromium for the page's
  Accessibility Object Model — the same reduced tree screen readers use —
  instead of raw HTML. It comes back as a compact, readable outline of roles
  and names (e.g. `heading "Example Domain" [level=1]`), which is what makes
  it cheap for an LLM to reason over compared to parsing the DOM directly.
- Visual debugging: every call also saves a numbered screenshot to
  `screenshots/`, so a human can replay the agent's "visual focus" step by
  step after the run instead of needing a live stream.
"""

import itertools
from pathlib import Path

from langchain_core.tools import tool
from playwright.sync_api import sync_playwright

SCREENSHOT_DIR = Path(__file__).parent / "screenshots"
MAX_SNAPSHOT_CHARS = 4000

_screenshot_counter = itertools.count(1)


@tool
def browse_page(url: str) -> str:
    """Open a URL in a real browser, return a simplified accessibility tree
    of the page (roles + names, not raw HTML), and save a screenshot for
    visual debugging."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=15_000)
        except Exception as exc:
            browser.close()
            return f"Could not load {url}: {exc}"

        snapshot = page.locator("body").aria_snapshot()

        SCREENSHOT_DIR.mkdir(exist_ok=True)
        shot_path = SCREENSHOT_DIR / f"{next(_screenshot_counter):03d}.png"
        page.screenshot(path=str(shot_path))

        browser.close()

    if len(snapshot) > MAX_SNAPSHOT_CHARS:
        snapshot = snapshot[:MAX_SNAPSHOT_CHARS] + "\n... (truncated)"

    return (
        f"Loaded {url}\nScreenshot saved to {shot_path}\n\n"
        f"Accessibility tree:\n{snapshot}"
    )
