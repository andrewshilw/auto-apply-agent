# Auto-Apply Agent

A job-application agent built with LangGraph, gaining tools over time. An
earlier, weather-only version is preserved as-is on the `week1-lab` git
branch.

A terminal-based ReAct agent: you type a message, an LLM-based router
decides whether you want the agent to act, quit, or are just saying
something unrelated (ignored), and requests are handled by reasoning about
whether to call a tool, observing the result, then answering. The whole
interactive session — including exiting — is driven by the graph itself
rather than a Python `while True` loop or hardcoded string checks. See the
module docstring in `agent.py` for the full graph shape.

## Tools

- `get_weather` (`tools.py`) — current weather for a city, via the free
  Open-Meteo geocoding + forecast APIs.
- `search_jobs` (`jobs_tool.py`) — searches LinkedIn + Indeed's public job
  search via [JobSpy](https://github.com/speedyapply/JobSpy); no login
  required.
- `browse_page` (`browser_tools.py`) — opens a URL with Playwright, returns
  a simplified accessibility-tree snapshot (`Locator.aria_snapshot()`)
  instead of raw HTML, and saves a screenshot to `screenshots/` (gitignored)
  for visual debugging. Expect some sites to return a Cloudflare/bot-block
  page instead of their real content — that's the intended behavior to
  observe, not a bug to route around; the agent reports what it saw rather
  than trying to defeat the block.

## Setup

Needs Python 3.10+ (`python-jobspy` requires it) — use `python3.12` or
newer, not an older system `python3`.

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
```

LangSmith tracing (optional): set `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`,
and `LANGCHAIN_PROJECT` in `.env`. LangGraph reports every node run to
LangSmith automatically when those are set — no code here talks to
LangSmith directly.

## Run

```bash
source venv/bin/activate
python main.py
```

## Files

- `tools.py`, `jobs_tool.py`, `browser_tools.py` — the three tools, see
  above.
- `agent.py` — the LangGraph graph:
  - `input` reads one line from the terminal.
  - `classify` asks an LLM to route the message as `ACT` / `QUIT` / `OTHER`.
    `QUIT` exits the graph, `OTHER` loops back to `input` without doing
    anything ("ignored"), and `ACT` proceeds to the ReAct loop. `ACT` is
    domain-agnostic on purpose: which tool to use is `reasoning`'s job, not
    the router's, so adding a tool never means updating the router.
  - `reasoning` (LLM decides whether to call a tool) and `action` (executes
    the tool call) are wired so `action` always routes back to `reasoning`
    before it's allowed to answer — that's what forces an Observation after
    every Action.
  - `respond` prints the final answer and loops back to `input` for the next
    turn.
- `main.py` — loads the system prompt and calls `graph.invoke()` once; the
  interactive loop and the exit condition both live in the graph, not in a
  Python `while True`.
