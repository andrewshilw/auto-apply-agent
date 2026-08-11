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
- `linkedin_job_search` (`linkedin_tool.py`) — logs into LinkedIn with a
  real, **visible** browser window (via the [agent-browser](https://agent-browser.dev)
  CLI, not a headless/terminal scrape) and searches LinkedIn's job board
  directly. This is the Week 2 lab: an agent that actually opens LinkedIn,
  logs in, and searches, rather than hitting a job-search API from the
  terminal. LinkedIn's User Agreement prohibits automated login/scraping
  and this can get an account banned — only use it with an account you're
  willing to risk. If your account logs in with a password, set
  `LINKEDIN_EMAIL` / `LINKEDIN_PASSWORD` in `.env` and the tool fills the
  form itself; for anything else (Google sign-in, 2FA, a security
  checkpoint), it pauses and asks you to finish logging in by hand in the
  visible window — either way this only has to happen once, since the
  session is saved (via agent-browser's `--session --restore`) and reused
  on later runs. Before every fill/click, it highlights the target element
  live in the browser window (agent-browser's `highlight` command draws a
  red box around it) so you can watch its "visual focus" move across the
  page in real time, and saves a numbered screenshot of each moment to
  `screenshots/` (gitignored) so the sequence can be replayed afterward.

## Setup

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
```

LangSmith tracing (optional): set `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`,
and `LANGCHAIN_PROJECT` in `.env`. LangGraph reports every node run to
LangSmith automatically when those are set — no code here talks to
LangSmith directly.

`linkedin_job_search` / `run_linkedin_lab.py` need the agent-browser CLI
(Node/npm, separate from the Python venv above), and `LINKEDIN_EMAIL` /
`LINKEDIN_PASSWORD` in `.env` if your account uses password login:

```bash
npm install -g agent-browser
agent-browser install   # downloads a Chrome build agent-browser drives
```

## Run

```bash
source venv/bin/activate
python main.py
```

Or run the LinkedIn lab directly (opens a real, visible browser window):

```bash
source venv/bin/activate
python run_linkedin_lab.py
```

## Files

- `tools.py`, `linkedin_tool.py` — the two tools, see above.
- `run_linkedin_lab.py` — standalone script for the Week 2 lab: log into
  LinkedIn, search "Java Engineer", print the top 5 titles + links. Run it
  directly rather than through `main.py`'s chat loop.
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
