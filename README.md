# Auto-Apply Agent — Week 1 Lab: Weather Agent

A terminal-based ReAct agent built with LangGraph, built as the Week 1 lab from
the 6-week roadmap. You type a message, an LLM-based router decides whether
you're asking about weather, asking to quit, or saying something else (which
is ignored), and weather questions are answered by reasoning about whether to
call a tool (Open-Meteo, no API key needed), observing the result, then
answering. The whole interactive session — including exiting — is driven by
the graph itself rather than a Python `while True` loop or hardcoded string
checks.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
```

## Run

```bash
source venv/bin/activate
python main.py
```

## Files

- `tools.py` — `get_weather` tool, backed by the free Open-Meteo geocoding +
  forecast APIs.
- `agent.py` — the LangGraph graph:
  - `input` reads one line from the terminal.
  - `classify` asks an LLM to route the message as `WEATHER` / `QUIT` /
    `OTHER` — this replaces hardcoded checks like
    `if city.lower() in {"quit", "exit"}` with a model decision. `QUIT`
    exits the graph, `OTHER` loops back to `input` without doing anything
    ("ignored"), and `WEATHER` proceeds to the ReAct loop.
  - `reasoning` (LLM decides whether to call a tool) and `action` (executes
    the tool call) are wired so `action` always routes back to `reasoning`
    before it's allowed to answer — that's what forces an Observation after
    every Action. See the module docstring in `agent.py` for more.
  - `respond` prints the final answer and loops back to `input` for the next
    turn.
- `main.py` — loads the system prompt and calls `graph.invoke()` once; the
  interactive loop and the exit condition both live in the graph, not in a
  Python `while True`.
