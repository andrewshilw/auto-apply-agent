# Auto-Apply Agent — Week 1 Lab: Weather Agent

A terminal-based ReAct agent built with LangGraph, built as the Week 1 lab from
the 6-week roadmap. Given a city name, it reasons about whether it needs to
look up live weather, calls a tool to fetch it (Open-Meteo, no API key
needed), observes the result, then answers.

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
- `agent.py` — the LangGraph graph: a `reasoning` node (LLM decides whether to
  call a tool) and an `action` node (executes the tool call), wired so the
  action always routes back to reasoning before the loop can end. That edge
  is what forces an Observation after every Action — see the module
  docstring in `agent.py`.
- `main.py` — terminal loop that reads a city, invokes the graph, prints the
  final answer.
