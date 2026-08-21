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
- `evaluate_job_listing` (`evaluation.py`) — the Week 3 lab: gives the
  agent "judgment" instead of blindly applying everywhere. Opens a job's
  detail page (`get_job_description_text` in `linkedin_tool.py`, via
  agent-browser's `read` so it isn't tied to LinkedIn's CSS), retrieves the
  resume chunks most relevant to that description from a ChromaDB
  collection (`vector_store.py`), and asks the LLM to score the match
  0-100 with concrete reasons (`JobEvaluation` in `evaluation.py`). A score
  **> 80** recommends APPLY, otherwise SKIP (`decide()`); either way this
  only logs the decision + reasons to `logs/evaluations.jsonl` and returns
  them — **it never clicks Apply or submits anything**, applying for real
  is a separate, explicit decision left to a human. `reasoning` decides
  *whether* to call this tool same as any other, but unlike `get_weather` /
  `linkedin_job_search` its execution is routed to a **dedicated
  `evaluate` graph node** (`agent.py`) rather than the generic `action`
  ToolNode: the score has to steer the graph itself (a real conditional
  edge to an `apply_action` or `fallback` node — the ">80 triggers Apply,
  otherwise auto-fallback" logic), not just come back as text for the LLM
  to interpret. See the graph shape in `agent.py`'s module docstring.
- `fill_application_form` (`form_fill.py`) — the Week 4 lab: automated ATS
  form filling. Opens a job application page (Greenhouse, Lever, or
  similar) in the same real, visible agent-browser window and runs its own
  self-contained LangGraph cycle — `identify` (snapshot the page's fields)
  `-> fill -> click_next -> identify` — until either a "Submit" control is
  found or there's no "Next"/"Continue" left. If the page has no fillable
  fields yet (Lever puts the job description and the actual form behind a
  separate "Apply for this job" link), `click_next` clicks that entry link
  for real too — it's pure navigation, and gated on zero fields being
  present so it can never be confused with the final Submit. **Semantic
  mapping** (`map_field_to_key`) matches each field's accessible-name label
  against keyword patterns (e.g. "Phone Number *" -> `phone_number`) to
  decide which key of the candidate's `ApplicantProfile` fills it; anything
  with no confident match (free-text essay questions, file uploads, a
  combined "Full name" field) is left alone and reported as skipped rather
  than guessed at. Every fill is verified, not just fired-and-forgotten:
  `_fill_and_verify` reads the value back *after forcing a blur*, because
  testing against a real Greenhouse posting turned up autocomplete-style
  fields (e.g. a Google-Places-style "Location" box) that accept a typed
  value right up until focus moves away, then silently clear themselves
  since no dropdown suggestion was ever selected — checking immediately
  after `fill` alone missed that and would have falsely reported success.
  Clicking "Next" is a real click (it's just page navigation), but the
  final Submit control is only **shadow-clicked** — highlighted and
  screenshotted to prove the targeting was accurate, never actually
  clicked — same "recommend, don't act" policy as `evaluate_job_listing`'s
  APPLY decision. Unlike `evaluate_job_listing`, its outcome doesn't need
  to steer the *main* agent graph (only its own internal loop), so it's
  bound as an ordinary domain-agnostic tool in `agent.py`, same as
  `linkedin_job_search`.
- `browser.py` — the agent-browser primitives (`run`/`snapshot`/`focus`/
  etc.) shared by `linkedin_tool.py` and `form_fill.py`, parameterized by
  an explicit session name per tool (LinkedIn needs a persistent logged-in
  session; ATS forms generally don't).
- `run_form_fill_lab.py` — standalone script for the Week 4 lab: fill out a
  real Greenhouse/Lever application form end-to-end (`python
  run_form_fill_lab.py <job_application_url>`) using the synthetic
  `sample_data/sample_applicant_profile.json`, and print a summary of what
  was filled, skipped, and shadow-clicked. Never submits.

### Resume structuring + vector store

- `resume.py` — parses a PDF resume with `pypdf`, then asks the LLM
  (`ChatAnthropic(...).with_structured_output(Resume)`) to turn the raw
  text into a structured `Resume` (skills, education, experience,
  projects). Cached to `resume_data/resume.json` (gitignored) so this only
  runs once per resume.
- `vector_store.py` — flattens the structured resume into one chunk per
  skill/education/experience/project entry and embeds them into a
  ChromaDB collection persisted at `chroma_db/` (gitignored), using
  Chroma's bundled local embedding model (no embeddings API key needed).
  `retrieve_relevant_resume_chunks(jd_text)` is the retrieval step the
  evaluation node uses instead of stuffing the whole resume into every
  prompt.
- By default this all runs against `sample_data/sample_resume.pdf`, a
  **synthetic** resume (fake name, fake companies) generated by
  `scripts/generate_sample_resume.py` — nobody's real data. To evaluate
  against your own resume, drop a PDF anywhere (e.g. `resume.pdf` at the
  repo root — `*.pdf` outside `sample_data/` is gitignored so it's never
  committed by accident), delete `resume_data/` and `chroma_db/` so they
  rebuild, and pass its path to `build_structured_resume(pdf_path=...)`.
- `run_evaluation_lab.py` — standalone script for the Week 3 experiment:
  search LinkedIn for a role, then for every candidate job open its detail
  page, extract the description, and print an APPLY/SKIP decision with
  reasons for each one. Run it directly (like `run_linkedin_lab.py`)
  rather than through `main.py`'s chat loop.

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

Or run the Week 3 evaluation lab (search + score every candidate job):

```bash
source venv/bin/activate
python run_evaluation_lab.py
```

Or run the Week 4 form-filling lab against a real Greenhouse/Lever
application page (never submits):

```bash
source venv/bin/activate
python run_form_fill_lab.py <job_application_url>
```

## Files

- `tools.py`, `linkedin_tool.py`, `evaluation.py`, `form_fill.py` — the four
  tools, see above.
- `browser.py` — agent-browser primitives shared by `linkedin_tool.py` and
  `form_fill.py`, see above.
- `resume.py`, `vector_store.py` — resume PDF parsing/structuring and the
  ChromaDB resume vector store, see "Resume structuring + vector store"
  above.
- `scripts/generate_sample_resume.py` — generates the synthetic
  `sample_data/sample_resume.pdf` used by default.
- `sample_data/sample_applicant_profile.json` — synthetic applicant contact
  info (name, email, phone, links, etc.) `form_fill.py` fills ATS forms
  with by default.
- `run_linkedin_lab.py` — standalone script for the Week 2 lab: log into
  LinkedIn, search "Java Engineer", print the top 5 titles + links. Run it
  directly rather than through `main.py`'s chat loop.
- `run_evaluation_lab.py` — standalone script for the Week 3 lab: search
  LinkedIn, then score every candidate job against the resume vector store
  and print an APPLY/SKIP decision with reasons for each.
- `run_form_fill_lab.py` — standalone script for the Week 4 lab: fill out a
  real Greenhouse/Lever application form end-to-end and print what was
  filled, skipped, and shadow-clicked. Never submits.
- `agent.py` — the LangGraph graph:
  - `input` reads one line from the terminal.
  - `classify` asks an LLM to route the message as `ACT` / `QUIT` / `OTHER`.
    `QUIT` exits the graph, `OTHER` loops back to `input` without doing
    anything ("ignored"), and `ACT` proceeds to the ReAct loop. `ACT` is
    domain-agnostic on purpose: which tool to use is `reasoning`'s job, not
    the router's, so adding a tool never means updating the router.
  - `reasoning` (LLM decides whether to call a tool) routes to one of two
    "Act" steps depending on which tool was called: `action` for
    domain-agnostic tools (`get_weather`, `linkedin_job_search`,
    `fill_application_form`), or `evaluate` specifically for
    `evaluate_job_listing`.
  - `action` executes the tool call generically and always routes back to
    `reasoning` — that's what forces an Observation after every Action.
    `fill_application_form` runs its own self-contained LangGraph cycle
    inside that single tool call (see `form_fill.py`), since unlike
    `evaluate_job_listing` its outcome doesn't need to steer this graph.
  - `evaluate` (the Week 3 JD Evaluation Node) runs the JD-vs-resume
    scoring pipeline, then a real conditional edge — not just prompt text —
    routes to `apply_action` (score `> 80`) or `fallback` (otherwise) based
    on the decision, mirroring the ">80 triggers Apply, otherwise
    auto-fallback" logic as graph structure. Both routes are thin (they
    never click anything) and loop back to `reasoning`, same as `action`.
  - `respond` prints the final answer and loops back to `input` for the next
    turn.
- `main.py` — loads the system prompt and calls `graph.invoke()` once; the
  interactive loop and the exit condition both live in the graph, not in a
  Python `while True`.
