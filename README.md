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
  `-> fill -> fill_dropdowns -> human_review -> click_next -> identify` —
  until either a "Submit" control is found or there's no "Next"/"Continue"
  left. If the page has no fillable fields yet (Lever puts the job
  description and the actual form behind a separate "Apply for this job"
  link), `click_next` clicks that entry link for real too — it's pure
  navigation, and gated on zero fields being present so it can never be
  confused with the final Submit. Either real click (Next/Continue or the
  entry link) waits for the new page to actually render before looping back
  to `identify` — confirmed live against a real Lever posting, snapshotting
  immediately after the click read back zero elements, well before the
  form had actually mounted. **Field identification is AI-driven, not a
  keyword table**:
  `identify` hands every fillable label on the page — plus the candidate's
  `ApplicantProfile` key names, nothing else — to an LLM
  (`map_fields_to_profile`), which decides which profile key (if any) each
  label corresponds to (e.g. "Phone Number *", "Mobile", or "Best contact
  number" all resolve to `phone_number`, without any `if "phone" in label`
  special-casing). A label the model can't confidently place — a free-text
  essay question, a file upload, a field the profile has no equivalent for
  — is left unmapped rather than guessed at. **Dropdowns get the same
  agent-takeover treatment one level further** (`fill_dropdowns_node`,
  `choose_dropdown_option`): Greenhouse's EEO-style dropdowns (work
  authorization, gender, race, veteran/disability status, ...) render
  wildly different *option* wording per posting — the profile might say
  `work_authorized: "Yes"` while the page's own option reads "I am
  authorized to work in the US without sponsorship" — so for every combobox
  the agent reads back whatever options *this* posting actually renders
  (opening it first if that's what reveals them — `browser.options_for`)
  and judges which one the profile supports, or declines rather than
  guessing (same never-guess contract, so demographic questions the profile
  has no field for come back skipped, not answered). Every fill is
  verified, not just fired-and-forgotten: `_fill_and_verify` reads a
  textbox's value back *after forcing a blur*, because testing against a
  real Greenhouse posting turned up autocomplete-style fields (e.g. a
  Google-Places-style "Location" box) that accept a typed value right up
  until focus moves away, then silently clear themselves since no dropdown
  suggestion was ever selected; dropdown picks get their own
  widget-specific verification (`_select_native_option` /
  `_select_custom_option`) since a real `<select>` and a custom
  react-select-style widget expose their current value completely
  differently. The click that picks a custom option is itself guarded
  (`_click_option`) with one retry against a freshly re-located ref before
  giving up — confirmed live on a real Greenhouse posting, where a
  ~250-option country-code picker could re-render during the LLM round-trip
  between reading its options and clicking one, invalidating the ref and
  raising outright instead of just failing verification; that retry is what
  turns it back into an ordinary "needs manual review" skip. Clicking
  "Next" is a real click (it's just page navigation), but the final Submit
  control is only **shadow-clicked** — highlighted and
  screenshotted to prove the targeting was accurate, never actually clicked
  — same "recommend, don't act" policy as `evaluate_job_listing`'s APPLY
  decision. **Exception handling is the Week 5 lab**: a CAPTCHA on the page
  (`_detect_captcha`, a name-based heuristic) or a free-text custom question
  the profile has no answer for (e.g. "Why do you love Java?" — flagged by
  the same field-identification call as `custom_question=true`, unlike a
  demographic dropdown question, which is still a deliberate skip) is
  escalated to a human instead of silently skipped. `human_review_node`
  raises this via LangGraph's dynamic `interrupt()` (not the static
  `interrupt_before` compile option, since which field needs a human is
  only known after reading the actual page) which suspends the whole graph
  — state and all — until `run_form_fill`'s CLI loop resumes it with
  `Command(resume=...)` after prompting in the terminal: solve the CAPTCHA
  yourself in the visible browser window and press Enter, or type an answer
  to fill in (blank to skip). See `run_human_review_lab.py` to try it
  end-to-end against a local mock page built for exactly this. Unlike
  `evaluate_job_listing`, this tool's outcome doesn't need to steer the
  *main* agent graph (only its own internal loop), so it's bound as an
  ordinary domain-agnostic tool in `agent.py`, same as `linkedin_job_search`.
- `browser.py` — the agent-browser primitives (`run`/`snapshot`/`focus`/
  `options_for`/etc.) shared by `linkedin_tool.py` and `form_fill.py`,
  parameterized by an explicit session name per tool (LinkedIn needs a
  persistent logged-in session; ATS forms generally don't).
- `run_form_fill_lab.py` — standalone script for the Week 4 lab. Run with no
  argument to fill out a local mock ATS page
  (`sample_data/mock_dropdown_application.html`, built to exercise the three
  dropdown shapes `fill_dropdowns_node` has to tell apart) using the
  synthetic `sample_data/sample_applicant_profile.json`, or pass a real
  Greenhouse/Lever application URL to run the same agent against a live
  posting (`python run_form_fill_lab.py <job_application_url>`). Prints a
  summary of what was filled, skipped, and shadow-clicked. Never submits.
- `run_human_review_lab.py` — standalone script for the Week 5 lab: same
  pipeline as `run_form_fill_lab.py`, pointed by default at
  `sample_data/mock_human_review_application.html`, a local mock ATS page
  built specifically to trigger both kinds of human-in-the-loop escalation
  — a mock CAPTCHA checkbox and a free-text "Why do you love Java?" custom
  question — so you can watch `form_fill.py`'s `human_review_node` pause
  the graph and prompt for input in the terminal, then resume and finish
  the form once you answer.

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

Or run the Week 4 form-filling lab — with no argument it fills out a local
mock ATS page; pass a real Greenhouse/Lever application URL to run against a
live posting instead (never submits either way):

```bash
source venv/bin/activate
python run_form_fill_lab.py [job_application_url]
```

Or run the Week 5 human-in-the-loop lab — a local mock page that deliberately
triggers a CAPTCHA pause and a custom-question pause, so you can see the
agent hand off to a human and resume:

```bash
source venv/bin/activate
python run_human_review_lab.py [job_application_url]
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
- `sample_data/mock_dropdown_application.html` — local mock ATS page used by
  `run_form_fill_lab.py`'s default (no-argument) run; built to exercise all
  three dropdown shapes `fill_dropdowns_node` has to tell apart without
  needing a live posting with matching fields on hand.
- `run_linkedin_lab.py` — standalone script for the Week 2 lab: log into
  LinkedIn, search "Java Engineer", print the top 5 titles + links. Run it
  directly rather than through `main.py`'s chat loop.
- `run_evaluation_lab.py` — standalone script for the Week 3 lab: search
  LinkedIn, then score every candidate job against the resume vector store
  and print an APPLY/SKIP decision with reasons for each.
- `run_form_fill_lab.py` — standalone script for the Week 4 lab: fill out an
  ATS application form end-to-end (a local mock page by default, or a real
  Greenhouse/Lever posting if a URL is passed) and print what was filled,
  skipped, and shadow-clicked. Never submits.
- `sample_data/mock_human_review_application.html` — local mock ATS page
  used by `run_human_review_lab.py`'s default (no-argument) run; includes a
  free-text custom question and a mock CAPTCHA checkbox specifically to
  exercise `human_review_node`'s two escalation paths.
- `run_human_review_lab.py` — standalone script for the Week 5 lab: same as
  `run_form_fill_lab.py`, but against a mock page built to trigger the
  human-in-the-loop pause for a CAPTCHA and a custom question.
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
