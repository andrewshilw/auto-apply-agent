"""Week 3 lab: the JD Evaluation Node.

Given a job description, retrieve the resume chunks most relevant to it
(RAG over the ChromaDB collection built in `vector_store.py`), ask the LLM
to score the match 0-100 with concrete reasons, then apply a threshold: a
score above `APPLY_THRESHOLD` recommends APPLY, otherwise SKIP. Neither
outcome clicks anything — this only logs a decision + reasons to
`logs/evaluations.jsonl` and returns them, so a human (or a future, more
autonomous version of this agent) decides what to do with the
recommendation.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from linkedin_tool import get_job_description_text
from vector_store import retrieve_relevant_resume_chunks

APPLY_THRESHOLD = 80
LOG_PATH = Path(__file__).parent / "logs" / "evaluations.jsonl"

EVAL_PROMPT = (
    "You are screening a candidate against a job description. Score the "
    "match from 0-100 based on how well the candidate's actual skills, "
    "projects, and experience (below) satisfy the job's stated "
    "requirements. Be specific — cite the resume entries and JD "
    "requirements you're comparing, not generic statements.\n\n"
    "## Candidate resume excerpts (retrieved as most relevant to this JD)\n"
    "{resume_context}\n\n"
    "## Job description\n"
    "{jd_text}"
)


class JobEvaluation(BaseModel):
    match_score: int = Field(description="0-100 fit score between the resume and the job description")
    reasons_for: list[str] = Field(
        default_factory=list, description="Concrete reasons the candidate fits, citing specific resume entries"
    )
    reasons_against: list[str] = Field(
        default_factory=list, description="Concrete gaps or mismatches between the resume and the JD"
    )


class Decision(BaseModel):
    recommendation: Literal["APPLY", "SKIP"]
    evaluation: JobEvaluation


def evaluate_job(jd_text: str, k: int = 8) -> JobEvaluation:
    """The Evaluation Node's core logic: retrieve relevant resume context,
    then score the JD against it."""
    resume_chunks = retrieve_relevant_resume_chunks(jd_text, k=k)
    resume_context = "\n".join(f"- {chunk}" for chunk in resume_chunks) or "(no resume data indexed)"
    # method="json_schema" (Claude's native structured-output feature) instead of the
    # default "function_calling" (forced tool-call parsing): on long, noisy real JD
    # text the default intermittently returned reasons_for/reasons_against as a raw
    # string instead of a list, failing Pydantic validation ~2/3 of the time in testing.
    llm = ChatAnthropic(model="claude-sonnet-5").with_structured_output(JobEvaluation, method="json_schema")
    result = llm.invoke(EVAL_PROMPT.format(resume_context=resume_context, jd_text=jd_text))
    assert isinstance(result, JobEvaluation)
    return result


def decide(evaluation: JobEvaluation, threshold: int = APPLY_THRESHOLD) -> Decision:
    recommendation = "APPLY" if evaluation.match_score > threshold else "SKIP"
    return Decision(recommendation=recommendation, evaluation=evaluation)


def log_decision(job_title: str, job_url: str, decision: Decision) -> None:
    """Background record of every decision made, independent of whatever
    the agent says in the chat transcript."""
    LOG_PATH.parent.mkdir(exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "job_title": job_title,
        "job_url": job_url,
        "match_score": decision.evaluation.match_score,
        "recommendation": decision.recommendation,
        "reasons_for": decision.evaluation.reasons_for,
        "reasons_against": decision.evaluation.reasons_against,
    }
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def format_decision(job_title: str, decision: Decision) -> str:
    header = f"[{decision.recommendation}] {job_title} — match score {decision.evaluation.match_score}/100"
    if decision.recommendation == "APPLY":
        label, reasons = "Reasons to apply", decision.evaluation.reasons_for
    else:
        label, reasons = "Reasons to skip", decision.evaluation.reasons_against
    lines = [header, f"{label}:"] + [f"  - {r}" for r in reasons]
    return "\n".join(lines)


def evaluate_and_decide(job_title: str, job_url: str, jd_text: str) -> Decision:
    decision = decide(evaluate_job(jd_text))
    log_decision(job_title, job_url, decision)
    return decision


def run_job_evaluation(job_title: str, job_url: str) -> tuple[str, Decision]:
    """Full pipeline for one job listing: open its detail page, extract the
    description, evaluate + decide + log. Shared by the `evaluate_job_listing`
    tool (used when a `ToolNode` executes it directly) and `agent.py`'s
    dedicated `evaluate` graph node (used when the ReAct loop routes there
    instead, so the APPLY/SKIP result can drive real conditional edges)."""
    jd_text = get_job_description_text(job_url)
    decision = evaluate_and_decide(job_title, job_url, jd_text)
    return format_decision(job_title, decision), decision


@tool
def evaluate_job_listing(job_title: str, job_url: str) -> str:
    """Open a LinkedIn job's detail page, compare its description against
    the candidate's resume (via the resume vector store), and return a
    match score with concrete reasons plus an APPLY/SKIP recommendation
    (threshold: score > 80 -> APPLY). This never submits an application —
    it only evaluates and records a recommendation in the background log
    for a human to act on."""
    text, _ = run_job_evaluation(job_title, job_url)
    return text
