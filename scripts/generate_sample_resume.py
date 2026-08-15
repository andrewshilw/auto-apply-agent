"""Generates a synthetic (fake person, fake companies) resume PDF for
development and testing, so the resume-parsing pipeline can be exercised
without anyone's real resume. Run directly:

    python scripts/generate_sample_resume.py

Writes to sample_data/sample_resume.pdf.
"""

from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

OUTPUT_PATH = Path(__file__).parent.parent / "sample_data" / "sample_resume.pdf"

RESUME_SECTIONS = [
    ("name", "Jordan Rivera"),
    ("contact", "jordan.rivera.dev@example.com | (555) 019-2837 | github.com/jrivera-dev"),
    ("heading", "Education"),
    ("body", "B.S. in Computer Science, Riverbend State University, 2021-2025, GPA 3.7/4.0"),
    ("heading", "Skills"),
    (
        "body",
        "Python, TypeScript, Java, SQL, React, FastAPI, Django, Docker, AWS (EC2, S3, Lambda), "
        "PostgreSQL, Redis, Git, LangChain, REST API design, CI/CD (GitHub Actions), pytest",
    ),
    ("heading", "Experience"),
    ("subheading", "Software Engineering Intern — Northlake Data Co. (Summer 2024)"),
    (
        "body",
        "Built a FastAPI service that ingested streaming sensor data into PostgreSQL, cutting "
        "average query latency by 40% through indexing and caching with Redis. Wrote pytest "
        "suites covering the ingestion pipeline and added them to a GitHub Actions CI workflow.",
    ),
    ("subheading", "Web Developer, Part-Time — Cedarline Student Media (2023-2024)"),
    (
        "body",
        "Rebuilt the campus news site's front end in React and TypeScript, replacing a legacy "
        "jQuery codebase. Deployed to AWS S3 + CloudFront and set up a Lambda-based contact form "
        "handler.",
    ),
    ("heading", "Projects"),
    ("subheading", "AutoTrack — Personal Finance Tracker"),
    (
        "body",
        "Django + PostgreSQL web app for tracking recurring subscriptions and spending trends, "
        "with a React dashboard and Dockerized local dev environment. Used by ~50 classmates.",
    ),
    ("subheading", "PromptBench — LLM Evaluation Harness"),
    (
        "body",
        "Python CLI tool (built with LangChain) that runs a suite of prompts against multiple "
        "LLM providers and scores responses against reference answers for regression testing.",
    ),
]


def build_pdf(output_path: Path = OUTPUT_PATH) -> Path:
    output_path.parent.mkdir(exist_ok=True)
    styles = getSampleStyleSheet()
    name_style = ParagraphStyle("Name", parent=styles["Title"], fontSize=20, spaceAfter=4)
    contact_style = ParagraphStyle("Contact", parent=styles["Normal"], fontSize=10, spaceAfter=16)
    heading_style = ParagraphStyle(
        "SectionHeading", parent=styles["Heading2"], spaceBefore=12, spaceAfter=6
    )
    subheading_style = ParagraphStyle(
        "SubHeading", parent=styles["Heading4"], spaceBefore=6, spaceAfter=2
    )
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, spaceAfter=6, leading=14)

    style_map = {
        "name": name_style,
        "contact": contact_style,
        "heading": heading_style,
        "subheading": subheading_style,
        "body": body_style,
    }

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=LETTER,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
        topMargin=0.8 * inch,
        bottomMargin=0.8 * inch,
    )
    flowables = []
    for kind, text in RESUME_SECTIONS:
        flowables.append(Paragraph(text, style_map[kind]))
        if kind == "name":
            flowables.append(Spacer(1, 2))

    doc.build(flowables)
    return output_path


if __name__ == "__main__":
    path = build_pdf()
    print(f"Wrote synthetic sample resume to {path}")
