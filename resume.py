"""Week 3 lab: turn a PDF resume into a structured JSON profile.

`parse_resume_pdf` extracts raw text with pypdf, then `structure_resume`
hands that text to the LLM with `.with_structured_output(Resume)` so the
model returns a `Resume` instance directly (skills, projects, education)
instead of free-form prose that would need separate parsing.
"""

import json
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field
from pypdf import PdfReader

DEFAULT_RESUME_PDF = Path(__file__).parent / "sample_data" / "sample_resume.pdf"
DEFAULT_RESUME_JSON = Path(__file__).parent / "resume_data" / "resume.json"


class Education(BaseModel):
    institution: str
    degree: str
    field_of_study: str = ""
    graduation_year: str = ""


class Project(BaseModel):
    name: str
    description: str = Field(description="What it does and the technologies used")
    skills_used: list[str] = Field(default_factory=list)


class Experience(BaseModel):
    title: str
    organization: str
    description: str = Field(description="Responsibilities and measurable impact")
    skills_used: list[str] = Field(default_factory=list)


class Resume(BaseModel):
    name: str = ""
    skills: list[str] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)


STRUCTURE_PROMPT = (
    "Extract this resume's contents into the given schema. Keep each "
    "skill as a short standalone term (e.g. \"Python\", not \"Python and "
    "SQL\"). Preserve concrete details in project/experience descriptions "
    "(technologies, scale, measurable impact) since they'll be used later "
    "to judge fit against job descriptions. Don't invent information that "
    "isn't in the text below.\n\n---\n\n{resume_text}"
)


def parse_resume_pdf(pdf_path: Path = DEFAULT_RESUME_PDF) -> str:
    """Extract raw text from a PDF resume."""
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def structure_resume(resume_text: str) -> Resume:
    """Ask the LLM to turn raw resume text into a structured `Resume`."""
    # method="json_schema": see evaluation.py's evaluate_job for why — the default
    # forced-tool-call parsing intermittently mis-serializes list fields on long input.
    llm = ChatAnthropic(model="claude-sonnet-5").with_structured_output(Resume, method="json_schema")
    result = llm.invoke(STRUCTURE_PROMPT.format(resume_text=resume_text))
    assert isinstance(result, Resume)
    return result


def save_resume(resume: Resume, path: Path = DEFAULT_RESUME_JSON) -> Path:
    path.parent.mkdir(exist_ok=True)
    path.write_text(resume.model_dump_json(indent=2))
    return path


def load_resume(path: Path = DEFAULT_RESUME_JSON) -> Resume:
    return Resume.model_validate(json.loads(path.read_text()))


def build_structured_resume(pdf_path: Path = DEFAULT_RESUME_PDF, json_path: Path = DEFAULT_RESUME_JSON) -> Resume:
    """Parse + structure a PDF resume, caching the JSON result so repeat
    runs (e.g. every agent startup) skip the LLM call unless the PDF or
    cache is missing."""
    if json_path.exists():
        return load_resume(json_path)
    resume_text = parse_resume_pdf(pdf_path)
    resume = structure_resume(resume_text)
    save_resume(resume, json_path)
    return resume


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    resume = build_structured_resume()
    print(resume.model_dump_json(indent=2))
