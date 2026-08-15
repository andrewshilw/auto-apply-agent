"""Week 3 lab: store the structured resume in ChromaDB so the evaluation
node can retrieve just the resume chunks relevant to a given job
description, instead of stuffing the whole resume into every prompt.

Chroma's default embedding function (a small local ONNX model, downloaded
once on first use) is used, so no embeddings API key is required. The
collection is persisted to disk so it only needs to be rebuilt when the
resume itself changes.
"""

from pathlib import Path

import chromadb

from resume import Resume

PERSIST_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "resume_chunks"


def _resume_to_documents(resume: Resume) -> tuple[list[str], list[dict], list[str]]:
    """Flatten a `Resume` into (documents, metadatas, ids) for Chroma —
    one chunk per skill/education/experience/project entry, so retrieval
    can surface exactly the entries relevant to a job description rather
    than the whole resume at once."""
    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for i, skill in enumerate(resume.skills):
        documents.append(f"Skill: {skill}")
        metadatas.append({"type": "skill", "name": skill})
        ids.append(f"skill-{i}")

    for i, edu in enumerate(resume.education):
        documents.append(
            f"Education: {edu.degree} in {edu.field_of_study} from {edu.institution} "
            f"({edu.graduation_year})"
        )
        metadatas.append({"type": "education", "name": edu.institution})
        ids.append(f"education-{i}")

    for i, exp in enumerate(resume.experience):
        skills = ", ".join(exp.skills_used)
        documents.append(
            f"Experience: {exp.title} at {exp.organization}. {exp.description} "
            f"Skills used: {skills}"
        )
        metadatas.append({"type": "experience", "name": exp.title})
        ids.append(f"experience-{i}")

    for i, project in enumerate(resume.projects):
        skills = ", ".join(project.skills_used)
        documents.append(f"Project: {project.name}. {project.description} Skills used: {skills}")
        metadatas.append({"type": "project", "name": project.name})
        ids.append(f"project-{i}")

    return documents, metadatas, ids


def build_resume_collection(resume: Resume, persist_dir: Path = PERSIST_DIR) -> chromadb.Collection:
    """(Re)build the resume collection from scratch so stale chunks from a
    previous resume never linger alongside new ones."""
    client = chromadb.PersistentClient(path=str(persist_dir))
    if COLLECTION_NAME in {c.name for c in client.list_collections()}:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME)

    documents, metadatas, ids = _resume_to_documents(resume)
    if documents:
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
    return collection


def get_resume_collection(persist_dir: Path = PERSIST_DIR) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_or_create_collection(COLLECTION_NAME)


def retrieve_relevant_resume_chunks(query_text: str, k: int = 8, persist_dir: Path = PERSIST_DIR) -> list[str]:
    """Return the `k` resume chunks most relevant to `query_text` (a job
    description), for use as retrieval-augmented context in the
    evaluation prompt."""
    collection = get_resume_collection(persist_dir)
    if collection.count() == 0:
        return []
    results = collection.query(query_texts=[query_text], n_results=min(k, collection.count()))
    return results["documents"][0]
