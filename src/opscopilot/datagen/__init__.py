"""Synthetic company data with known ground truth.

Every document is built from a template, so we know exactly which invoice
number, date or policy value is in it. That lets us score extraction and
retrieval automatically later, without labelling anything by hand.
"""
import json
from pathlib import Path

from opscopilot.datagen.documents import build_documents
from opscopilot.datagen.evalsets import build_emails, build_questions
from opscopilot.datagen.writers import write_doc


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_all(out: Path) -> dict:
    """Write documents, ground truth and eval sets under `out`. Returns counts."""
    docs = build_documents()
    questions = build_questions(docs)
    emails = build_emails()

    for doc in docs:
        write_doc(doc, out / "documents" / doc.collection)

    truth_dir = out / "ground_truth"
    eval_dir = out / "eval_sets"
    truth_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    truth = [
        {"filename": d.filename, "collection": d.collection, "doctype": d.doctype,
         "language": d.lang, "fields": d.truth}
        for d in docs
    ]
    (truth_dir / "documents.json").write_text(
        json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_jsonl(eval_dir / "questions.jsonl", questions)
    _write_jsonl(eval_dir / "emails.jsonl", emails)

    return {"documents": len(docs), "questions": len(questions), "emails": len(emails)}
