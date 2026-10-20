import json
from decimal import Decimal

from opscopilot.datagen import generate_all
from opscopilot.datagen.documents import build_documents
from opscopilot.datagen.evalsets import build_emails, build_questions


def test_counts():
    docs = build_documents()
    assert len(docs) == 28
    kinds = [d.doctype for d in docs]
    assert {k: kinds.count(k) for k in set(kinds)} == {
        "invoice": 8, "contract": 6, "hr_policy": 6, "meeting_minutes": 4, "technical_report": 4}
    assert len({d.filename for d in docs}) == 28


def test_questions():
    questions = build_questions(build_documents())
    assert len(questions) == 30
    answerable = [q for q in questions if q["answerable"]]
    assert len(answerable) == 24
    assert all(q["key_facts"] and q["expected_doc"] for q in answerable)
    assert all(not q["key_facts"] and q["expected_doc"] is None
               for q in questions if not q["answerable"])
    # the spec asks for at least 4 questions asked in the other language than the source
    assert sum(q["language"] != q["doc_language"] for q in answerable) >= 4


def test_emails():
    emails = build_emails()
    assert len(emails) == 20
    assert sum(e["adversarial"] for e in emails) == 3
    allowed = {"reply", "create_ticket", "forward", "schedule_meeting", "escalate", "archive"}
    assert {e["expected_action"] for e in emails} <= allowed


def test_invoice_numbers_unique_and_vat_is_22_percent():
    invoices = [d for d in build_documents() if d.doctype == "invoice"]
    assert len({d.truth["invoice_number"] for d in invoices}) == 8
    for d in invoices:
        total, vat = Decimal(d.truth["total_amount"]), Decimal(d.truth["vat_amount"])
        assert ((total - vat) * Decimal("0.22")).quantize(Decimal("0.01")) == vat


def test_same_seed_same_output(tmp_path):
    generate_all(tmp_path / "a")
    generate_all(tmp_path / "b")
    for rel in ["ground_truth/documents.json", "eval_sets/questions.jsonl",
                "eval_sets/emails.jsonl", "documents/meetings/minutes_01_it.txt"]:
        assert (tmp_path / "a" / rel).read_bytes() == (tmp_path / "b" / rel).read_bytes()


def test_ground_truth_file_matches_documents(tmp_path):
    generate_all(tmp_path)
    truth = json.loads((tmp_path / "ground_truth" / "documents.json").read_text(encoding="utf-8"))
    for entry in truth:
        assert (tmp_path / "documents" / entry["collection"] / entry["filename"]).exists()
