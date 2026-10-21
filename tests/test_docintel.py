import pytest
from pydantic import ValidationError

from opscopilot.docintel.analyze import analyze_text, classify
from opscopilot.docintel.schemas import InvoiceFields, normalize_amount, normalize_date
from opscopilot.llm.client import LLMClient


class ScriptedLLM(LLMClient):
    """Returns canned replies in order; records how many calls were made."""

    def __init__(self, replies):
        super().__init__("http://unused", "x", "fake")
        self.replies, self.calls = list(replies), 0

    def chat(self, messages, temperature=0.0, max_tokens=None):
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.mark.parametrize("raw, expected", [
    ("1.234,56 €", "1234.56"),
    ("EUR 1,234.56", "1234.56"),
    ("17910,00", "17910.00"),
    ("1234.5", "1234.50"),
    ("1,234", "1234.00"),
    ("12,5", "12.50"),
    (None, None),
])
def test_normalize_amount(raw, expected):
    assert normalize_amount(raw) == expected


def test_normalize_amount_rejects_text():
    with pytest.raises(ValueError):
        normalize_amount("a lot")


def test_normalize_date():
    assert normalize_date("2026-03-12") == "2026-03-12"
    for bad in ["12 marzo 2026", "2026-02-31", "12/03/2026"]:
        with pytest.raises(ValueError):
            normalize_date(bad)


def test_number_instead_of_string_is_accepted():
    fields = InvoiceFields.model_validate({"total_amount": 1220.5, "currency": "EUR"})
    assert fields.total_amount == "1220.50"


def test_bad_date_is_rejected_by_the_schema():
    with pytest.raises(ValidationError):
        InvoiceFields.model_validate({"issue_date": "24 febbraio 2026"})


def test_classify():
    llm = ScriptedLLM(['{"label": "invoice", "rationale": "has a number"}'])
    assert classify("Fattura n. 1", llm).label == "invoice"


def test_unknown_label_goes_through_the_repair_loop():
    llm = ScriptedLLM(['{"label": "memo", "rationale": "x"}',
                       '{"label": "other", "rationale": "x"}'])
    assert classify("text", llm).label == "other"
    assert llm.calls == 2


def test_analyze_invoice_end_to_end_with_a_repair():
    llm = ScriptedLLM([
        '{"label": "invoice", "rationale": "x"}',
        '{"invoice_number": "INV-1", "issue_date": "24 febbraio 2026"}',   # wrong date format
        '{"invoice_number": "INV-1", "issue_date": "2026-02-24", "total_amount": "1.220,00"}',
    ])
    result = analyze_text("Fattura", llm)
    assert result.error is None and result.label == "invoice"
    assert result.fields["issue_date"] == "2026-02-24"
    assert result.fields["total_amount"] == "1220.00"
    assert llm.calls == 3


def test_label_without_schema_skips_extraction():
    llm = ScriptedLLM(['{"label": "meeting_minutes", "rationale": "x"}'])
    result = analyze_text("Verbale", llm)
    assert (result.label, result.fields, result.error) == ("meeting_minutes", {}, None)
    assert llm.calls == 1


def test_classification_failure_is_returned_not_raised():
    result = analyze_text("x", ScriptedLLM([RuntimeError("quota exceeded")]))
    assert result.label is None and "classification failed" in result.error


def test_extraction_failure_keeps_the_label():
    llm = ScriptedLLM(['{"label": "contract", "rationale": "x"}'] + ["not json"] * 3)
    result = analyze_text("Contratto", llm)
    assert result.label == "contract" and result.fields is None
    assert "extraction failed" in result.error
