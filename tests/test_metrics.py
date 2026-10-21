import pytest

from opscopilot.evaluation.metrics import (
    accuracy,
    answer_correct,
    best_threshold,
    confusion,
    fact_found,
    field_matches,
    first_rank,
    retrieval_summary,
)


def test_fact_found_is_whole_token():
    assert fact_found("Sono 26 giorni [1].", ["26"])
    assert not fact_found("Sono 126 giorni", ["26"])
    assert not fact_found("Nel 2026 sono previsti", ["26"])
    assert not fact_found("circa 2.5 giorni", ["2"])
    assert not fact_found("totale 3.150,00", ["150"])


def test_fact_found_money_and_dates():
    assert fact_found("Il totale è 17.910,00 € [1]", ["17.910,00", "17,910.00", "17910.00"])
    assert fact_found("Scade il 12 MARZO 2026.", ["2026-03-12", "12 marzo 2026"])
    assert fact_found("by 12 March 2026", ["2026-03-12", "12 March 2026"])
    assert fact_found("write to ethics@alpina-systems.example.", ["ethics@alpina-systems.example"])


def test_answer_correct_needs_every_fact_group():
    facts = [["26"], ["cinque", "5"]]
    assert answer_correct("26 giorni, 5 riportabili", facts)
    assert not answer_correct("26 giorni", facts)
    assert not answer_correct("anything", [])


def test_ranks_and_summary():
    assert first_rank(["a", "b", "a"], "a") == 1
    assert first_rank(["a", "b"], "b") == 2
    assert first_rank(["a"], "z") is None
    summary = retrieval_summary([1, 2, None, 4])
    assert summary == {"hit@1": 0.25, "hit@3": 0.5, "hit@5": 0.75, "mrr": 0.438, "n": 4}


def test_best_threshold_separates_the_groups():
    result = best_threshold([0.9, 0.85, 0.8], [0.6, 0.7])
    assert result["balanced_accuracy"] == 1.0
    assert 0.7 < result["threshold"] <= 0.8


def test_confusion_and_accuracy():
    m = confusion(["a", "a", "b"], ["a", "b", "b"], ["a", "b"])
    assert m == {"a": {"a": 1, "b": 1}, "b": {"a": 0, "b": 1}}
    assert accuracy(["a", "a", "b"], ["a", "b", "b"]) == pytest.approx(0.667, abs=1e-3)


def test_field_matches():
    assert field_matches("Alpina Systems", "  alpina  systems. ")
    assert field_matches(["A", "B"], ["b", "a"])
    assert not field_matches(["A", "B"], ["A"])
    assert not field_matches("x", None)
    assert field_matches(48, "48")
