"""Pure functions that score the system. No database, no model, easy to test."""
import re
from collections import Counter

# ---------- answers ----------


def fact_found(answer: str, spellings: list[str]) -> bool:
    """True if any accepted spelling of a fact appears in the answer as a whole token.

    The lookarounds stop "26" from matching inside "126" or "2026", and "2" from
    matching the start of "2.5".
    """
    for spelling in spellings:
        pattern = (r"(?<!\w)(?<![.,]\d)" + re.escape(spelling) + r"(?!\w)(?![.,]\d)")
        if re.search(pattern, answer, flags=re.IGNORECASE):
            return True
    return False


def answer_correct(answer: str, key_facts: list[list[str]]) -> bool:
    """Every fact group needs at least one of its accepted spellings in the answer."""
    return bool(key_facts) and all(fact_found(answer, group) for group in key_facts)


# ---------- retrieval ----------


def first_rank(filenames: list[str], expected: str) -> int | None:
    """1-based position of the first retrieved chunk that comes from the expected file."""
    for rank, name in enumerate(filenames, start=1):
        if name == expected:
            return rank
    return None


def retrieval_summary(ranks: list[int | None], ks=(1, 3, 5)) -> dict:
    """hit@k: share of questions whose right document is in the top k.
    MRR: mean of 1/rank of the first right result (0 when it is missing)."""
    n = len(ranks)
    summary = {f"hit@{k}": round(sum(r is not None and r <= k for r in ranks) / n, 3) for k in ks}
    summary["mrr"] = round(sum(1 / r for r in ranks if r) / n, 3)
    summary["n"] = n
    return summary


def best_threshold(answerable: list[float], unanswerable: list[float]) -> dict:
    """The similarity cut-off that best separates answerable from unanswerable questions.

    Rule: 'answerable if best score >= t'. Chosen by balanced accuracy, the average of
    the share of answerable questions kept and unanswerable ones rejected.
    """
    best = {"threshold": 0.0, "balanced_accuracy": 0.0, "kept": 1.0, "rejected": 0.0}
    for t in sorted(set(answerable + unanswerable)):
        kept = sum(s >= t for s in answerable) / len(answerable)
        rejected = sum(s < t for s in unanswerable) / len(unanswerable)
        balanced = (kept + rejected) / 2
        if balanced > best["balanced_accuracy"]:
            best = {"threshold": round(t, 4), "balanced_accuracy": round(balanced, 3),
                    "kept": round(kept, 3), "rejected": round(rejected, 3)}
    return best


# ---------- classification ----------


def confusion(truth: list[str], predicted: list[str], labels: list[str]) -> dict:
    """matrix[true_label][predicted_label] = count."""
    counts = Counter(zip(truth, predicted))
    return {t: {p: counts[(t, p)] for p in labels} for t in labels}


def accuracy(truth: list, predicted: list) -> float:
    return round(sum(t == p for t, p in zip(truth, predicted)) / len(truth), 3)


# ---------- extraction ----------


def _norm(value) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().strip(".,;").casefold()


def field_matches(truth, predicted) -> bool:
    """Exact match after light normalisation (case, spaces, trailing punctuation).

    Lists (contract parties) are compared as sets, so order doesn't matter.
    A missing prediction is always wrong.
    """
    if predicted is None:
        return False
    if isinstance(truth, list):
        return isinstance(predicted, list) and {_norm(x) for x in truth} == {_norm(x) for x in predicted}
    return _norm(truth) == _norm(predicted)
