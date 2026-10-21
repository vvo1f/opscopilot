"""Classify a document, then extract fields for its type.

Two LLM calls per document: the second one needs the label from the first
(each type has its own schema). A single combined call would be cheaper but
mixes two tasks in one prompt, and is harder to evaluate separately.
"""
import logging
from dataclasses import dataclass

from opscopilot.config import settings
from opscopilot.docintel.schemas import EXTRACTORS, Classification
from opscopilot.llm.client import LLMClient
from opscopilot.prompts import load_prompt

log = logging.getLogger(__name__)


@dataclass
class Analysis:
    label: str | None
    fields: dict | None
    error: str | None = None


def _document_message(text: str) -> str:
    # tags mark where the untrusted document starts and ends
    return f"<document>\n{text[:settings.analysis_chars]}\n</document>"


def classify(text: str, llm: LLMClient) -> Classification:
    return llm.chat_json(
        [{"role": "system", "content": load_prompt("classify")},
         {"role": "user", "content": _document_message(text)}],
        Classification)


def extract(text: str, label: str, llm: LLMClient):
    schema = EXTRACTORS[label]
    system = load_prompt("extract").replace("{doctype}", label.replace("_", " "))
    return llm.chat_json(
        [{"role": "system", "content": system},
         {"role": "user", "content": _document_message(text)}],
        schema)


def analyze_text(text: str, llm: LLMClient) -> Analysis:
    """Never raises: a failure is returned as an error, so ingestion can go on."""
    try:
        label = classify(text, llm).label
    except Exception as exc:  # noqa: BLE001  (network, quota, bad output: ingestion must go on)
        log.warning("classification failed: %s", exc)
        return Analysis(None, None, f"classification failed: {exc}")

    if label not in EXTRACTORS:
        return Analysis(label, {})  # nothing to extract for this type

    try:
        fields = extract(text, label, llm).model_dump()
    except Exception as exc:  # noqa: BLE001
        log.warning("extraction failed: %s", exc)
        return Analysis(label, None, f"extraction failed: {exc}")
    return Analysis(label, fields)
