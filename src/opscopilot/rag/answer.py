"""Retrieval-augmented answering: search, build a prompt with sources, ask the LLM.

The model only sees the retrieved chunks, numbered, and is told to cite them.
That makes answers checkable, and lets the system say "not found" instead of
making something up. It reduces hallucination; it does not remove it.
"""
import re
from dataclasses import dataclass, field

from opscopilot.config import settings
from opscopilot.llm.client import LLMClient
from opscopilot.prompts import load_prompt
from opscopilot.rag.retriever import Source, search

NOT_FOUND_TOKEN = "NOT_FOUND"
NOT_FOUND_MESSAGE = {
    "it": "Non ho trovato la risposta nei documenti.",
    "en": "I could not find the answer in the documents.",
}
HR_COLLECTION = "hr"

# Tiny language guesser, enough to pick the language of the "not found" message.
# It counts common function words; it is not a real language detector.
IT_WORDS = {"il", "lo", "la", "gli", "le", "un", "una", "di", "del", "della", "dei", "delle",
            "che", "è", "quanti", "quante", "quanto", "qual", "quale", "quali", "chi", "come",
            "per", "entro", "nel", "nella", "sono", "esiste", "dove", "quando", "si", "può"}
EN_WORDS = {"the", "is", "are", "what", "how", "many", "much", "who", "which", "of", "for",
            "does", "do", "there", "an", "by", "to", "in", "at", "can", "when", "where"}


def guess_language(text: str) -> str:
    words = re.findall(r"[^\W\d_]+", text.lower())
    it = sum(w in IT_WORDS for w in words)
    en = sum(w in EN_WORDS for w in words)
    return "it" if it > en else "en"


def parse_citations(answer: str, n_sources: int) -> list[int]:
    """Source numbers cited in the answer: handles [1], [2] and [1, 2]. Out-of-range ones are dropped."""
    found = []
    for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer):
        for num in group.split(","):
            n = int(num)
            if 1 <= n <= n_sources and n not in found:
                found.append(n)
    return found


@dataclass
class Answer:
    answer: str
    found: bool                                     # False = the system abstained
    citations: list[Source] = field(default_factory=list)  # sources the answer cites
    retrieved: list[Source] = field(default_factory=list)  # everything that was retrieved


def build_user_message(question: str, sources: list[Source]) -> str:
    context = "\n\n".join(f"[{s.n}] ({s.filename}, page {s.page})\n{s.text}" for s in sources)
    return f"Sources:\n{context}\n\nQuestion: {question}"


def ask(question: str, llm: LLMClient, collection: str | None = None,
        k: int | None = None, search_fn=search, prompt_name: str | None = None) -> Answer:
    lang = guess_language(question)
    sources = search_fn(question, k=k, collection=collection)

    # abstain without calling the LLM when nothing relevant came back
    if not sources or sources[0].score < settings.min_similarity:
        return Answer(NOT_FOUND_MESSAGE[lang], False, [], sources)

    # asking the HR collection switches to the HR assistant persona
    # prompt_name lets the evaluation compare prompt variants; normal use leaves it empty
    system = load_prompt(prompt_name or ("hr_answer" if collection == HR_COLLECTION else "rag_answer"))
    reply = llm.chat(
        [{"role": "system", "content": system},
         {"role": "user", "content": build_user_message(question, sources)}],
        temperature=0.0,
    ).strip()

    if NOT_FOUND_TOKEN in reply:
        return Answer(NOT_FOUND_MESSAGE[lang], False, [], sources)

    cited = parse_citations(reply, len(sources))
    return Answer(reply, True, [s for s in sources if s.n in cited], sources)
