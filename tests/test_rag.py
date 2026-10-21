"""RAG logic with a scripted LLM and a fake search: no database, no model."""
from opscopilot.config import settings
from opscopilot.rag.answer import (
    NOT_FOUND_MESSAGE,
    ask,
    build_user_message,
    guess_language,
    parse_citations,
)
from opscopilot.rag.retriever import Source


class ScriptedLLM:
    def __init__(self, reply):
        self.reply, self.calls, self.messages = reply, 0, None

    def chat(self, messages, temperature=0.0, max_tokens=None):
        self.calls += 1
        self.messages = messages
        return self.reply


def sources(*scores):
    return [Source(n=i, document_id=i, filename=f"doc{i}.pdf", collection="hr", page=1,
                   text=f"text {i}", score=s) for i, s in enumerate(scores, start=1)]


def fake_search(result):
    return lambda question, k=None, collection=None: result


def test_answer_with_citations():
    llm = ScriptedLLM("Sono 26 giorni [1].")
    out = ask("Quanti giorni di ferie?", llm, search_fn=fake_search(sources(0.9, 0.8)))
    assert out.found and out.answer == "Sono 26 giorni [1]."
    assert [s.n for s in out.citations] == [1]
    assert len(out.retrieved) == 2


def test_prompt_contains_numbered_sources_and_question():
    llm = ScriptedLLM("ok [2]")
    ask("What is it?", llm, search_fn=fake_search(sources(0.9, 0.8)))
    user = llm.messages[1]["content"]
    assert "[1] (doc1.pdf, page 1)\ntext 1" in user and "[2] (doc2.pdf" in user
    assert user.endswith("Question: What is it?")


def test_not_found_from_the_model():
    llm = ScriptedLLM("NOT_FOUND")
    out = ask("Qual è la password del Wi-Fi?", llm, search_fn=fake_search(sources(0.8)))
    assert not out.found and out.answer == NOT_FOUND_MESSAGE["it"]
    assert out.citations == []


def test_not_found_message_follows_question_language():
    out = ask("What is the guest Wi-Fi password?", ScriptedLLM("NOT_FOUND"),
              search_fn=fake_search(sources(0.8)))
    assert out.answer == NOT_FOUND_MESSAGE["en"]


def test_no_results_means_no_llm_call():
    llm = ScriptedLLM("should not be used")
    out = ask("anything", llm, search_fn=fake_search([]))
    assert not out.found and llm.calls == 0


def test_similarity_threshold_skips_the_llm(monkeypatch):
    monkeypatch.setattr(settings, "min_similarity", 0.8)
    llm = ScriptedLLM("x [1]")
    out = ask("anything", llm, search_fn=fake_search(sources(0.7)))
    assert not out.found and llm.calls == 0
    out = ask("anything", llm, search_fn=fake_search(sources(0.85)))
    assert out.found and llm.calls == 1


def test_hr_collection_uses_the_hr_prompt():
    llm = ScriptedLLM("ok [1]")
    ask("question", llm, collection="hr", search_fn=fake_search(sources(0.9)))
    assert "HR assistant" in llm.messages[0]["content"]
    ask("question", llm, collection="finance", search_fn=fake_search(sources(0.9)))
    assert "HR assistant" not in llm.messages[0]["content"]


def test_parse_citations():
    assert parse_citations("A [1] and B [2, 3].", 3) == [1, 2, 3]
    assert parse_citations("Wrong [7] and [0], right [2] [2].", 3) == [2]
    assert parse_citations("no markers", 3) == []


def test_guess_language():
    assert guess_language("Quanti giorni di ferie spettano a un dipendente?") == "it"
    assert guess_language("How many days of remote work are allowed?") == "en"


def test_build_user_message_shape():
    msg = build_user_message("q?", sources(0.9))
    assert msg.startswith("Sources:\n[1] (doc1.pdf, page 1)")
