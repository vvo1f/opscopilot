import itertools

from opscopilot.ingestion.chunker import Chunk, chunk_pages, chunk_text
from opscopilot.ingestion.parsers import Page


def numbered_sentences(n):
    return " ".join(f"Sentence number {i} is here." for i in range(n))


def test_short_text_is_one_chunk():
    assert chunk_text("A short note.", 600, 100) == ["A short note."]


def test_no_chunk_exceeds_the_limit():
    chunks = chunk_text(numbered_sentences(200), max_chars=300, overlap=60)
    assert len(chunks) > 5
    assert all(len(c) <= 300 for c in chunks)


def test_consecutive_chunks_overlap():
    chunks = chunk_text(numbered_sentences(200), max_chars=300, overlap=60)
    for before, after in itertools.pairwise(chunks):
        # the start of each chunk was copied from the end of the previous one
        assert after[:20] in before


def test_no_overlap_when_zero():
    chunks = chunk_text(numbered_sentences(100), max_chars=300, overlap=0)
    assert all(len(c) <= 300 for c in chunks)
    joined = " ".join(chunks)
    assert joined.count("Sentence number 50 is here.") == 1


def test_prefers_paragraph_boundaries():
    text = ("First paragraph. " * 20).strip() + "\n\n" + ("Second paragraph. " * 20).strip()
    chunks = chunk_text(text, max_chars=400, overlap=0)
    assert chunks[0].startswith("First") and not chunks[0].endswith("Second paragraph.")
    assert any(c.startswith("Second") for c in chunks)


def test_text_without_separators_is_cut_blindly():
    chunks = chunk_text("x" * 1000, max_chars=300, overlap=0)
    assert [len(c) for c in chunks] == [300, 300, 300, 100]


def test_no_empty_chunks():
    assert all(c.strip() for c in chunk_text("a\n\n\n\n b\n\n   \n\nc", 5, 0))


def test_chunk_pages_keeps_page_numbers_and_global_index():
    pages = [Page(1, numbered_sentences(30)), Page(2, "Only a line on page two.")]
    chunks = chunk_pages(pages, max_chars=300, overlap=50)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert chunks[-1] == Chunk(page=2, index=len(chunks) - 1, text="Only a line on page two.")
    assert {c.page for c in chunks} == {1, 2}
