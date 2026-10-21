"""The retrieval harness, with a word-hashing stand-in for the embedding model.

Hashing words into a vector is a crude bag-of-words, so retrieval works on shared
words only. That is enough to check that the harness is wired correctly
(chunking, headers, filtering, ranking); the real quality numbers come from
`python eval/run_eval.py retrieval` with the real model.
"""
import re
import zlib

import numpy as np
import pytest

from opscopilot.datagen import generate_all
from opscopilot.datagen.documents import build_documents
from opscopilot.datagen.evalsets import build_questions
from opscopilot.evaluation.index import build_index
from opscopilot.evaluation.metrics import first_rank, retrieval_summary


class WordHashEmbedder:
    dim = 512

    def _vec(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            v[zlib.crc32(word.encode()) % self.dim] += 1
        return v / (np.linalg.norm(v) or 1)

    def embed_passages(self, texts, batch_size=16):
        return np.array([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture(scope="module")
def docs_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    generate_all(out)
    return out / "documents"


def test_index_has_chunks_with_metadata(docs_dir):
    index = build_index(docs_dir, WordHashEmbedder())
    assert len(index.chunks) > 28            # more chunks than documents
    assert {c["collection"] for c in index.chunks} == {"finance", "contracts", "hr", "meetings", "technical"}
    assert index.vectors.shape == (len(index.chunks), 512)


def test_headers_make_chunks_longer(docs_dir):
    with_header = build_index(docs_dir, WordHashEmbedder(), 150, 30, True)
    without = build_index(docs_dir, WordHashEmbedder(), 150, 30, False)
    assert sum(len(c["text"]) for c in with_header.chunks) > sum(len(c["text"]) for c in without.chunks)


def test_collection_filter(docs_dir):
    index = build_index(docs_dir, WordHashEmbedder())
    hits = index.search("remote work days per week", k=5, collection="hr")
    assert hits and {h.collection for h in hits} == {"hr"}
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_harness_finds_documents_by_shared_words(docs_dir):
    index = build_index(docs_dir, WordHashEmbedder())
    ranks = []
    for q in build_questions(build_documents()):
        if q["answerable"]:
            hits = index.search(q["question"], k=10)
            ranks.append(first_rank([h.filename for h in hits], q["expected_doc"]))
    summary = retrieval_summary(ranks)
    # same-language questions share words with their document, so even this crude
    # embedder finds many of them; the exact value is not the point
    assert summary["hit@5"] >= 0.5
