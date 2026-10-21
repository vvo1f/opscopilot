"""Evaluation runner. Every number in the README should come from here.

    python eval/run_eval.py retrieval                 # hit@k, MRR, similarity threshold (no LLM)
    python eval/run_eval.py chunks                    # chunk size x header sweep (no LLM)
    python eval/run_eval.py answers                   # RAG answers (30 LLM calls, cached)
    python eval/run_eval.py answers --prompt rag_plain
    python eval/run_eval.py answers --k 3 --scoped
    python eval/run_eval.py classification            # reads what ingestion stored in the database
    python eval/run_eval.py extraction                # same
    python eval/run_eval.py report                    # collect all results into eval/results/summary.md

Retrieval and answers use the in-memory index (same chunking and embeddings as
ingestion), so experiments never change your database.
"""
import argparse
import json
from datetime import date
from pathlib import Path

from cache_llm import CachedLLM

from opscopilot import store
from opscopilot.config import settings
from opscopilot.docintel.schemas import EXTRACTORS, LABELS
from opscopilot.embeddings import get_embedder
from opscopilot.evaluation.index import build_index
from opscopilot.evaluation.metrics import (
    accuracy,
    answer_correct,
    best_threshold,
    confusion,
    field_matches,
    first_rank,
    retrieval_summary,
)
from opscopilot.rag.answer import ask

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "eval" / "results"


def load_questions() -> list[dict]:
    path = DATA / "eval_sets" / "questions.jsonl"
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def load_truth() -> dict:
    rows = json.loads((DATA / "ground_truth" / "documents.json").read_text("utf-8"))
    return {r["filename"]: r for r in rows}


def save(name: str, config: dict, metrics: dict, table: list | None = None, rows: list | None = None):
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = {"name": name, "date": date.today().isoformat(), "config": config,
           "metrics": metrics, "table": table or [], "rows": rows or []}
    (RESULTS / f"{name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"saved eval/results/{name}.json")


def current_config() -> dict:
    return {"embedding_model": settings.embedding_model, "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap, "chunk_header": settings.chunk_header,
            "llm_model": settings.llm_model}


# ---------- retrieval ----------

def rank_questions(index, questions):
    """For each question: the rank of the right document, and the best similarity score."""
    out = []
    for q in questions:
        hits = index.search(q["question"], k=10)
        rank = first_rank([h.filename for h in hits], q["expected_doc"]) if q["answerable"] else None
        out.append({"q": q, "rank": rank, "best": hits[0].score})
    return out


def cmd_retrieval(args):
    index = build_index(DATA / "documents", get_embedder())
    ranked = rank_questions(index, load_questions())
    answerable = [r for r in ranked if r["q"]["answerable"]]
    cross = [r for r in answerable if r["q"]["language"] != r["q"]["doc_language"]]
    same = [r for r in answerable if r["q"]["language"] == r["q"]["doc_language"]]
    metrics = {
        "all": retrieval_summary([r["rank"] for r in answerable]),
        "question_in_same_language_as_document": retrieval_summary([r["rank"] for r in same]),
        "question_in_other_language": retrieval_summary([r["rank"] for r in cross]),
        "chunks_in_index": len(index.chunks),
        "similarity_threshold": best_threshold(
            [r["best"] for r in answerable],
            [r["best"] for r in ranked if not r["q"]["answerable"]]),
    }
    rows = [{"id": r["q"]["id"], "question": r["q"]["question"], "expected": r["q"]["expected_doc"],
             "rank": r["rank"], "best_score": round(r["best"], 3)} for r in ranked]
    save("retrieval", current_config(), metrics, rows=rows)
    missed = [r for r in rows if r["expected"] and (r["rank"] is None or r["rank"] > 3)]
    print(f"{len(missed)} questions have the right document outside the top 3:")
    for r in missed:
        print(f"  {r['id']} rank={r['rank']}  {r['question']}")


def cmd_chunks(args):
    embedder, questions, table = get_embedder(), load_questions(), []
    for size in (150, 300, 600):
        for header in (True, False):
            index = build_index(DATA / "documents", embedder, size, size // 5, header)
            ranked = rank_questions(index, questions)
            summary = retrieval_summary([r["rank"] for r in ranked if r["q"]["answerable"]])
            table.append({"chunk_size": size, "overlap": size // 5, "header": header,
                          "chunks": len(index.chunks), **{k: v for k, v in summary.items() if k != "n"}})
            print(table[-1])
    save("chunks", {**current_config(), "note": "overlap = chunk_size / 5"},
         {"questions": sum(q["answerable"] for q in questions)}, table=table)


# ---------- answers ----------

def cmd_answers(args):
    llm = CachedLLM(settings.llm_base_url, settings.llm_api_key, settings.llm_model,
                    timeout=settings.llm_timeout)
    index = build_index(DATA / "documents", get_embedder())
    prompt = None if args.prompt == "rag_answer" else args.prompt

    rows = []
    for q in load_questions()[:args.limit]:
        collection = q["collection"] if args.scoped and q["collection"] else None
        try:
            result = ask(q["question"], llm, collection, args.k, search_fn=index.search,
                         prompt_name=prompt)
        except Exception as exc:  # noqa: BLE001  (a failed call counts as a failed answer)
            print(f"{q['id']} failed: {exc}")
            rows.append({"id": q["id"], "answerable": q["answerable"], "error": str(exc)})
            continue
        row = {"id": q["id"], "question": q["question"], "answerable": q["answerable"],
               "cross_language": bool(q["answerable"] and q["language"] != q["doc_language"]),
               "found": result.found, "answer": result.answer}
        if q["answerable"]:
            row["correct"] = result.found and answer_correct(result.answer, q["key_facts"])
            row["doc_retrieved"] = any(s.filename == q["expected_doc"] for s in result.retrieved)
            row["cited"] = bool(result.citations)
            row["cited_right_doc"] = any(c.filename == q["expected_doc"] for c in result.citations)
        rows.append(row)
        print(f"{q['id']} {'ok ' if row.get('correct') else 'no '} {result.answer[:90]!r}")

    def rate(items, key):
        return round(sum(bool(r.get(key)) for r in items) / len(items), 3) if items else None

    ok = [r for r in rows if "error" not in r]
    ans = [r for r in ok if r["answerable"]]
    unans = [r for r in ok if not r["answerable"]]
    found = [r for r in ans if r["found"]]
    metrics = {
        "answerable_questions": len(ans),
        "answer_correct": rate(ans, "correct"),
        "answer_correct_same_language": rate([r for r in ans if not r["cross_language"]], "correct"),
        "answer_correct_cross_language": rate([r for r in ans if r["cross_language"]], "correct"),
        "right_document_retrieved": rate(ans, "doc_retrieved"),
        "wrongly_abstained": round(1 - rate(ans, "found"), 3) if ans else None,
        "citation_present": rate(found, "cited"),
        "citation_right_document": rate(found, "cited_right_doc"),
        "unanswerable_questions": len(unans),
        "abstained_on_unanswerable": round(1 - rate(unans, "found"), 3) if unans else None,
        "failed_calls": len(rows) - len(ok),
        "llm_cache": {"hits": llm.hits, "new_calls": llm.misses},
    }
    name = f"answers_{args.prompt}_k{args.k}" + ("_scoped" if args.scoped else "")
    save(name, {**current_config(), "k": args.k, "prompt": args.prompt, "scoped": args.scoped},
         metrics, rows=rows)


# ---------- classification and extraction (read from the database) ----------

def stored_documents() -> dict:
    with store.connect() as conn:
        rows = conn.execute("SELECT filename, label, fields FROM documents").fetchall()
    return {name: {"label": label, "fields": fields or {}} for name, label, fields in rows}


def cmd_classification(args):
    truth, stored = load_truth(), stored_documents()
    names = sorted(truth)
    true = [truth[n]["doctype"] for n in names]
    pred = [(stored.get(n) or {}).get("label") or "none" for n in names]
    labels = LABELS + ["none"]
    matrix = confusion(true, pred, labels)
    table = [{"true \\ predicted": t, **matrix[t]} for t in labels if any(matrix[t].values())]
    wrong = [{"file": n, "true": t, "predicted": p} for n, t, p in zip(names, true, pred) if t != p]
    save("classification", current_config(),
         {"documents": len(names), "accuracy": accuracy(true, pred), "misclassified": wrong},
         table=table)


def cmd_extraction(args):
    truth, stored = load_truth(), stored_documents()
    counts, rows, failed = {}, [], 0
    for name, entry in sorted(truth.items()):
        if entry["doctype"] not in EXTRACTORS:
            continue
        fields = (stored.get(name) or {}).get("fields") or {}
        failed += "error" in fields or name not in stored
        for field, expected in entry["fields"].items():
            predicted = fields.get(field)
            ok = field_matches(expected, predicted)
            slot = counts.setdefault((entry["doctype"], field), [0, 0])
            slot[0] += ok
            slot[1] += 1
            if not ok:
                rows.append({"file": name, "field": field, "expected": expected, "predicted": predicted})
    table = [{"type": t, "field": f, "correct": c, "total": n, "accuracy": round(c / n, 3)}
             for (t, f), (c, n) in sorted(counts.items())]
    total_c, total_n = sum(c for c, _ in counts.values()), sum(n for _, n in counts.values())
    per_type = {}
    for (t, _), (c, n) in counts.items():
        per_type.setdefault(t, [0, 0])
        per_type[t][0] += c
        per_type[t][1] += n
    save("extraction", current_config(),
         {"field_accuracy": round(total_c / total_n, 3),
          "per_type": {t: round(c / n, 3) for t, (c, n) in per_type.items()},
          "documents_with_failed_analysis": failed},
         table=table, rows=rows)


# ---------- report ----------

def md_table(rows: list[dict]) -> str:
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def flatten(prefix: str, value, out: list):
    if isinstance(value, dict):
        for k, v in value.items():
            flatten(f"{prefix}.{k}" if prefix else k, v, out)
    elif not isinstance(value, list):
        out.append({"metric": prefix, "value": value})


def cmd_report(args):
    note = ("Generated by `python eval/run_eval.py report`. Datasets are small and synthetic "
            "(see README), so read differences of a few points as noise.")
    parts = ["# Evaluation results", "", note, ""]
    for path in sorted(RESULTS.glob("*.json")):
        data = json.loads(path.read_text("utf-8"))
        parts += [f"## {data['name']}  ({data['date']})", "",
                  "Config: `" + json.dumps(data["config"], ensure_ascii=False) + "`", ""]
        flat = []
        flatten("", data["metrics"], flat)
        if flat:
            parts += [md_table(flat), ""]
        if data["table"]:
            parts += [md_table(data["table"]), ""]
    out = RESULTS / "summary.md"
    out.write_text("\n".join(parts), "utf-8")
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, fn in [("retrieval", cmd_retrieval), ("chunks", cmd_chunks),
                     ("classification", cmd_classification), ("extraction", cmd_extraction),
                     ("report", cmd_report)]:
        sub.add_parser(name).set_defaults(fn=fn)
    answers = sub.add_parser("answers")
    answers.add_argument("--k", type=int, default=5)
    answers.add_argument("--prompt", default="rag_answer", choices=["rag_answer", "rag_plain"])
    answers.add_argument("--scoped", action="store_true",
                         help="search only the question's own collection")
    answers.add_argument("--limit", type=int, default=100)
    answers.set_defaults(fn=cmd_answers)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
