"""Ingest every generated document.

    python scripts/ingest_folder.py             # add what is not stored yet
    python scripts/ingest_folder.py --reset     # drop the tables first (needed if EMBEDDING_DIM changed)
"""
import argparse
import logging
from pathlib import Path

from opscopilot import store
from opscopilot.ingestion.pipeline import ingest_file

logging.basicConfig(level=logging.WARNING)
ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--reset", action="store_true")
parser.add_argument("--folder", type=Path, default=ROOT / "data" / "documents")
args = parser.parse_args()

if args.reset:
    store.reset_schema()
else:
    store.init_schema()

files = sorted(p for p in args.folder.glob("*/*") if p.is_file())
if not files:
    raise SystemExit(f"no documents in {args.folder}. Run scripts/generate_data.py first.")

total = 0
for path in files:
    result = ingest_file(path, collection=path.parent.name)
    total += result.chunks
    state = "skipped" if result.skipped else f"{result.chunks} chunks"
    print(f"{result.collection:10} {result.filename:32} {state}")
print(f"done: {len(files)} files, {total} new chunks")
