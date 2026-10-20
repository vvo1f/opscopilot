"""Generate the synthetic documents, ground truth and eval sets.

    python scripts/generate_data.py            # writes into ./data
    python scripts/generate_data.py --out /tmp/data
"""
import argparse
from pathlib import Path

from opscopilot.datagen import generate_all

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, default=ROOT / "data")
args = parser.parse_args()

counts = generate_all(args.out)
print(f"wrote {counts['documents']} documents, {counts['questions']} questions, "
      f"{counts['emails']} emails to {args.out}")
