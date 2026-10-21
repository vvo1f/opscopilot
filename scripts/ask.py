"""Ask a question from the terminal.

    python scripts/ask.py "Quanti giorni di ferie spettano a un dipendente?"
    python scripts/ask.py "How many days of remote work?" --collection hr -k 3
"""
import argparse
import logging

from opscopilot.llm.factory import make_llm
from opscopilot.rag.answer import ask

logging.basicConfig(level=logging.WARNING)

parser = argparse.ArgumentParser()
parser.add_argument("question")
parser.add_argument("--collection")
parser.add_argument("-k", type=int, default=5)
args = parser.parse_args()

result = ask(args.question, make_llm(), args.collection, args.k)
print(f"\n{result.answer}\n")
print("retrieved:")
for s in result.retrieved:
    cited = "*" if s in result.citations else " "
    print(f" {cited} [{s.n}] {s.score:.3f}  {s.filename} p.{s.page}  {s.text[:70]!r}")
