"""Run before every push: python scripts/check_secrets.py [--history]

Scans the files git tracks for API keys and secret-looking values, and fails if
.env is tracked. --history also scans every past commit, because a key that was
committed once stays in the history even after the file is fixed.
"""
import subprocess
import sys
from pathlib import Path

from opscopilot.secretscan import scan_text

ROOT = Path(__file__).resolve().parents[1]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", check=True).stdout


problems = []
tracked = git("ls-files").splitlines()
if ".env" in tracked:
    problems.append(".env is tracked by git. Run: git rm --cached .env")

for name in tracked:
    path = ROOT / name
    if path.is_file() and path.stat().st_size < 1_000_000:
        try:
            problems += scan_text(name, path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            pass  # binary file

if "--history" in sys.argv:
    problems += scan_text("git history", git("log", "-p", "--all"), keys_only=True)

if problems:
    print("\n".join(problems))
    sys.exit(1)
print(f"ok: {len(tracked)} tracked files checked, no secrets found")
