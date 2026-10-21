"""LLM client that remembers its answers on disk.

Evaluation reruns the same prompts many times. With a free API quota that is
wasteful, and a run that dies halfway would lose everything. Every reply is
saved as soon as it arrives, keyed by the exact request, so a rerun only pays
for calls it has not made before.
"""
import hashlib
import json
from pathlib import Path

from opscopilot.llm.client import LLMClient

CACHE_FILE = Path(__file__).resolve().parent / "cache" / "llm_cache.json"


class CachedLLM(LLMClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hits = self.misses = 0
        CACHE_FILE.parent.mkdir(exist_ok=True)
        self._cache = json.loads(CACHE_FILE.read_text("utf-8")) if CACHE_FILE.exists() else {}

    def chat(self, messages, temperature=0.0, max_tokens=None):
        key = hashlib.sha256(json.dumps([self.model, messages, temperature],
                                        ensure_ascii=False).encode()).hexdigest()
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        reply = super().chat(messages, temperature, max_tokens)
        self.misses += 1
        self._cache[key] = reply
        CACHE_FILE.write_text(json.dumps(self._cache, ensure_ascii=False), "utf-8")
        return reply
