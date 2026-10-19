import pytest
from pydantic import BaseModel

from opscopilot.llm.client import LLMClient, LLMError


class Item(BaseModel):
    name: str
    qty: int


class ScriptedLLM(LLMClient):
    """Replaces the network call with a list of canned replies."""

    def __init__(self, replies):
        super().__init__("http://unused", "x", "fake")
        self.replies = list(replies)
        self.calls = 0

    def chat(self, messages, temperature=0.0, max_tokens=None):
        self.calls += 1
        return self.replies.pop(0)


def test_valid_json_first_try():
    llm = ScriptedLLM(['{"name": "bolt", "qty": 3}'])
    assert llm.chat_json([{"role": "user", "content": "x"}], Item) == Item(name="bolt", qty=3)
    assert llm.calls == 1


def test_json_inside_code_fence_is_accepted():
    llm = ScriptedLLM(['Sure!\n```json\n{"name": "nut", "qty": 5}\n```'])
    assert llm.chat_json([{"role": "user", "content": "x"}], Item).qty == 5


def test_repair_loop_recovers():
    llm = ScriptedLLM(['{"name": "bolt", "qty": "many"}', '{"name": "bolt", "qty": 2}'])
    result = llm.chat_json([{"role": "user", "content": "x"}], Item)
    assert result.qty == 2
    assert llm.calls == 2


def test_gives_up_after_repairs():
    llm = ScriptedLLM(["nope"] * 3)
    with pytest.raises(LLMError):
        llm.chat_json([{"role": "user", "content": "x"}], Item, max_repairs=2)
    assert llm.calls == 3


def test_schema_goes_into_existing_system_message():
    seen = {}

    class Spy(ScriptedLLM):
        def chat(self, messages, temperature=0.0, max_tokens=None):
            seen["messages"] = messages
            return '{"name": "a", "qty": 1}'

    Spy([]).chat_json([{"role": "system", "content": "Be brief."},
                       {"role": "user", "content": "x"}], Item)
    assert [m["role"] for m in seen["messages"]] == ["system", "user"]
    assert "Be brief." in seen["messages"][0]["content"]
    assert "qty" in seen["messages"][0]["content"]


def test_retries_after_rate_limit(monkeypatch):
    from types import SimpleNamespace

    import httpx
    from openai import RateLimitError

    request = httpx.Request("POST", "http://x")
    limited = RateLimitError("slow down", response=httpx.Response(429, request=request), body=None)
    ok = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="done"))],
                         usage=None)
    outcomes = [limited, ok]

    def fake_create(**kwargs):
        result = outcomes.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    sleeps = []
    monkeypatch.setattr("opscopilot.llm.client.time.sleep", sleeps.append)

    llm = LLMClient("http://unused", "x", "fake")
    monkeypatch.setattr(llm._client.chat.completions, "create", fake_create)

    assert llm.chat([{"role": "user", "content": "hi"}]) == "done"
    assert sleeps == [2.0]  # waited once, with the first backoff step
