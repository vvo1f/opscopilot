"""Thin wrapper around an OpenAI-compatible chat endpoint.

Why OpenAI-compatible: Ollama, Groq and Gemini all expose the same
/chat/completions request shape, so one client covers all of them and the
provider is just a base_url + api_key + model name in the settings.
"""
import json
import logging
import time
from typing import TypeVar

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when the model can't be reached or never returns usable output."""


def _extract_json(text: str) -> str:
    # Small models like to wrap the JSON in ```json fences or add a sentence
    # before it. Keep only what sits between the first '{' and the last '}'.
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return text
    return text[start : end + 1]


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout: float = 180.0, max_retries: int = 5):
        # The SDK has its own retry logic. We turn it off (max_retries=0) and do
        # the backoff ourselves so the behaviour and the logs are ours.
        self._client = OpenAI(base_url=base_url, api_key=api_key,
                              timeout=timeout, max_retries=0)
        self.model = model
        self.max_retries = max_retries

    @staticmethod
    def _wait_time(exc: Exception, attempt: int) -> float:
        # Exponential backoff: 2s, 4s, 8s, 16s, 30s (capped). If the server says
        # how long to wait (Retry-After header), trust that instead.
        response = getattr(exc, "response", None)
        retry_after = response.headers.get("retry-after") if response is not None else None
        if retry_after and retry_after.replace(".", "", 1).isdigit():
            return min(float(retry_after), 60.0)
        return min(2.0 * 2 ** attempt, 30.0)

    def chat(self, messages: list[dict], temperature: float = 0.0,
             max_tokens: int | None = None) -> str:
        """Send a chat request and return the text of the reply.

        temperature=0 makes the model pick the most likely token every time,
        which is what we want for extraction, classification and SQL.
        """
        kwargs = {"model": self.model, "messages": messages, "temperature": temperature}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                resp = self._client.chat.completions.create(**kwargs)
            except (RateLimitError, APITimeoutError, APIConnectionError) as exc:
                # 429 = rate limit (free tiers have per-minute quotas); timeouts and
                # refused connections happen e.g. while Ollama is loading the model.
                if attempt == self.max_retries:
                    raise LLMError(f"LLM call failed after {attempt + 1} attempts: {exc}") from exc
                wait = self._wait_time(exc, attempt)
                log.warning("LLM call failed (%s), retrying in %.0fs", type(exc).__name__, wait)
                time.sleep(wait)
                continue

            elapsed = time.perf_counter() - started
            usage = resp.usage  # some providers don't return it
            log.info("llm model=%s latency=%.1fs prompt_tokens=%s completion_tokens=%s",
                     self.model, elapsed,
                     usage.prompt_tokens if usage else "?",
                     usage.completion_tokens if usage else "?")
            return resp.choices[0].message.content or ""

        raise LLMError("unreachable")  # keeps type checkers happy

    def chat_json(self, messages: list[dict], schema: type[T], max_repairs: int = 2) -> T:
        """Ask for JSON and return it as a validated pydantic object.

        Models don't always produce valid JSON, especially small ones. When
        validation fails we show the model its own answer plus the error and ask
        for a fix (a "repair loop"). After max_repairs failed fixes we give up.
        """
        instruction = (
            "Reply with one JSON object that matches this JSON schema. "
            "No text before or after it.\n" + json.dumps(schema.model_json_schema())
        )
        # Some chat templates handle several system messages badly, so merge ours
        # into the existing system message when there is one.
        convo = [dict(m) for m in messages]
        if convo and convo[0]["role"] == "system":
            convo[0]["content"] += "\n\n" + instruction
        else:
            convo.insert(0, {"role": "system", "content": instruction})

        last_error = ""
        for attempt in range(max_repairs + 1):
            raw = self.chat(convo, temperature=0.0)
            try:
                # model_validate_json fails on broken JSON and on wrong fields alike
                return schema.model_validate_json(_extract_json(raw))
            except ValidationError as exc:
                last_error = str(exc)[:500]
                log.warning("JSON validation failed (attempt %d/%d)", attempt + 1, max_repairs + 1)
                convo.append({"role": "assistant", "content": raw})
                convo.append({"role": "user", "content":
                              f"That JSON is not valid: {last_error}\nReturn the corrected JSON only."})

        raise LLMError(f"No valid JSON after {max_repairs + 1} attempts. Last error: {last_error}")
