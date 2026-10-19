from opscopilot.config import settings
from opscopilot.llm.client import LLMClient


def make_llm() -> LLMClient:
    """Build the client from the current settings."""
    return LLMClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        timeout=settings.llm_timeout,
    )
