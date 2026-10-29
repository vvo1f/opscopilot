"""Email -> a proposed action. The action is only a proposal: a person approves it."""
import re

from opscopilot.llm.client import LLMClient
from opscopilot.prompts import load_prompt
from opscopilot.triage.schemas import ProposedAction

MAX_BODY_CHARS = 4000

# Phrases that address an AI assistant instead of a human reader. This is a cheap
# warning light, not a defence: it cannot catch every attack (a plain request for
# confidential data has no special phrase), which is why a person approves every action.
INJECTION_PATTERNS = [
    r"ignore (all |any )?(the |your )?(previous|prior|above|earlier) (rules|instructions)",
    r"ignora (tutte )?le (regole|istruzioni)",
    r"system instruction",
    r"istruzione per l'assistente",
    r"instruction for the assistant",
    r"you are now",
    r"set (the )?action to",
    r"action\s*=\s*\w+",
]


def detect_injection(text: str) -> list[str]:
    return [p for p in INJECTION_PATTERNS if re.search(p, text, flags=re.IGNORECASE)]


def build_email_message(sender: str, subject: str, body: str) -> str:
    # Otherwise an email could write "</email>" and continue outside the data block.
    def clean(text: str) -> str:
        return re.sub(r"</?\s*email\s*>", "[email-tag]", text, flags=re.IGNORECASE)

    return (f"<email>\nFrom: {clean(sender)}\nSubject: {clean(subject)}\n"
            f"Body: {clean(body[:MAX_BODY_CHARS])}\n</email>")


def triage_email(sender: str, subject: str, body: str, llm: LLMClient) -> tuple[ProposedAction, list[str]]:
    """Returns the proposed action and the injection phrases found in the email."""
    result = llm.chat_json(
        [{"role": "system", "content": load_prompt("triage")},
         {"role": "user", "content": build_email_message(sender, subject, body)}],
        ProposedAction)
    return result, detect_injection(f"{subject}\n{body}")
