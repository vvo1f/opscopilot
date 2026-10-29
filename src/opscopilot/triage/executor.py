"""Carries out an approved proposal. Here it only describes what it would do.

There is deliberately no network call, no email sending and no database write
other than the log text. A real integration would live here, behind the same
function, after the approval step.
"""
import logging

log = logging.getLogger(__name__)


def simulate(proposal: dict) -> str:
    action, dept, priority = proposal["action"], proposal["department"], proposal["priority"]
    messages = {
        "reply": f"would send the drafted reply to {proposal['sender']}",
        "create_ticket": f"would open a {priority}-priority ticket in the {dept} queue",
        "forward": f"would forward the email to {dept}",
        "schedule_meeting": f"would send {proposal['sender']} proposed meeting times",
        "escalate": f"would alert management ({priority} priority)",
        "archive": "would archive the email",
    }
    text = f"SIMULATED: {messages[action]}"
    log.info("proposal %s: %s", proposal["id"], text)
    return text
