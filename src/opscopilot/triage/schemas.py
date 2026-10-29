"""The only things the model is allowed to answer when triaging an email.

Using Literal types means anything outside these lists fails validation, so a
hijacked model cannot invent a new action such as "send_money". The repair loop
then asks again, and after three failures the email is not triaged at all.
"""
from typing import Literal

from pydantic import BaseModel, Field

Action = Literal["reply", "create_ticket", "forward", "schedule_meeting", "escalate", "archive"]
Priority = Literal["low", "medium", "high"]
Department = Literal["sales", "support", "hr", "finance", "it", "management", "none"]


class ProposedAction(BaseModel):
    action: Action
    priority: Priority
    department: Department
    summary: str = Field(max_length=500, description="At most two sentences")
    draft_reply: str | None = Field(None, max_length=1500,
                                    description="Only when action is reply, otherwise null")
