"""What we extract from each document type, as pydantic models.

The model is asked to return JSON matching these classes (the descriptions
below end up in the JSON schema inside the prompt). The validators reject
values in the wrong format, and the repair loop in LLMClient.chat_json then
shows the model its mistake and asks again.
"""
import re
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

LABELS = ["contract", "invoice", "hr_policy", "meeting_minutes", "technical_report", "other"]


class Classification(BaseModel):
    label: Literal["contract", "invoice", "hr_policy", "meeting_minutes", "technical_report", "other"]
    rationale: str


def normalize_date(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError(f"'{value}' is not an ISO date (YYYY-MM-DD)")
    date.fromisoformat(value)  # raises for impossible dates such as 2026-02-31
    return value


def normalize_amount(value: str | None) -> str | None:
    """'1.234,56 €', 'EUR 1,234.56' and 1234.5 all become '1234.56' or '1234.50'."""
    if value is None:
        return None
    text = re.sub(r"[^\d.,-]", "", value)  # drop currency symbols, letters and spaces
    if "," in text and "." in text:
        # the separator that comes last is the decimal one
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        # "1,234" is thousands, "12,5" or "12,50" is decimal
        after = text.split(",")[-1]
        text = text.replace(",", "") if len(after) == 3 else text.replace(",", ".")
    try:
        return f"{Decimal(text):.2f}"
    except InvalidOperation as exc:
        raise ValueError(f"'{value}' is not an amount") from exc


class _Fields(BaseModel):
    # the model sometimes returns 1234.5 instead of "1234.5"; accept numbers as text
    model_config = ConfigDict(coerce_numbers_to_str=True)


class InvoiceFields(_Fields):
    invoice_number: str | None = Field(None, description="Invoice number, for example INV-2026-123")
    issue_date: str | None = Field(None, description="Date the invoice was issued, YYYY-MM-DD")
    supplier: str | None = Field(None, description="Company that issued the invoice")
    customer: str | None = Field(None, description="Company that receives the invoice")
    total_amount: str | None = Field(None, description="Final total including VAT, as a decimal number")
    vat_amount: str | None = Field(None, description="VAT amount only, as a decimal number")
    currency: str | None = Field(None, description="Three-letter currency code, for example EUR")

    _date = field_validator("issue_date")(normalize_date)
    _amounts = field_validator("total_amount", "vat_amount")(normalize_amount)


class ContractFields(_Fields):
    parties: list[str] | None = Field(None, description="The two companies that sign the contract")
    effective_date: str | None = Field(None, description="Date the contract starts, YYYY-MM-DD")
    end_date: str | None = Field(None, description="Date the contract ends, YYYY-MM-DD")
    value: str | None = Field(None, description="Total fee of the contract, as a decimal number")

    _dates = field_validator("effective_date", "end_date")(normalize_date)
    _value = field_validator("value")(normalize_amount)


class HRPolicyFields(_Fields):
    policy_title: str | None = Field(None, description="Title of the policy, without the company name")
    effective_date: str | None = Field(None, description="Date the policy takes effect, YYYY-MM-DD")
    applies_to: str | None = Field(None, description="Who the policy applies to, as written in the text")

    _date = field_validator("effective_date")(normalize_date)


# Only these types are extracted. Minutes and reports are classified and indexed
# for search, but have no field schema in the MVP.
EXTRACTORS: dict[str, type[BaseModel]] = {
    "invoice": InvoiceFields,
    "contract": ContractFields,
    "hr_policy": HRPolicyFields,
}
