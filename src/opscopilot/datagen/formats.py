"""Date and money formatting in Italian and English style.

The same fact is written differently per language ("12 marzo 2026" vs
"March 12, 2026", "1.234,56" vs "1,234.56"). Extraction has to cope with that,
and the evaluation needs every accepted spelling of a fact, so we keep both.
"""
from datetime import date
from decimal import Decimal

IT_MONTHS = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio",
             "agosto", "settembre", "ottobre", "novembre", "dicembre"]
EN_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
             "August", "September", "October", "November", "December"]


def fmt_date(d: date, lang: str) -> str:
    if lang == "it":
        return f"{d.day} {IT_MONTHS[d.month - 1]} {d.year}"
    return f"{EN_MONTHS[d.month - 1]} {d.day}, {d.year}"


def date_variants(d: date) -> list[str]:
    """Every way a correct answer might spell this date."""
    return [d.isoformat(), fmt_date(d, "it"), fmt_date(d, "en"),
            f"{d.day} {EN_MONTHS[d.month - 1]} {d.year}"]


def _number(amount: Decimal, lang: str) -> str:
    text = f"{amount:,.2f}"  # 1,234.56
    if lang == "it":
        text = text.replace(",", "_").replace(".", ",").replace("_", ".")  # 1.234,56
    return text


def fmt_money(amount: Decimal, lang: str) -> str:
    if lang == "it":
        return f"{_number(amount, 'it')} €"
    return f"EUR {_number(amount, 'en')}"


def money_variants(amount: Decimal) -> list[str]:
    return [_number(amount, "it"), _number(amount, "en"), f"{amount:.2f}"]
