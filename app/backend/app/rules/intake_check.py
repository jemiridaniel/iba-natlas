"""Deterministic check of the intake model's key numbers against the worker's own words.

An 8B intake model can misread an explicit fact: on N-ATLaS we saw "Adult man 35 years, fever 5
days, RDT negative" become age 3.5 with fever days and RDT left empty, so the app asked
questions the note had already answered and the referral note said "3.5-year-old". When the
note states age, fever duration or the RDT result in an unambiguous form, the note wins.
Only exactly one distinct match counts; anything ambiguous leaves the model's value alone.
"""

from __future__ import annotations

import re

from backend.app.rules.text import ascii_punct
from backend.app.schemas import PatientCase

_NUM = r"(\d{1,3}(?:\.\d)?)"

# "35 years", "35yrs", "35-year-old", "35 yo", "aged 35", "pikin 2 years"; months -> years.
_AGE_YEARS = re.compile(
    rf"(?<![\w.]){_NUM}\s*-?\s*(?:years?|yrs?|yo|y/o|y\.o\.?)(?:\s*-?\s*old)?\b"
    rf"|\baged?\s+{_NUM}\b",
    re.IGNORECASE,
)
_AGE_MONTHS = re.compile(rf"(?<![\w.]){_NUM}\s*-?\s*(?:months?|mo|mths?)(?:\s*-?\s*old)?\b",
                         re.IGNORECASE)

# Duration of fever: "fever 5 days", "fever for 5 days", "hot body 4 days", "body dey hot 3 days",
# "5 days fever", "5 days of fever". The fever word and the number must be close together.
_FEVER = r"(?:fever|febrile|hot\s+body|body\s+(?:dey\s+|don\s+dey\s+)?hot|pyrexia)"
_DAYS = r"(?:days?|dys?)"
_FEVER_DAYS = re.compile(
    rf"{_FEVER}\W+(?:for\s+|since\s+|of\s+|x\s*)?(?:\w+\s+){{0,2}}?{_NUM}\s*-?\s*{_DAYS}\b"
    rf"|(?<![\w.]){_NUM}\s*-?\s*{_DAYS}\s+(?:of\s+|with\s+|history\s+of\s+)?{_FEVER}",
    re.IGNORECASE,
)

_RDT = re.compile(
    r"\b(?:m?rdt|malaria\s+(?:rdt|test))[\s:=-]*(?:result[\s:=-]*)?(?:is\s+|was\s+|na\s+|show(?:s|ed)?\s+)?"
    r"(positive|negative|pos|neg|\+ve|-ve|not\s+done)(?![\w])",
    re.IGNORECASE,
)
_RDT_VALUE = {"positive": "positive", "pos": "positive", "+ve": "positive",
              "negative": "negative", "neg": "negative", "-ve": "negative"}


def _one(values: set[float | str]) -> float | str | None:
    return next(iter(values)) if len(values) == 1 else None


def stated_age(text: str) -> float | None:
    years = {float(a or b) for a, b in _AGE_YEARS.findall(text)}
    months = {round(float(m) / 12, 1) for m in _AGE_MONTHS.findall(text)}
    if years and months:
        return None
    value = _one(years or months)
    return value if value is not None and 0 <= value <= 120 else None


def stated_fever_days(text: str) -> float | None:
    values = {float(a or b) for a, b in _FEVER_DAYS.findall(text)}
    return _one(values)


def stated_rdt(text: str) -> str | None:
    values = {_RDT_VALUE.get(m.lower(), "not_done") for m in _RDT.findall(text)}
    return _one(values)


def check_intake(case: PatientCase, text: str) -> tuple[PatientCase, list[str]]:
    """Return the case with explicit facts from `text` applied, and the fields corrected."""
    text = ascii_punct(text)
    found = {
        "age_years": stated_age(text),
        "fever_days": stated_fever_days(text),
        "rdt_result": stated_rdt(text),
    }
    update = {f: v for f, v in found.items() if v is not None and getattr(case, f) != v}
    return (case.model_copy(update=update) if update else case), sorted(update)
