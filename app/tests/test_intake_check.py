"""Explicit age, fever duration and RDT in the note override the intake model."""

import pytest

from backend.app.rules.intake_check import check_intake, stated_age, stated_fever_days, stated_rdt
from backend.app.schemas import PatientCase

LASSA = ("Adult man 35 years, fever 5 days, RDT negative, took coartem for 3 days but no "
         "improvement, headache and sore throat")


def test_misread_lassa_note_is_corrected():
    model = PatientCase(age_years=3.5, sex="male")
    case, fixed = check_intake(model, LASSA)
    assert (case.age_years, case.fever_days, case.rdt_result) == (35, 5, "negative")
    assert fixed == ["age_years", "fever_days", "rdt_result"]


def test_agreeing_model_is_left_alone():
    model = PatientCase(age_years=35, fever_days=5, rdt_result="negative")
    case, fixed = check_intake(model, LASSA)
    assert fixed == [] and case is model


@pytest.mark.parametrize("text,age", [
    ("Pikin 2 years, hot body 3 days", 2),
    ("35-year-old woman", 35),
    ("man aged 40, fever", 40),
    ("baby 18 months old", 1.5),
    ("28yo female", 28),
    ("fever 4 days", None),
    ("man 35 years, diabetic for 10 years", None),  # ambiguous: keep the model's value
])
def test_age(text, age):
    assert stated_age(text) == age


@pytest.mark.parametrize("text,days", [
    ("fever 5 days", 5),
    ("fever for 3 days", 3),
    ("Pikin 2 years, hot body 3 days, e dey convulse", 3),
    ("body dey hot 4 days", 4),
    ("2 days of fever", 2),
    ("fever, took coartem for 3 days", None),
    ("fever 2 days then 4 days of fever", None),
])
def test_fever_days(text, days):
    assert stated_fever_days(text) == days


@pytest.mark.parametrize("text,rdt", [
    ("RDT negative", "negative"),
    ("mRDT +ve", "positive"),
    ("RDT-negative", "negative"),
    ("malaria test was positive", "positive"),
    ("RDT na positive", "positive"),
    ("RDT not done", "not_done"),
    ("RDT positive, repeat RDT negative", None),
    ("no test", None),
])
def test_rdt(text, rdt):
    assert stated_rdt(text) == rdt
