"""Versioned Ibà API (/v1) for other apps, e.g. the Magana reading app.

Deterministic rules only: no model call, so answers are instant, free and repeatable.

- GET  /v1/health     liveness
- GET  /v1/version    service, API and rule versions
- GET  /v1/questions  ?mode=clinician|self_report&lang=en|ha|yo|ig
- POST /v1/assess     {mode, answers} -> priority + message code (+ rule detail for clinicians)

Every route needs an X-API-Key header listed in IBA_API_KEYS. Answers are never logged or
stored: only the mode, language and resulting priority are logged.
"""

import logging
import secrets
from datetime import date
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from backend.app.config import get_settings
from backend.app.questions import LASSA_SYMPTOM_OPTIONS, TRANSLATION_STATUS, Lang, Mode, questions
from backend.app.rules import danger_signs
from backend.app.rules.danger_signs import CLINICIAN_RULES_STATUS, CLINICIAN_RULES_VERSION
from backend.app.rules.self_report import (
    SELF_REPORT_RULES_VERSION,
    SELF_REPORT_STATUS,
    assess_self_report,
)
from backend.app.schemas import DangerSignCode, DangerSignHit, PatientCase, TriageLevel
from backend.app.tools.endemicity import Endemicity

log = logging.getLogger("iba.api.v1")

API_VERSION = "1"


def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    keys = [
        k.strip() for k in get_settings().iba_api_keys.get_secret_value().split(",") if k.strip()
    ]
    if not keys:
        raise HTTPException(503, {"code": "api_keys_not_configured"})
    # Compare against every key in constant time, so timing reveals nothing.
    if not x_api_key or not any(secrets.compare_digest(x_api_key, k) for k in keys):
        raise HTTPException(401, {"code": "invalid_api_key"})


router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])


def _service_version() -> str:
    try:
        return version("iba-triage")
    except PackageNotFoundError:
        return "0.1.0"


def _rules(mode: Mode) -> tuple[str, str]:
    if mode == "self_report":
        return SELF_REPORT_RULES_VERSION, SELF_REPORT_STATUS
    return CLINICIAN_RULES_VERSION, CLINICIAN_RULES_STATUS


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/version")
def get_version() -> dict:
    return {
        "service": _service_version(),
        "api": API_VERSION,
        "rules": {
            "clinician": {"version": CLINICIAN_RULES_VERSION, "status": CLINICIAN_RULES_STATUS},
            "self_report": {"version": SELF_REPORT_RULES_VERSION, "status": SELF_REPORT_STATUS},
        },
    }


@router.get("/questions")
def get_questions(mode: Annotated[Mode, Query()], lang: Annotated[Lang, Query()] = "en") -> dict:
    rule_version, rule_status = _rules(mode)
    return {
        "mode": mode,
        "lang": lang,
        "translation_status": TRANSLATION_STATUS[lang],
        "rule_version": rule_version,
        "rule_status": rule_status,
        "questions": questions(mode, lang),
    }


Answer = bool | float | str | list[str] | None


class AssessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Mode
    answers: dict[str, Answer] = Field(default_factory=dict)


class SelfReportOut(BaseModel):
    """Exactly this, nothing more: no diagnosis, no condition names, no danger-sign list."""

    model_config = ConfigDict(extra="forbid")

    priority: Literal["emergency", "urgent", "routine"]
    message: str
    rule_version: str


class ClinicianOut(BaseModel):
    priority: Literal["emergency", "urgent", "routine"]
    message: Literal["refer_now", "refer_within_24h", "no_rule_escalation"]
    triage_level: TriageLevel | None  # None: no rule escalated; clinical judgement decides
    danger_signs: list[DangerSignCode]
    findings: list[str]  # codes, e.g. "lassa_baseline"
    rule_version: str
    basis: Literal["rules_only"] = "rules_only"


def _invalid(question: str, reason: str) -> HTTPException:
    return HTTPException(422, {"code": "invalid_answer", "question": question, "reason": reason})


def _validate(mode: Mode, answers: dict[str, Answer]) -> dict[str, Answer]:
    """Check every answer against its question's type; unknown questions are refused."""
    spec = {q["id"]: q for q in questions(mode, "en")}
    clean: dict[str, Answer] = {}
    for qid, value in answers.items():
        q = spec.get(qid)
        if q is None:
            raise _invalid(qid, "unknown_question")
        if value is None:
            continue
        kind = q["type"]
        if kind == "yes_no" and not isinstance(value, bool):
            raise _invalid(qid, "expected_yes_no")
        if kind == "number" and (
            isinstance(value, bool) or not isinstance(value, int | float) or value < 0
        ):
            raise _invalid(qid, "expected_non_negative_number")
        if kind == "text" and (not isinstance(value, str) or len(value) > 60):
            raise _invalid(qid, "expected_short_text")
        if kind == "choice" and value not in {o["value"] for o in q["options"]}:
            raise _invalid(qid, "unknown_option")
        if kind == "multi_choice" and (
            not isinstance(value, list) or not set(value) <= {o["value"] for o in q["options"]}
        ):
            raise _invalid(qid, "unknown_option")
        clean[qid] = value
    return clean


@lru_cache(maxsize=1)
def _endemicity() -> Endemicity:
    return Endemicity.load()


_PRIORITY = {
    TriageLevel.REFER_NOW: ("emergency", "refer_now"),
    TriageLevel.REFER_24H: ("urgent", "refer_within_24h"),
}


def _assess_clinician(answers: dict[str, Answer]) -> ClinicianOut:
    case = PatientCase(
        fever_days=answers.get("fever_days"),
        temperature_c=answers.get("temperature_c"),
        state=answers.get("state"),
        rdt_result=answers.get("rdt_result"),
        antimalarial_no_response=answers.get("no_treatment_response"),
        symptoms=[s for s in answers.get("lassa_symptoms") or [] if s in LASSA_SYMPTOM_OPTIONS],
    )
    hits = [
        DangerSignHit(code=code, source="answer")
        for code in DangerSignCode
        if answers.get(code.value) is True
    ]
    signals = _endemicity().for_state(case.state, date.today())
    rules = danger_signs.assess(case, signals, extra_hits=hits)
    priority, message = _PRIORITY.get(rules.floor, ("routine", "no_rule_escalation"))
    findings = [f"lassa_{rules.lassa.basis}"] if rules.lassa else []
    return ClinicianOut(
        priority=priority,
        message=message,
        triage_level=rules.floor,
        danger_signs=[h.code for h in rules.danger_signs],
        findings=findings,
        rule_version=CLINICIAN_RULES_VERSION,
    )


@router.post("/assess", response_model=SelfReportOut | ClinicianOut)
def assess(body: AssessRequest) -> SelfReportOut | ClinicianOut:
    answers = _validate(body.mode, body.answers)
    if body.mode == "self_report":
        r = assess_self_report(answers)  # type: ignore[arg-type]
        out: SelfReportOut | ClinicianOut = SelfReportOut(
            priority=r.priority.value, message=r.message, rule_version=r.rule_version
        )
    else:
        out = _assess_clinician(answers)
    log.info("assess mode=%s priority=%s", body.mode, out.priority)  # never the answers
    return out
