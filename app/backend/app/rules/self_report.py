"""Self-report rules: a person (or their carer) answers yes/no questions about themselves.

PENDING CLINICAL SIGN-OFF BY DR. AMOS. These rules are a draft and must not be used with
real patients until he signs them off (record the sign-off in docs/CLINICAL_REVIEW.md and
bump SELF_REPORT_RULES_VERSION).

The output is deliberately minimal: a priority, a message code and the rule version. No
diagnosis, no condition names and no danger-sign list ever leave this module, because the
answer goes straight to a member of the public (e.g. through the Magana reading app).

Danger signs map one-to-one onto the clinician rules' DangerSignCode, so both modes escalate
on the same criteria (WHO / NMEP severe malaria features, WHO IMCI general danger signs).
"""

from dataclasses import dataclass
from enum import StrEnum

from backend.app.schemas import DangerSignCode

SELF_REPORT_RULES_VERSION = "self-report-0.1.0"
SELF_REPORT_STATUS = "pending clinical sign-off by Dr. Amos"

D = DangerSignCode

# Question id (plain-language) -> the clinician rule it stands for.
DANGER_QUESTIONS: dict[str, DangerSignCode] = {
    "fits": D.CONVULSIONS,
    "cannot_drink": D.UNABLE_TO_DRINK,
    "vomits_everything": D.VOMITING_EVERYTHING,
    "very_sleepy": D.LETHARGY_UNCONSCIOUS,
    "bleeding": D.BLEEDING,
    "yellow_eyes": D.JAUNDICE,
    "hard_breathing": D.RESPIRATORY_DISTRESS,
    "very_pale": D.SEVERE_PALLOR,
    "dark_urine": D.DARK_URINE,
    "too_weak": D.PROSTRATION,
}

# Fever this long, with no danger sign, gets a faster nurse call-back.
URGENT_FEVER_DAYS = 2


class Priority(StrEnum):
    EMERGENCY = "emergency"  # go to a clinic now / call 112
    URGENT = "urgent"  # nurse calls back within 2 hours
    ROUTINE = "routine"  # nurse calls back during working hours


# Message codes; the calling app words them in the user's language.
MESSAGES = {
    Priority.EMERGENCY: "go_to_clinic_now",
    Priority.URGENT: "nurse_call_within_2_hours",
    Priority.ROUTINE: "nurse_call_working_hours",
}


@dataclass(frozen=True)
class SelfReportResult:
    priority: Priority
    message: str
    rule_version: str = SELF_REPORT_RULES_VERSION


def assess_self_report(answers: dict[str, bool | float | None]) -> SelfReportResult:
    """Any danger sign -> emergency; fever for URGENT_FEVER_DAYS or more -> urgent; else routine.

    Unanswered questions count as "no": the caller decides which questions to ask, and a
    danger sign is only ever added, never inferred away.
    """
    if any(answers.get(q) is True for q in DANGER_QUESTIONS):
        priority = Priority.EMERGENCY
    elif answers.get("fever") is True and (answers.get("fever_days") or 0) >= URGENT_FEVER_DAYS:
        priority = Priority.URGENT
    else:
        priority = Priority.ROUTINE
    return SelfReportResult(priority, MESSAGES[priority])
