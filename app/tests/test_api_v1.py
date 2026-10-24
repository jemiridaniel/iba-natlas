"""The versioned /v1 API: API key, questions in four languages, self-report and clinician
assessment, and that answers never reach the logs. Deterministic rules only, no network."""

import logging

import pytest
from fastapi.testclient import TestClient

import backend.app.main as main
from backend.app.config import get_settings
from backend.app.questions import LANGS
from backend.app.rules.danger_signs import CLINICIAN_RULES_VERSION
from backend.app.rules.self_report import (
    DANGER_QUESTIONS,
    SELF_REPORT_RULES_VERSION,
    SELF_REPORT_STATUS,
)
from backend.app.schemas import DANGER_SIGN_LABELS, DangerSignCode

KEY = {"X-API-Key": "test-key-2"}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("IBA_API_KEYS", "test-key-1, test-key-2")
    get_settings.cache_clear()
    yield TestClient(main.app)
    get_settings.cache_clear()


def assess(client: TestClient, mode: str, **answers) -> dict:
    resp = client.post("/v1/assess", json={"mode": mode, "answers": answers}, headers=KEY)
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---------- API key


@pytest.mark.parametrize(
    "method, path",
    [
        ("get", "/v1/health"),
        ("get", "/v1/version"),
        ("get", "/v1/questions?mode=self_report"),
        ("post", "/v1/assess"),
    ],
)
def test_every_route_needs_a_valid_key(client: TestClient, method: str, path: str) -> None:
    for headers in ({}, {"X-API-Key": "wrong"}):
        resp = getattr(client, method)(path, headers=headers)
        assert resp.status_code == 401
        assert resp.json()["detail"] == {"code": "invalid_api_key"}


def test_no_configured_keys_closes_the_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IBA_API_KEYS", "")
    get_settings.cache_clear()
    try:
        resp = TestClient(main.app).get("/v1/health", headers=KEY)
        assert resp.status_code == 503
        assert resp.json()["detail"] == {"code": "api_keys_not_configured"}
    finally:
        get_settings.cache_clear()


def test_web_app_health_stays_open_for_the_container_check(client: TestClient) -> None:
    assert client.get("/health").status_code == 200


# ---------- health and version


def test_health_and_version(client: TestClient) -> None:
    assert client.get("/v1/health", headers=KEY).json() == {"status": "ok"}
    v = client.get("/v1/version", headers=KEY).json()
    assert v["api"] == "1"
    assert v["rules"]["self_report"] == {
        "version": SELF_REPORT_RULES_VERSION,
        "status": "pending clinical sign-off by Dr. Amos",
    }
    assert v["rules"]["clinician"]["version"] == CLINICIAN_RULES_VERSION


# ---------- questions


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("mode", ["clinician", "self_report"])
def test_questions_in_every_language(client: TestClient, mode: str, lang: str) -> None:
    body = client.get(f"/v1/questions?mode={mode}&lang={lang}", headers=KEY).json()
    assert body["mode"] == mode and body["lang"] == lang
    ids = [q["id"] for q in body["questions"]]
    assert len(ids) == len(set(ids))
    assert all(q["text"].strip() for q in body["questions"])
    for q in body["questions"]:
        for opt in q.get("options", []):
            assert opt["text"].strip()
    if lang != "en":
        assert body["translation_status"].startswith("draft")


def test_same_question_ids_in_every_language(client: TestClient) -> None:
    for mode in ("clinician", "self_report"):
        ids = {
            lang: [
                q["id"]
                for q in client.get(f"/v1/questions?mode={mode}&lang={lang}", headers=KEY).json()[
                    "questions"
                ]
            ]
            for lang in LANGS
        }
        assert all(v == ids["en"] for v in ids.values())


def test_self_report_questions_name_no_condition(client: TestClient) -> None:
    body = client.get("/v1/questions?mode=self_report&lang=en", headers=KEY).json()
    assert body["rule_status"] == SELF_REPORT_STATUS
    assert all("label" not in q for q in body["questions"])
    text = " ".join(q["text"].lower() for q in body["questions"])
    for word in ("malaria", "lassa", "jaundice", "convulsion", "cholera", "meningitis"):
        assert word not in text


def test_clinician_questions_cover_every_danger_sign(client: TestClient) -> None:
    body = client.get("/v1/questions?mode=clinician", headers=KEY).json()
    labels = {q["id"]: q.get("label") for q in body["questions"]}
    for code in DangerSignCode:
        assert labels[code.value] == DANGER_SIGN_LABELS[code]


def test_bad_mode_or_language_is_rejected(client: TestClient) -> None:
    assert client.get("/v1/questions?mode=doctor", headers=KEY).status_code == 422
    assert client.get("/v1/questions?mode=self_report&lang=fr", headers=KEY).status_code == 422


# ---------- self-report


@pytest.mark.parametrize("question", list(DANGER_QUESTIONS))
def test_any_danger_sign_is_an_emergency(client: TestClient, question: str) -> None:
    body = assess(client, "self_report", fever=False, **{question: True})
    assert body == {
        "priority": "emergency",
        "message": "go_to_clinic_now",
        "rule_version": SELF_REPORT_RULES_VERSION,
    }


def test_two_days_of_fever_is_urgent(client: TestClient) -> None:
    body = assess(client, "self_report", fever=True, fever_days=2)
    assert (body["priority"], body["message"]) == ("urgent", "nurse_call_within_2_hours")


def test_otherwise_routine(client: TestClient) -> None:
    for answers in ({}, {"fever": True, "fever_days": 1}, {"fever": False}):
        body = assess(client, "self_report", **answers)
        assert (body["priority"], body["message"]) == ("routine", "nurse_call_working_hours")


def test_self_report_result_has_nothing_but_priority_message_and_version(
    client: TestClient,
) -> None:
    body = assess(client, "self_report", fever=True, fever_days=4, bleeding=True, yellow_eyes=True)
    assert set(body) == {"priority", "message", "rule_version"}
    flat = str(body).lower()
    for word in ("lassa", "malaria", "jaundice", "bleeding", "diagnos"):
        assert word not in flat


# ---------- clinician


def test_clinician_danger_sign_refers_now(client: TestClient) -> None:
    body = assess(client, "clinician", fever_days=1, convulsions=True)
    assert body["priority"] == "emergency"
    assert body["message"] == "refer_now"
    assert body["triage_level"] == "refer_now"
    assert body["danger_signs"] == ["convulsions"]
    assert body["rule_version"] == CLINICIAN_RULES_VERSION
    assert body["basis"] == "rules_only"


def test_clinician_lassa_definition_in_an_endemic_state(client: TestClient) -> None:
    body = assess(
        client, "clinician", fever_days=4, state="Ondo", lassa_symptoms=["vomiting", "sore throat"]
    )
    assert body["triage_level"] == "refer_now"
    assert body["findings"] == ["lassa_baseline"]
    assert body["danger_signs"] == []


def test_clinician_no_rule_escalation_is_said_plainly(client: TestClient) -> None:
    body = assess(client, "clinician", fever_days=1, rdt_result="positive", state="Lagos")
    assert body["priority"] == "routine"
    assert body["message"] == "no_rule_escalation"
    assert body["triage_level"] is None


# ---------- validation


@pytest.mark.parametrize(
    "mode, answers, reason",
    [
        ("self_report", {"pregnant_horse": True}, "unknown_question"),
        ("self_report", {"fits": "yes"}, "expected_yes_no"),
        ("self_report", {"fever_days": -1}, "expected_non_negative_number"),
        ("self_report", {"fever_days": True}, "expected_non_negative_number"),
        ("clinician", {"rdt_result": "maybe"}, "unknown_option"),
        ("clinician", {"lassa_symptoms": ["headache"]}, "unknown_option"),
        ("clinician", {"state": "x" * 61}, "expected_short_text"),
        ("self_report", {"convulsions": True}, "unknown_question"),  # clinician id
    ],
)
def test_bad_answers_are_refused(client: TestClient, mode: str, answers: dict, reason: str) -> None:
    resp = client.post("/v1/assess", json={"mode": mode, "answers": answers}, headers=KEY)
    assert resp.status_code == 422
    assert resp.json()["detail"]["reason"] == reason


def test_extra_fields_are_refused(client: TestClient) -> None:
    resp = client.post(
        "/v1/assess", json={"mode": "self_report", "answers": {}, "name": "Amina"}, headers=KEY
    )
    assert resp.status_code == 422


# ---------- privacy


def test_answers_never_reach_the_logs(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    assess(client, "clinician", fever_days=17, state="Taraba", convulsions=True)
    assess(client, "self_report", fever=True, fever_days=13, bleeding=True)
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "assess mode=" in logged  # the endpoint does log, just not the answers
    for secret in ("17", "13", "Taraba", "convulsions", "bleeding"):
        assert secret not in logged
