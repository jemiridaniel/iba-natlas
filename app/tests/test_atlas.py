"""N-ATLAS build: Ibà-ATLAS triage step, patient-language explanation, endpoints, JSON schema."""

from pathlib import Path
from types import SimpleNamespace

from backend.app.graph.pipeline import run_triage, stream_triage
from backend.app.graph.prompts import ATLAS_SYSTEM
from backend.app.graph.state import ReasonOutput, TriageRequest
from backend.app.llm.client import LLMClient, structured_format
from backend.app.schemas import AtlasTriage, DangerSignCode, TriageLevel
from tests.pipeline_fakes import COMPOSE_EN, make_deps, make_settings
from tests.test_pipeline_e2e import MALARIA_INTAKE, MALARIA_REASON, MALARIA_TEXT

ROOT = Path(__file__).resolve().parents[2]
ATLAS = {
    "model_atlas": "iba-atlas",
    "model_prices": {
        m: {"input": 0.0, "output": 0.0} for m in ("fast", "mid", "reason", "iba-atlas")
    },
}


def _run(tmp_path, atlas_reply, **extra):
    replies = {
        "intake": MALARIA_INTAKE,
        "reason": MALARIA_REASON,
        "compose": COMPOSE_EN,
        "atlas": atlas_reply,
        **extra.pop("replies", {}),
    }
    deps, llm = make_deps(tmp_path, replies, search=None, **ATLAS, **extra)
    return run_triage(
        TriageRequest(text=MALARIA_TEXT, state="Lagos", **extra.pop("request", {})), deps
    ), llm


def test_atlas_prompt_matches_training_prompt() -> None:
    # The adapter was fine-tuned on natlas/prompts.py; any drift silently degrades it.
    ns: dict = {}
    exec((ROOT / "natlas" / "prompts.py").read_text(), ns)
    assert ns["SYSTEM_PROMPT"] == ATLAS_SYSTEM


def test_atlas_can_raise_urgency_and_says_so(tmp_path: Path) -> None:
    r, llm = _run(
        tmp_path,
        {
            "triage_level": "refer_24h",
            "danger_signs": [],
            "malaria_risk": "high",
            "reason": "Many malaria features.",
        },
    )
    assert r.triage_level == TriageLevel.REFER_24H
    assert "Ibà-ATLAS" in r.triage_rationale
    assert r.atlas is not None and r.atlas.malaria_risk == "high"
    assert "atlas" in llm.tasks()
    step = next(s for s in r.decision_trace.steps if s.step == "atlas_triage")
    assert step.model == "iba-atlas" and step.calls == 1


def test_atlas_can_never_lower_urgency(tmp_path: Path) -> None:
    reason = {**MALARIA_REASON, "triage_level": "refer_now", "triage_rationale": "Refer."}
    r, _ = _run(
        tmp_path,
        {
            "triage_level": "treat_monitor",
            "danger_signs": [],
            "malaria_risk": "standard",
            "reason": "Mild.",
        },
        replies={"reason": reason},
    )
    assert r.triage_level == TriageLevel.REFER_NOW
    assert "Ibà-ATLAS" not in r.triage_rationale


def test_atlas_danger_sign_is_shown_and_refers(tmp_path: Path) -> None:
    r, _ = _run(
        tmp_path,
        {
            "triage_level": "refer_now",
            "danger_signs": ["prostration", "made_up_sign"],
            "malaria_risk": "high",
            "reason": "Cannot sit.",
        },
    )
    assert r.triage_level == TriageLevel.REFER_NOW
    hit = next(h for h in r.danger_signs if h.code == DangerSignCode.PROSTRATION)
    assert hit.source == "llm_atlas"


def test_atlas_failure_never_blocks_the_result(tmp_path: Path) -> None:
    r, _ = _run(tmp_path, "not json at all")
    assert r.status == "complete"
    assert r.triage_level == TriageLevel.TREAT_MONITOR
    step = next(s for s in r.decision_trace.steps if s.step == "atlas_triage")
    assert step.note


def test_atlas_off_by_default(tmp_path: Path) -> None:
    deps, llm = make_deps(
        tmp_path,
        {"intake": MALARIA_INTAKE, "reason": MALARIA_REASON, "compose": COMPOSE_EN},
        search=None,
    )
    r = run_triage(TriageRequest(text=MALARIA_TEXT, state="Lagos"), deps)
    assert "atlas" not in llm.tasks() and r.atlas is None
    assert all(s.step != "atlas_triage" for s in r.decision_trace.steps)


def test_patient_explanation_in_yoruba(tmp_path: Path) -> None:
    atlas = {
        "triage_level": "treat_monitor",
        "danger_signs": [],
        "malaria_risk": "standard",
        "reason": "No danger signs.",
    }
    yoruba = {"text": "A ó tọ́jú rẹ níbí. Padà wá tí ara rẹ kò bá yá. Lo 80/480 mg."}
    r, llm = _run(
        tmp_path,
        atlas,
        patient_explanation=True,
        replies={"explain": yoruba},
        request={"patient_language": "yo"},
    )
    assert r.patient_explanation is not None
    assert r.patient_explanation.language == "yo"
    assert "80/480" not in r.patient_explanation.text  # doses stripped, as everywhere
    assert "Yoruba" in llm.messages_for("explain")[0]["content"]


def test_explanation_needs_the_setting(tmp_path: Path) -> None:
    atlas = {
        "triage_level": "treat_monitor",
        "danger_signs": [],
        "malaria_risk": "standard",
        "reason": "ok",
    }
    r, llm = _run(tmp_path, atlas, request={"patient_language": "ha"})
    assert r.patient_explanation is None and "explain" not in llm.tasks()


def test_stream_emits_atlas_event(tmp_path: Path) -> None:
    deps, _ = make_deps(
        tmp_path,
        {
            "intake": MALARIA_INTAKE,
            "reason": MALARIA_REASON,
            "compose": COMPOSE_EN,
            "atlas": {
                "triage_level": "treat_monitor",
                "danger_signs": [],
                "malaria_risk": "standard",
                "reason": "ok",
            },
        },
        search=None,
        **ATLAS,
    )
    events = dict(stream_triage(TriageRequest(text=MALARIA_TEXT, state="Lagos"), deps))
    assert events["atlas_triage"]["atlas"]["triage_level"] == "treat_monitor"
    assert events["final"]["atlas"]["malaria_risk"] == "standard"


def test_json_schema_mode_sends_the_step_schema(tmp_path: Path) -> None:
    deps, llm = make_deps(
        tmp_path,
        {"intake": MALARIA_INTAKE, "reason": MALARIA_REASON, "compose": COMPOSE_EN},
        search=None,
        llm_structured_output="json_schema",
    )
    run_triage(TriageRequest(text=MALARIA_TEXT, state="Lagos"), deps)
    kw = next(kw for t, kw in llm.calls if t == "reason")
    assert kw["response_format"] == structured_format(ReasonOutput)
    assert kw["response_format"]["json_schema"]["name"] == "ReasonOutput"


def test_atlas_schema_keeps_known_signs_only() -> None:
    a = AtlasTriage.model_validate(
        {"triage_level": "refer_now", "danger_signs": ["convulsions", "chest_pain"]}
    )
    assert a.danger_signs == [DangerSignCode.CONVULSIONS]


def test_chat_and_embeddings_use_separate_endpoints(tmp_path: Path) -> None:
    s = make_settings(
        tmp_path, chat_base_url="http://vllm:8000/v1", embed_base_url="http://embed:9000/v1"
    )
    c = LLMClient(s)
    assert str(c.client.base_url).startswith("http://vllm:8000")
    assert str(c.embed_client.base_url).startswith("http://embed:9000")


def test_embeddings_fall_back_to_token_factory(tmp_path: Path) -> None:
    s = make_settings(tmp_path, chat_base_url="http://vllm:8000/v1", nebius_api_key="k")
    c = LLMClient(s)
    assert "tokenfactory.nebius.com" in str(c.embed_client.base_url)
    assert "vllm" in str(c.client.base_url)


def test_injected_client_serves_both(tmp_path: Path) -> None:
    fake = SimpleNamespace()
    c = LLMClient(make_settings(tmp_path), client=fake)
    assert c.client is fake and c.embed_client is fake


def test_escalation_by_atlas_signs_is_credited_to_atlas(tmp_path: Path) -> None:
    # A note the text rules cannot read (Yorùbá): only Ibà-ATLAS finds the danger sign.
    reason = {**MALARIA_REASON, "triage_level": "refer_24h", "triage_rationale": "Refer soon."}
    r, _ = _run(
        tmp_path,
        {
            "triage_level": "refer_now",
            "danger_signs": ["unable_to_drink_or_breastfeed"],
            "malaria_risk": "high",
            "reason": "Danger sign present: refer now.",
        },
        replies={"reason": reason},
    )
    assert r.triage_level == TriageLevel.REFER_NOW
    assert "flagged by Ibà-ATLAS" in r.triage_rationale
    assert "safety rules" not in r.triage_rationale


def test_explain_prompt_forbids_treatment_advice() -> None:
    from backend.app.graph.prompts import EXPLAIN_SYSTEM

    assert "Never recommend or name any medicine" in EXPLAIN_SYSTEM
