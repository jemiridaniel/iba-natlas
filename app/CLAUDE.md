# CLAUDE.md — Ibà: Fever-Triage Copilot

> Working name "Ibà" (Yoruba for fever). Rename freely.
> Full design lives in `docs/SPEC.md`. Read it before starting any non-trivial task.

## What this is
A mobile-first copilot for **primary health care workers in Nigeria** (CHEWs, nurses at PHCs).
A worker describes a febrile patient by text (English or Nigerian Pidgin). The app returns:
- a triage level,
- the danger signs present,
- a refer / treat-and-monitor recommendation with guideline citations,
- a **differential** that accounts for **live outbreak context** in the patient's state (Lassa fever, cholera, meningitis, etc.),
- a printable referral note.

It is **decision support, not diagnosis**. A human clinician always decides.

**This copy is the N-ATLAS build** for the **National AI Innovation Challenge 2026** (NCAIR/NITDA,
Problem Statement 3: sectoral fine-tuning, health). It lives in `iba-natlas/app/`. The original
Nemotron build for the Nebius x NVIDIA hackathon is the separate `iba-triage` repo: never change
that repo from here.

## Hard rules (never break these)
1. **All text generation uses N-ATLaS** (`NCAIR1/N-ATLaS`) or the Ibà-ATLAS adapter derived from it, served by an OpenAI-compatible server (`CHAT_BASE_URL`: vLLM, or `natlas/serve.py`). Never wrap another general-purpose LLM: the challenge disqualifies that. Embeddings for guideline retrieval may use Token Factory (`EMBED_BASE_URL` unset).
1a. **Ibà-ATLAS can only raise urgency**, like the rules. Its system prompt in `graph/prompts.py` must stay identical to `natlas/prompts.py` (a test checks this).
2. **Danger signs are detected by deterministic rules** (`backend/app/rules/danger_signs.py`), both before and after the LLM. The LLM can never downgrade an escalation triggered by a rule.
3. **The LLM never generates drug doses.** Doses come only from the weight-band lookup table in `backend/app/rules/dosing.py`, which is sourced from the national guideline and cited.
4. **Model IDs are never hard-coded.** They come from env vars (`MODEL_FAST`, `MODEL_MID`, `MODEL_REASON`, `MODEL_ATLAS`, `MODEL_EMBED`). Verify IDs against `GET /v1/models`.
5. **N-ATLaS has an 8,192-token context and no reasoning mode.** Keep `REASONING_REASON=false` and prompts well under the window (the longest reason prompt measured was ~5,300 tokens). See `.env.natlas.example`.
6. **No real patient data, ever.** Use synthetic vignettes only. Don't log free-text patient descriptions in production mode.
7. **Never commit secrets** (`.env` is gitignored).
8. **Don't commit guideline PDFs** unless their licence allows it. Commit `data/sources.yaml` plus the ingest script instead.
9. **This is original work owned by Daniel / HeyPulse Technologies.** Do not copy code from any employer project.
10. **N-ATLaS licence:** derivatives keep the N-ATLaS Terms; credit Awarri Technologies and the Federal Ministry of Communications, Innovation and Digital Economy; renamed models end with "Powered by Awarri".

## Stack
- **Backend:** Python 3.12, FastAPI, LangGraph, Pydantic v2, `openai` SDK (pointed at Token Factory), `tavily-python`, NumPy for the vector index (no external vector DB).
- **Frontend:** React + Vite + TypeScript + Tailwind, installable PWA, chat-style UI. Served as static files by FastAPI in production.
- **Tooling:** `uv` for Python deps, `ruff` + `pytest`; `pnpm` for the frontend.
- **Serving:** any OpenAI-compatible server with N-ATLaS + the Ibà-ATLAS LoRA: vLLM where it installs, or `natlas/serve.py` (Transformers + PEFT, no schema enforcement: use `LLM_STRUCTURED_OUTPUT=off`). `natlas/kaggle_serve_demo.ipynb` runs `serve.py`, the app and a public link on a Kaggle T4 x2.

## Layout
```
backend/app/
  main.py            FastAPI app + static frontend
  config.py          env settings (pydantic-settings)
  schemas.py         PatientCase, TriageResult, Citation, OutbreakSignal...
  llm/client.py      Token Factory client, retries, disk cache, token/cost accounting
  llm/router.py      picks model per step; records model used, latency, cost
  graph/             LangGraph nodes: intake -> rules_pre -> retrieve -> outbreak -> reason -> rules_post -> compose
  rag/ingest.py      chunk + embed guideline docs -> data/index/
  rag/store.py       cosine search over the NumPy index
  tools/outbreak.py  Tavily search restricted to trusted domains + state filter, cached per state/day
  rules/danger_signs.py
  rules/dosing.py
frontend/            React PWA
eval/
  vignettes.jsonl    synthetic cases with gold labels
  run_eval.py        runs the pipeline in configs: routed / reason-only / fast-only
  report.py          metrics table + charts -> eval/results/
data/sources.yaml    guideline sources + licence notes
docs/SPEC.md
```

## Commands
```bash
uv sync                                         # install backend deps
uv run uvicorn backend.app.main:app --reload    # dev server
uv run pytest                                   # tests (the LLM is mocked by default)
uv run python -m backend.app.rag.ingest         # build the guideline index
uv run python -m eval.run_eval --config routed --limit 10
cd frontend && pnpm dev
```

## Conventions
- Every LLM call returns **structured JSON** validated by Pydantic. On a parse failure, retry once with the error message fed back, then fail safe (escalate to "refer").
- Every recommendation carries **citations** (guideline doc + section, or outbreak source URL + date).
- Every request logs: models used per step, tokens, latency, and estimated cost. These numbers feed the demo video and the README.
- Tests must mock Token Factory and Tavily. There is one opt-in live smoke test: `pytest -m live`.
- Keep the README current: how N-ATLaS is integrated, and setup and run steps.
- Don't add features beyond the task; list ideas under "Suggestions" in your summary instead.
