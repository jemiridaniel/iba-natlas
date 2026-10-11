# Ibà-ATLAS — NAIC 2026 submission

**National AI Innovation Challenge 2026 · Track B (Innovation & Enterprise) ·
Problem Statement 3: Sectoral Fine-Tuning (Health)**

| | |
|---|---|
| Repository | https://github.com/jemiridaniel/iba-natlas |
| Model (LoRA adapter on N-ATLaS) | https://huggingface.co/jemiridaniel/Iba-ATLAS-8B-LoRA-Powered-by-Awarri |
| Benchmark (raw outputs, every case) | [`results/`](results/) · scored by `python -m natlas.score` |
| Live demo | Kaggle notebook `natlas/kaggle_serve_demo.ipynb` (serves the model + app, opens a public link) |
| Video | _link to be added_ |

---

## 1. Summary (one paragraph)

Ibà-ATLAS is N-ATLaS fine-tuned to triage febrile patients from the short notes Nigerian
primary health care workers write, in English or Nigerian Pidgin. It powers Ibà, a fever-triage
copilot that runs entirely on N-ATLaS: base N-ATLaS reads the case, checks live outbreak reports
and reasons over national and WHO guidelines; Ibà-ATLAS gives an independent triage that can only
make the decision more urgent; and the patient receives a plain explanation in Yorùbá, Hausa or
Igbo. On a 710-case held-out benchmark, fine-tuning cut under-triage from 23.1% (base N-ATLaS) to
3.5% and raised danger-sign recall from 17.3% to 100%. Deterministic danger-sign rules and a
guideline dose table sit around the models. It is decision support, not diagnosis.

## 2. Problem

Community health extension workers and PHC nurses see most febrile patients in Nigeria first, often
alone and without a doctor to ask. Severe malaria, Lassa fever, meningitis and cholera can look
like ordinary fever early, and missing a danger sign — a child who cannot drink, a convulsion —
costs lives. Health workers write in English and Pidgin; patients often understand Yorùbá, Hausa
or Igbo better. General-purpose models are not tuned for either.

## 3. What we built (working artefact)

1. **Ibà-ATLAS** — a QLoRA adapter on N-ATLaS (8B), published on Hugging Face under the N-ATLaS
   Terms with "Powered by Awarri" in its name. Input: a worker's note. Output: one JSON object with
   triage level, danger signs, malaria risk and a one-line reason.
2. **Ibà on N-ATLAS** (`app/`) — a mobile-first web app (installable PWA). Every text-generation
   step uses N-ATLaS: intake (note → structured case), outbreak extraction from live NCDC/WHO
   search results, guideline-grounded reasoning with verified quotes, the referral note, and the
   patient explanation in Yorùbá/Hausa/Igbo. Ibà-ATLAS runs in parallel as an escalate-only check.
3. **Serving** (`natlas/serve.py`) — an OpenAI-compatible server for N-ATLaS + the adapter that
   constrains every reply to the step's JSON schema, so the 8B model cannot break the format. Runs
   on a free Kaggle T4 x2.
4. **Reproducible training and benchmark** (`natlas/`) — dataset builder, training, generation,
   scoring, and Kaggle notebooks for each step.

## 4. N-ATLAS integration evidence

| Component | How N-ATLaS is used | Where |
|---|---|---|
| Fine-tuned model | Ibà-ATLAS is a LoRA adapter trained on `NCAIR1/N-ATLaS` weights (4-bit QLoRA); it cannot run without the N-ATLaS base | `natlas/train.py`, HF model card |
| App: intake, outbreak extraction, reasoning, referral note | Base `NCAIR1/N-ATLaS` via the OpenAI-compatible server (`MODEL_FAST/MID/REASON`) | `app/.env.natlas.example`, decision trace |
| App: triage check | `iba-atlas` (the adapter) on the same server | `app/backend/app/graph/nodes.py` (`atlas_triage`) |
| App: patient explanation in Yorùbá / Hausa / Igbo | Base N-ATLaS, chosen for its Nigerian-language ability | `app/backend/app/graph/prompts.py` (`EXPLAIN_SYSTEM`) |
| Verification | Every response's decision trace lists the model per step; the server log lists every call | `kaggle_serve_demo.ipynb` cell 4 output |

No other general-purpose LLM generates any text. Guideline retrieval uses a text-embedding model
(Qwen3-Embedding-8B) to search the guideline index; it generates no text.

## 5. Real-world validation

**Benchmark: 710 held-out cases, base N-ATLaS vs Ibà-ATLAS** (required: 500+)

| | Base N-ATLaS | Ibà-ATLAS v1 | **Ibà-ATLAS v2 (submitted)** |
|---|---|---|---|
| Under-triage, all 710 (lower is better) | 23.1% | 5.9% | **3.5%** |
| Exact triage, all 710 | 50.6% | 86.1% | **93.5%** |
| Refer-now recall | 77.6% | 92.2% | **97.4%** |
| Danger-sign recall / precision | 17.3% / 19.3% | 93.0% / 99.1% | **100% / 98.8%** |
| Valid JSON | 99.9% | 100% | **100%** |
| Under-triage, 61 Nigerian vignettes | 31.1% | 44.3% | **18.0%** |
| Under-triage, 211 Pidgin cases | 31.3% | 4.3% | **1.9%** |

- **Data used:** 649 held-out records from a public, de-identified hospital malaria dataset
  (Kaggle; never trained on), the dataset from our malaria-llm-cdss study; 61 synthetic Nigerian
  fever vignettes (malaria, Lassa, cholera, meningitis, paediatric danger signs, Pidgin). No
  patient data from Nigerian clinics was collected or used; every case in the app demo is
  synthetic.
- **Live end-to-end runs** on the deployed app (Kaggle T4 x2): complete assessments with live
  NCDC/WHO outbreak search, retrieved guideline passages with quote-verified citations, and patient
  explanations in Yorùbá.
- **Native-language input (preliminary):** a note written entirely in Yorùbá ("child, 3, fever 4
  days, cannot drink, vomits everything"). Ibà-ATLAS — trained only on English and Pidgin — flagged
  both danger signs, and Ibà escalated to Refer now; the English/Pidgin text rules could not read
  the note. One case, reported as a demonstration, not a result.

**What we learned and fixed (v1 → v2).** v1 trained only on the hospital records, whose single
danger sign is confusion. It improved sharply on held-out hospital records but got worse on the
Nigerian vignettes: it had learned that confusion is the only danger sign. v2 added 628 synthetic
cases covering all ten WHO/IMCI danger signs and urgent features (stiff neck, bulging fontanelle,
non-fading rash, severe dehydration, Lassa exposure), every phrase checked against Ibà's
deterministic rules and screened against the test vignettes. v2 fixed severe malaria (6/10 →
10/10) and paediatric danger signs (2/6 → 6/6) without over-triaging uncomplicated malaria.

**Limitations, stated plainly.** The hospital records are Indian, not Nigerian, and their risk
label is a score threshold. Vignette gold labels await clinician review. The v2 data categories
were chosen after seeing v1's failures, so the vignettes are no longer a fully blind test for v2.
The app's long-form reasoning comes from base N-ATLaS and can misstate details; the triage
decision is protected by rules and Ibà-ATLAS, and a clinician always decides.

## 6. Impact potential

- Primary care is where most febrile illness in Nigeria is first seen; earlier, more consistent
  recognition of danger signs and outbreak-linked fevers (Lassa, cholera, meningitis) means
  faster referral.
- Explanations in Yorùbá, Hausa and Igbo help patients and caregivers understand why they are
  referred and what to do now.
- An open, N-ATLaS-derived health model and benchmark that other Nigerian health tools can reuse.

## 7. Scalability and sustainability

- Runs on modest hardware: an 8B model with a 168 MB adapter; demonstrated on free T4 GPUs.
- Open source; standard OpenAI-compatible serving, so it can move to any GPU host or a
  government cloud.
- Next steps, in order:
  1. **Clinician-written synthetic test set:** new CHEW-style cases written and labelled by the
     clinical lead, never seen in training (no patient data, so no ethics approval needed).
  2. **Real-world validation:** de-identified clinic cases, only after ethics approval from a
     health research ethics committee, compliance with the Nigeria Data Protection Act (NDPA)
     2023 and facility consent, with a clinician's label for every case.
  3. Fine-tuning the reasoning step; danger-sign rules and tested input in Yorùbá, Hausa and
     Igbo; then a supervised pilot with PHCs in one state.
- The N-ATLaS licence caps use at 1,000 active users per 30 days; a pilot fits, and larger roll-out
  would need a licence from Awarri.

## 8. Team

| Name | Role | Affiliation | GitHub |
|---|---|---|---|
| Jemiri Daniel Taiwo | ML engineering, data, app | HeyPulse Technologies; MSc Information Management, Ahmadu Bello University | [@jemiridaniel](https://github.com/jemiridaniel) |
| Amos Oluwasegun Oluba, B.Pharm, PharmD, MPSN | Clinical Lead: label review, clinical safety | Clinical Pharmacist, Federal Medical Centre, Epe, Lagos State | [@AmazingAmos](https://github.com/AmazingAmos) |

Prior work: Ibà fever-triage copilot (Nebius × NVIDIA hackathon build, Nemotron); malaria-llm-cdss,
an explainable malaria decision-support study.

## 9. Licence and attribution

Ibà-ATLAS is derived from **N-ATLaS** by Awarri Technologies and the Federal Ministry of
Communications, Innovation and Digital Economy, and is released under the N-ATLaS Terms of Use.
Code: MIT.
