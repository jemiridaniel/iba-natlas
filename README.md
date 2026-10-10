# Ibà-ATLAS: N-ATLAS fine-tuned for fever triage in Nigerian primary care

Submission for the **National AI Innovation Challenge 2026**, Problem Statement 3 (Sectoral
Fine-Tuning), health sector.

Ibà-ATLAS is [N-ATLaS](https://huggingface.co/NCAIR1/N-ATLaS) (Nigeria's open multilingual
LLM, Llama-3 8B) adapted with QLoRA to read a health worker's short note about a febrile
patient, in English or Nigerian Pidgin, and return a structured triage decision: triage level,
danger signs, malaria risk and a one-line reason. It is the model behind the Ibà fever-triage
copilot. **Decision support, not diagnosis: a clinician always decides.**

## Results (710-case held-out benchmark)

| | Base N-ATLaS | Ibà-ATLAS v1 | **Ibà-ATLAS v2** |
|---|---|---|---|
| **Under-triage, all 710** (lower is better) | 23.1% | 5.9% | **3.5%** |
| Exact triage, all 710 | 50.6% | 86.1% | **93.5%** |
| Refer-now recall | 77.6% | 92.2% | **97.4%** |
| Danger-sign recall / precision | 17.3% / 19.3% | 93.0% / 99.1% | **100% / 98.8%** |
| Valid JSON | 99.9% | 100% | **100%** |
| **Under-triage, 61 Nigerian vignettes** | 31.1% | 44.3% | **18.0%** |
| Exact triage, vignettes | 68.9% | 37.7% | **80.3%** |
| Under-triage, Pidgin cases (211) | 31.3% | 4.3% | **1.9%** |

Full tables: [results/report.md](results/report.md) (overall, per source, per language) and
[results/vignette_categories.md](results/vignette_categories.md). Raw model outputs for every
case are in `results/preds_*.jsonl`; `python -m natlas.score` recomputes every number.

**What happened between v1 and v2.** v1 trained only on the hospital records, whose single
danger sign is confusion. It improved sharply on held-out hospital records but got *worse* on
the Nigerian vignettes (under-triage 31% → 44%): it had learned that confusion is the only
danger sign. v2 adds 628 synthetic worker-style cases covering all ten danger signs and urgent
features without a danger-sign code (stiff neck, bulging fontanelle, non-fading rash, severe
dehydration, Lassa exposure), in English and Pidgin, plus hard negatives. Every phrase is
checked against Ibà's deterministic danger-sign rules, and 26 generated cases too similar to a
vignette were dropped. v2 fixed severe malaria (6/10 → 10/10 correct) and paediatric danger
signs (2/6 → 6/6) without over-triaging uncomplicated malaria (10/10 kept).

**Caveat on method.** The *categories* added in v2 were chosen after seeing where v1 failed on
the vignettes. No vignette text or near-copy was trained on, but the vignette set is therefore
no longer a fully blind test for v2. A fresh, clinician-written test set is the next step.

**Remaining misses (v2, 11 of 61):** 7 suspected-Lassa cases where the gold label assumes an
active outbreak in the patient's state. A text-only model cannot know that; in the app, the live
outbreak search and Lassa case-definition rules escalate these. Also 2 prolonged-fever cases
(possible typhoid/TB), 1 meningitis case and 1 Pidgin case.

> Status: October 2026. Gold labels for the vignettes and for the v0 hospital-record triage
> mapping are pending review by the clinical lead.

## Why fine-tune

Ibà needs a model that (1) always answers in a strict, machine-checkable format, (2) never
under-triages a patient with danger signs, and (3) understands how Nigerian health workers
actually write, including Pidgin. The benchmark measures exactly these against base N-ATLAS.

## Data

| Set | Source | Size |
|---|---|---|
| Train / val | De-identified hospital malaria admissions (Kaggle `programmer3/malaria-diagnosis-dataset`), as used in our malaria-llm-cdss study | 876 / 97 |
| Train / val (v2 only) | Synthetic worker-style cases: ten danger signs, urgent features, hard negatives (`natlas/augment.py`) | 628 / 70 |
| Benchmark A | Held-out records from the same dataset (never trained on) | 649 |
| Benchmark B | Ibà's synthetic Nigerian fever vignettes (malaria, Lassa, cholera, meningitis, paediatric danger signs, Pidgin) | 61 |
| **Benchmark total** | | **710** |

Labels for the hospital records: `malaria_risk` is the dataset's own high/standard risk label;
confusion maps to the danger sign *impaired consciousness*; triage is derived as danger sign →
refer now, high risk → refer within 24 h, otherwise treat and monitor (v0 mapping, under
clinical review). Each record is rendered as a short worker-style note, 30% in Pidgin.

Limitations we state up front: the hospital records come from Indian hospitals, not
Nigerian ones; their risk label is a score threshold, not a lab result; benchmark B is synthetic
and its gold labels are pending clinician review.

## Metrics

Valid-JSON rate, exact triage accuracy, **under-triage rate** (headline safety metric: invalid
output counts as missed), refer-now recall, danger-sign recall/precision, malaria-risk accuracy;
reported overall, per source and per language.

## Reproduce

```bash
pip install -r requirements.txt
python -m natlas.build_dataset                       # CPU, deterministic (seed 2026)
python -m natlas.generate --out results/preds_base.jsonl                                # GPU
python -m natlas.train --epochs 2                                                        # GPU
python -m natlas.generate --adapter outputs/iba-atlas-lora --out results/preds_ft.jsonl  # GPU
python -m natlas.score results/preds_base.jsonl results/preds_ft.jsonl                   # CPU
```

Or run `natlas/kaggle_iba_atlas.ipynb` on a free Kaggle T4.

## The app: Ibà on N-ATLAS

[`app/`](app/) is the Ibà fever-triage copilot running entirely on N-ATLaS: intake, outbreak
extraction, guideline-grounded reasoning and the referral note use base N-ATLaS; Ibà-ATLAS adds an
independent triage that can only raise urgency; and the patient gets a plain explanation in
Yorùbá, Hausa or Igbo. Deterministic danger-sign rules and a dose table that the model can never
override stay in place. Setup: [app/README.md](app/README.md#running-on-n-atlas).

| Notebook | What it does |
|---|---|
| `natlas/kaggle_iba_atlas.ipynb` | Fine-tune and benchmark (base vs Ibà-ATLAS) |
| `natlas/kaggle_serve_demo.ipynb` | vLLM serving N-ATLaS + adapter, the app, and a public demo link |

## Licence and attribution

Model derived from N-ATLaS and released under the N-ATLaS Terms of Use. Built on N-ATLaS by
Awarri Technologies and the Federal Ministry of Communications, Innovation and Digital Economy.
Model name carries "Powered by Awarri" as the Terms require. Code: MIT.

## Team

- Jemiri Daniel Taiwo: ML engineering
- (clinical lead to be confirmed)
