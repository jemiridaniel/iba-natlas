# Ibà-ATLAS: N-ATLAS fine-tuned for fever triage in Nigerian primary care

Submission for the **National AI Innovation Challenge 2026**, Problem Statement 3 (Sectoral
Fine-Tuning), health sector.

Ibà-ATLAS is [N-ATLaS](https://huggingface.co/NCAIR1/N-ATLaS) (Nigeria's open multilingual
LLM, Llama-3 8B) adapted with QLoRA to read a health worker's short note about a febrile
patient, in English or Nigerian Pidgin, and return a structured triage decision: triage level,
danger signs, malaria risk and a one-line reason. It is the model behind the Ibà fever-triage
copilot. **Decision support, not diagnosis: a clinician always decides.**

> Status: work in progress (October 2026). Benchmark numbers are added after the run.

## Why fine-tune

Ibà needs a model that (1) always answers in a strict, machine-checkable format, (2) never
under-triages a patient with danger signs, and (3) understands how Nigerian health workers
actually write, including Pidgin. The benchmark measures exactly these against base N-ATLAS.

## Data

| Set | Source | Size |
|---|---|---|
| Train / val | De-identified hospital malaria admissions (Kaggle `programmer3/malaria-diagnosis-dataset`), as used in our malaria-llm-cdss study | 876 / 97 |
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

## Licence and attribution

Model derived from N-ATLaS and released under the N-ATLaS Terms of Use. Built on N-ATLaS by
Awarri Technologies and the Federal Ministry of Communications, Innovation and Digital Economy.
Model name carries "Powered by Awarri" as the Terms require. Code: MIT.

## Team

- Jemiri Daniel Taiwo: ML engineering
- (clinical lead to be confirmed)
