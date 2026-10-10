---
base_model: NCAIR1/N-ATLaS
library_name: peft
license: other
license_name: n-atlas-terms-of-use
license_link: https://huggingface.co/NCAIR1/N-ATLaS
language: [en, pcm]
tags: [n-atlas, lora, medical, triage, nigeria, clinical-decision-support]
pipeline_tag: text-generation
---

# Ibà-ATLAS 8B LoRA (Powered by Awarri)

A LoRA adapter for **[N-ATLaS](https://huggingface.co/NCAIR1/N-ATLaS)**, Nigeria's multilingual
LLM, fine-tuned to triage febrile patients from the short notes primary health care workers write,
in English or Nigerian Pidgin. It returns one JSON object: triage level, danger signs, malaria
risk and a one-line reason. It powers the triage check in the
[Ibà fever-triage copilot](https://github.com/jemiridaniel/iba-natlas).

**Decision support, not diagnosis. A health worker or clinician always decides.** In Ibà, this
model can only raise urgency, never lower it, and deterministic danger-sign rules and a
guideline dose table sit around it.

## Results (710-case held-out benchmark)

| | Base N-ATLaS | **Ibà-ATLAS (this adapter)** |
|---|---|---|
| Under-triage, all 710 (lower is better) | 23.1% | **3.5%** |
| Exact triage, all 710 | 50.6% | **93.5%** |
| Refer-now recall | 77.6% | **97.4%** |
| Danger-sign recall / precision | 17.3% / 19.3% | **100% / 98.8%** |
| Under-triage, 61 Nigerian vignettes | 31.1% | **18.0%** |
| Under-triage, 211 Pidgin cases | 31.3% | **1.9%** |

Benchmark: 649 held-out hospital malaria records (never trained on) and 61 synthetic Nigerian
fever vignettes (malaria, Lassa, cholera, meningitis, paediatric danger signs, Pidgin). Full
tables, raw outputs for every case and the scoring code:
[github.com/jemiridaniel/iba-natlas](https://github.com/jemiridaniel/iba-natlas).

## Usage

Use exactly this system prompt (the adapter was trained on it):

```text
You are Ibà, a fever-triage assistant for primary health care workers in Nigeria.
The worker describes a febrile patient in English or Nigerian Pidgin.
Reply with ONE JSON object and nothing else, with these keys:
- "triage_level": one of ['refer_now', 'refer_24h', 'treat_monitor']
- "danger_signs": list drawn only from ['lethargy_or_unconsciousness', 'convulsions', 'unable_to_drink_or_breastfeed', 'vomiting_everything', 'prostration', 'respiratory_distress', 'dark_urine', 'jaundice', 'bleeding', 'severe_pallor'] (empty list if none)
- "malaria_risk": "high", "standard", or "not_applicable" (not_applicable when malaria is unlikely)
- "reason": one short sentence for the health worker
Rules: any danger sign means "refer_now". Never give drug doses. You support, a clinician decides.
```

With vLLM (OpenAI-compatible):

```bash
vllm serve NCAIR1/N-ATLaS --enable-lora --max-lora-rank 16 \
  --lora-modules iba-atlas=jemiridaniel/Iba-ATLAS-8B-LoRA-Powered-by-Awarri \
  --dtype half --max-model-len 8192
```

With Transformers + PEFT:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = """..."""  # paste the system prompt above, exactly

tok = AutoTokenizer.from_pretrained("NCAIR1/N-ATLaS")
base = AutoModelForCausalLM.from_pretrained("NCAIR1/N-ATLaS", dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, "jemiridaniel/Iba-ATLAS-8B-LoRA-Powered-by-Awarri")
messages = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Pikin 2 years, hot body 3 days, e no fit drink, RDT positive"}]
ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=200, do_sample=False)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
```

## Training

QLoRA (4-bit NF4, r=16, alpha=32, all attention and MLP projections), 2 epochs, lr 2e-4, one
Kaggle T4. Loss on the assistant reply only. Data (1,504 train / 167 val):

- 876 de-identified hospital malaria admissions (Kaggle `programmer3/malaria-diagnosis-dataset`),
  rendered as worker-style notes, 30% in Pidgin. Malaria risk is the dataset's own label.
- 628 synthetic cases covering all ten WHO/IMCI danger signs and urgent features without a
  danger-sign code (stiff neck, bulging fontanelle, non-fading rash, severe dehydration, Lassa
  exposure), with hard negatives. Every danger-sign phrase is checked against Ibà's
  deterministic rules; cases too similar to a benchmark vignette were removed.

## Limitations

- **Not clinically validated.** Gold labels for the vignettes, and the v0 mapping from hospital
  records to triage levels, are pending review by a clinician.
- The hospital records are from Indian hospitals, not Nigerian ones; their risk label is a
  score threshold, not a lab result.
- The synthetic categories were chosen after an earlier version failed on the vignettes, so the
  vignette set is not a fully blind test for this version.
- Text only: it cannot know local outbreak status. Most remaining vignette misses are suspected
  Lassa cases whose urgency depends on an active outbreak; Ibà's outbreak search and rules handle
  those.
- It never gives drug doses and must not be used on its own to make clinical decisions.

## Licence and attribution

Derived from **N-ATLaS** and released under the
[N-ATLaS Terms of Use](https://huggingface.co/NCAIR1/N-ATLaS). N-ATLaS is by **Awarri
Technologies** and the **Federal Ministry of Communications, Innovation and Digital Economy**
of Nigeria. Use is limited to organisations with at most 1,000 active end-users in a rolling
30-day period; larger deployments need a licence from Awarri.

Built by Jemiri Daniel Taiwo (HeyPulse Technologies) for the National AI Innovation Challenge 2026.
