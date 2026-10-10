"""Build the Ibà-ATLAS fine-tuning and benchmark sets.

Sources
  data/raw/Malaria_Dataset.csv   1,622 de-identified hospital malaria admissions (Kaggle:
                                 programmer3/malaria-diagnosis-dataset), used in the
                                 malaria-llm-cdss study.
  data/raw/iba_vignettes.jsonl   61 synthetic Nigerian fever vignettes with triage gold labels
                                 (from iba-triage). Test only, never trained on.

Labels for the hospital records
  malaria_risk   dataset's own label: Target=1 -> "high", 0 -> "standard" (Risk_Score >= 7).
  danger_signs   Confusion=1 -> lethargy_or_unconsciousness (WHO: impaired consciousness is a
                 feature of severe malaria).
  triage_level   any danger sign -> refer_now; else high risk -> refer_24h; else treat_monitor.
                 This mapping is v0 and must be confirmed by the clinical lead.

Each structured record is rendered as the short free-text note a health worker would type,
in English (70%) or Nigerian Pidgin (30%), so the model learns from the same kind of input
Ibà receives. The rendering is deterministic (seeded).

Outputs (data/processed/)
  train.jsonl, val.jsonl            chat-format examples for supervised fine-tuning
  benchmark.jsonl                   649 held-out hospital records + 61 Ibà vignettes (710)
  dataset_card.json                 counts and label distributions

v2 (default) also trains on synthetic worker-style cases covering all ten danger signs, plus
hard negatives (natlas/augment.py), checked against Ibà's danger-sign rules and screened against
the vignettes. The benchmark is identical in v1 and v2.

Run: python -m natlas.build_dataset [--version v1|v2]
"""

import argparse

import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd

from natlas.augment import generate, too_close
from natlas.prompts import SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
SEED = 2026
TEST_FRACTION = 0.40
VAL_FRACTION = 0.10  # of the non-test part
PIDGIN_FRACTION = 0.30

SYMPTOMS = {
    # column: (English phrases, Pidgin phrases)
    "Fever": (["fever", "high temperature", "hot body"], ["body dey hot", "fever"]),
    "Headache": (["headache"], ["head dey pain am", "headache"]),
    "Abdominal_Pain": (["abdominal pain", "stomach pain"], ["belle dey pain am"]),
    "General_Body_Malaise": (["general body weakness", "body malaise"], ["body no strong", "body dey weak am"]),
    "Dizziness": (["dizziness"], ["head dey turn am"]),
    "Vomiting": (["vomiting"], ["e dey vomit"]),
    "Confusion": (["confused", "not talking sense"], ["e dey confuse", "e no dey talk well"]),
    "Backache": (["back pain"], ["back dey pain am"]),
    "Chest_Pain": (["chest pain"], ["chest dey pain am"]),
    "Coughing": (["cough"], ["e dey cough"]),
    "Joint_Pain": (["joint pains"], ["joint dey pain am"]),
}


def _person(age: int, sex: str, pidgin: bool, rng: random.Random) -> str:
    female = sex.lower().startswith("f")
    if age < 13:
        noun = ("girl" if female else "boy") if not pidgin else ("pikin (girl)" if female else "pikin (boy)")
    elif pidgin:
        noun = "Woman" if female else "Man"
    else:
        noun = rng.choice(["Woman", "Female patient"] if female else ["Man", "Male patient"])
    return f"{noun} {age} years" if not pidgin else f"{noun} wey be {age} years"


def render_case(row: pd.Series, rng: random.Random) -> tuple[str, str]:
    pidgin = rng.random() < PIDGIN_FRACTION
    present = [c for c in SYMPTOMS if int(row[c]) == 1]
    rng.shuffle(present)
    phrases = [rng.choice(SYMPTOMS[c][1 if pidgin else 0]) for c in present]
    who = _person(int(row["Age"]), str(row["Sex"]), pidgin, rng)
    if not phrases:
        tail = "no clear complaint" if not pidgin else "e no talk wetin dey do am"
        return f"{who}, {tail}", "pcm" if pidgin else "en"
    joiner = ", " if not pidgin else ", "
    text = f"{who}, {joiner.join(phrases)}"
    return text, "pcm" if pidgin else "en"


def label_record(row: pd.Series) -> dict:
    danger = ["lethargy_or_unconsciousness"] if int(row["Confusion"]) == 1 else []
    risk = "high" if int(row["Target"]) == 1 else "standard"
    if danger:
        level, reason = "refer_now", "Altered consciousness is a danger sign: refer now."
    elif risk == "high":
        level, reason = "refer_24h", "Many malaria features together: high-risk malaria, refer within 24 hours."
    else:
        level, reason = "treat_monitor", "No danger signs and standard-risk malaria features: test, treat at PHC and review."
    return {"triage_level": level, "danger_signs": danger, "malaria_risk": risk, "reason": reason}


def chat_example(text: str, label: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
            {"role": "assistant", "content": json.dumps(label, ensure_ascii=False)},
        ]
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["v1", "v2"], default="v2")
    version = ap.parse_args().version
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW / "Malaria_Dataset.csv")
    df["stratum"] = df["Target"].astype(str) + "_" + df["Confusion"].astype(str)

    test_idx: list[int] = []
    for _, group in df.groupby("stratum"):
        idx = list(group.index)
        rng.shuffle(idx)
        test_idx += idx[: round(len(idx) * TEST_FRACTION)]
    test_set = set(test_idx)
    rest = [i for i in df.index if i not in test_set]
    rng.shuffle(rest)
    n_val = round(len(rest) * VAL_FRACTION)
    val_idx, train_idx = rest[:n_val], rest[n_val:]

    def records(indices: list[int], split: str) -> list[dict]:
        out = []
        for i in sorted(indices):
            row = df.loc[i]
            text, lang = render_case(row, rng)
            out.append({"id": f"hosp-{i:04d}", "source": "hospital", "split": split,
                        "language": lang, "text": text, "gold": label_record(row)})
        return out

    train, val, test = records(train_idx, "train"), records(val_idx, "val"), records(test_idx, "test")

    vignettes = []
    for line in (RAW / "iba_vignettes.jsonl").read_text().splitlines():
        v = json.loads(line)
        g = v["gold"]
        vignettes.append({
            "id": f"iba-{v['id']}", "source": "iba_vignette", "split": "test",
            "language": "pcm" if v["category"] == "pidgin_variant" else "en",
            "category": v["category"], "text": v["text"],
            "gold": {"triage_level": g["triage_level"], "danger_signs": g["danger_signs"],
                     "malaria_risk": None, "reason": None},
        })

    synthetic, dropped = [], 0
    if version == "v2":
        held_out = [v["text"] for v in vignettes]
        for i, (text, gold) in enumerate(generate()):
            if too_close(text, held_out):
                dropped += 1
                continue
            synthetic.append({"id": f"syn-{i:04d}", "source": "synthetic", "language": "pcm"
                              if " dey " in f" {text} " or "wey be" in text else "en",
                              "text": text, "gold": gold})
        n_syn_val = round(len(synthetic) * VAL_FRACTION)
        val = val + synthetic[:n_syn_val]
        train = train + synthetic[n_syn_val:]
        random.Random(SEED).shuffle(train)

    for name, rows in [("train", train), ("val", val)]:
        with open(OUT / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(chat_example(r["text"], r["gold"]), ensure_ascii=False) + "\n")
    with open(OUT / "benchmark.jsonl", "w") as f:
        for r in test + vignettes:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    card = {
        "version": version,
        "seed": SEED,
        "synthetic_used": len(synthetic),
        "synthetic_dropped_near_vignette": dropped,
        "train_sources": Counter(r["source"] for r in train),
        "train": len(train), "val": len(val),
        "benchmark_total": len(test) + len(vignettes),
        "benchmark_hospital": len(test), "benchmark_iba_vignettes": len(vignettes),
        "train_triage": Counter(r["gold"]["triage_level"] for r in train),
        "benchmark_hospital_triage": Counter(r["gold"]["triage_level"] for r in test),
        "benchmark_hospital_risk": Counter(r["gold"]["malaria_risk"] for r in test),
        "vignette_triage": Counter(r["gold"]["triage_level"] for r in vignettes),
        "language_train": Counter(r["language"] for r in train),
        "language_benchmark": Counter(r["language"] for r in test + vignettes),
    }
    (OUT / "dataset_card.json").write_text(json.dumps(card, indent=2))
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
