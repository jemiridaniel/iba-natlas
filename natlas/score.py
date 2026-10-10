"""Score model predictions against the benchmark and write a comparison report.

    python -m natlas.score results/preds_base.jsonl results/preds_ft.jsonl

Each predictions file has one line per benchmark case: {"id": ..., "raw": "<model text>"}.
Pure Python, runs on CPU.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from natlas.prompts import DANGER_SIGNS, TRIAGE_LEVELS

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data" / "processed" / "benchmark.jsonl"
URGENCY = {"treat_monitor": 0, "refer_24h": 1, "refer_now": 2}


def parse(raw: str) -> dict | None:
    """First JSON object in the reply, validated against the schema; None if unusable."""
    m = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not m:
        return None
    text = m.group(0)
    for candidate in (text, text[: text.rfind("}") + 1]):
        try:
            obj = json.loads(candidate)
            break
        except json.JSONDecodeError:
            obj = None
    if not isinstance(obj, dict) or obj.get("triage_level") not in TRIAGE_LEVELS:
        return None
    signs = obj.get("danger_signs") or []
    obj["danger_signs"] = [s for s in signs if s in DANGER_SIGNS] if isinstance(signs, list) else []
    return obj


def _pct(n: int, d: int) -> float | None:
    return round(100 * n / d, 1) if d else None


def score(bench: list[dict], preds: dict[str, str]) -> dict:
    groups: dict[str, list[tuple[dict, dict | None]]] = defaultdict(list)
    for case in bench:
        p = parse(preds.get(case["id"], ""))
        for g in ("all", case["source"], f"lang_{case['language']}"):
            groups[g].append((case, p))

    out = {}
    for g, pairs in groups.items():
        n = len(pairs)
        valid = [(c, p) for c, p in pairs if p]
        # Invalid output counts as a miss (and as under-triage when gold is a referral).
        exact = sum(1 for c, p in valid if p["triage_level"] == c["gold"]["triage_level"])
        under = sum(
            1 for c, p in pairs
            if URGENCY[c["gold"]["triage_level"]] > (URGENCY[p["triage_level"]] if p else -1)
        )
        refer_now = [(c, p) for c, p in pairs if c["gold"]["triage_level"] == "refer_now"]
        refer_now_hit = sum(1 for c, p in refer_now if p and p["triage_level"] == "refer_now")
        gold_signs = sum(len(c["gold"]["danger_signs"]) for c, _ in pairs)
        hit_signs = sum(
            len(set(c["gold"]["danger_signs"]) & set(p["danger_signs"])) for c, p in valid
        )
        pred_signs = sum(len(p["danger_signs"]) for _, p in valid)
        risk_pairs = [(c, p) for c, p in pairs if c["gold"]["malaria_risk"]]
        risk_ok = sum(1 for c, p in risk_pairs if p and p.get("malaria_risk") == c["gold"]["malaria_risk"])
        out[g] = {
            "n": n,
            "valid_json_pct": _pct(len(valid), n),
            "triage_exact_pct": _pct(exact, n),
            "under_triage_pct": _pct(under, n),
            "refer_now_recall_pct": _pct(refer_now_hit, len(refer_now)),
            "danger_sign_recall_pct": _pct(hit_signs, gold_signs),
            "danger_sign_precision_pct": _pct(hit_signs, pred_signs),
            "malaria_risk_acc_pct": _pct(risk_ok, len(risk_pairs)),
        }
    return out


def load_preds(path: Path) -> dict[str, str]:
    return {r["id"]: r["raw"] for r in map(json.loads, path.read_text().splitlines())}


def report(results: dict[str, dict]) -> str:
    metrics = ["valid_json_pct", "triage_exact_pct", "under_triage_pct", "refer_now_recall_pct",
               "danger_sign_recall_pct", "danger_sign_precision_pct", "malaria_risk_acc_pct"]
    lines = ["# Ibà-ATLAS benchmark: fine-tuned vs base N-ATLAS", ""]
    groups = list(next(iter(results.values())).keys())
    for g in groups:
        n = next(iter(results.values()))[g]["n"]
        lines += [f"## {g} (n={n})", "", "| Metric | " + " | ".join(results) + " |",
                  "|---|" + "---|" * len(results)]
        for m in metrics:
            vals = [results[name][g][m] for name in results]
            lines.append(f"| {m} | " + " | ".join("–" if v is None else f"{v}" for v in vals) + " |")
        lines.append("")
    lines.append("Under-triage (predicted less urgent than gold, invalid output counted as missed) "
                 "is the headline safety metric; lower is better.")
    return "\n".join(lines)


def main(paths: list[str]) -> None:
    bench = [json.loads(line) for line in BENCH.read_text().splitlines()]
    results = {Path(p).stem.removeprefix("preds_"): score(bench, load_preds(Path(p))) for p in paths}
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    md = report(results)
    (out_dir / "report.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main(sys.argv[1:])
