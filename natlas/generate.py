"""Run base or fine-tuned N-ATLAS over the benchmark and save raw replies.

    python -m natlas.generate --out results/preds_base.jsonl
    python -m natlas.generate --adapter outputs/iba-atlas-lora --out results/preds_ft.jsonl

Both runs use the same 4-bit loading, system prompt and greedy decoding, so the only
difference is the adapter. Resumable: cases already in --out are skipped.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from natlas.prompts import SYSTEM_PROMPT
from natlas.train import load_4bit, load_tokenizer

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data" / "processed" / "benchmark.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="NCAIR1/N-ATLaS")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    tok = load_tokenizer(args.base)
    tok.padding_side = "left"
    model = load_4bit(args.base)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    cases = [json.loads(line) for line in BENCH.read_text().splitlines()][: args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in out.read_text().splitlines()} if out.exists() else set()
    todo = [c for c in cases if c["id"] not in done]
    print(f"{len(todo)} to run ({len(done)} already done)")

    eos = [tok.eos_token_id]
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot, int) and eot != tok.unk_token_id:
        eos.append(eot)

    t0 = time.time()
    with open(out, "a") as f:
        for i in range(0, len(todo), args.batch):
            batch = todo[i: i + args.batch]
            prompts = [
                tok.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": c["text"]}],
                    add_generation_prompt=True, tokenize=False,
                )
                for c in batch
            ]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                     eos_token_id=eos, pad_token_id=tok.pad_token_id)
            for c, g in zip(batch, gen):
                raw = tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True)
                f.write(json.dumps({"id": c["id"], "raw": raw}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{i + len(batch)}/{len(todo)}  {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
