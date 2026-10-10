"""QLoRA fine-tune of N-ATLAS on the Ibà triage set. Fits one 16 GB GPU (Kaggle T4).

    python -m natlas.train [--base NCAIR1/N-ATLaS] [--out outputs/iba-atlas-lora] [--epochs 2]

Loss is computed on the assistant reply only (prompt tokens are masked), using N-ATLAS's own
chat template, so the adapter learns the output format and the triage mapping, not the prompt.
"""

import argparse
import json
import os
from pathlib import Path

# Must be set before CUDA initialises: reduces fragmentation on 16 GB cards.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"


def load_tokenizer(base: str):
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_4bit(base: str):
    return AutoModelForCausalLM.from_pretrained(
        base,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        ),
        torch_dtype=torch.float16,
        device_map={"": 0},
    )


def encode(tok, messages: list[dict], max_len: int) -> dict:
    # Render to text first: return types of apply_chat_template(tokenize=True) differ across
    # transformers versions, and the template already contains BOS.
    prompt_text = tok.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=False)
    full_text = tok.apply_chat_template(messages, tokenize=False)
    if not full_text.startswith(prompt_text):
        raise ValueError("chat template: prompt is not a prefix of the full conversation")
    prompt = tok(prompt_text, add_special_tokens=False)["input_ids"]
    full = tok(full_text, add_special_tokens=False)["input_ids"]
    if full[: len(prompt)] != prompt:
        raise ValueError("tokenisation: prompt tokens are not a prefix of the full conversation")
    full = full[:max_len]
    labels = [-100] * len(prompt) + full[len(prompt):]
    return {"input_ids": full, "labels": labels[: len(full)]}


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        width = max(len(b["input_ids"]) for b in batch)
        ids, labels, mask = [], [], []
        for b in batch:
            pad = width - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            mask.append([1] * len(b["input_ids"]) + [0] * pad)
        return {k: torch.tensor(v) for k, v in
                {"input_ids": ids, "labels": labels, "attention_mask": mask}.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="NCAIR1/N-ATLaS")
    ap.add_argument("--out", default="outputs/iba-atlas-lora")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    args = ap.parse_args()

    tok = load_tokenizer(args.base)

    def read(name: str) -> list[dict]:
        rows = [json.loads(line) for line in (DATA / f"{name}.jsonl").read_text().splitlines()]
        return [encode(tok, r["messages"], args.max_len) for r in rows]

    train, val = read("train"), read("val")
    print(f"train={len(train)} val={len(val)} max_tokens={max(len(x['input_ids']) for x in train)}")

    # Not peft's prepare_model_for_kbit_training: it upcasts the 128K-vocab embedding and
    # lm_head to fp32 (~4 GB), which does not fit a 16 GB T4 next to the activations.
    model = load_4bit(args.base)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    # fp16 mixed precision needs fp32 master weights for the (small) trainable LoRA params.
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out + "-ckpt",
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch,
            gradient_accumulation_steps=args.accum,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_steps=6,
            fp16=True,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="no",
            report_to="none",
            seed=2026,
        ),
        train_dataset=train,
        eval_dataset=val,
        data_collator=Collator(tok.pad_token_id),
    )
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    Path(args.out, "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    print(f"saved adapter to {args.out}")


if __name__ == "__main__":
    main()
