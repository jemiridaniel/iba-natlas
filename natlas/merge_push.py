"""Merge the LoRA adapter into N-ATLAS (fp16) and push to the Hugging Face Hub.

    python -m natlas.merge_push --adapter outputs/iba-atlas-lora \
        --repo jemiridaniel/Iba-ATLAS-8B-Powered-by-Awarri [--adapter-only]

N-ATLAS licence: derivatives are released under the same Terms, credit Awarri Technologies and
the Federal Ministry of Communications, Innovation and Digital Economy, and a renamed model
carries the suffix "Powered by Awarri". The repo is created PRIVATE; make it public yourself
after checking the model card.
"""

import argparse

import torch
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="NCAIR1/N-ATLaS")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--adapter-only", action="store_true", help="push only the LoRA adapter")
    args = ap.parse_args()

    api = HfApi()
    api.create_repo(args.repo, private=True, exist_ok=True)
    if args.adapter_only:
        api.upload_folder(folder_path=args.adapter, repo_id=args.repo)
        print(f"pushed adapter to {args.repo}")
        return

    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float16, device_map="cpu",
                                                low_cpu_mem_usage=True)
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    merged.push_to_hub(args.repo, private=True, safe_serialization=True, max_shard_size="2GB")
    AutoTokenizer.from_pretrained(args.adapter).push_to_hub(args.repo, private=True)
    print(f"pushed merged model to {args.repo}")


if __name__ == "__main__":
    main()
