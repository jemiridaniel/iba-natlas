"""Minimal OpenAI-compatible server for N-ATLaS + the Ibà-ATLAS adapter (Transformers + PEFT).

A fallback for GPUs or environments where vLLM is not available (e.g. Kaggle T4s). Serves:

    GET  /health
    GET  /v1/models                -> the base model and the adapter
    POST /v1/chat/completions      -> model = base id (adapter off) or adapter name (adapter on)

    python -m natlas.serve --port 8000
    python -m natlas.serve --load-4bit            # one 16 GB GPU

One model instance serves both names: the adapter is switched off for base requests. Requests
run one at a time (a lock), greedy when temperature is 0. `response_format` is accepted but not
enforced; the Ibà client validates every reply against its schema and retries once, so run the
app with LLM_STRUCTURED_OUTPUT=off against this server.
"""

from __future__ import annotations

import argparse
import threading
import time
import uuid
from contextlib import nullcontext
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int | None = 1024
    temperature: float | None = 0.0
    response_format: dict[str, Any] | None = None

    model_config = {"extra": "ignore"}


def load(base: str, adapter: str | None, adapter_name: str, load_4bit: bool):
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs: dict[str, Any] = {"dtype": torch.float16, "device_map": "auto"}
    if load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, adapter_name=adapter_name)
    model.eval()
    return tok, model


def build_app(tok, model, base_id: str, adapter_name: str | None, max_context: int) -> FastAPI:
    app = FastAPI(title="N-ATLaS server")
    lock = threading.Lock()
    stop_ids = [tok.eos_token_id]
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot, int) and eot != tok.unk_token_id:
        stop_ids.append(eot)
    names = [base_id] + ([adapter_name] if adapter_name else [])

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "models": names}

    @app.get("/v1/models")
    def models() -> dict:
        return {"object": "list", "data": [{"id": n, "object": "model", "owned_by": "local"} for n in names]}

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest) -> dict:
        if req.model not in names:
            raise HTTPException(404, f"unknown model {req.model!r}; served: {names}")
        prompt = tok.apply_chat_template([m.model_dump() for m in req.messages],
                                         add_generation_prompt=True, tokenize=False)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
        n_prompt = enc["input_ids"].shape[1]
        max_new = max(1, min(req.max_tokens or 1024, max_context - n_prompt))
        if n_prompt >= max_context:
            raise HTTPException(400, f"prompt is {n_prompt} tokens; context is {max_context}")
        greedy = not req.temperature
        use_adapter = adapter_name is not None and req.model == adapter_name
        ctx = nullcontext() if use_adapter or adapter_name is None else model.disable_adapter()
        start = time.perf_counter()
        with lock, ctx, torch.no_grad():
            out = model.generate(
                **enc.to(model.device), max_new_tokens=max_new, do_sample=not greedy,
                temperature=None if greedy else req.temperature, top_p=None,
                eos_token_id=stop_ids, pad_token_id=tok.pad_token_id)
        new = out[0][n_prompt:]
        text = tok.decode(new, skip_special_tokens=True)
        hit_stop = len(new) > 0 and int(new[-1]) in stop_ids
        finish = "stop" if hit_stop or len(new) < max_new else "length"
        print(f"{req.model} prompt={n_prompt} out={len(new)} {time.perf_counter() - start:.1f}s "
              f"finish={finish}", flush=True)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion",
            "created": int(time.time()), "model": req.model,
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": n_prompt, "completion_tokens": len(new),
                      "total_tokens": n_prompt + len(new)},
        }

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="NCAIR1/N-ATLaS")
    ap.add_argument("--adapter", default="jemiridaniel/Iba-ATLAS-8B-LoRA-Powered-by-Awarri")
    ap.add_argument("--adapter-name", default="iba-atlas")
    ap.add_argument("--no-adapter", action="store_true")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--max-context", type=int, default=8192)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    adapter = None if args.no_adapter else args.adapter
    tok, model = load(args.base, adapter, args.adapter_name, args.load_4bit)
    app = build_app(tok, model, args.base, args.adapter_name if adapter else None, args.max_context)
    print(f"serving {args.base}" + (f" + {args.adapter_name}" if adapter else "") + f" on :{args.port}",
          flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
