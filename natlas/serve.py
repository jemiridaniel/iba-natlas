"""Minimal OpenAI-compatible server for N-ATLaS + the Ibà-ATLAS adapter (Transformers + PEFT).

A fallback for GPUs or environments where vLLM is not available (e.g. Kaggle T4s). Serves:

    GET  /health
    GET  /v1/models                -> the base model and the adapter
    POST /v1/chat/completions      -> model = base id (adapter off) or adapter name (adapter on)

    python -m natlas.serve --port 8000
    python -m natlas.serve --load-4bit            # one 16 GB GPU

One model instance serves both names: the adapter is switched off for base requests. Requests
run one at a time (a lock), greedy when temperature is 0. A `response_format` of type
`json_schema` constrains decoding to that schema (lm-format-enforcer), like vLLM's structured
outputs, so run the app with LLM_STRUCTURED_OUTPUT=json_schema. Other response formats are
accepted and ignored.
"""

from __future__ import annotations

import argparse
import json
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

try:  # optional: schema-constrained decoding
    from natlas.constrain import allowed_tokens_fn, cap_schema, normalise_json, tokenizer_data
except ImportError:  # pragma: no cover
    tokenizer_data = None


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


def _is_json(text: str) -> bool:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return False
    try:
        json.loads(text[start : end + 1])
        return True
    except ValueError:
        return False


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


def build_app(tok, model, base_id: str, adapter_name: str | None, max_context: int,
              unconstrained: frozenset[str] = frozenset(), max_string: int = 200,
              max_items: int = 3) -> FastAPI:
    """`unconstrained`: schema titles to generate freely even when a json_schema is sent."""
    app = FastAPI(title="N-ATLaS server")
    lock = threading.Lock()
    stop_ids = [tok.eos_token_id]
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot, int) and eot != tok.unk_token_id:
        stop_ids.append(eot)
    names = [base_id] + ([adapter_name] if adapter_name else [])
    # Built once: maps every vocabulary token for the enforcer (a few seconds for 128K tokens).
    enforcer_data = None
    if tokenizer_data is not None:
        t0 = time.perf_counter()
        enforcer_data = tokenizer_data(tok, stop_ids)
        print(f"schema enforcement ready ({time.perf_counter() - t0:.0f}s)", flush=True)
    else:
        print("lm-format-enforcer not installed: response_format is ignored", flush=True)

    def schema_of(rf: dict[str, Any] | None) -> dict[str, Any] | None:
        if not rf or rf.get("type") != "json_schema":
            return None
        return (rf.get("json_schema") or {}).get("schema")

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
        schema = schema_of(req.response_format)
        constrain = {}
        if schema is not None and schema.get("title") in unconstrained:
            schema = None
        if schema is not None and enforcer_data is not None:
            constrain["prefix_allowed_tokens_fn"] = allowed_tokens_fn(
                enforcer_data, cap_schema(schema, max_string, max_items))
        start = time.perf_counter()
        with lock, ctx, torch.no_grad():
            out = model.generate(
                **enc.to(model.device), max_new_tokens=max_new, do_sample=not greedy,
                temperature=None if greedy else req.temperature, top_p=None,
                eos_token_id=stop_ids, pad_token_id=tok.pad_token_id, **constrain)
        new = out[0][n_prompt:]
        text = tok.decode(new, skip_special_tokens=True)
        if constrain:
            text = normalise_json(text)
        hit_stop = len(new) > 0 and int(new[-1]) in stop_ids
        finish = "stop" if hit_stop or len(new) < max_new else "length"
        print(f"{req.model} prompt={n_prompt} out={len(new)} {time.perf_counter() - start:.1f}s "
              f"finish={finish} schema={(schema or {}).get('title', '-') if constrain else '-'}",
              flush=True)
        if req.response_format and not _is_json(text):
            # Synthetic demo cases only: a preview helps diagnose a step that fails to parse.
            print(f"  not valid JSON, starts: {text[:300]!r}", flush=True)
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
    ap.add_argument("--max-string", type=int, default=200,
                    help="maxLength added to free-text fields of constrained schemas")
    ap.add_argument("--max-items", type=int, default=3,
                    help="maxItems added to arrays of constrained schemas")
    ap.add_argument("--unconstrained", default="ExplainOutput",
                    help="comma-separated schema titles to generate without constraints")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    adapter = None if args.no_adapter else args.adapter
    tok, model = load(args.base, adapter, args.adapter_name, args.load_4bit)
    free = frozenset(t.strip() for t in args.unconstrained.split(",") if t.strip())
    app = build_app(tok, model, args.base, args.adapter_name if adapter else None, args.max_context,
                    free, args.max_string, args.max_items)
    print(f"serving {args.base}" + (f" + {args.adapter_name}" if adapter else "") + f" on :{args.port}",
          flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
