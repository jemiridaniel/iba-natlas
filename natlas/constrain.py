"""JSON-schema constrained decoding for Transformers `generate()`, via lm-format-enforcer's core.

lm-format-enforcer's own Transformers integration imports `transformers.tokenization_utils`,
which Transformers 5 removed, so this re-implements its two small helpers against the core API.

    data = tokenizer_data(tok)                         # once per tokenizer (128K vocab: ~1 min)
    fn = allowed_tokens_fn(data, schema_dict)          # once per request
    model.generate(..., prefix_allowed_tokens_fn=fn)
"""

from __future__ import annotations

import functools
from typing import Any

import json

from lmformatenforcer import (
    CharacterLevelParserConfig,
    JsonSchemaParser,
    TokenEnforcer,
    TokenEnforcerTokenizerData,
)


def _regular_tokens(tok, vocab_size: int) -> list[tuple[int, str, bool]]:
    special = set(tok.all_special_ids)
    token_0 = tok.encode("0", add_special_tokens=False)[-1]
    out = []
    for idx in range(vocab_size):
        if idx in special:
            continue
        # Decode after a "0" and drop it, to see whether the token starts a new word (leading space).
        after_0 = tok.decode([token_0, idx])[1:]
        alone = tok.decode([idx])
        out.append((idx, after_0, len(after_0) > len(alone)))
    return out


def _decode(tok, tokens: list[int]) -> str:
    return tok.decode(tokens).rstrip("�")


def tokenizer_data(tok, stop_ids: list[int] | None = None) -> TokenEnforcerTokenizerData:
    """`stop_ids`: tokens that may end a finished JSON object (Llama 3 chat ends with <|eot_id|>)."""
    vocab_size = len(tok)
    return TokenEnforcerTokenizerData(
        _regular_tokens(tok, vocab_size), functools.partial(_decode, tok),
        stop_ids or tok.eos_token_id, False, vocab_size)


# Whitespace runs between JSON tokens: enough for pretty-printing, not for endless padding.
_CONFIG = CharacterLevelParserConfig(max_consecutive_whitespaces=6)


def allowed_tokens_fn(data: TokenEnforcerTokenizerData, schema: dict[str, Any]):
    enforcer = TokenEnforcer(data, JsonSchemaParser(schema, config=_CONFIG))

    def fn(batch_id: int, sent) -> list[int]:
        return enforcer.get_allowed_tokens(sent.tolist()).allowed_tokens

    return fn


def normalise_json(text: str) -> str:
    """Re-serialise constrained output as strict JSON.

    The enforcer admits raw control characters (e.g. a literal line break) inside strings, which
    strict parsers reject; a model writing a multi-line referral note produces exactly that.
    Parse leniently and dump strictly. Returns the text unchanged if it is not complete JSON.
    """
    try:
        return json.dumps(json.loads(text, strict=False), ensure_ascii=False)
    except ValueError:
        return text
