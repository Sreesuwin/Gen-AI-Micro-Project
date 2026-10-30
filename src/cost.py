"""Pricing model: real token counts multiplied by published per-million prices.

Costs are SIMULATED - no money actually changes hands - but the token counts
are real (produced by the model's own tokenizer during the run) and the rates
are the published Gemini Flash list prices, so the reported saving is the
saving you would get if you were paying per token.
"""

from __future__ import annotations

import os

# USD per 1M tokens. Edit these if the provider changes its list prices.
PRICING: dict[str, dict[str, float]] = {
    # Open-weight models hosted on the Gemini API (free of charge to call, but
    # the published list price is what we simulate cost against).
    "gemma-4-26b-a4b-it": {"input": 0.15, "output": 0.60},
    "gemma-4-31b-it": {"input": 0.15, "output": 0.60},
    # Current Gemini generation (verified October 2026 against the list price)
    "gemini-3.5-flash": {"input": 1.50, "output": 9.00},
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.00},
    "gemini-3.1-flash-lite": {"input": 0.10, "output": 0.40},
    # Older generations, kept so the table still documents earlier rates
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
}
DEFAULT_MODEL = os.environ.get("GENAI_MODEL", "gemma-4-26b-a4b-it")


def rates(model: str) -> dict[str, float]:
    if model in PRICING:
        return PRICING[model]
    # unknown model: fall back to the default's rates rather than crashing
    return PRICING[DEFAULT_MODEL]


def cost_usd(tokens_in: int, tokens_out: int, model: str = DEFAULT_MODEL) -> float:
    """Cost of a single request in US dollars."""
    r = rates(model)
    return (tokens_in / 1_000_000.0) * r["input"] + (tokens_out / 1_000_000.0) * r["output"]


def cost_per_million(tokens_in: int, tokens_out: int, model: str = DEFAULT_MODEL) -> dict[str, float]:
    r = rates(model)
    return {
        "input_usd": round((tokens_in / 1_000_000.0) * r["input"], 8),
        "output_usd": round((tokens_out / 1_000_000.0) * r["output"], 8),
        "total_usd": round(cost_usd(tokens_in, tokens_out, model), 8),
    }
