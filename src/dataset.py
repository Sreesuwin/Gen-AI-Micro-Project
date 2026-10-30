"""Dataset loading and validation."""

from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "data" / "dataset.json"

REQUIRED = ("id", "title", "long_prompt", "question", "reference_answer")


def load(path: pathlib.Path | str = DEFAULT_PATH) -> list[dict]:
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")
    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items:
        raise ValueError("dataset must be a non-empty JSON array")
    for i, item in enumerate(items):
        missing = [k for k in REQUIRED if not item.get(k)]
        if missing:
            raise ValueError(f"dataset item {i} missing fields: {missing}")
    return items


def corpus(items: list[dict]) -> list[str]:
    return [it["long_prompt"] for it in items]
