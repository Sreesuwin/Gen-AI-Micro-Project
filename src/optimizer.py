"""Aggregation of sweep results and constrained optimisation (objective 4)."""

from __future__ import annotations

import csv
import json
import pathlib
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

CSV_PATH = RESULTS_DIR / "results.csv"
RUNS_PATH = RESULTS_DIR / "runs.json"
SUMMARY_PATH = RESULTS_DIR / "summary.json"

# The constraint from the problem statement: pick the cheapest level that still
# clears a minimum quality bar.
DEFAULT_MIN_JUDGE = 4.0
DEFAULT_F1_FRACTION = 0.90  # at least 90% of the uncompressed baseline F1


def load_runs() -> list[dict]:
    if not RUNS_PATH.exists():
        return []
    return json.loads(RUNS_PATH.read_text(encoding="utf-8"))


def save_runs(runs: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_PATH.write_text(json.dumps(runs, indent=1), encoding="utf-8")


def write_csv(runs: list[dict]) -> pathlib.Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "item_id", "method", "level", "orig_tokens", "compressed_tokens",
        "compression_ratio", "tokens_in", "tokens_out", "cost_usd",
        "latency_ms", "f1", "precision", "recall", "judge_score", "judge_reason",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in runs:
            writer.writerow(r)
    return CSV_PATH


def _mean(values: list[float]) -> float:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def aggregate(runs: list[dict]) -> list[dict]:
    """Mean metrics per (method, level) cell."""
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for r in runs:
        groups[(r["method"], float(r["level"]))].append(r)

    rows = []
    for (method, level), items in sorted(groups.items(), key=lambda kv: (kv[0][0], -kv[0][1])):
        rows.append(
            {
                "method": method,
                "level": level,
                "n": len(items),
                "mean_orig_tokens": _mean([i["orig_tokens"] for i in items]),
                "mean_compressed_tokens": _mean([i["compressed_tokens"] for i in items]),
                "mean_compression_ratio": _mean([i["compression_ratio"] for i in items]),
                "mean_tokens_in": _mean([i["tokens_in"] for i in items]),
                "mean_tokens_out": _mean([i["tokens_out"] for i in items]),
                "mean_cost_usd": round(sum(i["cost_usd"] for i in items), 6),
                "mean_f1": _mean([i["f1"] for i in items]),
                "mean_judge": _mean([i["judge_score"] for i in items if i.get("judge_score")]),
                "n_judged": len([i for i in items if i.get("judge_score")]),
                "mean_latency_ms": _mean([i["latency_ms"] for i in items]),
            }
        )
    return rows


def baseline_stats(runs: list[dict]) -> dict:
    base = [r for r in runs if r["method"] == "baseline"]
    if not base:
        return {"mean_f1": 0.0, "mean_cost_usd": 0.0, "mean_tokens_in": 0.0, "n": 0}
    return {
        "n": len(base),
        "mean_f1": _mean([r["f1"] for r in base]),
        "mean_judge": _mean([r["judge_score"] for r in base if r.get("judge_score")]),
        "mean_cost_usd": round(sum(r["cost_usd"] for r in base), 6),
        "mean_tokens_in": _mean([r["tokens_in"] for r in base]),
        "mean_tokens_out": _mean([r["tokens_out"] for r in base]),
    }


def select_best(
    summary_rows: list[dict],
    baseline: dict,
    min_judge: float = DEFAULT_MIN_JUDGE,
    f1_fraction: float = DEFAULT_F1_FRACTION,
) -> dict:
    """Objective 4: cheapest (method, level) that satisfies the quality constraint.

    Constraint:  mean judge score >= min_judge   AND   mean F1 >= f1_fraction * baseline F1.
    Objective:   minimise cost.
    """
    f1_floor = baseline.get("mean_f1", 0.0) * f1_fraction
    candidates = [
        r for r in summary_rows
        if r["method"] != "baseline"
        and r["mean_judge"] >= min_judge
        and r["mean_f1"] >= f1_floor
    ]

    if not candidates:
        # fall back to the highest-quality cell so the report can say why
        fallback = max(summary_rows, key=lambda r: (r["mean_judge"], r["mean_f1"]))
        return {
            "found": False,
            "constraint": {"min_judge": min_judge, "f1_fraction": f1_fraction, "f1_floor": round(f1_floor, 4)},
            "reason": "No compressed configuration satisfied both constraints; showing the best available quality instead.",
            "best_available": fallback,
        }

    best = min(candidates, key=lambda r: r["mean_cost_usd"])
    base_cost = baseline.get("mean_cost_usd", 0.0) or 1e-9
    base_tokens = baseline.get("mean_tokens_in", 0.0) or 1e-9
    return {
        "found": True,
        "constraint": {"min_judge": min_judge, "f1_fraction": f1_fraction, "f1_floor": round(f1_floor, 4)},
        "baseline": baseline,
        "selected": best,
        "cost_reduction_pct": round(100 * (1 - best["mean_cost_usd"] / base_cost), 2),
        "input_token_reduction_pct": round(100 * (1 - best["mean_tokens_in"] / base_tokens), 2),
        "quality_retained_pct": round(100 * best["mean_f1"] / baseline["mean_f1"], 2) if baseline.get("mean_f1") else 0.0,
        "judge_gap": round(min_judge - best["mean_judge"], 3),
        "n_candidates": len(candidates),
    }


def build_summary(
    runs: list[dict],
    min_judge: float = DEFAULT_MIN_JUDGE,
    f1_fraction: float = DEFAULT_F1_FRACTION,
) -> dict:
    rows = aggregate(runs)
    baseline = baseline_stats(runs)
    selection = select_best(rows, baseline, min_judge, f1_fraction)
    summary = {
        "n_runs": len(runs),
        "n_items": len({r["item_id"] for r in runs}),
        "baseline": baseline,
        "selection": selection,
        "cells": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary
