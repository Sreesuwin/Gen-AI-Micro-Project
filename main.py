"""CLI entry point for the Smart Prompt Optimization project.

    python main.py --smoke      quick end-to-end check (API key + 3 items)
    python main.py --run-all    full sweep, judging, optimisation, charts, report
    python main.py --report     rebuild summary/charts/report from stored runs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src import compressor, cost, dataset, llm, metrics, optimizer

RESULTS = optimizer.RESULTS_DIR
WORKERS = int(os.environ.get("GENAI_WORKERS", "3"))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------- sweeps

def run_sweep(items: list[dict], levels: list[float], judge_filter: bool = False) -> list[dict]:
    """Call the model for every (item x method x level) and measure everything.

    Requests run in parallel (the free tier allows far more than one request at
    a time); the run stops early, keeping what it has, if the daily quota runs out.
    """
    idf = compressor.build_idf(dataset.corpus(items))
    model = llm.model_name()

    combos: list[tuple[dict, str, float]] = []
    for item in items:
        combos.append((item, "baseline", 1.0))
        for method in compressor.COMPRESSIBLE_METHODS:
            for level in levels:
                combos.append((item, method, level))

    total = len(combos)
    runs: list[dict] = []
    quota_hit = False

    def work(idx: int, combo: tuple[dict, str, float]):
        item, method, level = combo
        ctx = compressor.compress(item["long_prompt"], method, level, idf, item["question"])
        out = llm.generate(compressor.build_prompt(item, ctx), model=model)
        return idx, item, method, level, out

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(work, i, c): i for i, c in enumerate(combos)}
        for fut in as_completed(futures):
            item, method, level = combos[futures[fut]]
            try:
                _, item, method, level, out = fut.result()
            except llm.QuotaExceededError as err:
                quota_hit = True
                log(f"  ! API daily quota exhausted - stopping the sweep early "
                    f"({len(runs)} runs kept). Resume later with --run-all --resume")
                log(f"    {str(err)[:160]}")
                pool.shutdown(cancel_futures=True)
                break
            except RuntimeError as err:
                log(f"  ! {item['id']}/{method}/{level}: {str(err)[:160]}")
                continue

            f1 = metrics.token_f1(out["text"], item["reference_answer"])
            runs.append({
                "item_id": item["id"],
                "method": method,
                "level": level,
                "orig_tokens": 0,
                "compressed_tokens": out["tokens_in"],
                "compression_ratio": 1.0,
                "tokens_in": out["tokens_in"],
                "tokens_out": out["tokens_out"],
                "cost_usd": round(cost.cost_usd(out["tokens_in"], out["tokens_out"], model), 8),
                "latency_ms": out["latency_ms"],
                "f1": f1["f1"],
                "precision": f1["precision"],
                "recall": f1["recall"],
                "judge_score": None,
                "judge_reason": "",
                "answer": out["text"],
            })
            if len(runs) % 10 == 0 or len(runs) == total:
                log(f"  sweep {len(runs)}/{total} calls done")

    if quota_hit:
        log("  sweep halted by quota; stored partial results")
    return runs


def _fill_baseline_tokens(runs: list[dict]) -> None:
    """Set orig_tokens and compression_ratio using the baseline for each item."""
    base = {r["item_id"]: r["tokens_in"] for r in runs if r["method"] == "baseline"}
    for r in runs:
        orig = base.get(r["item_id"], r["tokens_in"])
        r["orig_tokens"] = orig
        r["compressed_tokens"] = r["tokens_in"]
        r["compression_ratio"] = round(r["tokens_in"] / orig, 4) if orig else 1.0


def run_judging(runs: list[dict], items: list[dict], max_levels_per_method: int = 4) -> int:
    """Judge the baseline plus the most promising levels of every method.

    Cells are chosen by token-F1 so that every configuration which could plausibly
    satisfy the optimiser's F1 floor also has a judge score.
    """
    by_id = {it["id"]: it for it in items}
    base_f1 = {
        r["item_id"]: r["f1"] for r in runs if r["method"] == "baseline"
    }
    baseline_mean = sum(base_f1.values()) / len(base_f1) if base_f1 else 0.0

    # aggregate F1 per cell
    cells: dict[tuple[str, float], list[float]] = {}
    for r in runs:
        cells.setdefault((r["method"], r["level"]), []).append(r["f1"])
    cell_f1 = {k: sum(v) / len(v) for k, v in cells.items()}

    chosen: set[tuple[str, float]] = {("baseline", 1.0)}
    for method in compressor.COMPRESSIBLE_METHODS:
        eligible = [(lv, f) for (m, lv), f in cell_f1.items()
                    if m == method and f >= 0.80 * baseline_mean]
        eligible.sort(key=lambda t: -t[0])  # most compression first among eligible
        for lv, _ in eligible[:max_levels_per_method]:
            chosen.add((method, lv))

    targets = [r for r in runs if (r["method"], r["level"]) in chosen and r.get("judge_score") is None]
    log(f"  judging {len(targets)} runs across {len(chosen)} cells")

    def work(r: dict):
        item = by_id[r["item_id"]]
        verdict = metrics.judge(item["question"], r["answer"], item["reference_answer"])
        r["judge_score"] = verdict["score"]
        r["judge_reason"] = verdict["reason"]
        return r

    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(work, r) for r in targets]
        for fut in as_completed(futures):
            try:
                fut.result()
                done += 1
            except llm.QuotaExceededError:
                log(f"  ! quota exhausted during judging after {done} runs - "
                    f"rerun `python main.py --report` later to finish")
                pool.shutdown(cancel_futures=True)
                break
            except RuntimeError as err:
                log(f"  ! judge failed: {str(err)[:160]}")
                continue
            if done % 10 == 0 or done == len(targets):
                log(f"  judge {done}/{len(targets)}")
    return done


# ----------------------------------------------------------------- outputs

def make_charts(summary: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RESULTS.mkdir(parents=True, exist_ok=True)
    cells = summary["cells"]
    methods = sorted({c["method"] for c in cells})
    written: list[str] = []

    def series(method: str, key: str) -> tuple[list[float], list[float]]:
        pts = sorted([(c["level"], c[key]) for c in cells if c["method"] == method], reverse=True)
        return [p[0] for p in pts], [p[1] for p in pts]

    # 1. cost vs level
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in methods:
        x, y = series(m, "mean_cost_usd")
        ax.plot(x, y, marker="o", label=m)
    ax.set_xlabel("Retention level (fraction of prompt kept)")
    ax.set_ylabel("Mean cost per query (USD)")
    ax.set_title("API cost vs. compression level")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = RESULTS / "cost_vs_level.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(str(p))

    # 2. quality vs level (F1 + judge)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for m in methods:
        x, y = series(m, "mean_f1")
        axes[0].plot(x, y, marker="o", label=m)
    axes[0].set_title("Token-F1 vs. compression level")
    axes[0].set_xlabel("Retention level")
    axes[0].set_ylabel("Mean Token-F1")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    for m in methods:
        pts = sorted([(c["level"], c["mean_judge"]) for c in cells
                      if c["method"] == m and c["n_judged"]], reverse=True)
        if pts:
            axes[1].plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=m)
    axes[1].set_title("LLM judge score (1-5) vs. compression level")
    axes[1].set_xlabel("Retention level")
    axes[1].set_ylabel("Mean judge score")
    axes[1].set_ylim(0, 5.5)
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    p = RESULTS / "quality_vs_level.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(str(p))

    # 3. cost/quality Pareto scatter
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in methods:
        pts = [c for c in cells if c["method"] == m]
        ax.scatter([c["mean_cost_usd"] for c in pts],
                   [c["mean_f1"] for c in pts], label=m, s=60)
        for c in pts:
            ax.annotate(f"{c['level']}", (c["mean_cost_usd"], c["mean_f1"]),
                        fontsize=7, alpha=0.7)
    base = summary["baseline"]
    if base.get("n"):
        ax.scatter([base["mean_cost_usd"]], [base["mean_f1"]], marker="*", s=260,
                   c="gold", edgecolors="k", label="baseline (no compression)")
    ax.set_xlabel("Mean cost per query (USD)")
    ax.set_ylabel("Mean Token-F1")
    ax.set_title("Cost / quality Pareto frontier (labels = retention level)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = RESULTS / "pareto.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(str(p))
    return written


def make_report(summary: dict) -> str:
    sel = summary["selection"]
    base = summary["baseline"]
    lines = [
        "# Smart Prompt Optimization - Results Report",
        "",
        f"- Dataset items: **{summary['n_items']}**",
        f"- Total model calls measured: **{summary['n_runs']}**",
        f"- Model: `{llm.model_name()}`",
        "",
        "## Baseline (no compression)",
        "",
        f"- Mean input tokens: {base.get('mean_tokens_in', 0):.1f}",
        f"- Mean cost per query: ${base.get('mean_cost_usd', 0):.6f}",
        f"- Mean Token-F1: {base.get('mean_f1', 0):.4f}",
        f"- Mean judge score: {base.get('mean_judge', 0) or 'n/a'}",
        "",
        "## All configurations",
        "",
        "| method | level | ratio | tokens in | cost/query | F1 | judge |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in summary["cells"]:
        judge = f"{c['mean_judge']:.2f}" if c["n_judged"] else "-"
        lines.append(
            f"| {c['method']} | {c['level']} | {c['mean_compression_ratio']:.3f} | "
            f"{c['mean_tokens_in']:.1f} | ${c['mean_cost_usd']:.6f} | {c['mean_f1']:.4f} | {judge} |"
        )

    lines += ["", "## Optimisation result", "",
              f"- Constraint: mean judge score >= **{sel['constraint']['min_judge']}** and "
              f"mean F1 >= **{sel['constraint']['f1_fraction']:.0%}** of baseline "
              f"(floor {sel['constraint']['f1_floor']})"]

    if sel.get("found"):
        s = sel["selected"]
        lines += [
            "",
            f"**Selected configuration: method = `{s['method']}`, level = {s['level']}**",
            "",
            f"- Compression ratio achieved: {s['mean_compression_ratio']:.3f}",
            f"- Input tokens: {base.get('mean_tokens_in', 0):.1f} -> {s['mean_tokens_in']:.1f} "
            f"(**{sel['input_token_reduction_pct']}% fewer**)",
            f"- Cost per query: ${base.get('mean_cost_usd', 0):.6f} -> ${s['mean_cost_usd']:.6f} "
            f"(**{sel['cost_reduction_pct']}% cheaper**)",
            f"- Token-F1 retained: {sel['quality_retained_pct']}% of baseline",
            f"- Judge score: {s['mean_judge']:.2f} (threshold {sel['constraint']['min_judge']})",
            f"- Feasible candidates satisfying the constraint: {sel['n_candidates']}",
        ]
    else:
        best = sel.get("best_available", {})
        lines += ["", "**No configuration satisfied both constraints.** Best available:",
                  f"- {best.get('method')} @ {best.get('level')} "
                  f"(F1 {best.get('mean_f1')}, judge {best.get('mean_judge')})",
                  "", sel.get("reason", "")]

    lines += ["", "## Charts", "",
              "- `results/cost_vs_level.png`", "- `results/quality_vs_level.png`",
              "- `results/pareto.png`", ""]
    text = "\n".join(lines)
    (RESULTS / "report.md").write_text(text, encoding="utf-8")
    return text


# ----------------------------------------------------------------- commands

def cmd_smoke(args: argparse.Namespace) -> None:
    log("health check...")
    info = llm.healthcheck()
    log(f"  model={info['model']} reply={info['text']!r} "
        f"tokens_in={info['tokens_in']} tokens_out={info['tokens_out']} "
        f"exact_count={info['exact_count_tokens']}")

    items = dataset.load()
    smoke_items = items[: args.smoke_items]
    idf = compressor.build_idf(dataset.corpus(items))

    log("compression sanity check (no API calls):")
    for method in compressor.COMPRESSIBLE_METHODS:
        ctx = compressor.compress(smoke_items[0]["long_prompt"], method, 0.45, idf,
                                  smoke_items[0]["question"])
        log(f"  {method:9s} @0.45 -> {len(ctx)} chars "
            f"(orig {len(smoke_items[0]['long_prompt'])})")

    log("running one real call per method...")
    runs = run_sweep(smoke_items, levels=[0.45])
    _fill_baseline_tokens(runs)
    for r in runs:
        log(f"  {r['item_id']:18s} {r['method']:9s} lv={r['level']} "
            f"in={r['tokens_in']} out={r['tokens_out']} ratio={r['compression_ratio']:.3f} "
            f"f1={r['f1']:.3f} cost=${r['cost_usd']:.6f}")
    log("SMOKE OK")


def cmd_run_all(args: argparse.Namespace) -> None:
    items = dataset.load()
    log(f"dataset: {len(items)} items, model: {llm.model_name()}, "
        f"levels: {compressor.LEVELS}")

    existing = optimizer.load_runs()
    key = lambda r: (r["item_id"], r["method"], r["level"])  # noqa: E731
    done_keys = {key(r) for r in existing} if args.resume and existing else set()
    runs = [r for r in existing if key(r) in done_keys]

    new_runs = run_sweep(items, compressor.LEVELS)
    all_keys = {key(r) for r in new_runs}
    runs = [r for r in runs if key(r) not in all_keys] + new_runs
    _fill_baseline_tokens(runs)
    optimizer.save_runs(runs)
    log(f"sweep complete: {len(runs)} runs stored")

    if not args.no_judge:
        run_judging(runs, items, max_levels_per_method=args.max_judged_levels)
        _fill_baseline_tokens(runs)
        optimizer.save_runs(runs)

    optimizer.write_csv(runs)
    summary = optimizer.build_summary(runs, min_judge=args.min_judge,
                                      f1_fraction=args.f1_fraction)
    charts = make_charts(summary)
    make_report(summary)
    log(f"wrote {optimizer.CSV_PATH}")
    log(f"wrote {optimizer.SUMMARY_PATH}")
    for c in charts:
        log(f"wrote {c}")
    log(f"wrote {RESULTS / 'report.md'}")
    _print_selection(summary)


def cmd_report(args: argparse.Namespace) -> None:
    runs = optimizer.load_runs()
    if not runs:
        log("no stored runs found - run --run-all first")
        sys.exit(1)
    if not args.no_judge:
        items = dataset.load()
        run_judging(runs, items, max_levels_per_method=args.max_judged_levels)
        optimizer.save_runs(runs)
    optimizer.write_csv(runs)
    summary = optimizer.build_summary(runs, min_judge=args.min_judge,
                                      f1_fraction=args.f1_fraction)
    make_charts(summary)
    make_report(summary)
    _print_selection(summary)


def _print_selection(summary: dict) -> None:
    sel = summary["selection"]
    if sel.get("found"):
        s = sel["selected"]
        log("=== OPTIMISATION RESULT ===")
        log(f"  selected      : {s['method']} @ level {s['level']}")
        log(f"  compression   : ratio {s['mean_compression_ratio']:.3f}")
        log(f"  input tokens  : {summary['baseline'].get('mean_tokens_in', 0):.1f} -> "
            f"{s['mean_tokens_in']:.1f} ({sel['input_token_reduction_pct']}% fewer)")
        log(f"  cost/query    : ${summary['baseline'].get('mean_cost_usd', 0):.6f} -> "
            f"${s['mean_cost_usd']:.6f} ({sel['cost_reduction_pct']}% cheaper)")
        log(f"  quality       : F1 {sel['quality_retained_pct']}% of baseline, "
            f"judge {s['mean_judge']:.2f} >= {sel['constraint']['min_judge']}")
    else:
        log(f"  NO FEASIBLE CONFIGURATION: {sel.get('reason', '')}")
        best = sel.get("best_available")
        if best:
            log(f"  best available : {best['method']} @ {best['level']} "
                f"F1={best['mean_f1']} judge={best['mean_judge']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart Prompt Optimization")
    parser.add_argument("--smoke", action="store_true", help="quick end-to-end check")
    parser.add_argument("--run-all", action="store_true", help="full sweep + judging + report")
    parser.add_argument("--report", action="store_true", help="rebuild outputs from stored runs")
    parser.add_argument("--smoke-items", type=int, default=3)
    parser.add_argument("--min-judge", type=float, default=optimizer.DEFAULT_MIN_JUDGE)
    parser.add_argument("--f1-fraction", type=float, default=optimizer.DEFAULT_F1_FRACTION)
    parser.add_argument("--max-judged-levels", type=int, default=4)
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM judge pass")
    parser.add_argument("--resume", action="store_true", help="keep previously stored runs")
    args = parser.parse_args()

    llm.load_env()
    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        cmd_smoke(args)
    elif args.run_all:
        cmd_run_all(args)
    elif args.report:
        cmd_report(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
