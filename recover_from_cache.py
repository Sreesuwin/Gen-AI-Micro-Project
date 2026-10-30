"""
recover_from_cache.py - Rebuilds all outputs from cache.json, no API calls.
Run with: python recover_from_cache.py
"""
from __future__ import annotations
import hashlib, json, os, pathlib, sys, time

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import compressor, cost, dataset, metrics, optimizer
from src.llm import load_env, model_name

CACHE_PATH = ROOT / "results" / "cache.json"

def cache_key(model, prompt):
    return hashlib.sha1(f"{model}||{prompt}".encode("utf-8")).hexdigest()

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def build_runs_from_cache():
    load_env()
    model = model_name()
    if not CACHE_PATH.exists():
        log("ERROR: results/cache.json not found."); sys.exit(1)
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    log(f"Loaded cache: {len(cache)} entries")
    items = dataset.load()
    idf = compressor.build_idf(dataset.corpus(items))
    log(f"Dataset: {len(items)} items, model: {model}")
    runs = []
    hits = misses = 0
    for item in items:
        combos = [("baseline", 1.0)]
        for method in compressor.COMPRESSIBLE_METHODS:
            for level in compressor.LEVELS:
                combos.append((method, level))
        for method, level in combos:
            ctx = compressor.compress(item["long_prompt"], method, level, idf, item["question"])
            prompt = compressor.build_prompt(item, ctx)
            k = cache_key(model, prompt)
            if k not in cache:
                misses += 1; continue
            hits += 1
            entry = cache[k]
            f1 = metrics.token_f1(entry["text"], item["reference_answer"])
            tin, tout = entry["tokens_in"], entry["tokens_out"]
            runs.append({
                "item_id": item["id"], "method": method, "level": level,
                "orig_tokens": 0, "compressed_tokens": tin, "compression_ratio": 1.0,
                "tokens_in": tin, "tokens_out": tout,
                "cost_usd": round(cost.cost_usd(tin, tout, model), 8),
                "latency_ms": entry.get("latency_ms", 0),
                "f1": f1["f1"], "precision": f1["precision"], "recall": f1["recall"],
                "judge_score": None, "judge_reason": "", "answer": entry["text"],
            })
    log(f"Cache hits: {hits}  |  misses (skipped): {misses}")
    base_tokens = {r["item_id"]: r["tokens_in"] for r in runs if r["method"] == "baseline"}
    for r in runs:
        orig = base_tokens.get(r["item_id"], r["tokens_in"])
        r["orig_tokens"] = orig; r["compressed_tokens"] = r["tokens_in"]
        r["compression_ratio"] = round(r["tokens_in"] / orig, 4) if orig else 1.0
    return runs

def make_charts(summary):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = summary["cells"]
    methods = sorted({c["method"] for c in cells})
    def series(method, key):
        pts = sorted([(c["level"], c[key]) for c in cells if c["method"] == method], reverse=True)
        return [p[0] for p in pts], [p[1] for p in pts]
    # Cost
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in methods:
        x, y = series(m, "mean_cost_usd"); ax.plot(x, y, marker="o", label=m)
    ax.set_xlabel("Retention level"); ax.set_ylabel("Mean cost per query (USD)")
    ax.set_title("API cost vs. compression level"); ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(ROOT / "results" / "cost_vs_level.png", dpi=130); plt.close(fig)
    log("Saved -> results/cost_vs_level.png")
    # F1
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in methods:
        x, y = series(m, "mean_f1"); ax.plot(x, y, marker="o", label=m)
    ax.set_title("Token-F1 vs. compression level"); ax.set_xlabel("Retention level")
    ax.set_ylabel("Mean Token-F1"); ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(ROOT / "results" / "quality_vs_level.png", dpi=130); plt.close(fig)
    log("Saved -> results/quality_vs_level.png")
    # Pareto
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in methods:
        pts = [c for c in cells if c["method"] == m]
        ax.scatter([c["mean_cost_usd"] for c in pts], [c["mean_f1"] for c in pts], label=m, s=60)
        for c in pts:
            ax.annotate(f"{c['level']}", (c["mean_cost_usd"], c["mean_f1"]), fontsize=7, alpha=0.7)
    base = summary["baseline"]
    if base.get("n"):
        ax.scatter([base["mean_cost_usd"]], [base["mean_f1"]], marker="*", s=260, c="gold", edgecolors="k", label="baseline")
    ax.set_xlabel("Mean cost per query (USD)"); ax.set_ylabel("Mean Token-F1")
    ax.set_title("Cost / quality Pareto frontier"); ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(ROOT / "results" / "pareto.png", dpi=130); plt.close(fig)
    log("Saved -> results/pareto.png")

def main():
    runs = build_runs_from_cache()
    if not runs: log("No runs recovered."); sys.exit(1)
    optimizer.save_runs(runs); log(f"Saved {len(runs)} runs -> results/runs.json")
    optimizer.write_csv(runs); log("Saved -> results/results.csv")
    summary = optimizer.build_summary(runs)
    log(f"Saved -> results/summary.json  ({summary['n_items']} items, {summary['n_runs']} runs)")
    make_charts(summary)
    # Report
    sel = summary["selection"]; base = summary["baseline"]
    lines = [
        "# Smart Prompt Optimization - Results Report",
        f"\n- Items covered: **{summary['n_items']}** | Runs: **{summary['n_runs']}** | Model: `{model_name()}`",
        "\n> Note: LLM-as-Judge skipped (cache-only run). Selection based on Token-F1.\n",
        f"## Baseline\n- Tokens: {base.get('mean_tokens_in',0):.1f}  Cost/query: ${base.get('mean_cost_usd',0):.6f}  F1: {base.get('mean_f1',0):.4f}\n",
        "## All configurations\n| method | level | ratio | tokens_in | cost/query | F1 |\n|---|---|---|---|---|---|",
    ]
    for c in summary["cells"]:
        lines.append(f"| {c['method']} | {c['level']} | {c['mean_compression_ratio']:.3f} | {c['mean_tokens_in']:.1f} | ${c['mean_cost_usd']:.6f} | {c['mean_f1']:.4f} |")
    lines.append("\n## Optimisation Result")
    if sel.get("found"):
        s = sel["selected"]
        lines += [f"\n**Winner: `{s['method']}` @ level {s['level']}**",
                  f"- Input tokens: {base.get('mean_tokens_in',0):.1f} → {s['mean_tokens_in']:.1f} ({sel['input_token_reduction_pct']}% fewer)",
                  f"- Cost/query: ${base.get('mean_cost_usd',0):.6f} → ${s['mean_cost_usd']:.6f} ({sel['cost_reduction_pct']}% cheaper)",
                  f"- F1 retained: {sel['quality_retained_pct']}% of baseline"]
    else:
        lines.append(f"\nNo winner (judge scores absent): {sel.get('reason','')}")
    (ROOT / "results" / "report.md").write_text("\n".join(lines), encoding="utf-8")
    log("Saved -> results/report.md")
    log("\n=== FINAL RESULT ===")
    log(f"  Items  : {summary['n_items']} / 13")
    log(f"  Runs   : {summary['n_runs']}")
    log(f"  Base   : {base.get('mean_tokens_in',0):.1f} tok, ${base.get('mean_cost_usd',0):.6f}, F1={base.get('mean_f1',0):.4f}")
    if sel.get("found"):
        s = sel["selected"]
        log(f"  WINNER : {s['method']} @ {s['level']}  |  {sel['input_token_reduction_pct']}% fewer tokens  |  {sel['cost_reduction_pct']}% cheaper  |  F1 retained {sel['quality_retained_pct']}%")
    else:
        log(f"  No winner: {sel.get('reason','')}")
    log("=== DONE ===")

if __name__ == "__main__":
    main()
