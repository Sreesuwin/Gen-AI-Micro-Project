"""Streamlit dashboard for the Smart Prompt Optimization project.

Run with:  streamlit run app.py

Reads the stored results (results/runs.json + summary.json) and also
provides a live interactive prompt compression playground.
"""

from __future__ import annotations

import html
import json
import math
import pathlib

import pandas as pd
import streamlit as st

from src import compressor, cost, dataset, llm, metrics as qmetrics, optimizer

llm.load_env()

# ------------------------------------------------------------------ page config
st.set_page_config(
    page_title="Smart Prompt Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------ custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Hero gradient header */
.hero-header {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 16px;
    padding: 2.5rem 2rem 2rem 2rem;
    margin-bottom: 1.5rem;
    color: white;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.hero-header h1 { font-size: 2rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
.hero-header p  { font-size: 0.95rem; color: #a8b2d8; margin: 0.4rem 0 0 0; }

/* Section headers */
.section-title {
    font-size: 1.15rem; font-weight: 600; color: #e2e8f0;
    border-left: 4px solid #7c3aed; padding-left: 0.75rem;
    margin: 1.5rem 0 1rem 0;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(145deg, #1e1b4b, #312e81);
    border: 1px solid #4338ca44;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
.metric-card .value { font-size: 1.6rem; font-weight: 700; color: #a5b4fc; }
.metric-card .label { font-size: 0.75rem; color: #94a3b8; margin-top: 0.2rem; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-card .delta-good { font-size: 0.85rem; color: #34d399; font-weight: 600; }
.metric-card .delta-bad  { font-size: 0.85rem; color: #f87171; font-weight: 600; }

/* Method pill badges */
.badge-baseline { background:#374151; color:#d1d5db; padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
.badge-lexical  { background:#1e3a5f; color:#60a5fa; padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
.badge-sentence { background:#1e3a2f; color:#34d399; padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }

/* Compressed text box */
.prompt-box {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1rem;
    font-family: 'Courier New', monospace;
    font-size: 0.82rem;
    color: #e2e8f0;
    min-height: 200px;
    max-height: 400px;
    overflow-y: auto;
    white-space: pre-wrap;
    line-height: 1.6;
}

/* Winner banner */
.winner-banner {
    background: linear-gradient(135deg, #065f46, #047857);
    border: 1px solid #10b981;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    color: white;
    margin-bottom: 1rem;
}
.winner-banner h3 { margin: 0 0 0.3rem 0; font-size: 1.1rem; }
.winner-banner p  { margin: 0; color: #a7f3d0; font-size: 0.88rem; }

/* Progress bar for compression */
.compress-bar-outer {
    background: #1e293b; border-radius: 6px; height: 8px; margin: 6px 0;
}
.compress-bar-inner {
    background: linear-gradient(90deg, #7c3aed, #a855f7);
    height: 8px; border-radius: 6px; transition: width 0.4s;
}

/* Scrollable dataframe */
div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* Tab styling */
button[data-baseweb="tab"] { font-weight: 500 !important; }

/* Divider */
hr { border-color: #1e293b !important; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ constants
ROOT   = pathlib.Path(__file__).resolve().parent
RESULTS = optimizer.RESULTS_DIR
MODEL  = "gemma-4-26b-a4b-it"

PRICING = {"input": 0.15, "output": 0.60}   # $/1M tokens

# Models the LIVE evaluation may use, in order: the project's experiment model
# first, then a healthy fallback so the interactive demo does not fail when the
# free tier degrades. Both sides of a comparison always use the same model.
LIVE_MODELS = [MODEL, "gemini-3.1-flash-lite"]

METHOD_COLORS = {
    "baseline": ("🔘", "#6b7280"),
    "lexical":  ("🔵", "#3b82f6"),
    "sentence": ("🟢", "#10b981"),
}

# ------------------------------------------------------------------ data load
@st.cache_data
def load_data():
    runs     = optimizer.load_runs()
    items    = {it["id"]: it for it in dataset.load()}
    idf      = compressor.build_idf(dataset.corpus(list(items.values())))
    summary  = (
        json.loads(optimizer.SUMMARY_PATH.read_text(encoding="utf-8"))
        if optimizer.SUMMARY_PATH.exists()
        else {"cells": [], "baseline": {}, "selection": {}}
    )
    return runs, items, idf, summary

runs, items, idf, summary = load_data()

# ------------------------------------------------------------------ hero
st.markdown("""
<div class="hero-header">
  <h1>⚡ Smart Prompt Optimizer</h1>
  <p>Compress long prompts at multiple levels · Measure token savings, API cost & quality · Find the cheapest configuration that preserves answer quality</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — PROMPT CHECKER & SHORTENER (interactive)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🎮 Prompt Checker &amp; Shortener</div>', unsafe_allow_html=True)
st.caption(
    "Paste a prompt — with or without a document. The app checks it, shortens it, "
    "shows the original next to the shortened version and scores the result on "
    "compression, cost and quality metrics. Offline metrics are instant and free; "
    "live evaluation makes real model calls."
)

ALLOWED_DOCS = ["txt", "md", "csv", "json", "log", "py", "yml", "yaml", "ini", "cfg"]


def read_upload(uploaded) -> str:
    """Decode an uploaded text document (returns '' when nothing was uploaded)."""
    if uploaded is None:
        return ""
    try:
        return uploaded.getvalue().decode("utf-8", errors="ignore")
    except Exception:
        return ""


# --- Input panel
with st.container():
    inp_col, ctrl_col = st.columns([2, 1], gap="large")

    with inp_col:
        uploaded_file = st.file_uploader(
            "📄 Upload a document (optional) — txt / md / csv / json / log / py",
            type=ALLOWED_DOCS,
            key="doc_upload",
        )
        doc_paste = st.text_area(
            "📄 …or paste the document / context here (optional)",
            height=110,
            placeholder="Leave empty to check and shorten the prompt itself.",
            key="doc_paste",
        )
        user_prompt = st.text_area(
            "✏️ Your prompt (required)",
            height=150,
            placeholder="Paste the prompt you want checked and shortened — docs, policy text, specs, instructions...",
            key="user_prompt_input",
        )
        user_question = st.text_input(
            "❓ Relevance hint (optional) — words the shortened text must keep",
            placeholder="e.g. return policy refund deadline",
            key="user_question_input",
        )

    with ctrl_col:
        st.markdown("#### ⚙️ Compression Settings")

        play_method = st.radio(
            "Method",
            options=compressor.METHODS,
            format_func=lambda m: {
                "baseline": "🔘 Baseline (no compression)",
                "lexical":  "🔵 Lexical  (token scoring)",
                "sentence": "🟢 Sentence (TF-IDF similarity)",
            }[m],
            key="play_method",
            index=2,
        )

        play_level = st.select_slider(
            "Retention level",
            options=compressor.LEVELS,
            value=0.45,
            format_func=lambda v: f"{int(v*100)}% kept" if v < 1.0 else "100% (no change)",
            key="play_level",
        )

        st.markdown("---")
        compare_all = st.checkbox(
            "📊 Compare all methods side-by-side",
            value=True,
            key="compare_all",
        )
        live_eval = st.checkbox(
            "🤖 Live evaluation — send both versions to the model and compare the answers (2 API calls)",
            value=False,
            key="live_eval",
            help="Measures the real tokens, cost and latency of both versions and scores how well the answer survived compression.",
        )
        judge_eval = st.checkbox(
            "🏅 Also score the shortened answer with the LLM judge (+1 API call)",
            value=True,
            key="judge_eval",
            disabled=not live_eval,
        )

        run_playground = st.button(
            "⚡ Check & Shorten Prompt",
            type="primary",
            use_container_width=True,
            key="run_playground",
        )

# --- Evaluate (only on button click: live evaluation costs real API calls)
def resolve_inputs() -> tuple[str, str, str]:
    """Return (source_text, question, mode).

    document mode -> the uploaded/pasted document is what gets shortened and the
                     prompt is kept verbatim as the question the model must answer.
    prompt mode   -> there is no document, so the prompt itself is shortened.
    """
    doc = ("\\n".join(x for x in [read_upload(uploaded_file), doc_paste.strip()] if x)).strip()
    prompt = (user_prompt or "").strip()
    hint = (user_question or "").strip()
    if doc:
        return doc, prompt or hint, "document"
    return prompt, hint, "prompt"


def measure(source: str, question: str, method: str, level: float,
            idf: dict, mode: str) -> dict:
    """Compress one configuration and score it with every offline metric."""
    ctx = compressor.compress(source, method, level, idf, question)
    orig_tok = compressor.estimate_tokens(source)
    comp_tok = compressor.estimate_tokens(ctx)
    ratio = comp_tok / max(1, orig_tok)
    saved_tok = orig_tok - comp_tok

    # Cost model: same assumed answer length (200 tokens) on both sides, so the
    # difference shown is caused purely by the input side being shorter.
    fixed_out = 200
    orig_cost = cost.cost_usd(orig_tok, fixed_out, MODEL)
    comp_cost = cost.cost_usd(comp_tok, fixed_out, MODEL)

    fidelity = qmetrics.token_f1(ctx, source)          # wording kept (P/R/F1)
    key_src = question or source[:160]                 # topic the text must protect
    kw = qmetrics.keyword_coverage(key_src, ctx, source)   # key-term survival
    info = qmetrics.information_retention(source, ctx, idf)   # IDF-weighted mass
    sents = qmetrics.sentence_stats(source, ctx)       # whole sentences kept

    return {
        "method": method, "level": level, "mode": mode,
        "source": source, "question": question, "compressed": ctx,
        "orig_tokens": orig_tok, "comp_tokens": comp_tok,
        "saved_tokens": saved_tok, "ratio": ratio,
        "orig_cost": orig_cost, "comp_cost": comp_cost,
        "savings_pct": round((1 - comp_cost / orig_cost) * 100, 1) if orig_cost else 0.0,
        "fidelity": fidelity, "keywords": kw,
        "info_retention": info, "sentences": sents,
        "verdict": qmetrics.verdict(ratio, fidelity["f1"], kw["coverage"]),
    }


def run_live(res: dict, with_judge: bool) -> dict:
    """Send the original and the shortened version to the model, compare answers.

    Both requests are made with the SAME model so the comparison is fair. If the
    primary model is unavailable (the free tier returns 500/504 under load) the
    whole comparison is retried on the fallback model rather than failing.
    """
    if res["mode"] == "document":
        base_prompt = compressor.build_prompt({"question": res["question"]}, res["source"])
        comp_prompt = compressor.build_prompt({"question": res["question"]}, res["compressed"])
        judge_q = res["question"] or "Answer the question about the context."
    else:
        base_prompt, comp_prompt = res["source"], res["compressed"]
        judge_q = res["question"] or res["source"][:160]

    errors: list[str] = []
    out_b = out_c = None
    used_model = None
    for candidate in LIVE_MODELS:
        try:
            out_b = llm.generate(base_prompt, model=candidate)
            out_c = llm.generate(comp_prompt, model=candidate)
            used_model = candidate
            break
        except Exception as err:  # noqa: BLE001 - try the next model
            errors.append(f"{candidate}: {str(err)[:90]}")
    if used_model is None:
        raise RuntimeError(" | ".join(errors))

    cost_b = cost.cost_usd(out_b["tokens_in"], out_b["tokens_out"], used_model)
    cost_c = cost.cost_usd(out_c["tokens_in"], out_c["tokens_out"], used_model)

    judge = None
    if with_judge:
        try:
            judge = qmetrics.judge(judge_q, out_c["text"], out_b["text"], model=used_model)
        except Exception as err:  # noqa: BLE001 - judge is a bonus, not a blocker
            judge = {"score": 0, "reason": f"judge unavailable: {str(err)[:80]}"}

    return {
        "base": out_b, "comp": out_c,
        "base_prompt": base_prompt, "comp_prompt": comp_prompt,
        "model": used_model,
        "base_cost": cost_b, "comp_cost": cost_c,
        "cost_saving_pct": round((1 - cost_c / cost_b) * 100, 1) if cost_b else 0.0,
        "answer_f1": qmetrics.token_f1(out_c["text"], out_b["text"]),
        "judge": judge,
    }


if run_playground:
    source_text, question_text, mode = resolve_inputs()
    if not source_text:
        st.warning("⚠️ Paste a prompt first — that is what gets checked and shortened.")
    else:
        user_idf = compressor.build_idf(dataset.corpus(list(items.values())) + [source_text])
        with st.spinner("Compressing and measuring..."):
            chosen = measure(source_text, question_text, play_method, play_level, user_idf, mode)
            all_methods = [measure(source_text, question_text, m, play_level, user_idf, mode)
                           for m in compressor.METHODS]
            sweep = [measure(source_text, question_text, play_method, lv, user_idf, mode)
                     for lv in compressor.LEVELS]
        live = None
        if live_eval and play_method != "baseline":
            try:
                with st.spinner("Calling the model with both versions..."):
                    live = run_live(chosen, judge_eval)
            except Exception as err:  # noqa: BLE001 - report, never crash the app
                st.warning(f"⚠️ Live evaluation failed: {str(err)[:220]}")
        st.session_state["playground_result"] = {
            "chosen": chosen, "all_methods": all_methods, "sweep": sweep,
            "live": live, "method": play_method, "level": play_level,
            "compare_all": compare_all, "mode": mode,
        }

R = st.session_state.get("playground_result")
if R:
    with st.container():
        result = R["chosen"]
        play_method = R["method"]
        play_level = R["level"]
        compare_all = R["compare_all"]
        prompt_text = result["source"]
        question_text = result["question"]

        if live_eval and play_method == "baseline":
            st.info("Live evaluation compares the shortened text against the original — "
                    "pick **lexical** or **sentence** to enable it.")

    # ── Verdict banner ────────────────────────────────────────────────────────
        v = result["verdict"]
        banner = f"**{v['label']}** — {v['text']}"
        {"pass": st.success, "warn": st.warning, "fail": st.error}.get(v["level"], st.info)(banner)

    # ── KPI bar ───────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(f"**Results for `{play_method}` compression @ {int(play_level*100)}% retention**")

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Original tokens",    f"{result['orig_tokens']:,}")
        k2.metric("Compressed tokens",  f"{result['comp_tokens']:,}",
                  delta=f"-{result['saved_tokens']:,} tokens",
                  delta_color="inverse")
        k3.metric("Retention ratio",    f"{result['ratio']:.1%}")
        k4.metric("Est. cost savings",  f"{result['savings_pct']:.1f}%",
                  delta="vs uncompressed",
                  delta_color="inverse" if result['savings_pct'] > 0 else "normal")
        k5.metric("Cost / query (est.)",
                  f"${result['comp_cost']:.6f}",
                  delta=f"was ${result['orig_cost']:.6f}",
                  delta_color="inverse")

        # Compression progress bar
        bar_pct = int(result['ratio'] * 100)
        st.markdown(
            f"<div style='font-size:0.78rem;color:#94a3b8;margin-top:0.3rem;'>"
            f"Compression progress — keeping {bar_pct}% of original</div>"
            f"<div class='compress-bar-outer'>"
            f"<div class='compress-bar-inner' style='width:{bar_pct}%'></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Quality metrics (offline, no API calls) ────────────────────────
        fid, kw = result["fidelity"], result["keywords"]
        sent = result["sentences"]
        st.markdown("**🧮 Quality checks on the shortened text**")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Wording kept (Token-F1)", f"{fid['f1']:.1%}",
                  delta=f"precision {fid['precision']:.0%} · recall {fid['recall']:.0%}")
        q2.metric("Key terms kept", f"{kw['coverage']:.0%}",
                  delta=f"{len(kw['covered'])}/{len(kw['terms'])} terms")
        q3.metric("Information retained (IDF-weighted)",
                  f"{result['info_retention']:.1%}")
        q4.metric("Whole sentences kept", f"{sent['kept']}/{sent['original']}",
                  delta=f"{sent['retention']:.0%} of sentences")

        if kw["terms"]:
            kept_html = " ".join(
                f"<span style='background:#064e3b;color:#34d399;padding:2px 8px;"
                f"border-radius:12px;margin-right:4px;font-size:0.8rem;'>"
                f"✓ {html.escape(t)}</span>" for t in kw["covered"]
            ) or "<i>none</i>"
            lost_html = " ".join(
                f"<span style='background:#4c0519;color:#f87171;padding:2px 8px;"
                f"border-radius:12px;margin-right:4px;font-size:0.8rem;'>"
                f"✗ {html.escape(t)}</span>" for t in kw["missing"]
            ) or "<i>none</i>"
            st.markdown(
                "<div style='font-size:0.85rem;margin:0.4rem 0 0.2rem 0;'>"
                f"<b style='color:#94a3b8'>KEY TERMS KEPT:</b> {kept_html}<br>"
                f"<b style='color:#94a3b8'>KEY TERMS LOST:</b> {lost_html}</div>",
                unsafe_allow_html=True,
            )

        # ── Side-by-side text comparison ─────────────────────────────────────
        st.markdown("#### 📄 Original vs Shortened")
        left_col, right_col = st.columns(2, gap="medium")

        with left_col:
            st.markdown(
                f"**🔘 Original** &nbsp; `{result['orig_tokens']:,} tokens`  &nbsp; "
                f"`~${result['orig_cost']:.6f}`",
                unsafe_allow_html=True,
            )
            st.text_area(
                "original_text", value=prompt_text,
                height=320, label_visibility="collapsed",
            )

        with right_col:
            icon = METHOD_COLORS.get(play_method, ("🔘", ""))[0]
            st.markdown(
                f"**{icon} {play_method.capitalize()}** &nbsp; `{result['comp_tokens']:,} tokens` "
                f"&nbsp; <span style='color:#34d399;font-size:0.85rem;'>"
                f"▼ {result['savings_pct']:.0f}% cheaper</span>",
                unsafe_allow_html=True,
            )
            st.text_area(
                "compressed_text", value=result["compressed"],
                height=320, label_visibility="collapsed",
            )

        # ── Multi-method comparison ───────────────────────────────────────────
        if compare_all:
            st.markdown("#### 📊 All Methods at This Level — Head-to-Head")

            all_results = R["all_methods"]

            tab_labels = [
                f"{'🔘' if r['method']=='baseline' else '🔵' if r['method']=='lexical' else '🟢'} "
                f"{r['method'].capitalize()}"
                for r in all_results
            ]
            tabs = st.tabs(tab_labels)

            for tab, res in zip(tabs, all_results):
                with tab:
                    ta1, ta2, ta3, ta4 = tab.columns(4)
                    ta1.metric("Tokens",       f"{res['comp_tokens']:,}",
                               delta=f"-{res['saved_tokens']:,}", delta_color="inverse")
                    ta2.metric("Ratio",        f"{res['ratio']:.1%}")
                    ta3.metric("Cost savings", f"{res['savings_pct']:.1f}%",
                               delta_color="inverse")
                    ta4.metric("Cost/query",   f"${res['comp_cost']:.6f}")

                    ta5, ta6 = tab.columns(2)
                    ta5.metric("Key terms kept", f"{res['keywords']['coverage']:.0%}",
                               delta=f"F1 {res['fidelity']['f1']:.2f}")
                    ta6.metric("Verdict", res["verdict"]["label"])

                    tab.text_area(
                        f"tab_comp_{res['method']}",
                        value=res["compressed"],
                        height=260,
                        label_visibility="collapsed",
                    )

            # Summary comparison table
            comp_df = pd.DataFrame([
                {
                    "Method":          r["method"],
                    "Tokens (in)":     r["comp_tokens"],
                    "Tokens saved":    r["saved_tokens"],
                    "Retention":       f"{r['ratio']:.1%}",
                    "Cost savings %":  f"{r['savings_pct']:.1f}%",
                    "Cost/query":      f"${r['comp_cost']:.6f}",
                    "Token-F1":        f"{r['fidelity']['f1']:.3f}",
                    "Key terms kept":  f"{r['keywords']['coverage']:.0%}",
                    "Info retained":   f"{r['info_retention']:.1%}",
                    "Verdict":         r["verdict"]["label"],
                }
                for r in all_results
            ])
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # ── All levels sweep for chosen method ───────────────────────────────
        with st.expander(f"📈 Sweep all retention levels for `{play_method}` method"):
            level_results = R["sweep"]
            sweep_df = pd.DataFrame([
                {
                    "Level":          r["level"],
                    "% kept":         f"{int(r['level']*100)}%",
                    "Tokens":         r["comp_tokens"],
                    "Tokens saved":   r["saved_tokens"],
                    "Retention":      f"{r['ratio']:.1%}",
                    "Cost savings %": f"{r['savings_pct']:.1f}%",
                    "Token-F1":       f"{r['fidelity']['f1']:.3f}",
                    "Key terms kept": f"{r['keywords']['coverage']:.0%}",
                    "Info retained":  f"{r['info_retention']:.1%}",
                    "Verdict":        r["verdict"]["label"],
                }
                for r in level_results
            ])
            st.dataframe(sweep_df, use_container_width=True, hide_index=True)

            # Mini bar chart of token count vs level
            chart_df = pd.DataFrame({
                "Retention level": [r["level"] for r in level_results],
                "Compressed tokens": [r["comp_tokens"] for r in level_results],
            }).set_index("Retention level")
            st.bar_chart(chart_df)

        # ── Live evaluation: both versions really sent to the model ───────
        if R.get("live"):
            live = R["live"]
            st.markdown("---")
            st.markdown("#### 🤖 Live Evaluation — original vs shortened, answered by the model")

            lf1 = live["answer_f1"]
            e1, e2, e3, e4, e5 = st.columns(5)
            e1.metric("Tokens in (original)", f"{live['base']['tokens_in']:,}",
                      delta=f"{live['base']['tokens_out']:,} out · "
                            f"{live['base']['latency_ms']/1000:.1f}s")
            e2.metric("Tokens in (shortened)", f"{live['comp']['tokens_in']:,}",
                      delta=f"{live['comp']['tokens_out']:,} out · "
                            f"{live['comp']['latency_ms']/1000:.1f}s",
                      delta_color="inverse")
            e3.metric("Real cost saving", f"{live['cost_saving_pct']:.1f}%",
                      delta=f"${live['base_cost']:.6f} → ${live['comp_cost']:.6f}",
                      delta_color="inverse")
            e4.metric("Answer similarity (Token-F1)", f"{lf1['f1']:.3f}",
                      delta=f"precision {lf1['precision']:.2f} · recall {lf1['recall']:.2f}")
            if live.get("judge") and live["judge"]["score"]:
                e5.metric("Judge score", f"{live['judge']['score']}/5",
                          delta=str(live["judge"]["reason"])[:48])
            else:
                e5.metric("Cost / query", f"${live['comp_cost']:.6f}",
                          delta=f"was ${live['base_cost']:.6f}", delta_color="inverse")

            st.caption(f"Both answers were produced by the same model (`{live['model']}`) "
                       "so the comparison is fair. Re-run if the provider was busy.")

            if live["comp"]["tokens_out"] >= 2040:
                st.caption("⚠️ The shortened run hit the output cap, so its answer may be "
                           "truncated — judge the similarity score with that in mind.")

            la, ra = st.columns(2, gap="medium")
            with la:
                st.markdown("**🔘 Answer from the ORIGINAL prompt**")
                st.text_area("live_base_answer", live["base"]["text"], height=260,
                             label_visibility="collapsed")
            with ra:
                st.markdown("**🟢 Answer from the SHORTENED prompt**")
                st.text_area("live_comp_answer", live["comp"]["text"], height=260,
                             label_visibility="collapsed")

            with st.expander("🔎 What exactly was sent to the model"):
                st.markdown("**Original request**")
                st.code(live["base_prompt"][:6000], language=None)
                st.markdown("**Shortened request**")
                st.code(live["comp_prompt"][:6000], language=None)


st.divider()

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — EXPERIMENT RESULTS (from stored runs)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">📊 Experiment Results (Stored Runs)</div>', unsafe_allow_html=True)

if not runs:
    st.warning(
        "No experiment results yet. Run the sweep first:\n\n"
        "`python main.py --run-all`"
    )
else:
    base = summary.get("baseline", {})
    sel  = summary.get("selection", {})

    # KPI row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Runs measured",        summary.get("n_runs", len(runs)))
    m2.metric("Dataset items",        summary.get("n_items", len(items)))
    m3.metric("Baseline tokens",      f"{base.get('mean_tokens_in', 0):.0f}")
    m4.metric("Baseline cost/query",  f"${base.get('mean_cost_usd', 0):.5f}")
    m5.metric("Baseline Token-F1",    f"{base.get('mean_f1', 0):.3f}")

    st.markdown("---")

    # Recommendation
    st.markdown('<div class="section-title">🏆 Recommended Configuration</div>', unsafe_allow_html=True)

    if sel.get("found"):
        s = sel["selected"]
        st.markdown(f"""
        <div class="winner-banner">
          <h3>✅ Winner: <code>{s['method']}</code> @ retention level {s['level']}</h3>
          <p>Cheapest configuration that satisfies the quality constraint.</p>
        </div>
        """, unsafe_allow_html=True)
        w1, w2, w3, w4, w5 = st.columns(5)
        w1.metric("Method",            s["method"].upper())
        w2.metric("Retention level",   f"{s['level']:.2f}")
        w3.metric("Input tokens",      f"{base.get('mean_tokens_in',0):.0f} → {s['mean_tokens_in']:.0f}",
                  delta=f"-{sel['input_token_reduction_pct']}%", delta_color="inverse")
        w4.metric("Cost per query",    f"${s['mean_cost_usd']:.5f}",
                  delta=f"-{sel['cost_reduction_pct']}%", delta_color="inverse")
        w5.metric("Quality retained",  f"{sel['quality_retained_pct']:.1f}% F1",
                  delta=f"judge {s['mean_judge']:.2f}")
    else:
        st.error("No configuration satisfied both constraints.")
        if sel.get("best_available"):
            ba = sel["best_available"]
            st.info(
                f"**Best available** (without meeting all constraints): "
                f"`{ba['method']}` @ level {ba['level']}  |  "
                f"F1 = {ba['mean_f1']}  |  judge = {ba['mean_judge']}"
            )
        if sel.get("reason"):
            st.caption(sel["reason"])

    # Re-run optimizer controls
    with st.expander("🔧 Adjust optimiser constraints"):
        constraint = sel.get("constraint", {})
        oc1, oc2 = st.columns(2)
        min_judge  = oc1.slider("Min judge score (1–5)", 0.0, 5.0,
                                float(constraint.get("min_judge", optimizer.DEFAULT_MIN_JUDGE)), 0.1)
        f1_frac    = oc2.slider("Min F1 as fraction of baseline", 0.5, 1.0,
                                float(constraint.get("f1_fraction", optimizer.DEFAULT_F1_FRACTION)), 0.01)
        if st.button("Recompute selection", key="recompute_btn"):
            new_summary = optimizer.build_summary(runs, min_judge=min_judge, f1_fraction=f1_frac)
            st.session_state["summary"] = new_summary
            st.rerun()
    summary = st.session_state.get("summary", summary)

    st.markdown("---")

    # Charts
    st.markdown('<div class="section-title">📈 Cost & Quality Charts</div>', unsafe_allow_html=True)
    chart_tab1, chart_tab2, chart_tab3 = st.tabs(["💰 Cost vs Level", "🎯 Quality vs Level", "⚖️ Pareto Frontier"])
    for tab, name in zip([chart_tab1, chart_tab2, chart_tab3],
                         ["cost_vs_level.png", "quality_vs_level.png", "pareto.png"]):
        path = RESULTS / name
        if path.exists():
            tab.image(str(path), use_container_width=True)
        else:
            tab.info(f"`{name}` not yet generated — run `python main.py --run-all`.")

    st.markdown("---")

    # All configurations table
    st.markdown('<div class="section-title">🗃️ All Configurations</div>', unsafe_allow_html=True)
    cells = summary.get("cells", [])
    if cells:
        df = pd.DataFrame(cells)
        show = df[[
            "method", "level", "n", "mean_compression_ratio",
            "mean_tokens_in", "mean_tokens_out",
            "mean_cost_usd", "mean_f1", "mean_judge", "n_judged",
        ]].copy()
        show.columns = [
            "method", "level", "n", "ratio",
            "tokens_in", "tokens_out",
            "cost/query (USD)", "Token-F1", "judge", "judged runs",
        ]
        st.dataframe(
            show.style.background_gradient(subset=["Token-F1"], cmap="Greens")
                      .background_gradient(subset=["cost/query (USD)"], cmap="Reds_r")
                      .format({"ratio": "{:.3f}", "Token-F1": "{:.4f}", "cost/query (USD)": "${:.6f}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"💡 **judge = 0** means the LLM-as-Judge pass has not run yet (requires live API calls). "
            "Run `python main.py --report` when the API is stable to fill in judge scores."
        )

    # Raw runs table
    with st.expander("🔍 Raw runs table"):
        raw = pd.DataFrame(runs)
        raw["answer"] = raw["answer"].str.slice(0, 200)
        st.dataframe(
            raw[[
                "item_id", "method", "level", "tokens_in", "tokens_out",
                "cost_usd", "compression_ratio", "f1", "judge_score", "answer",
            ]],
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # ── Dataset item prompt inspector ─────────────────────────────────────────
    st.markdown('<div class="section-title">🔬 Dataset Prompt Inspector</div>', unsafe_allow_html=True)
    st.caption("Select a dataset item to re-compress and preview exactly what the model received.")

    pi_c1, pi_c2, pi_c3 = st.columns([1, 1, 2])
    pi_method  = pi_c1.selectbox("Method", compressor.METHODS, index=1, key="pi_method")
    pi_level   = pi_c2.selectbox(
        "Retention level", compressor.LEVELS,
        index=compressor.LEVELS.index(0.45), key="pi_level"
    )
    pi_item_id = pi_c3.selectbox(
        "Dataset item", list(items.keys()),
        format_func=lambda k: f"{k}  —  {items[k]['title']}",
        key="pi_item_id"
    )
    pi_item = items[pi_item_id]
    pi_ctx  = (
        pi_item["long_prompt"]
        if pi_method == "baseline"
        else compressor.compress(pi_item["long_prompt"], pi_method, pi_level, idf, pi_item["question"])
    )

    po1, po2, po3 = st.columns(3)
    pi_orig_tok = compressor.estimate_tokens(pi_item["long_prompt"])
    pi_comp_tok = compressor.estimate_tokens(pi_ctx)
    po1.metric("Original tokens (est.)", f"{pi_orig_tok:,}")
    po2.metric("Compressed tokens (est.)", f"{pi_comp_tok:,}",
               delta=f"-{pi_orig_tok - pi_comp_tok:,}", delta_color="inverse")
    po3.metric("Retention",
               f"{pi_comp_tok / max(1, pi_orig_tok):.1%}")

    pl, pr = st.columns(2)
    with pl:
        st.markdown("**📄 Original prompt**")
        st.text_area("pi_orig", pi_item["long_prompt"], height=380,
                     label_visibility="collapsed")
    with pr:
        icon2 = METHOD_COLORS.get(pi_method, ("🔘",""))[0]
        st.markdown(f"**{icon2} {pi_method.capitalize()} @ {pi_level}**")
        st.text_area("pi_comp", pi_ctx, height=380,
                     label_visibility="collapsed")

    st.markdown(f"**Question:** {pi_item['question']}")
    with st.expander("📖 Reference answer"):
        st.write(pi_item["reference_answer"])

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#475569;font-size:0.8rem;padding:0.5rem 0;'>"
    "Smart Prompt Optimization · GenAI Micro Project (7CS 1018) · Alliance School of Advanced Computing"
    "</div>",
    unsafe_allow_html=True,
)
