# How to Run the Prompt Checker & Shortener

A complete, step-by-step guide to the website: how to start it, what every button
does, how to send your own prompt, how to read the results, and how to regenerate
the experiment data behind it.

---

## 1. What you need

| Requirement | Notes |
|---|---|
| Python 3.11+ | `python --version` |
| Packages | `pip install -r requirements.txt` (streamlit, pandas, matplotlib, google-genai, python-dotenv) |
| API key | **Only** for live evaluation and for re-running the experiment. Everything else works offline. |
| Browser | Any modern browser - the app is served locally. |

---

## 2. Start the website

```bash
pip install -r requirements.txt
streamlit run app.py
```

The terminal prints:

```
Local URL: http://localhost:8501
Network URL: http://<your-ip>:8501
```

Open the **Local URL**. The page loads instantly - nothing is computed until you
press a button.

**Where the API key goes (pick one):**

```text
Option A (local):  create API_KEY.env or .env in the project folder
                   either  GEMINI_API_KEY=your_key_here
                   or just the bare key on one line

Option B (local):  .streamlit/secrets.toml
                   GEMINI_API_KEY = "your_key_here"

Option C (deployed): Streamlit Cloud -> Settings -> Secrets, same line as Option B
```

Resolution order: local file -> Streamlit secrets -> `GOOGLE_API_KEY`.

---

## 3. The page at a glance

| Part | Purpose |
|---|---|
| **Prompt Checker & Shortener** | The interactive tool: paste a prompt, shorten it, get metrics |
| **Experiment Results** | Charts and tables from the offline sweep (`main.py --run-all`) |
| **Dataset Prompt Inspector** | Preview exactly what the model received for any dataset item |
| **Footer** | Project attribution |

Everything in the first section is instant and free unless you switch on live
evaluation.

---

## 4. Workflow A - shorten a prompt (no document)

1. Leave the document boxes **empty**.
2. Paste your text into **✏️ Your prompt (required)**.
3. Optionally fill **❓ Relevance hint** with the words the shortened text must keep
   (e.g. `refund deadline electronics`). Without it, the first 160 characters of
   your text act as the topic.
4. Choose a **Method** and a **Retention level** (start with `sentence` @ `45% kept`).
5. Press **⚡ Check & Shorten Prompt**.

What happens: your prompt itself is compressed and becomes the shortened prompt -
nothing else is added to it.

You will see, in order:

1. A **verdict banner** (green / orange / red / grey).
2. A **KPI row**: original tokens, shortened tokens, retention ratio, estimated cost
   saving, estimated cost per query.
3. A **compression bar** showing how much of the original is kept.
4. **Quality checks**: wording kept (Token-F1), key terms kept, information retained
   (IDF-weighted), whole sentences kept - plus green ✓ / red ✗ chips for each key term.
5. **Original vs Shortened** side by side, with token counts and cost on each header.
6. **Head-to-head tabs** (if *Compare all methods* is on) and a summary table.
7. **Sweep all retention levels** expander: all 7 levels for your method, table + chart.

---

## 5. Workflow B - shorten a document, keep the prompt

1. **Upload a document** (`.txt .md .csv .json .log .py .yml .yaml .ini .cfg`) **or**
   paste it into *📄 …or paste the document / context here*.
2. Put your question/instruction in **✏️ Your prompt** - in this mode the prompt is
   **never compressed**, it is passed to the model verbatim.
3. Press **⚡ Check & Shorten Prompt**.

What happens: only the document is shortened, then the app assembles the request as

```text
<shortened document>

QUESTION: <your prompt>

If the answer is not stated in the context, reply exactly: not stated in context.

Answer in at most 60 words, covering every detail needed to answer the question correctly.
```

The two fixed instructions keep answers the same length at every compression level
(so the cost comparison stays fair) and stop the model from reasoning forever when a
heavily shortened context no longer contains the answer.

*Upload and paste are concatenated if you use both.*

---

## 6. Workflow C - live evaluation (real model calls)

Tick **🤖 Live evaluation** (and optionally **🏅 …LLM judge**) before pressing the
button. This sends **both** versions to the model and compares the answers.

You get a **🤖 Live Evaluation** panel with:

| Metric | Meaning |
|---|---|
| Tokens in (original) | Real tokenizer count of the uncompressed request (+ output tokens and latency) |
| Tokens in (shortened) | Real tokenizer count of the compressed request |
| Real cost saving | Actual cost of both requests using the real output length |
| Answer similarity (Token-F1) | How well the answer survived compression |
| Judge score | 1-5 quality score of the shortened answer against the original answer |
| Model used | Both sides always use the **same** model, shown under the metrics |

Below it: both answers side by side, and an expander **🔎 What exactly was sent to
the model** showing the two full requests.

Notes:
- The project model is tried first; if the free tier errors, the app automatically
  falls back to a second model **for both requests** so the comparison stays fair.
- A failed call shows a warning - every offline metric is still there, just press the
  button again. Repeated identical requests are cached and cost nothing.
- If an answer hits the output cap, a note appears under the metrics.

---

## 7. Every control, explained

### Inputs

| Control | Required | What it does |
|---|---|---|
| Document upload | No | Long material to shorten (text file types) |
| Paste document box | No | Same as upload, for quick pasting; combined with the upload |
| Your prompt | Yes | The instruction/question. Shortened when there is no document, kept verbatim in document mode |
| Relevance hint | No | Words to protect; also drives key-term coverage. Defaults to the first 160 characters of the source |

### Settings

| Control | Options | Effect |
|---|---|---|
| Method | `baseline`, `lexical`, `sentence` | Which algorithm shortens the text (see §8) |
| Retention level | `100%`, `90%`, `75%`, `60%`, `45%`, `30%`, `15%` kept | Target size of the shortened text |
| Compare all methods | on/off | Head-to-head tabs + summary table for all three methods at the chosen level |
| Live evaluation | on/off | 2 real API calls (original vs shortened) |
| LLM judge | on/off (needs live evaluation) | +1 API call scoring the answer 1-5 |

---

## 8. The three methods and when to use them

| Method | What it does | Best for |
|---|---|---|
| **baseline** | Nothing - the control group | Seeing what the original costs and scores |
| **lexical** | Scores every word (question terms, numbers, ALL-CAPS, rare/high-IDF words score high; stopwords score ~0) and keeps the best until the character budget is spent | Keyword-heavy extraction; aggressive cuts. Reads telegraphic, so quality metrics usually drop fast |
| **sentence** | Splits into sentences, scores each by TF-IDF cosine similarity to your prompt (headings always kept), keeps the best that fit, restores original order | Prose, policies, docs - keeps full grammatical statements, so quality holds up at lower levels |

**Levels** are target sizes: `45% kept` aims to shorten the text to about 45% of its
original length. `100%` means no change.

Rule of thumb: start at `sentence @ 60%`, drop to `45%` or `30%` while the verdict
stays green, and only try `lexical` when you care more about size than readability.

---

## 9. How to read every number

### Size and cost (estimated)

| Metric | How it is calculated |
|---|---|
| Original / shortened tokens | `round(characters / 4)` - a tokenizer-independent estimate |
| Retention ratio | `shortened tokens / original tokens` |
| Tokens saved | `original - shortened` |
| Cost / query | `tokens_in/1e6 x $0.15 + 200/1e6 x $0.60` (a fixed 200-token answer on both sides, so the difference is caused only by the input) |
| Est. cost savings | `(1 - shortened cost / original cost) x 100` |

Live evaluation shows the provider's **real** token counts instead, so those two can
differ slightly - that is expected.

### Quality (offline, free)

| Metric | How it is calculated | Good value |
|---|---|---|
| Wording kept (Token-F1) | Multiset overlap of words: precision = overlap/shortened words, recall = overlap/original words, F1 = harmonic mean | ≥ 0.85 |
| Key terms kept | First 15 unique topic words after removing stopwords/question words/short tokens; only terms present in the original are scored | 100% |
| Information retained (IDF-weighted) | Weighted mass of surviving words, `idf = ln((n+1)/(df+0.5))+1`, divided by the total | ≥ 0.8 |
| Whole sentences kept | Original sentences that appear verbatim / total | higher = more faithful |

Terms missing from the original are listed as *absent*, never counted against you -
which is why **baseline always shows 100%**.

### The verdict

Checked in this order:

| # | Condition | Banner |
|---|---|---|
| 1 | retention ≥ 0.999 | NO COMPRESSION (grey) |
| 2 | key terms < 50% **or** F1 < 0.60 | NOT SAFE TO USE (red) |
| 3 | key terms < 100% **or** F1 < 0.85 | USABLE WITH CARE (orange) |
| 4 | otherwise | SAFE TO USE (green) |

---

## 10. Experiment Results section

Built from the stored sweep, so it needs no API calls.

| Element | What it shows |
|---|---|
| KPI row | Runs measured, dataset size, baseline tokens / cost / F1 |
| 🏆 Recommended configuration | The **cheapest** (method, level) with mean judge ≥ **4.0** and mean F1 ≥ **90%** of the uncompressed baseline |
| 🔧 Adjust optimiser constraints | Move *Min judge score* and *Min F1 fraction*, then **Recompute selection** - re-runs the selection logic only, no API calls |
| Chart tabs | 💰 Cost vs Level, 🎯 Quality vs Level (F1 and judge), ⚖️ Pareto frontier (each point labelled with its level) |
| 🗃️ All configurations | Per-cell averages: ratio, tokens, cost, Token-F1, judge |
| 🔍 Raw runs | Every individual call, with its answer truncated to 200 chars |

A judge value of `0` in the table means the judge pass has not run yet - see §11.

---

## 11. Regenerating the results (command line)

```bash
# quick end-to-end check: API key + compression sanity + 3 items
python main.py --smoke

# full experiment: sweep -> judging -> optimisation -> charts -> report
python main.py --run-all

# same, but keep the runs already measured (never re-bills a cached call)
python main.py --run-all --resume

# rebuild summary / charts / report from stored runs (no sweep)
python main.py --report
```

| Flag | Meaning |
|---|---|
| `--resume` | Keep stored runs, fill only what is missing |
| `--no-judge` | Skip the LLM-judge pass |
| `--min-judge 4.0` | Minimum acceptable judge score |
| `--f1-fraction 0.90` | Minimum F1 as a fraction of baseline |
| `--max-judged-levels 4` | How many levels per method get judged |
| `--smoke-items 3` | Dataset items used by `--smoke` |

Outputs land in `results/`: `runs.json`, `summary.json`, `results.csv`,
`report.md`, `cost_vs_level.png`, `quality_vs_level.png`, `pareto.png`.

**Environment variables:** `GENAI_MODEL` (default `gemma-4-26b-a4b-it`),
`GENAI_MIN_INTERVAL` (4.1), `GENAI_MAX_RETRIES` (3), `GENAI_TIMEOUT_MS` (120000),
`GENAI_MAX_OUTPUT_TOKENS` (2048), `GENAI_WORKERS` (3).

> Tip: a full sweep is ~195 calls. If it is interrupted, `--resume` picks up exactly
> where it stopped using `results/cache.json`, and cached calls are free.

---

## 12. Dataset Prompt Inspector

1. Pick a **Method**, a **Retention level**, and one of the **13 dataset items**.
2. The panel shows original vs shortened token counts and retention, then both texts
   side by side.
3. The item's question is printed below, with its reference answer in an expander.

Use it to see exactly what the model receives at any configuration - useful when
explaining why a method scored the way it did.

---

## 13. Deploying

See the *Deployment* section of `README.md`. Short version: point Streamlit Cloud at
this repo, branch `main`, main file `app.py`, and put the key in
**Settings → Secrets** as `GEMINI_API_KEY = "..."`. The committed `results/` folder
means the charts appear immediately, and the app runs without a key except for live
evaluation.

---

## 14. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| "Live evaluation failed ... 500 / 504" | Free-tier provider under load | Press the button again; a fallback model is used automatically and offline metrics are unaffected |
| "No API key found" | No key file and no secrets | Create `API_KEY.env`, or add secrets (§2), then restart |
| Nothing happens on click | Empty prompt | Paste text into **Your prompt** first |
| Shortened text shows NOT SAFE TO USE | Level too aggressive for that method | Raise the level or switch to `sentence` |
| Experiment section says "No experiment results yet" | `results/runs.json` missing | `python main.py --run-all` |
| Chart tab says "not yet generated" | PNGs missing | `python main.py --report` |
| Judge column shows 0 | Judge pass not run | `python main.py --report` (needs quota) |
| Estimate ≠ live numbers | Estimate uses chars/4 and a fixed 200-token answer | Expected |
| `use_container_width` deprecation warnings | Streamlit version notice | Harmless, can be ignored |

---

## 15. Quick reference

```text
START        streamlit run app.py            -> http://localhost:8501
SHORTEN      paste prompt (+ optional document) -> ⚡ Check & Shorten Prompt
COMPARE      keep "Compare all methods" on   -> tabs + summary table
GO DEEP      expand "Sweep all retention levels" -> 7 levels at once
PROVE IT     tick "Live evaluation"          -> real tokens, cost, answer F1, judge
SELF-TEST    python main.py --smoke          -> is the key + pipeline healthy?
REBUILD      python main.py --run-all --resume -> sweep, charts, report
RE-READ      Website_Operation_Guide.docx    -> full technical manual with formulas
```
