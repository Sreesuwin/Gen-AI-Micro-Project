# Smart Prompt Optimization for Reducing LLM API Cost

Generative AI micro project (7CS 1018) - Alliance School of Advanced Computing.

Long prompts cost money: every extra input token is charged, adds latency and
consumes context. This project compresses a long prompt at several levels,
measures what that does to **tokens, API cost and answer quality**, and then
picks the cheapest compression level that still clears a minimum quality bar.

## What it does

1. **Compresses** a long prompt two different ways, at 7 retention levels
   (1.00 down to 0.15):
   - `baseline` - the untouched prompt (the control group)
   - `lexical` - token-level scoring (LLMLingua style): keeps numbers, proper
     nouns, question terms and rare/high-IDF words, drops the rest
   - `sentence` - TF-IDF sentence scoring against the question (Liskavets
     style context-aware selection), top sentences kept in original order
2. **Measures** every run: input tokens, output tokens, latency, simulated API
   cost (real token counts x published list price) and Token-F1 against a
   reference answer.
3. **Judges** the promising configurations with an LLM-as-judge on a 1-5 rubric.
4. **Optimises**: cheapest (method, level) with mean judge score >= 4.0 and
   mean F1 >= 90% of the uncompressed baseline.
5. **Reports**: CSV, JSON summary, three charts and a Markdown report, plus a
   Streamlit dashboard.

## Setup

```bash
pip install -r requirements.txt
```

Create a file called `.env` (or `API_KEY.env`) in the project root containing
your Google AI Studio key, either as a dotenv variable or as the bare key:

```
GEMINI_API_KEY=your_key_here
```

## Usage

```bash
# 1. quick end-to-end check (key + 3 dataset items)
python main.py --smoke

# 2. full experiment: sweep, judging, optimisation, charts, report
python main.py --run-all

# 3. rebuild summary/charts/report from stored runs (no API calls)
python main.py --report

# 4. dashboard
streamlit run app.py
```

Useful flags:

| flag | meaning |
|---|---|
| `--resume` | keep previously stored runs and only fill in what is missing |
| `--no-judge` | skip the LLM-judge pass |
| `--min-judge 4.0` | minimum acceptable judge score (default 4.0) |
| `--f1-fraction 0.90` | minimum F1 as a fraction of baseline (default 0.90) |
| `--max-judged-levels 4` | how many levels per method get judged |

Environment variables: `GENAI_MODEL` (default `gemma-4-26b-a4b-it`),
`GENAI_WORKERS` (default 3), `GENAI_TIMEOUT_MS` (default 120000),
`GENAI_MIN_INTERVAL` (default 4.1), `GENAI_MAX_OUTPUT_TOKENS` (default 2048).

## Files

```
main.py                  CLI: --smoke / --run-all / --report
app.py                   Streamlit dashboard
src/compressor.py        the compression algorithms (written from scratch)
src/llm.py               Gemini wrapper: auth, cache, throttle, retries, quota
src/cost.py              pricing table and cost model
src/metrics.py           Token-F1 and LLM-as-judge scoring
src/optimizer.py         aggregation + constrained selection
src/dataset.py           dataset loading
data/dataset.json        13 long-prompt QA items (900-1700 tokens each)
results/                 cache.json, runs.json, results.csv, summary.json,
                         report.md, *.png
Project_Implementation_Plan.docx   the plan this implementation follows
```

## Notes and honest limitations

- **Cost is simulated**: real token counts multiplied by published per-million
  prices (see `src/cost.py`), not an actual invoice.
- **Model choice**: the free tier allows only ~20 requests/day on the Gemini
  Flash models, so the sweep runs on `gemma-4-26b-a4b-it`, which is served
  free of charge with a far larger allowance. Every condition uses the same
  model, which is what the comparison requires.
- **Hidden reasoning**: Gemma charges for reasoning tokens it never shows, so
  output cost varies between runs. The prompt fixes a 60-word answer cap so the
  comparison stays input-driven, and both input and output cost are reported
  separately.
- The LLM judge belongs to the same model family as the generator, so Token-F1
  is reported alongside it as an independent check.
- The dataset has 13 hand-written items; results are indicative, not
  statistically powered.
