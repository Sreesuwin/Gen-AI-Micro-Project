"""Quality metrics: Token-F1 (free, offline) and LLM-as-judge (Gemini, 1-5)."""

from __future__ import annotations

import json
import re
from collections import Counter

from . import llm
from .compressor import STOPWORDS, split_sentences

_WORD_RE = re.compile(r"[a-z0-9]+")

# Interrogatives / auxiliaries: never treated as content key terms.
QUESTION_WORDS = {
    "what", "when", "where", "which", "who", "whom", "whose", "how", "why",
    "is", "are", "was", "were", "do", "does", "did", "can", "could", "will",
    "would", "shall", "should", "may", "might", "must", "have", "has", "had",
}

# Explicit 1-5 rubric so the judge is consistent across runs.
JUDGE_RUBRIC = """You are a strict evaluator. Score the CANDIDATE answer against the REFERENCE answer.

Score 5: fully correct and complete - every fact in the reference is present, nothing contradicted.
Score 4: mostly correct - at least one reference detail missing, no factual errors.
Score 3: partially correct - the main point is right but several details are missing or vague.
Score 2: largely incorrect - only a minor overlap with the reference.
Score 1: wrong, empty, or does not answer the question.

Reply with ONLY a JSON object of the form {"score": <integer 1-5>, "reason": "<one short sentence>"}."""


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def token_f1(prediction: str, reference: str) -> dict:
    """Token-level precision / recall / F1 against the reference answer."""
    pred = _tokens(prediction)
    ref = _tokens(reference)
    if not pred or not ref:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    ref_counts: dict[str, int] = {}
    for t in ref:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    overlap = 0
    for t in pred:
        if ref_counts.get(t, 0) > 0:
            ref_counts[t] -= 1
            overlap += 1
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


# ---------------------------------------------------------------------------
# Offline evaluation metrics used by the interactive playground.
# All of them are pure text statistics: no API call, no cost, instant.
# ---------------------------------------------------------------------------

def key_terms(text: str, limit: int = 15) -> list[str]:
    """Important words of `text`, in first-appearance order.

    Stopwords, question words and 1-2 character tokens are dropped, duplicates
    removed. When a relevance hint / question is available these are the words
    a shortened prompt must not lose.
    """
    out: list[str] = []
    for w in _tokens(text):
        if w in STOPWORDS or w in QUESTION_WORDS or len(w) <= 2:
            continue
        if w not in out:
            out.append(w)
        if len(out) >= limit:
            break
    return out


def keyword_coverage(topic: str, compressed: str, reference: str | None = None) -> dict:
    """Fraction of the topic's key words that survived compression.

    When `reference` (the original, uncompressed text) is given, only key terms
    that actually occur in it are scored: a word the original never contained
    cannot be blamed on the compression, so it is reported separately as
    ``absent`` instead of being counted as lost.

    Returns {terms, covered, missing, absent, coverage}.
    """
    terms = key_terms(topic)
    have = set(_tokens(compressed))
    if reference is None:
        reference_terms = set(terms)
    else:
        reference_terms = set(_tokens(reference))
    scored = [t for t in terms if t in reference_terms]
    absent = [t for t in terms if t not in reference_terms]
    covered = [t for t in scored if t in have]
    missing = [t for t in scored if t not in have]
    coverage = len(covered) / len(scored) if scored else 1.0
    return {
        "terms": scored,
        "covered": covered,
        "missing": missing,
        "absent": absent,
        "coverage": round(coverage, 4),
    }


def information_retention(original: str, compressed: str, idf: dict | None = None) -> float:
    """IDF-weighted share of the original's information content that survived.

    Every token of the original carries its inverse-document-frequency weight
    (rare words weigh more than common ones). The score is the weighted mass of
    the tokens that are still present in the compressed text, divided by the
    total weighted mass. 1.0 = nothing of value lost, 0.0 = everything lost.
    """
    idf = idf or {}
    remaining = Counter(_tokens(compressed))
    total = 0.0
    kept = 0.0
    for tok in _tokens(original):
        w = float(idf.get(tok, 1.0))
        total += w
        if remaining.get(tok, 0) > 0:
            remaining[tok] -= 1
            kept += w
    if total == 0.0:
        return 0.0
    return round(kept / total, 4)


def sentence_stats(original: str, compressed: str) -> dict:
    """How many of the original sentences are still present, word for word."""
    orig_sentences = [s for s in split_sentences(original) if s]
    if not orig_sentences:
        return {"original": 0, "kept": 0, "retention": 1.0}
    kept = sum(1 for s in orig_sentences if s in compressed)
    return {
        "original": len(orig_sentences),
        "kept": kept,
        "retention": round(kept / len(orig_sentences), 4),
    }


def verdict(retention: float, content_f1: float, kw_coverage: float) -> dict:
    """PASS / WARN / FAIL judgement for one compressed prompt.

    Parameters
        retention   compressed tokens / original tokens (1.0 = unchanged)
        content_f1  token-level F1 of the shortened text against the original
        kw_coverage fraction of the key terms still present (0-1)

    Rules (in order)
        retention >= 0.999                       -> neutral, nothing was cut
        kw_coverage < 0.50 or content_f1 < 0.60  -> fail, too much lost
        kw_coverage < 1.00 or content_f1 < 0.85  -> warn, usable with care
        otherwise                                -> pass, safe to use
    """
    if retention >= 0.999:
        return {
            "level": "neutral",
            "label": "NO COMPRESSION",
            "text": "The text was not shortened, so nothing can be lost.",
        }
    if kw_coverage < 0.50 or content_f1 < 0.60:
        return {
            "level": "fail",
            "label": "NOT SAFE TO USE",
            "text": (f"Only {kw_coverage:.0%} of the key terms and "
                     f"{content_f1:.0%} of the wording survived - the prompt would "
                     "likely lose the answer. Raise the retention level or try "
                     "another method."),
        }
    if kw_coverage < 1.00 or content_f1 < 0.85:
        missing = 1 - kw_coverage
        return {
            "level": "warn",
            "label": "USABLE WITH CARE",
            "text": (f"{content_f1:.0%} of the wording survived and "
                     f"{missing:.0%} of the key terms were dropped - check the "
                     "shortened text before sending it."),
        }
    return {
        "level": "pass",
        "label": "SAFE TO USE",
        "text": (f"All key terms kept and {content_f1:.0%} of the original wording "
                 "retained - the shortened prompt is a faithful stand-in."),
    }


def judge(question: str, answer: str, reference: str, model: str | None = None) -> dict:
    """Score an answer 1-5 with an LLM judge. Returns {score, reason}."""
    model = model or llm.model_name()
    prompt = (
        f"{JUDGE_RUBRIC}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE ANSWER:\n{reference}\n\n"
        f"CANDIDATE ANSWER:\n{answer}\n"
    )
    raw = llm.generate(prompt, model=model)["text"]
    score, reason = _parse_judge(raw)
    return {"score": score, "reason": reason}


def _parse_judge(raw: str) -> tuple[int, str]:
    match = re.search(r"\{.*\}", raw, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            score = int(data.get("score", 0))
            reason = str(data.get("reason", ""))[:200]
            if 1 <= score <= 5:
                return score, reason
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    nums = re.findall(r"[1-5]", raw)
    if nums:
        return int(nums[0]), raw[:120]
    return 0, raw[:120]
