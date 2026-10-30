"""Prompt compression methods written from scratch.

Three methods:
    baseline - original prompt, untouched (ratio 1.0, the control group)
    lexical  - token-level scoring (LLMLingua style): keep numbers, proper nouns,
               question terms and rare / high-IDF words, drop stopwords and
               low-information tokens
    sentence - TF-IDF sentence scoring against the question (Liskavets style
               context-aware sentence selection); top ranked sentences are kept
               and restored to their original order

Compression levels are retention ratios: the compressed prompt should be about
`ratio` times the size of the original.
"""

from __future__ import annotations

import math
import re
from collections import Counter

LEVELS = [1.0, 0.9, 0.75, 0.6, 0.45, 0.3, 0.15]
METHODS = ["baseline", "lexical", "sentence"]
COMPRESSIBLE_METHODS = ["lexical", "sentence"]

# Rough tokenizer-independent size model: ~4 characters per token for English.
CHARS_PER_TOKEN = 4.0

_WORD_RE = re.compile(r"[A-Za-z0-9_]+|[^A-Za-z0-9_\s]")

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "of", "to",
    "in", "on", "at", "by", "for", "with", "from", "as", "is", "are", "was",
    "were", "be", "been", "being", "am", "it", "its", "this", "that", "these",
    "those", "there", "here", "we", "you", "your", "they", "their", "he", "she",
    "his", "her", "them", "us", "our", "i", "me", "my", "do", "does", "did",
    "not", "no", "so", "than", "too", "very", "can", "will", "just", "into",
    "over", "under", "up", "down", "out", "off", "all", "any", "each", "such",
    "only", "own", "same", "s", "t", "don", "should", "now", "also", "may",
    "must", "shall", "would", "could", "about", "above", "after", "again",
    "between", "both", "during", "before", "because", "until", "while",
}

# High-frequency English words: capped low so compression prefers content words.
_COMMON = {
    "the", "of", "and", "to", "in", "a", "is", "that", "for", "it", "with",
    "as", "was", "on", "be", "at", "by", "this", "are", "from", "or", "an",
    "not", "but", "if", "we", "you", "all", "can", "had", "has", "have",
    "will", "would", "there", "their", "which", "one", "been", "were", "when",
    "who", "his", "her", "she", "they", "them", "these", "so", "some", "may",
    "each", "than", "then", "into", "any", "may", "must", "should", "also",
    "such", "only", "other", "more", "most", "after", "before", "between",
    "during", "about", "while", "where", "what", "how", "why", "its", "our",
    "your", "out", "up", "down", "over", "under", "again", "once", "very",
    "just", "because", "until", "while", "same", "own", "too", "now",
}


# Public handles so other modules (metrics, the dashboard) can reuse them.
STOPWORDS = _STOPWORDS


def split_sentences(text: str) -> list[str]:
    """Sentence splitter, exposed so the metrics module can count them."""
    return _split_sentences(text)


def estimate_tokens(text: str) -> int:
    """Tokenizer-independent token estimate (about 4 characters per token)."""
    if not text:
        return 0
    return max(1, round(len(text) / CHARS_PER_TOKEN))


def build_idf(documents: list[str]) -> dict[str, float]:
    """Smooth inverse document frequency over the dataset corpus.

    A word appearing in many documents is uninformative for one particular
    prompt, so it gets a low idf and is dropped first by the lexical method.
    """
    n = max(1, len(documents))
    df: Counter[str] = Counter()
    for doc in documents:
        words = {m.group(0).lower() for m in _WORD_RE.finditer(doc) if m.group(0).isalnum()}
        for w in words:
            df[w] += 1
    return {w: math.log((n + 1) / (c + 0.5)) + 1.0 for w, c in df.items()}


def _default_idf(text: str) -> dict[str, float]:
    return {w: 1.0 for w in {m.group(0).lower() for m in _WORD_RE.finditer(text)}}


def _score_word(word: str, idf: dict[str, float], question_words: set[str], max_idf: float) -> float:
    """Score one word between 0 (drop) and 1 (keep at all costs)."""
    raw = word
    w = word.lower()

    if w in question_words:
        return 1.0
    if any(ch.isdigit() for ch in raw):
        return 1.0
    if len(raw) > 1 and raw.isupper():
        return 0.95
    if w in _STOPWORDS:
        return 0.05
    if raw[0].isupper() and len(raw) > 1:
        base = 0.85
    else:
        base = 0.45

    idf_val = idf.get(w, max_idf)
    idf_norm = idf_val / max_idf if max_idf else 0.5
    score = base + 0.35 * idf_norm

    if len(w) >= 8:
        score += 0.10
    if len(w) <= 2:
        score -= 0.15
    if w in _COMMON:
        score = min(score, 0.30)

    return max(0.0, min(1.0, score))


def _compress_lexical(text: str, ratio: float, idf: dict[str, float] | None, question: str) -> str:
    if ratio >= 1.0:
        return text

    idf = idf or _default_idf(text)
    max_idf = max(idf.values()) if idf else 1.0
    question_words = {m.group(0).lower() for m in _WORD_RE.finditer(question) if m.group(0).isalnum()}

    lines = text.split("\n")
    budget_chars = max(20, int(len(text) * ratio))

    # Score every word token once, globally, then take the best until the budget.
    candidates: list[tuple[float, int, int, str]] = []
    for li, line in enumerate(lines):
        pos = 0
        for m in _WORD_RE.finditer(line):
            tok = m.group(0)
            if tok.isalnum():
                score = _score_word(tok, idf, question_words, max_idf)
                candidates.append((score, li, pos, tok))
            pos += 1

    candidates.sort(key=lambda t: (-t[0], t[1], t[2]))

    kept: set[tuple[int, int]] = set()
    used = 0
    for score, li, pos, tok in candidates:
        cost = len(tok) + 1
        if used + cost > budget_chars:
            continue
        kept.add((li, pos))
        used += cost

    # Rebuild, keeping punctuation only where it still frames kept words.
    out_lines: list[str] = []
    for li, line in enumerate(lines):
        parts: list[str] = []
        prev_word_kept = False
        pos = 0
        for m in _WORD_RE.finditer(line):
            tok = m.group(0)
            if tok.isalnum():
                if (li, pos) in kept:
                    parts.append(tok)
                    prev_word_kept = True
                else:
                    prev_word_kept = False
            else:
                # sentence enders / separators survive only between kept words
                if prev_word_kept and tok in ".;:!?)-":
                    parts.append(tok)
                elif prev_word_kept and tok in ",(":
                    continue
                else:
                    prev_word_kept = False
            pos += 1
        joined = " ".join(parts).strip()
        if joined:
            out_lines.append(joined)

    result = "\n".join(out_lines)
    return result if result else text[: max(40, budget_chars)]


def _split_sentences(text: str) -> list[str]:
    lines = text.split("\n")
    sentences: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        chunks = re.split(r"(?<=[.!?])\s+", line)
        sentences.extend(c for c in chunks if c)
    return sentences


def _tfidf_vec(sentence: str, idf: dict[str, float], max_idf: float) -> dict[str, float]:
    words = [m.group(0).lower() for m in _WORD_RE.finditer(sentence) if m.group(0).isalnum()]
    if not words:
        return {}
    counts = Counter(words)
    total = sum(counts.values())
    vec = {}
    for w, c in counts.items():
        vec[w] = (c / total) * idf.get(w, max_idf)
    return vec


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[k] * b[k] for k in a if k in b)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _compress_sentence(text: str, ratio: float, idf: dict[str, float] | None, question: str) -> str:
    if ratio >= 1.0:
        return text

    idf = idf or _default_idf(text)
    max_idf = max(idf.values()) if idf else 1.0
    sentences = _split_sentences(text)
    if len(sentences) <= 2:
        return text

    q_vec = _tfidf_vec(question, idf, max_idf)
    scored = []
    for idx, s in enumerate(sentences):
        # short ALL-CAPS / heading lines are structural, always worth keeping
        is_heading = len(s) < 60 and (s.isupper() or (s.endswith(":") and len(s.split()) <= 6))
        score = _cosine(_tfidf_vec(s, idf, max_idf), q_vec)
        if is_heading:
            score = 1.1
        # penalise very long low-signal sentences slightly
        scored.append((score, idx, s))

    budget_chars = max(20, int(len(text) * ratio))
    order = sorted(range(len(scored)), key=lambda i: (-scored[i][0], scored[i][1]))
    chosen: set[int] = set()
    used = 0
    for i in order:
        s = scored[i][2]
        if used + len(s) + 1 > budget_chars:
            continue
        chosen.add(i)
        used += len(s) + 1

    kept_sentences = [scored[i][2] for i in sorted(chosen)]
    result = " ".join(kept_sentences)
    return result if result else text[: max(40, budget_chars)]


def compress(
    text: str,
    method: str,
    ratio: float,
    idf: dict[str, float] | None = None,
    question: str = "",
) -> str:
    """Compress `text` using `method` down to approximately `ratio` of its size."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if method == "baseline" or ratio >= 1.0:
        return text
    if method == "lexical":
        return _compress_lexical(text, ratio, idf, question)
    return _compress_sentence(text, ratio, idf, question)


def build_prompt(item: dict, compressed_context: str) -> str:
    """Assemble the final prompt: compressed context + the (never compressed) question.

    Two instructions are part of the fixed prompt that EVERY condition receives:

    * the answer-length cap keeps output token counts comparable across
      compression levels, so the cost difference we measure is driven by input;
    * the "not stated" escape hatch stops the model from spinning when a heavily
      compressed context really does lack the answer - without it the model
      loops in its hidden reasoning until the request times out.
    """
    return (
        f"{compressed_context}\n\n"
        f"QUESTION: {item['question']}\n\n"
        f"If the answer is not stated in the context, reply exactly: "
        f"not stated in context.\n\n"
        f"Answer in at most 60 words, covering every detail needed to answer "
        f"the question correctly."
    )
