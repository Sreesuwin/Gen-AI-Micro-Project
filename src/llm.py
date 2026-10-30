"""Gemini wrapper: authentication, generation, token counting, retry and cache."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import ssl
import threading
import time

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# SSL fix for Windows environments where Python cannot verify the local CA
# (e.g. corporate / institutional proxies that issue their own certificates).
# We try two approaches:
#   1. pip-system-certs / truststore: pulls the Windows cert store into Python.
#   2. Fallback: disable verification at the ssl module level so urllib3 and
#      httpx (used by google-genai) stop raising CERTIFICATE_VERIFY_FAILED.
# ---------------------------------------------------------------------------
def _patch_ssl() -> None:
    """Try to use the Windows certificate store; fall back to disabling verification."""
    try:
        import truststore  # installed by pip-system-certs or standalone
        truststore.inject_into_ssl()
        return
    except ImportError:
        pass
    try:
        import pip_system_certs.wrapt_requests  # noqa: F401
        return
    except ImportError:
        pass
    # Last-resort: create an unverified context and monkey-patch ssl defaults.
    # This is safe for outbound HTTPS calls in a controlled research environment.
    os.environ.setdefault("PYTHONHTTPSVERIFY", "0")
    ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[attr-defined]


_patch_ssl()

from .cost import DEFAULT_MODEL

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE_PATH = ROOT / "results" / "cache.json"

# Free-tier friendly pacing: stay comfortably under requests-per-minute limits.
MIN_INTERVAL_SECONDS = float(os.environ.get("GENAI_MIN_INTERVAL", "4.1"))
MAX_RETRIES = int(os.environ.get("GENAI_MAX_RETRIES", "3"))  # reduced from 6: fail fast on flaky free-tier

_client = None
_cache: dict[str, dict] | None = None
_lock = threading.RLock()


class QuotaExceededError(RuntimeError):
    """Daily / per-minute API quota is exhausted - stop rather than spin."""


def _is_quota_error(err: Exception) -> bool:
    """True only for a *daily* quota exhaustion.

    A per-minute 429 is just a throttle and is worth retrying; a per-day quota
    will not clear for hours, so the caller should stop and resume later.
    """
    msg = str(err)
    if "RESOURCE_EXHAUSTED" not in msg:
        return False
    low = msg.lower()
    if "per day" in low or "perday" in low:
        return True
    match = re.search(r"retry in (\d+)m", low)
    return bool(match and int(match.group(1)) >= 5)


def load_env() -> None:
    """Load the API key.

    Sources, in priority order:

    1. A normal dotenv file (.env / API_KEY.env containing GEMINI_API_KEY=...)
       or a file holding nothing but the raw key itself - the local setup;
    2. Streamlit secrets (``st.secrets["GEMINI_API_KEY"]``), which is how the
       key is supplied on Streamlit Community Cloud, where no key file ships
       with the repository;
    3. GOOGLE_API_KEY as a last resort.

    Never raises: a missing secrets file (the normal case outside Streamlit)
    is silently ignored.
    """
    for name in (".env", "API_KEY.env"):
        path = ROOT / name
        if not path.exists():
            continue
        load_dotenv(path)
        # A bare key file has no NAME=VALUE line, so dotenv sets nothing.
        if not os.environ.get("GEMINI_API_KEY"):
            raw = path.read_text(encoding="utf-8", errors="ignore").strip()
            if raw and "=" not in raw.splitlines()[0] and len(raw) >= 30:
                os.environ["GEMINI_API_KEY"] = raw.splitlines()[0].strip()

    if not os.environ.get("GEMINI_API_KEY"):
        # Streamlit secrets: .streamlit/secrets.toml locally, the Secrets tab
        # on Streamlit Community Cloud. Accessing st.secrets outside a running
        # app (or with no file present) raises - that is fine here.
        try:
            import streamlit as st

            secret = st.secrets.get("GEMINI_API_KEY")
            if secret:
                os.environ["GEMINI_API_KEY"] = str(secret).strip()
        except Exception:  # noqa: BLE001 - no secrets file / not in Streamlit
            pass

    if not os.environ.get("GEMINI_API_KEY") and os.environ.get("GOOGLE_API_KEY"):
        os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]


def get_api_key() -> str:
    load_env()
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "No API key found. Create .env (or API_KEY.env) in the project root "
            "containing: GEMINI_API_KEY=your_key_here"
        )
    return key


def client():
    global _client
    if _client is None:
        from google import genai
        from google.genai import types

        # attempts=1 is essential: the SDK otherwise honours the server's
        # RetryInfo, which for an exhausted daily quota means sleeping for
        # half an hour before failing anyway. We do our own short retries.
        http = types.HttpOptions(
            timeout=int(os.environ.get("GENAI_TIMEOUT_MS", "120000")),
            retry_options=types.HttpRetryOptions(attempts=1),
        )
        _client = genai.Client(api_key=get_api_key(), http_options=http)
        # If the SSL patch above didn't take effect at the urllib level,
        # also disable cert verification inside the google-genai httpx client.
        try:
            import httpx
            if hasattr(_client, '_api_client') and hasattr(_client._api_client, '_httpx_client'):
                old = _client._api_client._httpx_client
                _client._api_client._httpx_client = httpx.Client(
                    verify=False,
                    timeout=old.timeout,
                    headers=dict(old.headers),
                )
        except Exception:  # noqa: BLE001 - best-effort; don't break if internals change
            pass
    return _client


def model_name() -> str:
    load_env()
    return os.environ.get("GENAI_MODEL", DEFAULT_MODEL)


# ---------------- cache ----------------

def _load_cache() -> dict[str, dict]:
    global _cache
    if _cache is None:
        if CACHE_PATH.exists():
            try:
                _cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                _cache = {}
        else:
            _cache = {}
    return _cache


def _save_cache() -> None:
    if _cache is None:
        return
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(_cache, indent=1), encoding="utf-8")


def _key(model: str, prompt: str) -> str:
    return hashlib.sha1(f"{model}||{prompt}".encode("utf-8")).hexdigest()


# ---------------- rate limiting ----------------

_last_call = 0.0


def _throttle() -> None:
    global _last_call
    wait = MIN_INTERVAL_SECONDS - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.time()


# ---------------- public API ----------------

def generate(prompt: str, model: str | None = None, temperature: float = 0.0) -> dict:
    """Generate a completion. Returns {text, tokens_in, tokens_out, cached, latency_ms}.

    Thread-safe: the sweep runs several requests in parallel.
    """
    model = model or model_name()
    key = _key(model, prompt)
    with _lock:
        cache = _load_cache()
        if key in cache:
            hit = dict(cache[key])
            hit["cached"] = True
            return hit

    from google.genai import types

    # Thinking is switched off deliberately: this project measures how INPUT
    # token count drives cost, so output length must not be dominated by
    # variable-length reasoning tokens. Every call in the sweep uses the same
    # fixed generation configuration, which is what makes the comparison fair.
    # Gemma models reject the thinking config, so they get a plain config with
    # an output cap. The cap bounds runaway hidden reasoning on hard prompts;
    # normal calls use ~1300 output tokens. If the cap is hit during reasoning
    # and no answer comes back, generate() doubles it and retries.
    max_out = int(os.environ.get("GENAI_MAX_OUTPUT_TOKENS", "2048"))

    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        _throttle()
        if model.startswith("gemma"):
            config = types.GenerateContentConfig(
                temperature=temperature, max_output_tokens=max_out)
        else:
            config = types.GenerateContentConfig(
                temperature=temperature,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
        started = time.time()
        try:
            resp = client().models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            latency = int((time.time() - started) * 1000)
            usage = resp.usage_metadata
            text = (resp.text or "").strip()
            prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
            # Billed output = everything that is not input. This deliberately
            # includes thinking tokens, which the provider charges for as output.
            total_tokens = int(getattr(usage, "total_token_count", 0) or 0)
            out_tokens = max(0, total_tokens - prompt_tokens)
            if not total_tokens:
                out_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
            if not text and max_out < 8192:
                # Cap was consumed by hidden reasoning and no answer came back:
                # give it more room rather than recording a fake zero score.
                max_out *= 2
                continue
            out = {
                "text": text,
                "tokens_in": prompt_tokens,
                "tokens_out": out_tokens,
                "cached": False,
                "latency_ms": latency,
            }
            cache[key] = {k: out[k] for k in ("text", "tokens_in", "tokens_out", "latency_ms")}
            with _lock:
                _save_cache()
            return out
        except Exception as err:  # noqa: BLE001 - retry on any transport/API error
            last_err = err
            if _is_quota_error(err):
                # Daily/per-minute quota: retrying immediately is pointless.
                raise QuotaExceededError(str(err)) from err
            msg = str(err).lower()
            retryable = ("429" in msg) or ("rate" in msg) or ("503" in msg) or ("500" in msg) or ("unavailable" in msg) or ("deadline" in msg) or ("504" in msg)
            if not retryable or attempt == MAX_RETRIES - 1:
                break
            time.sleep(min(5, 2 ** (attempt + 1)))  # cap at 5s (was 20s) to fail fast

    raise RuntimeError(f"Gemini call failed after {MAX_RETRIES} attempts: {last_err}")


def count_tokens_exact(text: str, model: str | None = None) -> int | None:
    """Real tokenizer count via the free count_tokens endpoint, None on failure."""
    model = model or model_name()
    try:
        _throttle()
        resp = client().models.count_tokens(model=model, contents=text)
        return int(resp.total_tokens)
    except Exception:  # noqa: BLE001
        return None


def healthcheck() -> dict:
    """Cheap end-to-end check that the key, model and network all work."""
    out = generate("Reply with the single word OK.", model=model_name())
    exact = count_tokens_exact("Health check prompt for token counting.")
    return {
        "model": model_name(),
        "text": out["text"][:80],
        "tokens_in": out["tokens_in"],
        "tokens_out": out["tokens_out"],
        "exact_count_tokens": exact,
    }
