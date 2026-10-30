"""Diagnostic: find which free-tier models are usable and their daily quota.

Temporary script - not part of the project deliverable.
"""

import re
import time

from google import genai
from google.genai import types

from src import llm

http = types.HttpOptions(timeout=20000, retry_options=types.HttpRetryOptions(attempts=1))
client = genai.Client(api_key=llm.get_api_key(), http_options=http)
cfg = types.GenerateContentConfig(temperature=0)

CANDIDATES = [
    "gemma-4-26b-a4b-it",
    "gemma-4-31b-it",
    "gemini-flash-latest",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.1-flash-lite-preview",
    "gemini-3.1-pro-preview",
]


def out(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


alive = []
for m in CANDIDATES:
    try:
        r = client.models.generate_content(model=m, contents="Reply: OK", config=cfg)
        out(f"{m} OK total={r.usage_metadata.total_token_count}")
        alive.append(m)
    except Exception as e:  # noqa: BLE001
        out(f"{m} ERR {str(e)[:90]!r}")
    time.sleep(1.5)

out(f"ALIVE={alive}")

for m in alive:
    ok = 0
    hit = None
    for _ in range(60):
        try:
            client.models.generate_content(model=m, contents="Reply: OK", config=cfg)
            ok += 1
        except Exception as e:  # noqa: BLE001
            s = str(e)
            if "429" in s or "RESOURCE_EXHAUSTED" in s:
                qv = re.search(r"quotaValue[^0-9]*(\d+)", s)
                hit = qv.group(1) if qv else "unknown"
                break
        time.sleep(1.0)
    out(f"QUOTA {m}: ok_before_stop={ok} quotaValue={hit}")
