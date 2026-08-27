"""
nim_audit.py - NVIDIA NIM AI Confidence Auditor
Optimized for low-latency responses using 8B model + minimal prompts + in-memory cache.
"""
import os
import socket
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

NIM_API_KEY  = os.getenv("NVIDIA_NIM_API_KEY", "")
NIM_MODEL    = os.getenv("NIM_MODEL", "meta/llama-3.3-70b-instruct")
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

# In-memory cache: breed name → encyclopedia dict (survives the session)
_encyclopedia_cache: dict = {}

# Shared client (created once, reused every call)
_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=NIM_API_KEY, base_url=NIM_BASE_URL)
    return _client


def is_online() -> bool:
    """Fast 1-second TCP check — no DNS, no HTTP."""
    try:
        socket.setdefaulttimeout(1)
        socket.create_connection(("8.8.8.8", 53))
        return True
    except OSError:
        return False


def _guard() -> str | None:
    """Returns an error string if we cannot make API calls, else None."""
    if not NIM_API_KEY or "PASTE" in NIM_API_KEY:
        return "NO KEY"
    if not is_online():
        return "OFFLINE"
    return None


# ──────────────────────────────────────────────
# AUDIT CALL  (target: <2 s)
# ──────────────────────────────────────────────
def run_nim_audit(results: list) -> dict:
    err = _guard()
    if err == "NO KEY":
        return {"verdict": "NO KEY",  "reason": "NVIDIA_NIM_API_KEY not set in .env", "alert": "NO", "error": None}
    if err == "OFFLINE":
        return {"verdict": "OFFLINE", "reason": "No internet — NIM unavailable.",     "alert": "NO", "error": None}

    breed      = results[0]['breed'].replace('_', ' ')
    confidence = results[0]['probability']
    runner_up  = results[1]['breed'].replace('_', ' ') if len(results) > 1 else "N/A"
    runner_conf = results[1]['probability'] if len(results) > 1 else 0.0
    gap        = confidence - runner_conf

    prompt = f"""You are a livestock AI auditor. Evaluate this cattle breed classification result:

Primary: {breed} — {confidence:.1f}% confidence
Runner-up: {runner_up} — {runner_conf:.1f}% confidence
Confidence gap: {gap:.1f}%

Rules:
- TRUSTED if confidence >= 80% AND gap >= 20%
- REVIEW if confidence is between 45-79%, OR gap < 20%
- REJECT if confidence < 45%

Respond in exactly this format (3 lines only, no extra text):
VERDICT: TRUSTED
REASON: <one sentence, max 15 words explaining your verdict>
ALERT: NO"""

    try:
        resp = _get_client().chat.completions.create(
            model=NIM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=80,
            stop=["\n\n"],
        )
        raw = resp.choices[0].message.content.strip()
        parsed = {"verdict": "UNKNOWN", "reason": raw, "alert": "NO", "error": None}
        for line in raw.splitlines():
            if line.startswith("VERDICT:"):
                parsed["verdict"] = line.split(":", 1)[1].strip()
            elif line.startswith("REASON:"):
                parsed["reason"] = line.split(":", 1)[1].strip()
            elif line.startswith("ALERT:"):
                parsed["alert"] = line.split(":", 1)[1].strip()
        return parsed
    except Exception as e:
        return {"verdict": "ERROR", "reason": str(e), "alert": "NO", "error": str(e)}


# ──────────────────────────────────────────────
# ENCYCLOPEDIA CALL  (target: <3 s, cached)
# ──────────────────────────────────────────────
def get_breed_encyclopedia(breed: str) -> dict:
    # Cache hit — return instantly, no API call
    if breed in _encyclopedia_cache:
        return _encyclopedia_cache[breed]

    err = _guard()
    if err == "NO KEY":
        return {"error": "NVIDIA_NIM_API_KEY not configured in .env"}
    if err == "OFFLINE":
        return {"error": "Offline — connect to internet to load encyclopedia."}

    # Explicit multi-line format — required for 70B to follow correctly
    prompt = f"""You are an expert in Indian cattle breeds. Generate a factsheet for: "{breed}"

You MUST respond with exactly these 6 lines and nothing else. Each field on its own separate line:

ORIGIN: <state or country of origin only>
MILK_YIELD: <average daily litres, or 'Draught breed'>
HEAT_TOLERANCE: <LOW or MEDIUM or HIGH>
ECONOMIC_VALUE: <one short sentence on primary use>
PHYSICAL_TRAITS: <2 to 3 key features>
FUN_FACT: <one interesting fact>"""

    try:
        resp = _get_client().chat.completions.create(
            model=NIM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=160,
        )
        raw = resp.choices[0].message.content.strip()
        parsed: dict = {"error": None}
        for line in raw.splitlines():
            line = line.strip()
            if ":" in line:
                key, _, val = line.partition(":")
                key_clean = key.strip().lower().replace(" ", "_")
                val_clean = val.strip()
                if key_clean and val_clean:
                    parsed[key_clean] = val_clean
        _encyclopedia_cache[breed] = parsed
        return parsed
    except Exception as e:
        return {"error": str(e)}
