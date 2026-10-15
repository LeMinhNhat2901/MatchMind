"""
Unified LLM client for MatchMind.

Default provider is **Gemini** (google-genai). Anthropic and OpenAI are kept as
optional fallbacks — selected by ``settings.llm_provider``.

Every caller uses ``call_llm(...)`` and gets back ``(text, total_tokens)``.
"""
from __future__ import annotations

import base64
import logging
import os
import re
import time
from datetime import date
from pathlib import Path

from matchmind.config import settings

logger = logging.getLogger(__name__)

# google-genai logs "AFC is enabled / AFC remote call N is done" at INFO on the
# root logger for every request — quieten it.
for _n in ("google_genai", "google.genai", "google_genai.models"):
    logging.getLogger(_n).setLevel(logging.WARNING)

_PITCH_IMAGE_HINT = (
    "The attached image shows the current match situation on the pitch "
    "(red = home, blue = away, white = ball; the highlighted player is the focus). "
    "It is supporting context only — the numbers in the text are the primary evidence.\n\n"
)


def call_llm(
    prompt: str,
    *,
    image_path: str | None = None,
    max_tokens: int = 1000,
    temperature: float | None = None,
    json_mode: bool = False,
    thinking_budget: int | None = None,
    model: str | None = None,
    provider: str | None = None,
) -> tuple[str, int]:
    """Call the configured LLM provider.

    Args:
        prompt: user text.
        image_path: optional PNG to send as a multimodal part.
        max_tokens: output token cap.
        temperature: sampling temperature (None = provider default).
        json_mode: ask the provider to emit application/json (Gemini/OpenAI).
        thinking_budget: Gemini 2.5 only — 0 disables "thinking" for cheap calls.
        model: override settings.llm_model.
        provider: override settings.llm_provider (pass `model` too — the
            default model name belongs to the default provider).

    Returns:
        (text, total_tokens)

    With LLM_FALLBACK_CHAIN set (and no explicit provider/model), providers are
    tried in order: see _call_with_failover.
    """
    kwargs = dict(image_path=image_path, max_tokens=max_tokens, temperature=temperature,
                  json_mode=json_mode, thinking_budget=thinking_budget)
    if provider is None and model is None and settings.llm_fallback_chain.strip():
        return _call_with_failover(prompt, **kwargs)
    provider = (provider or settings.llm_provider or "gemini").lower()
    return _call_provider(provider, model or settings.llm_model, prompt, **kwargs)


# ── Provider failover ────────────────────────────────────────────────────────
#
# Free tiers run out (Groq: 200k tokens/day) or go down (Gemini 503 "high
# demand") mid-session — both happened on 2026-09-28 and blocked the chat.
# LLM_FALLBACK_CHAIN lists "provider:model" entries tried in order. A daily
# quota error takes that provider out until tomorrow (no pointless retry on
# every later call); any other error just moves this one call to the next entry.

# OpenAI-compatible endpoints: (base_url, settings attribute holding the key).
_OPENAI_COMPATIBLE: dict[str, tuple[str, str]] = {
    "cerebras": ("https://api.cerebras.ai/v1", "cerebras_api_key"),
    "mistral": ("https://api.mistral.ai/v1", "mistral_api_key"),
    "openrouter": ("https://openrouter.ai/api/v1", "openrouter_api_key"),
}
_ACCEPTS_REASONING_EFFORT = {"cerebras"}
_KEY_ATTR: dict[str, str] = {
    "gemini": "gemini_api_key", "groq": "groq_api_key", "openai": "openai_api_key",
    "anthropic": "anthropic_api_key", **{p: attr for p, (_, attr) in _OPENAI_COMPATIBLE.items()},
}
_exhausted: dict[str, date] = {}          # provider -> day its daily quota ran out
_last_provider: str | None = None


def parse_fallback_chain(spec: str) -> list[tuple[str, str]]:
    """"groq:qwen/qwen3.8-27b, cerebras:qwen-3.8-27b" -> [(provider, model), ...].
    Splits on the FIRST colon only, so OpenRouter's "model:free" names survive."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if part and ":" in part:
            provider, model = part.split(":", 1)
            out.append((provider.strip().lower(), model.strip()))
    return out


def is_daily_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(s in msg for s in ("tokens per day", "requests per day", "per day (", "(tpd)", "(rpd)", "daily limit"))


def last_provider() -> str | None:
    """Which provider answered the most recent call_llm (for traces/logs)."""
    return _last_provider


def _has_key(provider: str) -> bool:
    attr = _KEY_ATTR.get(provider)
    return bool(attr and getattr(settings, attr, ""))


def _call_with_failover(prompt: str, **kwargs) -> tuple[str, int]:
    global _last_provider
    today = date.today()
    failures: list[str] = []
    for provider, model in parse_fallback_chain(settings.llm_fallback_chain):
        if _exhausted.get(provider) == today:
            failures.append(f"{provider}: daily quota used up")
            continue
        if not _has_key(provider):
            failures.append(f"{provider}: no API key in .env")
            continue
        try:
            result = _call_provider(provider, model, prompt, **kwargs)
        except Exception as exc:  # noqa: BLE001 — any provider error moves on to the next one
            if is_daily_quota_error(exc):
                _exhausted[provider] = today
                logger.warning(f"[llm] {provider} daily quota exhausted — skipping it until tomorrow")
            else:
                logger.warning(f"[llm] {provider} failed ({str(exc)[:120]}) — trying next provider")
            failures.append(f"{provider}: {str(exc)[:160]}")
            continue
        _last_provider = provider
        return result
    raise RuntimeError("All LLM providers failed: " + " | ".join(failures))


def _call_provider(provider: str, model: str, prompt: str, *, image_path=None, max_tokens=1000,
                   temperature=None, json_mode=False, thinking_budget=None) -> tuple[str, int]:
    global _last_provider
    if provider == "gemini":
        result = _call_gemini(prompt, image_path, max_tokens, temperature, json_mode, thinking_budget, model)
    elif provider == "anthropic":
        result = _call_anthropic(prompt, image_path, max_tokens, temperature, model)
    elif provider == "openai":
        result = _call_openai(prompt, image_path, max_tokens, temperature, json_mode, model)
    elif provider == "groq":
        result = _call_groq(prompt, image_path, max_tokens, temperature, json_mode, model)
    elif provider in _OPENAI_COMPATIBLE:
        result = _call_openai_compatible(provider, prompt, max_tokens, temperature, json_mode, model,
                                         thinking_budget=thinking_budget)
    else:
        raise ValueError(
            f"Unknown LLM provider {provider!r}. Use one of: gemini | anthropic | openai | groq | "
            + " | ".join(_OPENAI_COMPATIBLE)
        )
    _last_provider = provider
    return result


def _openai_client(base_url: str, api_key: str, **kwargs):
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key, **kwargs)


def _call_openai_compatible(
    provider: str, prompt: str, max_tokens: int, temperature: float | None, json_mode: bool, model: str,
    thinking_budget: int | None = None,
) -> tuple[str, int]:
    """Cerebras / Mistral / OpenRouter — all speak the OpenAI chat API. Text only:
    the pitch image is supporting context; the numbers in the prompt are primary.

    thinking_budget=0 (the project's "cheap call, no thinking" signal) maps to
    reasoning_effort="none" on providers that accept it: qwen-3.8-27b on Cerebras
    otherwise spends hidden reasoning tokens INSIDE max_tokens (measured: 76 on a
    one-line JSON request), so small calls come back empty.
    """
    base_url, key_attr = _OPENAI_COMPATIBLE[provider]
    api_key = getattr(settings, key_attr, "")
    if not api_key:
        raise ValueError(f"{key_attr.upper()} not set in .env")
    client = _openai_client(base_url=base_url, api_key=api_key, timeout=60.0, max_retries=0)
    kwargs: dict = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if thinking_budget == 0 and provider in _ACCEPTS_REASONING_EFFORT:
        kwargs["reasoning_effort"] = "none"
    resp = client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content or ""
    if not text.strip():
        # An empty completion (e.g. the budget went to hidden reasoning) is a
        # failure, not an answer — raising lets the failover chain move on.
        raise RuntimeError(f"{provider} returned an empty completion (max_tokens={max_tokens})")
    tokens = int(resp.usage.total_tokens) if resp.usage else 0
    return text, tokens


def call_narrator_llm(prompt: str, **kwargs) -> tuple[str, int]:
    """call_llm for the user-facing narration/translation step.

    Uses NARRATOR_LLM_PROVIDER/NARRATOR_LLM_MODEL when set, and retries once on
    the default provider if that override fails (e.g. a free-tier daily quota),
    so configuring a better-but-scarcer model can never break the chat.
    """
    override_provider = settings.narrator_llm_provider.strip()
    override_model = settings.narrator_llm_model.strip()
    if override_provider or override_model:
        try:
            return call_llm(
                prompt,
                provider=override_provider or None,
                model=override_model or None,
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001 — fall back to the default provider below
            logger.warning(f"Narrator LLM override failed, using default provider: {exc}")
    return call_llm(prompt, **kwargs)


def _image_bytes(image_path: str | None) -> bytes | None:
    if image_path and Path(image_path).exists():
        return Path(image_path).read_bytes()
    return None


def _is_rate_limit_error(exc: Exception) -> bool:
    if getattr(exc, "status_code", None) == 429:
        return True
    msg = str(exc)
    return "429" in msg or "rate_limit_exceeded" in msg or "Request too large" in msg


def _parse_groq_token_limit(msg: str) -> int | None:
    """Parse "Limit 1000" out of a Groq OTPM rate-limit error message."""
    m = re.search(r"[Ll]imit[:\s]+(\d+)", msg)
    return int(m.group(1)) if m else None


def _parse_retry_seconds(msg: str, default: float = 5.0) -> float:
    """Parse a "try again in Xs" / "retry after Xs" hint, else a short default."""
    m = re.search(r"(?:try again in|retry(?:\s+after)?)\s*([\d.]+)\s*s", msg, re.IGNORECASE)
    return float(m.group(1)) if m else default


# ── Groq (fastest / cheapest) ───────────────────────────────────────────────

def _call_groq(
    prompt: str,
    image_path: str | None,
    max_tokens: int,
    temperature: float | None,
    json_mode: bool,
    model: str,
) -> tuple[str, int]:
    """
    Call Groq API.

    Free/on-demand Groq tiers enforce a per-model, per-minute OUTPUT token cap
    (OTPM) that can be far below a Gemini-sized max_tokens (e.g. 1000/min for
    some models) — a request asking for more is rejected outright (429), not
    throttled. This adapts instead of hardcoding one model's limit: on a 429
    it parses the model's actual "Limit N" from the error and retries once
    with max_tokens clamped to that, then falls back to a short backoff+retry
    for ordinary rate limiting.

    Text models (default): llama-3.3-70b-versatile, mixtral-8x7b-32768,
                           llama-3.1-70b-versatile, gemma2-9b-it
    Vision model:          llama-3.2-90b-vision-preview (used when image_path set)
    """
    try:
        from groq import Groq
    except ImportError as exc:
        raise ImportError("groq not installed. Run: pip install groq") from exc

    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY not set in .env")

    client = Groq(api_key=settings.groq_api_key)

    # Groq vision models are decommissioned or unavailable — always use text-only.
    # The pitch image info is embedded as context in the prompt by the reasoner node.
    if image_path and Path(image_path).exists():
        logger.info("[groq] image_path provided but using text-only (no Groq vision model available)")
        # Prepend a note so the LLM knows image context was intended
        text_prompt = (
            "[PITCH IMAGE: A tactical diagram was generated. Key spatial data is encoded "
            "in the statistics below — use those numbers as primary evidence.]\n\n"
            + prompt
        )
    else:
        text_prompt = prompt

    # Always string content for Groq
    content = str(text_prompt)

    # Auto-select model — use groq_model setting, reject non-Groq model names
    _groq_prefixes = ("llama", "mixtral", "gemma", "whisper", "qwen", "groq/", "openai/gpt-oss", "meta-", "allam", "canopy")
    if model and any(model.lower().startswith(p) for p in _groq_prefixes):
        use_model = model
    else:
        use_model = settings.groq_model

    resp_fmt = {"type": "json_object"} if json_mode else None

    kwargs: dict = {
        "model": use_model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if resp_fmt:
        kwargs["response_format"] = resp_fmt

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            resp = client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""
            tokens = int(resp.usage.total_tokens) if resp.usage else 0
            logger.debug(f"[groq] model={use_model} tokens={tokens} max_tokens={kwargs['max_tokens']}")
            return text, tokens
        except Exception as exc:
            # A daily quota won't recover within any sensible retry; fail fast so
            # the failover chain (or the caller) moves on.
            if not _is_rate_limit_error(exc) or is_daily_quota_error(exc) or attempt == max_attempts - 1:
                raise
            msg = str(exc)
            limit = _parse_groq_token_limit(msg)
            if limit is not None and kwargs["max_tokens"] > limit:
                new_cap = max(256, limit - 50)  # small safety margin
                logger.warning(
                    f"[groq] OTPM limit for {use_model} is {limit}; "
                    f"reducing max_tokens {kwargs['max_tokens']} -> {new_cap} and retrying"
                )
                kwargs["max_tokens"] = new_cap
            else:
                wait = min(_parse_retry_seconds(msg), 20.0)
                logger.warning(f"[groq] rate limited; waiting {wait:.1f}s (attempt {attempt + 1}/{max_attempts})")
                time.sleep(wait)

    raise RuntimeError("unreachable")  # loop always returns or raises


# ── Gemini (default) ─────────────────────────────────────────────────────────

def _call_gemini(
    prompt: str,
    image_path: str | None,
    max_tokens: int,
    temperature: float | None,
    json_mode: bool,
    thinking_budget: int | None,
    model: str,
) -> tuple[str, int]:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover
        raise ImportError("google-genai not installed. Run: pip install google-genai") from exc

    api_key = settings.gemini_api_key or os.environ.get("GOOGLE_API_KEY") or None
    client = genai.Client(api_key=api_key)

    parts: list = []
    img = _image_bytes(image_path)
    if img is not None:
        parts.append(types.Part.from_bytes(data=img, mime_type="image/png"))
        parts.append(_PITCH_IMAGE_HINT + prompt)
    else:
        parts.append(prompt)

    cfg: dict = {"max_output_tokens": max_tokens}
    if temperature is not None:
        cfg["temperature"] = temperature
    if json_mode:
        cfg["response_mime_type"] = "application/json"
    if thinking_budget is not None:
        try:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
        except Exception:  # older SDK / model without thinking control
            pass

    resp = client.models.generate_content(
        model=model,
        contents=parts,
        config=types.GenerateContentConfig(**cfg),
    )

    try:
        text = resp.text or ""
    except Exception as exc:  # blocked / no candidate
        logger.warning("Gemini returned no text: %s", exc)
        text = ""

    tokens = 0
    meta = getattr(resp, "usage_metadata", None)
    if meta is not None and getattr(meta, "total_token_count", None):
        tokens = int(meta.total_token_count)
    return text, tokens


# ── Anthropic (fallback) ────────────────────────────────────────────────────

def _call_anthropic(
    prompt: str,
    image_path: str | None,
    max_tokens: int,
    temperature: float | None,
    model: str,
) -> tuple[str, int]:
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover
        raise ImportError("anthropic not installed. Run: pip install anthropic") from exc

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    content: list = []
    img = _image_bytes(image_path)
    if img is not None:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(img).decode(),
                },
            }
        )
        content.append({"type": "text", "text": _PITCH_IMAGE_HINT + prompt})
    else:
        content.append({"type": "text", "text": prompt})

    kwargs: dict = {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": content}]}
    if temperature is not None:
        kwargs["temperature"] = temperature
    resp = client.messages.create(**kwargs)
    return resp.content[0].text, int(resp.usage.input_tokens + resp.usage.output_tokens)


# ── OpenAI (fallback) ──────────────────────────────────────────────────────

def _call_openai(
    prompt: str,
    image_path: str | None,
    max_tokens: int,
    temperature: float | None,
    json_mode: bool,
    model: str,
) -> tuple[str, int]:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise ImportError("openai not installed. Run: pip install openai") from exc

    client = OpenAI(api_key=settings.openai_api_key)
    content: list = []
    img = _image_bytes(image_path)
    if img is not None:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64.b64encode(img).decode()}"},
            }
        )
        content.append({"type": "text", "text": _PITCH_IMAGE_HINT + prompt})
    else:
        content.append({"type": "text", "text": prompt})

    kwargs: dict = {
        "model": model if model.startswith(("gpt-", "o1", "o3", "o4")) else "gpt-4o",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content}],
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or "", int(resp.usage.total_tokens)
