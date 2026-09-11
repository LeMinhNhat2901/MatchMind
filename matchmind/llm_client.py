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

    Returns:
        (text, total_tokens)
    """
    provider = (settings.llm_provider or "gemini").lower()
    model = model or settings.llm_model

    if provider == "gemini":
        return _call_gemini(prompt, image_path, max_tokens, temperature, json_mode, thinking_budget, model)
    if provider == "anthropic":
        return _call_anthropic(prompt, image_path, max_tokens, temperature, model)
    if provider == "openai":
        return _call_openai(prompt, image_path, max_tokens, temperature, json_mode, model)
    if provider == "groq":
        return _call_groq(prompt, image_path, max_tokens, temperature, json_mode, model)
    raise ValueError(
        f"Unknown LLM_PROVIDER={settings.llm_provider!r}. Use one of: gemini | anthropic | openai | groq"
    )


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
            if not _is_rate_limit_error(exc) or attempt == max_attempts - 1:
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
