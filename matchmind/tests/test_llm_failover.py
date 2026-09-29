"""LLM provider failover chain (LLM_FALLBACK_CHAIN). No network: providers are faked."""
from __future__ import annotations

import pytest

from matchmind import llm_client
from matchmind.llm_client import call_llm, is_daily_quota_error, parse_fallback_chain

DAILY = "Error code: 429 - Rate limit reached for model on tokens per day (TPD): Limit 200000, Used 198955"


@pytest.fixture
def chain(monkeypatch):
    """Install a chain + fake providers; returns the list of attempted providers."""
    calls: list[str] = []
    behaviour: dict[str, object] = {}

    def fake_call_provider(provider, model, prompt, **kw):
        calls.append(provider)
        outcome = behaviour.get(provider, "ok")
        if isinstance(outcome, Exception):
            raise outcome
        return f"{provider}:{model}", 7

    monkeypatch.setattr(llm_client, "_call_provider", fake_call_provider)
    monkeypatch.setattr(llm_client, "_exhausted", {})
    monkeypatch.setattr(llm_client.settings, "llm_fallback_chain",
                        "groq:qwen/qwen3.8-27b, cerebras:qwen-3.8-27b, mistral:mistral-small-latest")
    for key in ("groq_api_key", "cerebras_api_key", "mistral_api_key"):
        monkeypatch.setattr(llm_client.settings, key, "k")
    return calls, behaviour


def test_parse_chain_keeps_colons_inside_model_names():
    assert parse_fallback_chain("groq:qwen/qwen3.8-27b, openrouter:nvidia/nemotron:free") == [
        ("groq", "qwen/qwen3.8-27b"), ("openrouter", "nvidia/nemotron:free")]
    assert parse_fallback_chain("") == []


def test_daily_quota_detection():
    assert is_daily_quota_error(RuntimeError(DAILY))
    assert not is_daily_quota_error(RuntimeError("429 tokens per minute"))
    assert not is_daily_quota_error(RuntimeError("503 UNAVAILABLE"))


def test_first_provider_answers(chain):
    calls, _ = chain
    assert call_llm("p") == ("groq:qwen/qwen3.8-27b", 7)
    assert calls == ["groq"]
    assert llm_client.last_provider() == "groq"


def test_transient_error_fails_over_for_this_call_only(chain):
    calls, behaviour = chain
    behaviour["groq"] = RuntimeError("503 UNAVAILABLE")
    assert call_llm("p")[0] == "cerebras:qwen-3.8-27b"
    behaviour["groq"] = "ok"
    call_llm("p")
    assert calls == ["groq", "cerebras", "groq"]          # groq tried again next call


def test_daily_quota_marks_provider_exhausted_for_the_day(chain):
    calls, behaviour = chain
    behaviour["groq"] = RuntimeError(DAILY)
    call_llm("p")
    call_llm("p")
    assert calls == ["groq", "cerebras", "cerebras"]      # no second attempt at groq today


def test_provider_without_key_is_skipped(chain, monkeypatch):
    calls, _ = chain
    monkeypatch.setattr(llm_client.settings, "groq_api_key", "")
    monkeypatch.setattr(llm_client.settings, "cerebras_api_key", "")   # key not filled in yet
    assert call_llm("p")[0] == "mistral:mistral-small-latest"
    assert calls == ["mistral"]


def test_all_providers_failing_raises_with_every_reason(chain):
    _, behaviour = chain
    behaviour.update(groq=RuntimeError(DAILY), cerebras=RuntimeError("503"), mistral=RuntimeError("401 bad key"))
    with pytest.raises(RuntimeError) as err:
        call_llm("p")
    assert "groq" in str(err.value) and "cerebras" in str(err.value) and "mistral" in str(err.value)


def test_explicit_provider_bypasses_the_chain(chain):
    calls, _ = chain
    assert call_llm("p", provider="gemini", model="gemini-3.6-flash")[0] == "gemini:gemini-3.6-flash"
    assert calls == ["gemini"]


def test_reasoner_trace_records_which_provider_answered(monkeypatch):
    from matchmind.agent.nodes import reasoner_node
    from matchmind.situation_engine.encoder import encode_situation
    from matchmind.tests.fixtures.sample_data import make_sample_match_state

    def fake_llm(prompt, **kw):
        llm_client._last_provider = "cerebras"
        return "{}", 5

    monkeypatch.setattr(reasoner_node, "call_llm", fake_llm)
    ms, focus = make_sample_match_state()
    state = {"match_state": ms, "focus_player_id": focus, "question": "q", "trace": [],
             "situation_features": encode_situation(ms, focus), "retrieved_tactics": [],
             "candidate_actions": [], "counterfactual_text": "", "evidence_set": {}}
    out = reasoner_node.reasoner_node(state)
    assert out["trace"][-1]["provider"] == "cerebras"


def test_groq_daily_quota_raises_immediately_without_retry(monkeypatch):
    """Groq's TPD 429 carries "Limit 200000" and used to be retried with sleeps
    (the 2026-09-28 eval spent ~64 s/case this way). Fail fast so failover moves on."""
    import sys
    import types

    attempts = []

    class FakeGroq:
        def __init__(self, api_key):
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            attempts.append(kwargs)
            raise RuntimeError(DAILY)

    monkeypatch.setitem(sys.modules, "groq", types.SimpleNamespace(Groq=FakeGroq))
    monkeypatch.setattr(llm_client.settings, "groq_api_key", "k")
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: pytest.fail("should not sleep on a daily quota"))
    with pytest.raises(RuntimeError):
        llm_client._call_groq("p", None, 100, 0.0, False, "qwen/qwen3.8-27b")
    assert len(attempts) == 1


def test_openai_compatible_provider_uses_base_url_and_json_mode(monkeypatch):
    seen = {}

    class FakeCompletions:
        def create(self, **kwargs):
            seen.update(kwargs)

            class R:
                class _C:
                    class message:
                        content = '{"ok": true}'
                choices = [_C]

                class usage:
                    total_tokens = 42
            return R

    class FakeClient:
        def __init__(self, base_url, api_key, **kw):
            seen["base_url"], seen["api_key"] = base_url, api_key
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(llm_client, "_openai_client", FakeClient)
    monkeypatch.setattr(llm_client.settings, "mistral_api_key", "secret")
    text, tokens = llm_client._call_openai_compatible("mistral", "p", 100, 0.2, True, "mistral-small-latest")
    assert (text, tokens) == ('{"ok": true}', 42)
    assert seen["base_url"] == "https://api.mistral.ai/v1" and seen["api_key"] == "secret"
    assert seen["response_format"] == {"type": "json_object"}
    assert seen["model"] == "mistral-small-latest"
