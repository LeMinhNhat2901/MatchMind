"""
MatchMind — Central configuration using Pydantic Settings.
Loads from .env file automatically.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def __repr_args__(self):
        """Never print secrets: a failing test once dumped every API key into the log."""
        for name, value in super().__repr_args__():
            secret = name is not None and (name.endswith("_api_key") or name in {"database_url", "redis_url"})
            yield name, ("***" if secret and value else value)

    # ── LLM ──────────────────────────────────────────────────
    # MatchMind supports multiple providers. Set LLM_PROVIDER in .env.
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")

    # OpenAI-compatible fallbacks (llm_client._OPENAI_COMPATIBLE). Empty = skipped by the chain.
    cerebras_api_key: str = Field(default="", alias="CEREBRAS_API_KEY")
    mistral_api_key: str = Field(default="", alias="MISTRAL_API_KEY")
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    # Ordered "provider:model" list tried by call_llm; providers without a key or
    # with today's quota used up are skipped. Empty = LLM_PROVIDER/LLM_MODEL only.
    llm_fallback_chain: str = Field(default="", alias="LLM_FALLBACK_CHAIN")

    llm_provider: str = Field(default="gemini", alias="LLM_PROVIDER")  # gemini | anthropic | openai | groq
    llm_model: str = Field(default="gemini-2.5-flash", alias="LLM_MODEL")
    groq_model: str = Field(default="qwen/qwen3.8-27b", alias="GROQ_MODEL")
    # Optional override for the user-facing narration/translation step only
    # (narrator_node, translation.translate_from_english) — e.g. a model that
    # writes better Vietnamese than the default. Empty = use LLM_PROVIDER/LLM_MODEL.
    # On any error with the override, callers retry once on the default provider.
    narrator_llm_provider: str = Field(default="", alias="NARRATOR_LLM_PROVIDER")
    narrator_llm_model: str = Field(default="", alias="NARRATOR_LLM_MODEL")

    # ── Vector Store ─────────────────────────────────────────
    chroma_persist_dir: str = Field(default="./tactics_db", alias="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field(
        default="football_tactics", alias="CHROMA_COLLECTION_NAME"
    )

    # ── Embedding ────────────────────────────────────────────
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    # "chroma_default" uses ChromaDB's ONNX EF (avoids sentence-transformers crash on Python 3.13)
    # "sentence_transformers" uses SentenceTransformer directly (more flexible, may crash in threads)
    embedding_backend: str = Field(default="chroma_default", alias="EMBEDDING_BACKEND")

    # ── Databases ────────────────────────────────────────────
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    database_url: str = Field(
        default="postgresql+asyncpg://matchmind:matchmind@localhost:5432/matchmind",
        alias="DATABASE_URL",
    )

    # ── Agent Config ─────────────────────────────────────────
    agent_max_retries: int = Field(default=2, alias="AGENT_MAX_RETRIES")
    # Fallback threshold for MiniLM + a ~30-doc KB. Run scripts/calibrate_retrieval.py
    # to replace this with a data-driven value (written to retrieval_calibration_file).
    retrieval_confidence_threshold: float = Field(
        default=0.35, alias="RETRIEVAL_CONFIDENCE_THRESHOLD"
    )
    retrieval_calibration_file: Path = Field(
        default=Path("./output/retrieval_calibration.json"),
        alias="RETRIEVAL_CALIBRATION_FILE",
    )
    retrieval_top_k: int = Field(default=3, alias="RETRIEVAL_TOP_K")
    retrieval_phase_filter: bool = Field(default=True, alias="RETRIEVAL_PHASE_FILTER")

    # ── Retrieval V1 (matchmind/retrieval/) ──────────────────
    # "hybrid" = query builder + dense + BM25 + condition match + RRF + rerank.
    # "dense"  = legacy single-query Chroma search (kept for A/B benchmarks).
    retrieval_strategy: str = Field(default="hybrid", alias="RETRIEVAL_STRATEGY")
    retrieval_pool_size: int = Field(default=20, alias="RETRIEVAL_POOL_SIZE")
    # Empty string disables reranking (fusion order is used as-is).
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL"
    )
    # Final score = alpha * rerank + (1 - alpha) * normalised RRF score. 1.0 = rerank only.
    # 0.5 since the KB tripled (331 concepts): at 0.3 the cross-encoder was outvoted by
    # BM25 hits on long, generic concepts. Golden bench 2026-10-08 (exact dense):
    # 0.3/0.5/0.7/0.85 -> nDCG@5 0.660/0.696/0.694/0.686, hit@3 0.778/0.926/0.926/0.889.
    rerank_blend_alpha: float = Field(default=0.5, alias="RERANK_BLEND_ALPHA")
    # Example questions (doc2query). Both off until the owner picks a config: on
    # 2026-10-15 BM25-with-questions raised held-out nDCG@5 0.501 -> 0.538 but
    # lowered golden 0.700 -> 0.687 (agreed rule: adopt only if both sets improve).
    retrieval_bm25_questions: bool = Field(default=False, alias="RETRIEVAL_BM25_QUESTIONS")
    retrieval_question_weight: float = Field(default=0.0, alias="RETRIEVAL_QUESTION_WEIGHT")
    # 1.0 = off. Golden bench (Task 10): 0.88/0.92 cost nDCG@5 0.6599 -> 0.6572, because
    # near-duplicate concepts (e.g. several high-press concepts) are often all relevant.
    retrieval_diversity_max_sim: float = Field(default=1.0, alias="RETRIEVAL_DIVERSITY_MAX_SIM")

    # ── Example Library (matchmind/examples/) ────────────────
    statsbomb_open_data_dir: Path = Field(
        default=Path("../open-data/data"), alias="STATSBOMB_OPEN_DATA_DIR"
    )
    example_library_dir: Path = Field(
        default=Path("./output/example_library"), alias="EXAMPLE_LIBRARY_DIR"
    )
    example_top_k: int = Field(default=5, alias="EXAMPLE_TOP_K")

    # ── Paths ─────────────────────────────────────────────────
    knowledge_base_dir: Path = Field(
        default=Path("./matchmind/knowledge_base/concepts"),
        alias="KNOWLEDGE_BASE_DIR",
    )
    pitch_images_dir: Path = Field(
        default=Path("./output/pitch_images"), alias="PITCH_IMAGES_DIR"
    )
    evaluation_output_dir: Path = Field(
        default=Path("./output/evaluation"), alias="EVALUATION_OUTPUT_DIR"
    )

    # ── Logging ───────────────────────────────────────────────
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    def ensure_output_dirs(self) -> None:
        """Create output directories if they don't exist."""
        self.pitch_images_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def effective_retrieval_threshold(self) -> float:
        """Calibrated threshold if scripts/calibrate_retrieval.py has been run,
        otherwise the configured fallback."""
        try:
            import json

            if self.retrieval_calibration_file.exists():
                data = json.loads(self.retrieval_calibration_file.read_text())
                return float(data["threshold"])
        except Exception:
            pass
        return self.retrieval_confidence_threshold


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance. Call once at startup."""
    settings = Settings()
    settings.ensure_output_dirs()
    return settings


# Convenience singleton
settings = get_settings()
