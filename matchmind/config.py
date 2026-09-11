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

    # ── LLM ──────────────────────────────────────────────────
    # MatchMind supports multiple providers. Set LLM_PROVIDER in .env.
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")

    llm_provider: str = Field(default="gemini", alias="LLM_PROVIDER")  # gemini | anthropic | openai | groq
    llm_model: str = Field(default="gemini-2.5-flash", alias="LLM_MODEL")
    groq_model: str = Field(default="qwen/qwen3.8-27b", alias="GROQ_MODEL")

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
