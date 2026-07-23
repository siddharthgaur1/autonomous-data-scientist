"""Env-driven settings, validated at import time.

Every secret and path the system needs is declared here. Missing required vars
raise at startup rather than surfacing as an AttributeError deep inside an agent.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Required fields have no default and fail loudly."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    openai_api_key: str = Field(..., description="LLM API key (OpenAI or any compatible provider).")
    # Any OpenAI-compatible endpoint. Leave empty for OpenAI itself; set it to run
    # on a free tier instead — Groq (https://api.groq.com/openai/v1), OpenRouter,
    # Together, or a local Ollama (http://localhost:11434/v1). The whole agent
    # graph then runs without paid OpenAI usage.
    openai_base_url: str = Field(default="", description="OpenAI-compatible base URL; empty = OpenAI.")
    # Redis is a convenience (resumable checkpoints), not a requirement: the runner
    # already degrades to a keyless in-memory run if it is unreachable. Defaulted so
    # the app imports and runs with nothing but an API key.
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis URL for checkpoints and run cache.")
    db_path: Path = Field(default=Path("data/runs.db"), description="SQLite file holding run history.")
    runs_dir: Path = Field(
        default=Path("runs"), description="Root for per-run artifacts."
    )

    reasoning_model: str = Field(default="gpt-4o", description="Model for planning.")
    cheap_model: str = Field(default="gpt-4o-mini", description="Model for sub-tasks.")

    max_run_cost_usd: float = Field(
        default=2.0, description="Hard cap on LLM spend per run."
    )
    sandbox_timeout_s: int = Field(
        default=120, description="Wall-clock cap per sandboxed execution."
    )
    sandbox_memory_mb: int = Field(
        default=2048, description="Address-space cap per sandboxed execution."
    )
    confidence_threshold: float = Field(
        default=0.6, description="Below this, the run escalates to a human."
    )

    @field_validator("openai_api_key")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("OPENAI_API_KEY is set but empty.")
        return v

    def run_dir(self, run_id: str) -> Path:
        """Return (and create) the artifact directory for a run."""
        d = self.runs_dir / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process. Raises ValidationError if a var is missing."""
    return Settings()
