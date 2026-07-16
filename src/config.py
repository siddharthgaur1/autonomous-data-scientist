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

    openai_api_key: str = Field(..., description="OpenAI API key.")
    redis_url: str = Field(..., description="Redis URL for checkpoints and run cache.")
    db_path: Path = Field(..., description="SQLite file holding run history.")
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
