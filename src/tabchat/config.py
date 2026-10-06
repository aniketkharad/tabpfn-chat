"""Configuration management for tabchat using pydantic-settings."""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys (fail fast on startup if missing or empty)
    GEMINI_API_KEY: str = Field(..., description="Google GenAI API Key")
    TABPFN_TOKEN: str = Field(..., description="Prior Labs TabPFN API Token")

    # LLM Models
    LLM_PRIMARY_MODEL: str = Field("gemini-3.8-flash", description="Primary conversational model")
    LLM_FALLBACK_MODEL: str = Field("gemini-3.8-flash-lite", description="Fallback model on 429 burst errors")

    # Local Storage Directory
    DATA_DIR: Path = Field(Path("data"), description="Base directory for sessions and caches")

    # Operational Boundaries
    MAX_UPLOAD_BYTES: int = Field(1_048_576, description="Max CSV upload size in bytes (1 MB)")
    MAX_ROWS: int = Field(200, description="Max rows per uploaded CSV")
    MAX_COLS: int = Field(50, description="Max columns per uploaded CSV")
    MAX_STR_LEN: int = Field(64, description="Max string length per cell")
    MAX_SESSION_TURNS: int = Field(15, description="Max chat turns per session")

    # Rate Limiting & Concurrency
    GEMINI_RPM_LIMIT: int = Field(12, description="Max Gemini requests per minute")
    GEMINI_CONCURRENCY: int = Field(1, description="Max concurrent active Gemini calls")
    TABPFN_FIT_TIMEOUT_SECONDS: int = Field(90, description="TabPFN fit execution timeout in seconds")

    @field_validator("GEMINI_API_KEY", "TABPFN_TOKEN")
    @classmethod
    def validate_non_empty(cls, v: str, info) -> str:
        if not v or not v.strip():
            raise ValueError(f"{info.field_name} must be a non-empty string")
        return v.strip()

    @field_validator("DATA_DIR")
    @classmethod
    def validate_data_dir(cls, v: Path) -> Path:
        path = Path(v).resolve()
        try:
            path.mkdir(parents=True, exist_ok=True)
            # Test write access
            test_file = path / ".write_test"
            test_file.touch(exist_ok=True)
            test_file.unlink(missing_ok=True)
        except Exception as e:
            raise ValueError(f"Cannot create or write to DATA_DIR '{path}': {e}") from e
        return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton instance of Settings."""
    return Settings()
