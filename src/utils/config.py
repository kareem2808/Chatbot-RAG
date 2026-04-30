"""Manajemen konfigurasi aplikasi menggunakan Pydantic BaseSettings.

Memvalidasi variabel environment dari file .env dan menyediakan
akses terpusat ke seluruh konfigurasi sistem RAG.

Contoh penggunaan:
    >>> from src.utils.config import get_settings
    >>> settings = get_settings()
    >>> print(settings.gemini_api_key.get_secret_value())
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Final

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ──────────────────────────────────────────────
#  Konstanta default
# ──────────────────────────────────────────────
_DEFAULT_CHROMA_DB_PATH: Final[str] = "./data/chroma_db"
_DEFAULT_MAX_RETRIES: Final[int] = 3
_DEFAULT_RATE_LIMIT_PER_MINUTE: Final[int] = 60
_DEFAULT_LOG_LEVEL: Final[str] = "INFO"
_DEFAULT_LOG_DIR: Final[str] = "./logs"


class Settings(BaseSettings):
    """Konfigurasi utama aplikasi, divalidasi dari .env.

    Attributes:
        gemini_api_key: API key untuk Google Gemini 1.5 Flash.
        chroma_db_path: Path ke direktori ChromaDB.
        max_retries: Jumlah maksimum retry untuk panggilan API.
        rate_limit_per_minute: Batas request per menit.
        log_level: Level logging (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: Direktori penyimpanan file log.
        app_name: Nama aplikasi untuk identifikasi di log.
        environment: Environment deployment (development, staging, production).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Wajib Berdasarkan Provider ───────────
    llm_provider: str = Field(
        default="gemini",
        description="Provider LLM utama: 'gemini' atau 'groq'",
    )
    gemini_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="API key untuk Google Gemini 1.5 Flash (Opsional jika provider=groq)",
    )
    groq_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="API key untuk Groq (Opsional jika provider=gemini)",
    )
    groq_model_name: str = Field(
        default="llama-3.1-8b-instant",
        description="Nama model default untuk Groq",
    )

    # ── Opsional dengan default ──────────────
    chroma_db_path: Path = Field(
        default=Path(_DEFAULT_CHROMA_DB_PATH),
        description="Path ke direktori penyimpanan ChromaDB",
    )
    max_retries: int = Field(
        default=_DEFAULT_MAX_RETRIES,
        ge=0,
        le=10,
        description="Jumlah maksimum retry untuk panggilan API",
    )
    rate_limit_per_minute: int = Field(
        default=_DEFAULT_RATE_LIMIT_PER_MINUTE,
        ge=1,
        le=600,
        description="Batas request per menit",
    )
    log_level: str = Field(
        default=_DEFAULT_LOG_LEVEL,
        description="Level logging aplikasi",
    )
    log_dir: Path = Field(
        default=Path(_DEFAULT_LOG_DIR),
        description="Direktori penyimpanan file log",
    )
    app_name: str = Field(
        default="rag-system",
        description="Nama aplikasi untuk identifikasi di log",
    )
    environment: str = Field(
        default="development",
        description="Environment deployment aktif",
    )
    api_keys: str = Field(
        default="",
        description=(
            "API keys untuk autentikasi (comma-separated). "
            "Kosongkan untuk menonaktifkan auth."
        ),
    )
    cors_origins: str = Field(
        default="",
        description=(
            "Allowed CORS origins (comma-separated). "
            "Kosongkan untuk menggunakan default berdasarkan environment."
        ),
    )

    # ── Validator ────────────────────────────
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        """Validasi log_level terhadap level standar Python logging."""
        allowed_levels: set[str] = {
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }
        normalised: str = value.upper().strip()
        if normalised not in allowed_levels:
            msg = (
                f"log_level '{value}' tidak valid. "
                f"Pilih dari: {', '.join(sorted(allowed_levels))}"
            )
            raise ValueError(msg)
        return normalised

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, value: str) -> str:
        """Validasi environment terhadap nilai yang diizinkan."""
        allowed_envs: set[str] = {"development", "staging", "production"}
        normalised: str = value.lower().strip()
        if normalised not in allowed_envs:
            msg = (
                f"environment '{value}' tidak valid. "
                f"Pilih dari: {', '.join(sorted(allowed_envs))}"
            )
            raise ValueError(msg)
        return normalised

    @model_validator(mode="after")
    def validate_api_keys(self) -> Settings:
        """Pastikan API key yang sesuai dengan provider tersedia."""
        provider = self.llm_provider.lower().strip()
        if provider == "gemini":
            if not self.gemini_api_key.get_secret_value().strip():
                raise ValueError("GEMINI_API_KEY wajib diisi jika LLM_PROVIDER='gemini'")
        elif provider == "groq":
            if not self.groq_api_key.get_secret_value().strip():
                raise ValueError("GROQ_API_KEY wajib diisi jika LLM_PROVIDER='groq'")
        else:
            raise ValueError(f"LLM_PROVIDER '{provider}' tidak didukung. Pilih 'gemini' atau 'groq'")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Buat dan cache instance Settings (singleton via lru_cache).

    Returns:
        Instance Settings yang telah tervalidasi.

    Raises:
        pydantic.ValidationError: Jika konfigurasi tidak valid.
    """
    return Settings()  # type: ignore[call-arg]
