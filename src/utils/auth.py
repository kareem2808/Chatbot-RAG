"""Autentikasi API Key untuk endpoint yang dilindungi.

Mendukung multi-tenant API keys yang disimpan di .env sebagai
comma-separated string. Auth **diaktifkan secara default** —
untuk menonaktifkan, set API_KEYS="" (string kosong) di .env.

Contoh penggunaan di route:
    @router.post("/chat")
    async def chat(api_key: str = Depends(verify_api_key)):
        ...

Contoh .env:
    API_KEYS=key-abc123,key-def456
"""

from __future__ import annotations

import logging
import secrets
from typing import Optional

from fastapi import HTTPException, Request, status

from src.utils.config import get_settings

logger: logging.Logger = logging.getLogger(__name__)


def _get_valid_keys() -> set[str]:
    """Ambil set API keys valid dari konfigurasi.

    Returns:
        Set of valid API key strings. Empty set jika auth dinonaktifkan.
    """
    settings = get_settings()
    raw_keys: str = settings.api_keys.strip()
    if not raw_keys:
        return set()
    return {k.strip() for k in raw_keys.split(",") if k.strip()}


def is_auth_enabled() -> bool:
    """Cek apakah autentikasi API key aktif.

    Returns:
        True jika ada minimal 1 API key terkonfigurasi.
    """
    return len(_get_valid_keys()) > 0


async def verify_api_key(request: Request) -> Optional[str]:
    """FastAPI dependency: verifikasi API key dari header X-API-Key.

    Jika API_KEYS kosong di .env, autentikasi dilewati (dev mode).
    Jika API_KEYS terisi, header X-API-Key wajib ada dan valid.

    Args:
        request: FastAPI Request object.

    Returns:
        API key yang terverifikasi, atau None jika auth dinonaktifkan.

    Raises:
        HTTPException 401: Jika API key tidak valid atau tidak ada.
    """
    valid_keys: set[str] = _get_valid_keys()

    # Auth dinonaktifkan — skip
    if not valid_keys:
        return None

    # Ambil key dari header
    api_key: str | None = request.headers.get("X-API-Key")

    if not api_key:
        logger.warning(
            "Request tanpa API key ditolak",
            extra={"path": request.url.path},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "missing_api_key",
                "message": (
                    "API key diperlukan. Sertakan header "
                    "'X-API-Key: <your-key>' dalam request."
                ),
            },
        )

    # Constant-time comparison untuk mencegah timing attack
    is_valid: bool = any(
        secrets.compare_digest(api_key.encode(), key.encode())
        for key in valid_keys
    )

    if not is_valid:
        logger.warning(
            "API key tidak valid",
            extra={
                "path": request.url.path,
                "key_preview": f"{api_key[:4]}***" if len(api_key) > 4 else "***",
            },
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "invalid_api_key",
                "message": "API key tidak valid.",
            },
        )

    logger.debug(
        "API key terverifikasi",
        extra={"path": request.url.path},
    )
    return api_key
