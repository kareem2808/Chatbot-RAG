"""Rate limiter berbasis token bucket untuk FastAPI.

Implementasi in-memory token bucket yang thread-safe dan async-ready.
Dapat digunakan sebagai FastAPI dependency untuk membatasi jumlah
request per client per menit.

Contoh penggunaan standalone:
    >>> from src.utils.rate_limiter import RateLimiter
    >>> limiter = RateLimiter(max_requests=60, window_seconds=60)
    >>> allowed = await limiter.acquire("client-ip-192.168.1.1")

Contoh penggunaan sebagai FastAPI dependency:
    >>> from fastapi import Depends, FastAPI
    >>> from src.utils.rate_limiter import RateLimiter, rate_limit_dependency
    >>>
    >>> app = FastAPI()
    >>> limiter = RateLimiter(max_requests=60, window_seconds=60)
    >>>
    >>> @app.get("/ask", dependencies=[Depends(rate_limit_dependency(limiter))])
    >>> async def ask_endpoint():
    ...     return {"answer": "..."}
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from fastapi import HTTPException, Request, status


# ──────────────────────────────────────────────
#  Token Bucket per Client
# ──────────────────────────────────────────────
@dataclass
class _TokenBucket:
    """State internal untuk satu client.

    Attributes:
        tokens: Jumlah token yang tersedia saat ini.
        max_tokens: Kapasitas maksimum bucket.
        refill_rate: Jumlah token yang ditambahkan per detik.
        last_refill: Timestamp terakhir kali token di-refill.
    """

    tokens: float
    max_tokens: float
    refill_rate: float
    last_refill: float = field(default_factory=time.monotonic)

    def consume(self) -> bool:
        """Coba konsumsi satu token dari bucket.

        Melakukan refill terlebih dahulu berdasarkan waktu yang telah
        berlalu sejak refill terakhir, kemudian mencoba mengambil token.

        Returns:
            ``True`` jika token berhasil dikonsumsi, ``False`` jika habis.
        """
        now: float = time.monotonic()
        elapsed: float = now - self.last_refill

        # Refill token berdasarkan waktu yang berlalu
        self.tokens = min(
            self.max_tokens,
            self.tokens + elapsed * self.refill_rate,
        )
        self.last_refill = now

        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

    @property
    def retry_after(self) -> float:
        """Hitung detik yang harus ditunggu sebelum token tersedia.

        Returns:
            Waktu tunggu dalam detik (minimum 0).
        """
        if self.tokens >= 1.0:
            return 0.0
        deficit: float = 1.0 - self.tokens
        return deficit / self.refill_rate


# ──────────────────────────────────────────────
#  Rate Limiter
# ──────────────────────────────────────────────
class RateLimiter:
    """In-memory rate limiter berbasis token bucket algorithm.

    Thread-safe melalui ``asyncio.Lock``. Setiap client (diidentifikasi
    dengan string key, misalnya IP address) mendapat bucket independen.

    Args:
        max_requests: Jumlah maksimum request yang diizinkan per window.
        window_seconds: Durasi window dalam detik.
        cleanup_interval: Interval pembersihan bucket idle (detik).
    """

    def __init__(
        self,
        max_requests: int = 60,
        window_seconds: int = 60,
        cleanup_interval: int = 300,
    ) -> None:
        self._max_requests: int = max_requests
        self._window_seconds: int = window_seconds
        self._cleanup_interval: int = cleanup_interval
        self._refill_rate: float = max_requests / window_seconds

        self._buckets: dict[str, _TokenBucket] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

    @property
    def max_requests(self) -> int:
        """Jumlah maksimum request per window."""
        return self._max_requests

    @property
    def window_seconds(self) -> int:
        """Durasi window dalam detik."""
        return self._window_seconds

    async def acquire(self, client_id: str) -> bool:
        """Coba acquire satu token untuk client tertentu.

        Args:
            client_id: Identifier unik client (misal: IP address).

        Returns:
            ``True`` jika request diizinkan, ``False`` jika rate-limited.
        """
        async with self._lock:
            bucket: _TokenBucket = self._get_or_create_bucket(client_id)
            return bucket.consume()

    async def get_retry_after(self, client_id: str) -> float:
        """Dapatkan waktu tunggu (detik) sebelum client bisa request lagi.

        Args:
            client_id: Identifier unik client.

        Returns:
            Waktu tunggu dalam detik. ``0.0`` jika langsung bisa request.
        """
        async with self._lock:
            bucket: _TokenBucket | None = self._buckets.get(client_id)
            if bucket is None:
                return 0.0
            return bucket.retry_after

    async def get_remaining(self, client_id: str) -> int:
        """Dapatkan jumlah request tersisa untuk client tertentu.

        Args:
            client_id: Identifier unik client.

        Returns:
            Jumlah request yang tersisa (integer, minimum 0).
        """
        async with self._lock:
            bucket: _TokenBucket | None = self._buckets.get(client_id)
            if bucket is None:
                return self._max_requests
            # Refill dulu untuk akurasi
            now: float = time.monotonic()
            elapsed: float = now - bucket.last_refill
            current_tokens: float = min(
                bucket.max_tokens,
                bucket.tokens + elapsed * bucket.refill_rate,
            )
            return int(current_tokens)

    async def reset(self, client_id: str) -> None:
        """Reset rate limit untuk client tertentu.

        Args:
            client_id: Identifier unik client yang akan di-reset.
        """
        async with self._lock:
            self._buckets.pop(client_id, None)

    async def cleanup_stale_buckets(self) -> int:
        """Hapus bucket yang sudah idle melebihi cleanup_interval.

        Returns:
            Jumlah bucket yang dihapus.
        """
        async with self._lock:
            now: float = time.monotonic()
            stale_keys: list[str] = [
                key
                for key, bucket in self._buckets.items()
                if (now - bucket.last_refill) > self._cleanup_interval
            ]
            for key in stale_keys:
                del self._buckets[key]
            return len(stale_keys)

    def _get_or_create_bucket(self, client_id: str) -> _TokenBucket:
        """Ambil bucket yang ada atau buat baru untuk client.

        Args:
            client_id: Identifier unik client.

        Returns:
            Instance ``_TokenBucket`` untuk client tersebut.
        """
        if client_id not in self._buckets:
            self._buckets[client_id] = _TokenBucket(
                tokens=float(self._max_requests),
                max_tokens=float(self._max_requests),
                refill_rate=self._refill_rate,
            )
        return self._buckets[client_id]

    def __repr__(self) -> str:
        return (
            f"RateLimiter("
            f"max_requests={self._max_requests}, "
            f"window_seconds={self._window_seconds}, "
            f"active_clients={len(self._buckets)})"
        )


# ──────────────────────────────────────────────
#  FastAPI Dependency Factory
# ──────────────────────────────────────────────
def rate_limit_dependency(
    limiter: RateLimiter,
    *,
    client_id_header: str | None = None,
) -> Callable[[Request], Awaitable[None]]:
    """Buat FastAPI dependency untuk rate limiting.

    Secara default menggunakan ``request.client.host`` sebagai client_id.
    Jika ``client_id_header`` diberikan, maka header tersebut yang dipakai
    (berguna jika di belakang reverse proxy / load balancer).

    Args:
        limiter: Instance ``RateLimiter`` yang digunakan.
        client_id_header: Nama header HTTP untuk identifikasi client
            (misal: ``X-Forwarded-For``). Jika ``None``, pakai IP.

    Returns:
        Async callable yang kompatibel sebagai FastAPI ``Depends``.

    Raises:
        HTTPException: Status 429 jika client melebihi rate limit.
    """

    async def _dependency(request: Request) -> None:
        # Tentukan client_id
        client_id: str = _resolve_client_id(request, client_id_header)

        # Coba acquire token
        allowed: bool = await limiter.acquire(client_id)

        if not allowed:
            retry_after: float = await limiter.get_retry_after(client_id)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "rate_limit_exceeded",
                    "message": (
                        f"Terlalu banyak request. Coba lagi dalam "
                        f"{retry_after:.1f} detik."
                    ),
                    "retry_after_seconds": round(retry_after, 1),
                },
                headers={"Retry-After": str(int(retry_after) + 1)},
            )

    return _dependency


def _resolve_client_id(
    request: Request,
    header_name: str | None,
) -> str:
    """Resolve client identifier dari request.

    Args:
        request: FastAPI/Starlette Request object.
        header_name: Nama header opsional untuk identifikasi.

    Returns:
        String identifier client.
    """
    if header_name is not None:
        header_value: str | None = request.headers.get(header_name)
        if header_value:
            # Ambil IP pertama jika comma-separated (X-Forwarded-For)
            return header_value.split(",")[0].strip()

    if request.client is not None:
        return request.client.host

    return "unknown"
