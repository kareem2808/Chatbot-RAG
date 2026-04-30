"""Tests untuk RateLimiter."""

from __future__ import annotations

import asyncio
import time

import pytest

from src.utils.rate_limiter import RateLimiter


# ──────────────────────────────────────────────
#  Inisialisasi
# ──────────────────────────────────────────────
class TestRateLimiterInit:
    """Test inisialisasi RateLimiter."""

    def test_default_values(self) -> None:
        """Default values harus benar."""
        limiter = RateLimiter()
        assert limiter.max_requests == 60
        assert limiter.window_seconds == 60

    def test_custom_values(self) -> None:
        """Harus menerima custom values."""
        limiter = RateLimiter(max_requests=10, window_seconds=30)
        assert limiter.max_requests == 10
        assert limiter.window_seconds == 30

    def test_repr(self) -> None:
        """repr() harus informatif."""
        limiter = RateLimiter(max_requests=5)
        r = repr(limiter)
        assert "max_requests=5" in r


# ──────────────────────────────────────────────
#  acquire
# ──────────────────────────────────────────────
class TestAcquire:
    """Test metode acquire."""

    @pytest.mark.asyncio
    async def test_first_request_allowed(self) -> None:
        """Request pertama harus selalu diizinkan."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        allowed = await limiter.acquire("client-1")
        assert allowed is True

    @pytest.mark.asyncio
    async def test_within_limit(self) -> None:
        """Request dalam batas harus diizinkan."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            allowed = await limiter.acquire("client-2")
            assert allowed is True

    @pytest.mark.asyncio
    async def test_exceeds_limit(self) -> None:
        """Request melebihi batas harus ditolak."""
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            await limiter.acquire("client-3")

        # Request ke-4 harus ditolak
        allowed = await limiter.acquire("client-3")
        assert allowed is False

    @pytest.mark.asyncio
    async def test_different_clients_independent(self) -> None:
        """Setiap client punya bucket independen."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)

        # Client A habiskan kuota
        await limiter.acquire("client-a")
        await limiter.acquire("client-a")
        a_blocked = not await limiter.acquire("client-a")

        # Client B masih bisa
        b_allowed = await limiter.acquire("client-b")

        assert a_blocked is True
        assert b_allowed is True


# ──────────────────────────────────────────────
#  get_retry_after
# ──────────────────────────────────────────────
class TestGetRetryAfter:
    """Test metode get_retry_after."""

    @pytest.mark.asyncio
    async def test_new_client_zero(self) -> None:
        """Client baru harus punya retry_after 0."""
        limiter = RateLimiter(max_requests=5)
        retry = await limiter.get_retry_after("new-client")
        assert retry == 0.0

    @pytest.mark.asyncio
    async def test_exhausted_client_positive(self) -> None:
        """Client yang habis kuota harus punya retry_after > 0."""
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        await limiter.acquire("exhaust-client")
        await limiter.acquire("exhaust-client")  # ini ditolak

        retry = await limiter.get_retry_after("exhaust-client")
        assert retry > 0.0


# ──────────────────────────────────────────────
#  get_remaining
# ──────────────────────────────────────────────
class TestGetRemaining:
    """Test metode get_remaining."""

    @pytest.mark.asyncio
    async def test_new_client_full(self) -> None:
        """Client baru harus punya remaining = max_requests."""
        limiter = RateLimiter(max_requests=10)
        remaining = await limiter.get_remaining("fresh-client")
        assert remaining == 10

    @pytest.mark.asyncio
    async def test_after_usage(self) -> None:
        """Remaining harus berkurang setelah acquire."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        await limiter.acquire("used-client")
        await limiter.acquire("used-client")

        remaining = await limiter.get_remaining("used-client")
        assert remaining <= 4  # Mungkin sedikit refill terjadi


# ──────────────────────────────────────────────
#  reset
# ──────────────────────────────────────────────
class TestReset:
    """Test metode reset."""

    @pytest.mark.asyncio
    async def test_reset_restores_quota(self) -> None:
        """Reset harus mengembalikan kuota penuh."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)

        # Habiskan kuota
        await limiter.acquire("reset-client")
        await limiter.acquire("reset-client")
        blocked = not await limiter.acquire("reset-client")
        assert blocked

        # Reset
        await limiter.reset("reset-client")

        # Harus bisa lagi
        allowed = await limiter.acquire("reset-client")
        assert allowed is True


# ──────────────────────────────────────────────
#  cleanup_stale_buckets
# ──────────────────────────────────────────────
class TestCleanup:
    """Test metode cleanup_stale_buckets."""

    @pytest.mark.asyncio
    async def test_cleanup_removes_stale(self) -> None:
        """Cleanup harus menghapus bucket yang idle."""
        # cleanup_interval sangat kecil agar langsung stale
        limiter = RateLimiter(
            max_requests=5,
            window_seconds=60,
            cleanup_interval=0,  # Semua bucket langsung dianggap stale
        )

        await limiter.acquire("stale-client")
        removed = await limiter.cleanup_stale_buckets()
        assert removed >= 1

    @pytest.mark.asyncio
    async def test_cleanup_keeps_active(self) -> None:
        """Cleanup tidak boleh menghapus bucket yang masih aktif."""
        limiter = RateLimiter(
            max_requests=5,
            window_seconds=60,
            cleanup_interval=3600,  # 1 jam
        )

        await limiter.acquire("active-client")
        removed = await limiter.cleanup_stale_buckets()
        assert removed == 0
