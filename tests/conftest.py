"""Shared fixtures untuk test suite."""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create a single event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_documents() -> list[dict[str, str]]:
    """Fixture: daftar dokumen sampel sederhana."""
    return [
        {
            "text": "Kami menjual kopi arabika premium seharga Rp 45.000 per pack.",
            "metadata": {"source": "katalog", "category": "kopi"},
        },
        {
            "text": (
                "Jam buka toko: Senin-Jumat 08.00-17.00, "
                "Sabtu 09.00-15.00, Minggu tutup."
            ),
            "metadata": {"source": "info", "category": "jam_buka"},
        },
        {
            "text": (
                "Kami menerima pembayaran tunai, transfer bank BCA/BRI, "
                "dan QRIS (GoPay, OVO, Dana)."
            ),
            "metadata": {"source": "kebijakan", "category": "pembayaran"},
        },
    ]


@pytest.fixture
def long_document() -> str:
    """Fixture: dokumen panjang untuk testing chunking."""
    paragraphs = [
        f"Paragraf {i}: Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        f"Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        f"Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris."
        for i in range(20)
    ]
    return "\n\n".join(paragraphs)
