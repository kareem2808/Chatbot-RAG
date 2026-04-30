"""Tests untuk FastAPI routes (/health, /ingest, /chat).

Menggunakan FastAPI TestClient dengan mocked dependencies untuk
menghindari panggilan ke Gemini API dan ChromaDB sebenarnya.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.core.document_processor import DocumentProcessor, ProcessedChunk
from src.core.retriever import HybridRetriever, RetrievalResult
from src.core.router import QueryIntent, QueryRouter
from src.core.llm_engine import GeminiEngine
from src.utils.rate_limiter import RateLimiter
from src.api.routes import init_dependencies


# ──────────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────────
@pytest.fixture
def mock_retriever() -> MagicMock:
    """Mock HybridRetriever."""
    retriever = MagicMock(spec=HybridRetriever)
    retriever.list_collections = AsyncMock(return_value=["toko-demo"])
    retriever.search = AsyncMock(
        return_value=[
            RetrievalResult(
                document_id="doc-1",
                text="Kopi arabika Rp 45.000 per pack",
                metadata={"source": "katalog"},
                score=0.85,
                semantic_rank=1,
                keyword_rank=2,
            )
        ]
    )
    retriever.add_documents = AsyncMock(return_value=1)
    return retriever


@pytest.fixture
def mock_query_router() -> MagicMock:
    """Mock QueryRouter."""
    router = MagicMock(spec=QueryRouter)
    router.classify = AsyncMock(return_value=QueryIntent.IN_DOMAIN)
    return router


@pytest.fixture
def mock_llm_engine() -> MagicMock:
    """Mock GeminiEngine."""
    engine = MagicMock(spec=GeminiEngine)
    engine.generate_query_variations = AsyncMock(
        return_value=["harga kopi", "berapa kopi", "kopi arabika harga"]
    )
    engine.filter_relevant_chunks = AsyncMock(
        return_value=["Kopi arabika Rp 45.000 per pack"]
    )
    engine.generate_answer = AsyncMock(
        return_value="Kopi arabika kami dijual seharga Rp 45.000 per pack."
    )
    return engine


@pytest.fixture
def mock_rate_limiter() -> RateLimiter:
    """Real RateLimiter dengan limit tinggi (tidak akan menolak)."""
    return RateLimiter(max_requests=1000, window_seconds=60)


@pytest.fixture
def mock_document_processor() -> MagicMock:
    """Mock DocumentProcessor."""
    processor = MagicMock(spec=DocumentProcessor)
    processor.process_documents = AsyncMock(
        return_value=[
            ProcessedChunk(
                chunk_id="chunk-001",
                document_id="doc-test",
                text="Test dokumen chunk",
                metadata={"source": "test"},
                chunk_index=0,
                total_chunks=1,
            )
        ]
    )
    return processor


@pytest.fixture
def client(
    mock_retriever: MagicMock,
    mock_query_router: MagicMock,
    mock_llm_engine: MagicMock,
    mock_rate_limiter: RateLimiter,
    mock_document_processor: MagicMock,
) -> TestClient:
    """TestClient dengan semua dependency dimock."""
    # Patch Settings agar tidak perlu .env
    with patch("src.api.main.get_settings") as mock_settings, \
         patch("src.api.routes.get_settings") as mock_route_settings, \
         patch("src.api.main.HybridRetriever", return_value=mock_retriever), \
         patch("src.api.main.QueryRouter", return_value=mock_query_router), \
         patch("src.api.main.GeminiEngine", return_value=mock_llm_engine), \
         patch("src.api.main.RateLimiter", return_value=mock_rate_limiter), \
         patch("src.api.main.DocumentProcessor", return_value=mock_document_processor):

        settings = MagicMock()
        settings.app_name = "test-app"
        settings.environment = "development"
        settings.chroma_db_path = "./data/test_chroma"
        settings.gemini_api_key = MagicMock()
        settings.gemini_api_key.get_secret_value.return_value = "test-key"
        settings.max_retries = 1
        settings.rate_limit_per_minute = 1000
        mock_settings.return_value = settings
        mock_route_settings.return_value = settings

        app = create_app()
        return TestClient(app)


# ──────────────────────────────────────────────
#  GET /health
# ──────────────────────────────────────────────
class TestHealthEndpoint:
    """Test endpoint /api/v1/health."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """Health check harus return 200."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_response_fields(self, client: TestClient) -> None:
        """Response harus mengandung field yang benar."""
        response = client.get("/api/v1/health")
        data = response.json()
        assert "status" in data
        assert "chroma_db" in data
        assert "environment" in data
        assert data["status"] == "healthy"


# ──────────────────────────────────────────────
#  POST /ingest
# ──────────────────────────────────────────────
class TestIngestEndpoint:
    """Test endpoint /api/v1/ingest."""

    def test_ingest_success(self, client: TestClient) -> None:
        """Ingest yang valid harus return 200."""
        response = client.post(
            "/api/v1/ingest",
            json={
                "store_id": "toko-test",
                "documents": [
                    {
                        "text": "Ini dokumen test untuk ingest.",
                        "metadata": {"source": "test"},
                    }
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["store_id"] == "toko-test"
        assert data["documents_received"] == 1
        assert data["chunks_created"] >= 1

    def test_ingest_empty_documents(self, client: TestClient) -> None:
        """Ingest tanpa dokumen harus return 422."""
        response = client.post(
            "/api/v1/ingest",
            json={
                "store_id": "toko-test",
                "documents": [],
            },
        )
        assert response.status_code == 422

    def test_ingest_invalid_store_id(self, client: TestClient) -> None:
        """Store ID dengan karakter ilegal harus return 422."""
        response = client.post(
            "/api/v1/ingest",
            json={
                "store_id": "invalid store!!",
                "documents": [{"text": "Test"}],
            },
        )
        assert response.status_code == 422

    def test_ingest_empty_text(self, client: TestClient) -> None:
        """Dokumen dengan teks kosong harus return 422."""
        response = client.post(
            "/api/v1/ingest",
            json={
                "store_id": "toko-test",
                "documents": [{"text": ""}],
            },
        )
        assert response.status_code == 422


# ──────────────────────────────────────────────
#  POST /chat
# ──────────────────────────────────────────────
class TestChatEndpoint:
    """Test endpoint /api/v1/chat."""

    def test_chat_success(self, client: TestClient) -> None:
        """Chat yang valid harus return 200 dengan jawaban."""
        response = client.post(
            "/api/v1/chat",
            json={
                "user_query": "Berapa harga kopi arabika?",
                "store_id": "toko-demo",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert data["store_id"] == "toko-demo"
        assert "intent" in data
        assert "processing_time_ms" in data

    def test_chat_empty_query(self, client: TestClient) -> None:
        """Query kosong harus return 422."""
        response = client.post(
            "/api/v1/chat",
            json={
                "user_query": "",
                "store_id": "toko-demo",
            },
        )
        assert response.status_code == 422

    def test_chat_invalid_store_id(self, client: TestClient) -> None:
        """Store ID dengan karakter ilegal harus return 422."""
        response = client.post(
            "/api/v1/chat",
            json={
                "user_query": "Test query",
                "store_id": "invalid store!!",
            },
        )
        assert response.status_code == 422

    def test_chat_missing_fields(self, client: TestClient) -> None:
        """Request tanpa field wajib harus return 422."""
        response = client.post("/api/v1/chat", json={})
        assert response.status_code == 422
