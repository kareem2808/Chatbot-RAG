"""FastAPI application entry point.

Inisialisasi FastAPI, konfigurasi CORS middleware, dan wiring
seluruh dependency (retriever, router, LLM engine, rate limiter).

Jalankan dengan:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

Production:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from src.api.routes import init_dependencies, router
from src.core.document_processor import DocumentProcessor
from src.core.llm_engine import LLMEngine
from src.core.retriever import HybridRetriever
from src.core.router import QueryRouter
from src.utils.config import Settings, get_settings
from src.utils.logger import get_logger
from src.utils.rate_limiter import RateLimiter

# ──────────────────────────────────────────────
#  Logger (dikonfigurasi sedini mungkin)
# ──────────────────────────────────────────────
logger: logging.Logger = get_logger(__name__)


# ──────────────────────────────────────────────
#  Request ID Middleware
# ──────────────────────────────────────────────
class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware yang menambahkan unique Request ID ke setiap request.

    Menghasilkan UUID v4 per request dan menyertakannya di:
    - `request.state.request_id` (untuk akses internal)
    - Header `X-Request-ID` di response (untuk tracing)
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request, attach ID, log timing."""
        # Generate atau ambil dari header (jika sudah ada dari reverse proxy)
        request_id: str = request.headers.get(
            "X-Request-ID", str(uuid.uuid4())
        )
        request.state.request_id = request_id

        start_time: float = time.monotonic()

        response: Response = await call_next(request)

        # Hitung durasi
        duration_ms: float = (time.monotonic() - start_time) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"

        # Log request/response
        logger.info(
            "Request selesai",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )

        return response


# ──────────────────────────────────────────────
#  Lifespan (startup & shutdown)
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle: startup dan shutdown.

    Startup:
    - Load konfigurasi dari .env
    - Inisialisasi semua dependency (retriever, router, engine, limiter)
    - Wire dependency ke routes

    Shutdown:
    - Cleanup resources
    """
    settings: Settings = get_settings()

    logger.info(
        "Memulai aplikasi",
        extra={
            "app_name": settings.app_name,
            "environment": settings.environment,
        },
    )

    # ── Inisialisasi dependency ──────────────
    retriever = HybridRetriever(
        persist_directory=str(settings.chroma_db_path),
    )

    query_router = QueryRouter(
        provider=settings.llm_provider,
        api_key=settings.gemini_api_key.get_secret_value(),
        groq_api_key=settings.groq_api_key.get_secret_value(),
        groq_model_name=settings.groq_model_name,
        max_retries=settings.max_retries,
    )

    llm_engine = LLMEngine(
        provider=settings.llm_provider,
        api_key=settings.gemini_api_key.get_secret_value(),
        groq_api_key=settings.groq_api_key.get_secret_value(),
        groq_model_name=settings.groq_model_name,
        max_retries=settings.max_retries,
    )

    rate_limiter = RateLimiter(
        max_requests=settings.rate_limit_per_minute,
        window_seconds=60,
    )

    document_processor = DocumentProcessor()

    # Wire ke routes
    init_dependencies(
        retriever=retriever,
        query_router=query_router,
        llm_engine=llm_engine,
        rate_limiter=rate_limiter,
        document_processor=document_processor,
    )

    # Log auth status
    from src.utils.auth import is_auth_enabled
    if is_auth_enabled():
        logger.info("API key authentication AKTIF")
    else:
        logger.info("API key authentication NONAKTIF (API_KEYS kosong)")

    logger.info("Semua dependency berhasil diinisialisasi")

    yield  # ← Aplikasi berjalan

    # ── Shutdown / Cleanup ───────────────────
    logger.info("Aplikasi sedang dimatikan, membersihkan resources...")
    await rate_limiter.cleanup_stale_buckets()
    logger.info("Cleanup selesai, aplikasi dimatikan")


# ──────────────────────────────────────────────
#  FastAPI App
# ──────────────────────────────────────────────
def create_app() -> FastAPI:
    """Factory function untuk membuat instance FastAPI.

    Returns:
        Instance FastAPI yang terkonfigurasi lengkap.
    """
    settings: Settings = get_settings()

    application = FastAPI(
        title="RAG UMKM Assistant API",
        description=(
            "REST API untuk asisten AI toko UMKM berbasis "
            "Retrieval-Augmented Generation (RAG) dengan "
            "Gemini 1.5 Flash."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── Middleware (order matters — outermost first) ──
    application.add_middleware(RequestIDMiddleware)
    _configure_cors(application, settings)

    # ── Register Routes ──────────────────────
    application.include_router(router, prefix="/api/v1")

    return application


def _configure_cors(app: FastAPI, settings: Settings) -> None:
    """Konfigurasi CORS middleware berdasarkan environment.

    Prioritas:
    1. cors_origins dari .env (jika diisi)
    2. Fallback: permissive untuk dev, restricted untuk production

    Args:
        app: Instance FastAPI.
        settings: Konfigurasi aplikasi.
    """
    # Gunakan cors_origins dari config jika tersedia
    if settings.cors_origins.strip():
        allowed_origins: list[str] = [
            o.strip() for o in settings.cors_origins.split(",") if o.strip()
        ]
    elif settings.environment == "production":
        # Production default: batasi origin
        allowed_origins = [
            "https://yourdomain.com",
            "https://app.yourdomain.com",
        ]
    else:
        # Development & staging: permissive
        allowed_origins = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        max_age=600,  # Cache preflight 10 menit
    )

    logger.info(
        "CORS middleware dikonfigurasi",
        extra={
            "environment": settings.environment,
            "allowed_origins": allowed_origins,
        },
    )


# ──────────────────────────────────────────────
#  App Instance
# ──────────────────────────────────────────────
app: FastAPI = create_app()
