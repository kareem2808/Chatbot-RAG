"""FastAPI routes: endpoint /chat dan /health.

Mendefinisikan endpoint REST API yang menghubungkan UI dengan
Core AI pipeline. Endpoint /chat mengintegrasikan:
- Rate limiter (dari src.utils)
- Query router (intent classification)
- RAG Fusion (query expansion)
- Hybrid retriever (semantic + BM25)
- CRAG filtering + LLM answer generation

Contoh request POST /chat:
    {
        "user_query": "Berapa harga kopi arabika?",
        "store_id": "toko-budi"
    }
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from src.core.document_processor import DocumentProcessor, RawDocument
from src.core.llm_engine import LLMEngine
from src.core.retriever import HybridRetriever
from src.core.router import QueryIntent, QueryRouter
from src.utils.auth import verify_api_key
from src.utils.config import Settings, get_settings
from src.utils.rate_limiter import RateLimiter, rate_limit_dependency

logger: logging.Logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Request / Response Schemas
# ──────────────────────────────────────────────
class ChatRequest(BaseModel):
    """Schema request untuk endpoint /chat.

    Attributes:
        user_query: Pertanyaan pengguna.
        store_id: ID collection ChromaDB (mewakili toko/klien UMKM).
    """

    user_query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Pertanyaan pengguna",
        examples=["Berapa harga kopi arabika?"],
    )
    store_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="ID toko/collection (alfanumerik, underscore, dash)",
        examples=["toko-budi"],
    )


class IngestDocument(BaseModel):
    """Schema dokumen tunggal untuk ingestion.

    Attributes:
        text: Konten teks dokumen.
        metadata: Metadata tambahan (opsional).
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=50_000,
        description="Konten teks dokumen",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata tambahan (source, author, dsb.)",
    )


class IngestRequest(BaseModel):
    """Schema request untuk endpoint /ingest.

    Attributes:
        store_id: ID collection ChromaDB target.
        documents: Daftar dokumen yang akan di-ingest.
    """

    store_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="ID toko/collection target",
        examples=["toko-budi"],
    )
    documents: list[IngestDocument] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Daftar dokumen yang akan di-ingest",
    )


class IngestResponse(BaseModel):
    """Schema response untuk endpoint /ingest.

    Attributes:
        store_id: ID toko target.
        documents_received: Jumlah dokumen yang diterima.
        chunks_created: Jumlah chunk yang dihasilkan.
        processing_time_ms: Waktu pemrosesan dalam milidetik.
    """

    store_id: str
    documents_received: int
    chunks_created: int
    processing_time_ms: float


class ChatResponse(BaseModel):
    """Schema response untuk endpoint /chat.

    Attributes:
        answer: Jawaban yang dihasilkan oleh AI.
        store_id: ID toko yang ditargetkan.
        intent: Klasifikasi intent query.
        num_sources: Jumlah chunk konteks yang digunakan.
        processing_time_ms: Waktu pemrosesan dalam milidetik.
    """

    answer: str
    store_id: str
    intent: str
    num_sources: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Schema response untuk endpoint /health.

    Attributes:
        status: Status keseluruhan server.
        chroma_db: Status koneksi ChromaDB.
        environment: Environment deployment aktif.
        version: Versi aplikasi.
    """

    status: str
    chroma_db: str
    environment: str
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Schema response untuk error.

    Attributes:
        error: Kode error.
        message: Pesan error yang user-friendly.
        detail: Detail teknis tambahan (opsional).
    """

    error: str
    message: str
    detail: str | None = None


# ──────────────────────────────────────────────
#  Shared State (di-inject via dependency)
# ──────────────────────────────────────────────
_retriever: HybridRetriever | None = None
_query_router: QueryRouter | None = None
_llm_engine: LLMEngine | None = None
_rate_limiter: RateLimiter | None = None
_document_processor: DocumentProcessor | None = None


def init_dependencies(
    retriever: HybridRetriever,
    query_router: QueryRouter,
    llm_engine: LLMEngine,
    rate_limiter: RateLimiter,
    document_processor: DocumentProcessor,
) -> None:
    """Inisialisasi dependency yang digunakan oleh routes.

    Dipanggil dari main.py saat startup.

    Args:
        retriever: Instance HybridRetriever.
        query_router: Instance QueryRouter.
        llm_engine: Instance GeminiEngine.
        rate_limiter: Instance RateLimiter.
        document_processor: Instance DocumentProcessor.
    """
    global _retriever, _query_router, _llm_engine, _rate_limiter, _document_processor  # noqa: PLW0603
    _retriever = retriever
    _query_router = query_router
    _llm_engine = llm_engine
    _rate_limiter = rate_limiter
    _document_processor = document_processor


def _get_retriever() -> HybridRetriever:
    if _retriever is None:
        raise RuntimeError("HybridRetriever belum diinisialisasi")
    return _retriever


def _get_query_router() -> QueryRouter:
    if _query_router is None:
        raise RuntimeError("QueryRouter belum diinisialisasi")
    return _query_router


def _get_llm_engine() -> GeminiEngine:
    if _llm_engine is None:
        raise RuntimeError("GeminiEngine belum diinisialisasi")
    return _llm_engine


def _get_rate_limiter() -> RateLimiter:
    if _rate_limiter is None:
        raise RuntimeError("RateLimiter belum diinisialisasi")
    return _rate_limiter


def _get_document_processor() -> DocumentProcessor:
    if _document_processor is None:
        raise RuntimeError("DocumentProcessor belum diinisialisasi")
    return _document_processor


# ──────────────────────────────────────────────
#  Router
# ──────────────────────────────────────────────
router = APIRouter(tags=["RAG Chat"])


# ── POST /chat ───────────────────────────────
@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        429: {"model": ErrorResponse, "description": "Rate Limited"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
    summary="Tanya jawab AI untuk toko UMKM",
    description=(
        "Kirim pertanyaan pengguna beserta store_id. "
        "Sistem akan mengklasifikasi intent, mengambil konteks "
        "dari ChromaDB, dan menghasilkan jawaban menggunakan Gemini."
    ),
)
async def chat_endpoint(
    request: Request,
    body: ChatRequest,
    retriever: HybridRetriever = Depends(_get_retriever),
    query_router: QueryRouter = Depends(_get_query_router),
    llm_engine: GeminiEngine = Depends(_get_llm_engine),
    limiter: RateLimiter = Depends(_get_rate_limiter),
    _api_key: str | None = Depends(verify_api_key),
) -> ChatResponse:
    """Endpoint utama untuk RAG chat pipeline.

    Flow:
    1. Rate limiting check
    2. Intent classification (router)
    3. RAG Fusion query expansion
    4. Hybrid search (semantic + BM25)
    5. CRAG context filtering
    6. Answer generation
    """
    start_time: float = time.monotonic()

    # ── 1. Rate Limiting ─────────────────────
    client_id: str = _resolve_client_id(request)
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

    try:
        # ── 2. Intent Classification ─────────
        intent: QueryIntent = await query_router.classify(body.user_query)

        if intent == QueryIntent.OUT_OF_DOMAIN:
            elapsed: float = (time.monotonic() - start_time) * 1000
            return ChatResponse(
                answer=(
                    "Maaf, pertanyaan Anda di luar cakupan layanan toko ini. "
                    "Silakan ajukan pertanyaan seputar produk, harga, atau "
                    "layanan toko."
                ),
                store_id=body.store_id,
                intent=intent.value,
                num_sources=0,
                processing_time_ms=round(elapsed, 2),
            )

        # ── 3. RAG Fusion: Query Expansion ───
        query_variations: list[str] = (
            await llm_engine.generate_query_variations(body.user_query)
        )

        # ── 4. Hybrid Search ─────────────────
        all_chunks: list[str] = []
        seen_ids: set[str] = set()

        for query_var in query_variations:
            try:
                results = await retriever.search(
                    collection_name=body.store_id,
                    query=query_var,
                    top_k=5,
                )
                for result in results:
                    if result.document_id not in seen_ids:
                        seen_ids.add(result.document_id)
                        all_chunks.append(result.text)
            except ValueError:
                # Collection tidak ditemukan
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error": "collection_not_found",
                        "message": (
                            f"Toko '{body.store_id}' tidak ditemukan. "
                            f"Pastikan store_id valid."
                        ),
                    },
                )

        if not all_chunks:
            elapsed = (time.monotonic() - start_time) * 1000
            return ChatResponse(
                answer=(
                    "Maaf, saya tidak menemukan informasi yang relevan "
                    "untuk pertanyaan Anda. Silakan coba dengan "
                    "pertanyaan yang lebih spesifik."
                ),
                store_id=body.store_id,
                intent=intent.value,
                num_sources=0,
                processing_time_ms=round(elapsed, 2),
            )

        # ── 5. CRAG: Filter Relevansi ────────
        relevant_chunks: list[str] = (
            await llm_engine.filter_relevant_chunks(
                query=body.user_query,
                chunks=all_chunks[:10],  # Limit untuk efisiensi
            )
        )

        # Fallback ke semua chunks jika filter terlalu agresif
        context_chunks: list[str] = (
            relevant_chunks if relevant_chunks else all_chunks[:5]
        )

        # ── 6. Generate Answer ───────────────
        answer: str = await llm_engine.generate_answer(
            query=body.user_query,
            context_chunks=context_chunks,
        )

        elapsed = (time.monotonic() - start_time) * 1000

        logger.info(
            "Chat request berhasil diproses",
            extra={
                "store_id": body.store_id,
                "intent": intent.value,
                "num_query_variations": len(query_variations),
                "num_retrieved": len(all_chunks),
                "num_relevant": len(relevant_chunks),
                "processing_time_ms": round(elapsed, 2),
            },
        )

        return ChatResponse(
            answer=answer,
            store_id=body.store_id,
            intent=intent.value,
            num_sources=len(context_chunks),
            processing_time_ms=round(elapsed, 2),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Error saat memproses chat request",
            extra={
                "store_id": body.store_id,
                "query_preview": body.user_query[:80],
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "internal_error",
                "message": (
                    "Terjadi kesalahan internal. Silakan coba lagi "
                    "beberapa saat kemudian."
                ),
            },
        ) from exc


# ── GET /health ──────────────────────────────
@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check server",
    description=(
        "Periksa status server dan koneksi ChromaDB. "
        "Penting untuk monitoring Docker/VPS deployment."
    ),
)
async def health_check(
    retriever: HybridRetriever = Depends(_get_retriever),
) -> HealthResponse:
    """Health check endpoint untuk monitoring.

    Memeriksa:
    - Status server FastAPI
    - Koneksi ke ChromaDB (heartbeat)
    """
    settings: Settings = get_settings()

    # Cek koneksi ChromaDB
    chroma_status: str = "unknown"
    try:
        collections = await retriever.list_collections()
        chroma_status = f"connected ({len(collections)} collections)"
    except Exception as exc:
        chroma_status = f"error: {exc!s}"
        logger.warning(
            "ChromaDB health check gagal",
            extra={"error": str(exc)},
        )

    return HealthResponse(
        status="healthy",
        chroma_db=chroma_status,
        environment=settings.environment,
    )


# ── POST /ingest ─────────────────────────────
@router.post(
    "/ingest",
    response_model=IngestResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        429: {"model": ErrorResponse, "description": "Rate Limited"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
    summary="Ingest dokumen ke collection toko",
    description=(
        "Upload dokumen teks ke collection ChromaDB. "
        "Dokumen akan dipecah menjadi chunk-chunk optimal "
        "dan di-index untuk semantic + keyword search."
    ),
)
async def ingest_endpoint(
    request: Request,
    body: IngestRequest,
    retriever: HybridRetriever = Depends(_get_retriever),
    processor: DocumentProcessor = Depends(_get_document_processor),
    limiter: RateLimiter = Depends(_get_rate_limiter),
    _api_key: str | None = Depends(verify_api_key),
) -> IngestResponse:
    """Endpoint untuk memasukkan dokumen ke RAG pipeline.

    Flow:
    1. Rate limiting check
    2. Konversi ke RawDocument
    3. Proses chunking via DocumentProcessor
    4. Upsert chunks ke ChromaDB via HybridRetriever
    """
    start_time: float = time.monotonic()

    # ── 1. Rate Limiting ─────────────────────
    client_id: str = _resolve_client_id(request)
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

    try:
        # ── 2. Konversi ke RawDocument ───────
        raw_documents: list[RawDocument] = [
            RawDocument(text=doc.text, metadata=doc.metadata)
            for doc in body.documents
        ]

        # ── 3. Chunking ─────────────────────
        processed_chunks = await processor.process_documents(raw_documents)

        if not processed_chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "no_valid_documents",
                    "message": "Semua dokumen kosong atau tidak valid.",
                },
            )

        # ── 4. Upsert ke ChromaDB ───────────
        chunk_texts: list[str] = [c.text for c in processed_chunks]
        chunk_ids: list[str] = [c.chunk_id for c in processed_chunks]
        chunk_metadatas: list[dict[str, Any]] = [
            c.metadata for c in processed_chunks
        ]

        await retriever.add_documents(
            collection_name=body.store_id,
            documents=chunk_texts,
            metadatas=chunk_metadatas,
            ids=chunk_ids,
        )

        elapsed: float = (time.monotonic() - start_time) * 1000

        logger.info(
            "Dokumen berhasil di-ingest",
            extra={
                "store_id": body.store_id,
                "documents_received": len(body.documents),
                "chunks_created": len(processed_chunks),
                "processing_time_ms": round(elapsed, 2),
            },
        )

        return IngestResponse(
            store_id=body.store_id,
            documents_received=len(body.documents),
            chunks_created=len(processed_chunks),
            processing_time_ms=round(elapsed, 2),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Error saat ingest dokumen",
            extra={
                "store_id": body.store_id,
                "num_documents": len(body.documents),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "ingest_error",
                "message": (
                    "Terjadi kesalahan saat memproses dokumen. "
                    "Silakan coba lagi."
                ),
            },
        ) from exc


# ── GET /collections ──────────────────────────
@router.get(
    "/collections",
    response_model=list[str],
    summary="Daftar semua collection (toko)",
    description="Mengembalikan daftar nama semua collection yang tersedia di ChromaDB.",
)
async def list_collections(
    retriever: HybridRetriever = Depends(_get_retriever),
    _api_key: str | None = Depends(verify_api_key),
) -> list[str]:
    """Endpoint untuk mendapatkan daftar semua collection."""
    try:
        collections: list[str] = await retriever.list_collections()
        return collections
    except Exception as exc:
        logger.error(
            "Error saat mengambil daftar collections",
            extra={"error_type": type(exc).__name__, "error": str(exc)},
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "internal_error",
                "message": "Gagal mengambil daftar collections.",
            },
        ) from exc


# ── DELETE /collections/{store_id} ───────────
@router.delete(
    "/collections/{store_id}",
    response_model=dict[str, str],
    summary="Hapus collection toko",
    description="Hapus collection beserta seluruh dokumen di dalamnya.",
)
async def delete_collection(
    store_id: str,
    retriever: HybridRetriever = Depends(_get_retriever),
    _api_key: str | None = Depends(verify_api_key),
) -> dict[str, str]:
    """Endpoint untuk menghapus collection tertentu."""
    try:
        await retriever.delete_collection(store_id)
        logger.info(
            "Collection dihapus via API",
            extra={"store_id": store_id},
        )
        return {"status": "deleted", "store_id": store_id}
    except Exception as exc:
        logger.error(
            "Error saat menghapus collection",
            extra={"store_id": store_id, "error": str(exc)},
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "delete_failed",
                "message": (
                    f"Gagal menghapus collection '{store_id}'. "
                    f"Pastikan collection ada."
                ),
            },
        ) from exc


# ── Helper ───────────────────────────────────
def _resolve_client_id(request: Request) -> str:
    """Resolve client identifier dari request untuk rate limiting.

    Args:
        request: FastAPI Request object.

    Returns:
        String identifier client.
    """
    # Cek header X-Forwarded-For (untuk reverse proxy)
    forwarded: str | None = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()

    if request.client is not None:
        return request.client.host

    return "unknown"
