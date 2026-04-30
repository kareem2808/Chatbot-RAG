"""Hybrid Retriever: Semantic Search (ChromaDB) + BM25 Keyword Search.

Mengombinasikan vector similarity search dari ChromaDB dengan
exact keyword matching BM25 menggunakan Reciprocal Rank Fusion (RRF)
untuk hasil retrieval yang lebih robust.

Mendukung multi-tenancy dasar melalui parameter collection_name
yang mewakili setiap toko/klien UMKM.

Contoh penggunaan:
    >>> from src.core.retriever import HybridRetriever
    >>> retriever = HybridRetriever(persist_directory="./data/chroma_db")
    >>> await retriever.add_documents(
    ...     collection_name="toko-budi",
    ...     documents=["Kami menjual kopi arabika...", "Jam buka: 08.00-17.00"],
    ...     metadatas=[{"source": "faq"}, {"source": "info"}],
    ...     ids=["doc-1", "doc-2"],
    ... )
    >>> results = await retriever.search(
    ...     collection_name="toko-budi",
    ...     query="jam buka toko",
    ...     top_k=5,
    ... )
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logger: logging.Logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
#  Konstanta
# ──────────────────────────────────────────────
_DEFAULT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
_DEFAULT_PERSIST_DIR: str = "./data/chroma_db"
_DEFAULT_TOP_K: int = 5
_RRF_K: int = 60  # Reciprocal Rank Fusion constant


# ──────────────────────────────────────────────
#  Result Type
# ──────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Hasil pencarian tunggal dari hybrid retriever.

    Attributes:
        document_id: ID dokumen di ChromaDB.
        text: Konten teks dokumen.
        metadata: Metadata terkait dokumen.
        score: Skor gabungan RRF (semakin tinggi semakin relevan).
        semantic_rank: Peringkat dari semantic search (None jika tidak ada).
        keyword_rank: Peringkat dari BM25 search (None jika tidak ada).
    """

    document_id: str
    text: str
    metadata: dict[str, Any]
    score: float
    semantic_rank: int | None = None
    keyword_rank: int | None = None


# ──────────────────────────────────────────────
#  Hybrid Retriever
# ──────────────────────────────────────────────
@dataclass
class HybridRetriever:
    """Retriever hybrid yang menggabungkan semantic + keyword search.

    Menggunakan ChromaDB untuk persistent vector storage dan BM25
    untuk exact keyword matching. Hasil digabungkan menggunakan
    Reciprocal Rank Fusion (RRF).

    Args:
        persist_directory: Path ke direktori ChromaDB.
        embedding_model_name: Nama model sentence-transformers.
        semantic_weight: Bobot untuk semantic search dalam RRF (0.0-1.0).
        keyword_weight: Bobot untuk keyword search dalam RRF (0.0-1.0).
    """

    persist_directory: str = _DEFAULT_PERSIST_DIR
    embedding_model_name: str = _DEFAULT_EMBEDDING_MODEL
    semantic_weight: float = 0.6
    keyword_weight: float = 0.4

    # Internal state (tidak di-expose ke constructor)
    _chroma_client: chromadb.ClientAPI = field(init=False, repr=False)
    _embedding_model: SentenceTransformer = field(init=False, repr=False)
    _bm25_indices: dict[str, _BM25Index] = field(
        init=False, default_factory=dict, repr=False
    )
    _lock: asyncio.Lock = field(
        init=False, default_factory=asyncio.Lock, repr=False
    )

    def __post_init__(self) -> None:
        """Inisialisasi ChromaDB client dan embedding model."""
        if not (0.0 <= self.semantic_weight <= 1.0):
            msg = f"semantic_weight harus antara 0.0-1.0, got {self.semantic_weight}"
            raise ValueError(msg)
        if not (0.0 <= self.keyword_weight <= 1.0):
            msg = f"keyword_weight harus antara 0.0-1.0, got {self.keyword_weight}"
            raise ValueError(msg)

        logger.info(
            "Menginisialisasi HybridRetriever",
            extra={
                "persist_directory": self.persist_directory,
                "embedding_model": self.embedding_model_name,
            },
        )

        # ChromaDB persistent client
        self._chroma_client = chromadb.PersistentClient(
            path=self.persist_directory
        )

        # Embedding model (lokal, no API calls)
        self._embedding_model = SentenceTransformer(
            self.embedding_model_name
        )

        logger.info("HybridRetriever siap digunakan")

    # ── Public API: Document Management ──────
    async def add_documents(
        self,
        collection_name: str,
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> int:
        """Tambahkan dokumen ke collection tertentu (multi-tenant).

        Args:
            collection_name: Nama collection (mewakili toko/klien UMKM).
            documents: Daftar teks dokumen.
            metadatas: Metadata per dokumen (opsional).
            ids: ID per dokumen (opsional, auto-generate jika kosong).

        Returns:
            Jumlah dokumen yang berhasil ditambahkan.

        Raises:
            ValueError: Jika documents kosong atau panjang tidak konsisten.
        """
        if not documents:
            msg = "Daftar dokumen tidak boleh kosong"
            raise ValueError(msg)

        if ids and len(ids) != len(documents):
            msg = (
                f"Jumlah ids ({len(ids)}) harus sama dengan "
                f"jumlah documents ({len(documents)})"
            )
            raise ValueError(msg)

        if metadatas and len(metadatas) != len(documents):
            msg = (
                f"Jumlah metadatas ({len(metadatas)}) harus sama dengan "
                f"jumlah documents ({len(documents)})"
            )
            raise ValueError(msg)

        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()

        # Generate embeddings di thread pool (CPU-intensive)
        embeddings: list[list[float]] = await loop.run_in_executor(
            None,
            lambda: self._embedding_model.encode(
                documents, show_progress_bar=False
            ).tolist(),
        )

        # Generate IDs jika tidak disediakan
        if ids is None:
            ids = [f"{collection_name}_doc-{i}" for i in range(len(documents))]

        # Upsert ke ChromaDB
        async with self._lock:
            collection: Collection = self._chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            await loop.run_in_executor(
                None,
                lambda: collection.upsert(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                ),
            )

            # Rebuild BM25 index untuk collection ini
            await self._rebuild_bm25_index(collection_name, collection)

        logger.info(
            "Dokumen berhasil ditambahkan",
            extra={
                "collection": collection_name,
                "num_documents": len(documents),
            },
        )
        return len(documents)

    async def search(
        self,
        collection_name: str,
        query: str,
        top_k: int = _DEFAULT_TOP_K,
    ) -> list[RetrievalResult]:
        """Hybrid search: gabungan semantic + BM25 dengan RRF.

        Args:
            collection_name: Nama collection target.
            query: Query pencarian dari pengguna.
            top_k: Jumlah hasil teratas yang dikembalikan.

        Returns:
            Daftar ``RetrievalResult`` terurut berdasarkan skor RRF.

        Raises:
            ValueError: Jika query kosong.
            CollectionNotFoundError: Jika collection tidak ditemukan.
        """
        query = query.strip()
        if not query:
            msg = "Query pencarian tidak boleh kosong"
            raise ValueError(msg)

        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()

        # Ambil collection
        try:
            collection: Collection = self._chroma_client.get_collection(
                name=collection_name
            )
        except Exception as exc:
            msg = f"Collection '{collection_name}' tidak ditemukan"
            logger.error(msg, extra={"error": str(exc)})
            raise ValueError(msg) from exc

        # Fetch size — ambil lebih banyak dari top_k untuk reranking
        fetch_k: int = min(top_k * 3, collection.count() or top_k)
        if fetch_k == 0:
            return []

        # ── Semantic search (async) ──────────
        query_embedding: list[float] = await loop.run_in_executor(
            None,
            lambda: self._embedding_model.encode(
                query, show_progress_bar=False
            ).tolist(),
        )

        semantic_results: dict[str, Any] = await loop.run_in_executor(
            None,
            lambda: collection.query(
                query_embeddings=[query_embedding],
                n_results=fetch_k,
                include=["documents", "metadatas", "distances"],
            ),
        )

        # ── BM25 keyword search ─────────────
        keyword_results: list[tuple[str, str, dict[str, Any]]] = (
            await self._bm25_search(collection_name, collection, query, fetch_k)
        )

        # ── Reciprocal Rank Fusion ──────────
        fused: list[RetrievalResult] = self._reciprocal_rank_fusion(
            semantic_results=semantic_results,
            keyword_results=keyword_results,
            top_k=top_k,
        )

        logger.info(
            "Hybrid search selesai",
            extra={
                "collection": collection_name,
                "query_length": len(query),
                "num_results": len(fused),
            },
        )
        return fused

    async def delete_collection(self, collection_name: str) -> None:
        """Hapus collection beserta semua dokumennya.

        Args:
            collection_name: Nama collection yang akan dihapus.
        """
        async with self._lock:
            loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self._chroma_client.delete_collection(
                    name=collection_name
                ),
            )
            self._bm25_indices.pop(collection_name, None)

        logger.info(
            "Collection dihapus",
            extra={"collection": collection_name},
        )

    async def list_collections(self) -> list[str]:
        """Daftar semua collection (tenant) yang tersedia.

        Returns:
            Daftar nama collection.
        """
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        collections = await loop.run_in_executor(
            None, self._chroma_client.list_collections
        )
        return [str(c) for c in collections]

    # ── Internal: BM25 ───────────────────────
    async def _bm25_search(
        self,
        collection_name: str,
        collection: Collection,
        query: str,
        top_k: int,
    ) -> list[tuple[str, str, dict[str, Any]]]:
        """Jalankan BM25 keyword search pada collection.

        Args:
            collection_name: Nama collection.
            collection: ChromaDB Collection object.
            query: Query pencarian.
            top_k: Jumlah hasil teratas.

        Returns:
            List of (id, document, metadata) tuples.
        """
        async with self._lock:
            if collection_name not in self._bm25_indices:
                await self._rebuild_bm25_index(collection_name, collection)

        bm25_index: _BM25Index | None = self._bm25_indices.get(
            collection_name
        )
        if bm25_index is None or bm25_index.is_empty:
            return []

        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        results: list[tuple[str, str, dict[str, Any]]] = (
            await loop.run_in_executor(
                None, bm25_index.search, query, top_k
            )
        )
        return results

    async def _rebuild_bm25_index(
        self,
        collection_name: str,
        collection: Collection,
    ) -> None:
        """Rebuild BM25 index dari seluruh dokumen dalam collection.

        Args:
            collection_name: Nama collection.
            collection: ChromaDB Collection object.
        """
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        all_data: dict[str, Any] = await loop.run_in_executor(
            None,
            lambda: collection.get(include=["documents", "metadatas"]),
        )

        ids: list[str] = all_data.get("ids", [])
        documents: list[str] = all_data.get("documents", [])
        metadatas: list[dict[str, Any]] = all_data.get("metadatas", []) or [
            {} for _ in ids
        ]

        if not documents:
            self._bm25_indices[collection_name] = _BM25Index.empty()
            return

        self._bm25_indices[collection_name] = _BM25Index.build(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        logger.debug(
            "BM25 index direbuild",
            extra={
                "collection": collection_name,
                "num_documents": len(documents),
            },
        )

    # ── Internal: Reciprocal Rank Fusion ─────
    def _reciprocal_rank_fusion(
        self,
        semantic_results: dict[str, Any],
        keyword_results: list[tuple[str, str, dict[str, Any]]],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Gabungkan hasil semantic dan keyword search menggunakan RRF.

        Skor RRF = sum( weight / (k + rank) ) untuk setiap sumber.

        Args:
            semantic_results: Hasil query dari ChromaDB.
            keyword_results: Hasil dari BM25 search.
            top_k: Jumlah hasil akhir.

        Returns:
            Daftar RetrievalResult terurut berdasarkan skor.
        """
        scores: dict[str, float] = {}
        doc_data: dict[str, tuple[str, dict[str, Any]]] = {}
        semantic_ranks: dict[str, int] = {}
        keyword_ranks: dict[str, int] = {}

        # ── Skor dari semantic search ────────
        sem_ids: list[str] = (semantic_results.get("ids") or [[]])[0]
        sem_docs: list[str] = (semantic_results.get("documents") or [[]])[0]
        sem_metas: list[dict[str, Any]] = (
            semantic_results.get("metadatas") or [[]]
        )[0]

        for rank, (doc_id, text, meta) in enumerate(
            zip(sem_ids, sem_docs, sem_metas, strict=False)
        ):
            rrf_score: float = self.semantic_weight / (_RRF_K + rank + 1)
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_score
            doc_data[doc_id] = (text, meta or {})
            semantic_ranks[doc_id] = rank + 1

        # ── Skor dari keyword search ─────────
        for rank, (doc_id, text, meta) in enumerate(keyword_results):
            rrf_score = self.keyword_weight / (_RRF_K + rank + 1)
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_score
            if doc_id not in doc_data:
                doc_data[doc_id] = (text, meta)
            keyword_ranks[doc_id] = rank + 1

        # ── Sort dan truncate ────────────────
        sorted_ids: list[str] = sorted(
            scores, key=lambda x: scores[x], reverse=True
        )[:top_k]

        results: list[RetrievalResult] = []
        for doc_id in sorted_ids:
            text, meta = doc_data[doc_id]
            results.append(
                RetrievalResult(
                    document_id=doc_id,
                    text=text,
                    metadata=meta,
                    score=scores[doc_id],
                    semantic_rank=semantic_ranks.get(doc_id),
                    keyword_rank=keyword_ranks.get(doc_id),
                )
            )

        return results


# ──────────────────────────────────────────────
#  BM25 Index (internal)
# ──────────────────────────────────────────────
@dataclass
class _BM25Index:
    """Wrapper internal untuk BM25 index per-collection.

    Attributes:
        bm25: Instance BM25Okapi.
        ids: Daftar document IDs (sejajar dengan corpus).
        documents: Daftar teks dokumen asli.
        metadatas: Daftar metadata per dokumen.
        corpus_tokens: Corpus yang sudah di-tokenize.
    """

    bm25: BM25Okapi | None
    ids: list[str]
    documents: list[str]
    metadatas: list[dict[str, Any]]
    corpus_tokens: list[list[str]]

    @property
    def is_empty(self) -> bool:
        """Cek apakah index kosong."""
        return self.bm25 is None or len(self.ids) == 0

    @classmethod
    def empty(cls) -> _BM25Index:
        """Buat index kosong."""
        return cls(
            bm25=None,
            ids=[],
            documents=[],
            metadatas=[],
            corpus_tokens=[],
        )

    @classmethod
    def build(
        cls,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> _BM25Index:
        """Bangun BM25 index dari dokumen.

        Args:
            ids: Daftar document IDs.
            documents: Daftar teks dokumen.
            metadatas: Daftar metadata.

        Returns:
            Instance ``_BM25Index`` yang siap digunakan.
        """
        corpus_tokens: list[list[str]] = [
            _tokenize(doc) for doc in documents
        ]
        bm25 = BM25Okapi(corpus_tokens)
        return cls(
            bm25=bm25,
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            corpus_tokens=corpus_tokens,
        )

    def search(
        self,
        query: str,
        top_k: int,
    ) -> list[tuple[str, str, dict[str, Any]]]:
        """Cari dokumen menggunakan BM25.

        Args:
            query: Query pencarian.
            top_k: Jumlah hasil teratas.

        Returns:
            List of (id, document, metadata) tuples, terurut dari
            yang paling relevan.
        """
        if self.bm25 is None:
            return []

        query_tokens: list[str] = _tokenize(query)
        scores: list[float] = self.bm25.get_scores(query_tokens).tolist()

        # Dapatkan top_k indeks dengan skor tertinggi
        indexed_scores: list[tuple[int, float]] = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        results: list[tuple[str, str, dict[str, Any]]] = []
        for idx, score in indexed_scores:
            if score > 0.0:
                results.append(
                    (self.ids[idx], self.documents[idx], self.metadatas[idx])
                )

        return results


# ──────────────────────────────────────────────
#  Tokenizer sederhana
# ──────────────────────────────────────────────
def _tokenize(text: str) -> list[str]:
    """Tokenize teks menjadi daftar kata lowercase.

    Menghapus karakter non-alfanumerik dan mengubah ke lowercase
    untuk normalisasi sederhana yang cocok untuk BM25.

    Args:
        text: Teks yang akan di-tokenize.

    Returns:
        Daftar token (kata) lowercase.
    """
    return re.findall(r"\w+", text.lower())
