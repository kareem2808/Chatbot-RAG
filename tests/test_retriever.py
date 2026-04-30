"""Tests untuk HybridRetriever."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.core.retriever import HybridRetriever, RetrievalResult


@pytest.fixture
def temp_chroma_dir(tmp_path: Path) -> str:
    """Fixture: direktori ChromaDB sementara."""
    chroma_dir = tmp_path / "test_chroma_db"
    chroma_dir.mkdir()
    return str(chroma_dir)


@pytest.fixture
def retriever(temp_chroma_dir: str) -> HybridRetriever:
    """Fixture: HybridRetriever dengan direktori sementara."""
    return HybridRetriever(persist_directory=temp_chroma_dir)


# ──────────────────────────────────────────────
#  Inisialisasi
# ──────────────────────────────────────────────
class TestRetrieverInit:
    """Test inisialisasi HybridRetriever."""

    def test_default_weights(self, retriever: HybridRetriever) -> None:
        """Default weights harus 0.6 semantic, 0.4 keyword."""
        assert retriever.semantic_weight == 0.6
        assert retriever.keyword_weight == 0.4

    def test_invalid_semantic_weight(self, temp_chroma_dir: str) -> None:
        """semantic_weight di luar 0-1 harus raise ValueError."""
        with pytest.raises(ValueError, match="semantic_weight"):
            HybridRetriever(
                persist_directory=temp_chroma_dir,
                semantic_weight=1.5,
            )

    def test_invalid_keyword_weight(self, temp_chroma_dir: str) -> None:
        """keyword_weight di luar 0-1 harus raise ValueError."""
        with pytest.raises(ValueError, match="keyword_weight"):
            HybridRetriever(
                persist_directory=temp_chroma_dir,
                keyword_weight=-0.1,
            )


# ──────────────────────────────────────────────
#  add_documents
# ──────────────────────────────────────────────
class TestAddDocuments:
    """Test metode add_documents."""

    @pytest.mark.asyncio
    async def test_add_single_document(
        self, retriever: HybridRetriever
    ) -> None:
        """Harus bisa menambah 1 dokumen."""
        count = await retriever.add_documents(
            collection_name="test-store",
            documents=["Kopi arabika Rp 45.000"],
            metadatas=[{"source": "test"}],
            ids=["doc-1"],
        )
        assert count == 1

    @pytest.mark.asyncio
    async def test_add_multiple_documents(
        self, retriever: HybridRetriever
    ) -> None:
        """Harus bisa menambah banyak dokumen."""
        docs = ["Dokumen satu", "Dokumen dua", "Dokumen tiga"]
        count = await retriever.add_documents(
            collection_name="test-store",
            documents=docs,
        )
        assert count == 3

    @pytest.mark.asyncio
    async def test_add_empty_raises(
        self, retriever: HybridRetriever
    ) -> None:
        """Daftar dokumen kosong harus raise ValueError."""
        with pytest.raises(ValueError, match="kosong"):
            await retriever.add_documents(
                collection_name="test-store",
                documents=[],
            )

    @pytest.mark.asyncio
    async def test_mismatched_ids_raises(
        self, retriever: HybridRetriever
    ) -> None:
        """Jumlah ids != jumlah documents harus raise ValueError."""
        with pytest.raises(ValueError, match="sama dengan"):
            await retriever.add_documents(
                collection_name="test-store",
                documents=["doc1", "doc2"],
                ids=["id1"],
            )

    @pytest.mark.asyncio
    async def test_mismatched_metadatas_raises(
        self, retriever: HybridRetriever
    ) -> None:
        """Jumlah metadatas != jumlah documents harus raise ValueError."""
        with pytest.raises(ValueError, match="sama dengan"):
            await retriever.add_documents(
                collection_name="test-store",
                documents=["doc1", "doc2"],
                metadatas=[{"a": 1}],
            )


# ──────────────────────────────────────────────
#  search
# ──────────────────────────────────────────────
class TestSearch:
    """Test metode search."""

    @pytest.mark.asyncio
    async def test_search_returns_results(
        self, retriever: HybridRetriever
    ) -> None:
        """Search harus mengembalikan hasil setelah dokumen ditambahkan."""
        await retriever.add_documents(
            collection_name="test-store",
            documents=[
                "Kopi arabika premium Rp 45.000 per pack",
                "Jam buka toko Senin-Jumat 08.00-17.00",
                "Kami menerima pembayaran tunai dan transfer",
            ],
            ids=["doc-1", "doc-2", "doc-3"],
        )

        results = await retriever.search(
            collection_name="test-store",
            query="harga kopi",
            top_k=3,
        )

        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)
        # Dokumen kopi harus ada di hasil
        kopi_found = any("kopi" in r.text.lower() for r in results)
        assert kopi_found

    @pytest.mark.asyncio
    async def test_search_empty_query_raises(
        self, retriever: HybridRetriever
    ) -> None:
        """Query kosong harus raise ValueError."""
        with pytest.raises(ValueError, match="kosong"):
            await retriever.search(
                collection_name="test-store",
                query="",
            )

    @pytest.mark.asyncio
    async def test_search_nonexistent_collection(
        self, retriever: HybridRetriever
    ) -> None:
        """Search di collection yang tidak ada harus raise ValueError."""
        with pytest.raises(ValueError, match="tidak ditemukan"):
            await retriever.search(
                collection_name="nonexistent-store",
                query="test query",
            )

    @pytest.mark.asyncio
    async def test_search_result_has_scores(
        self, retriever: HybridRetriever
    ) -> None:
        """Hasil search harus memiliki skor RRF."""
        await retriever.add_documents(
            collection_name="score-test",
            documents=["Dokumen test untuk skor"],
            ids=["doc-score"],
        )
        results = await retriever.search(
            collection_name="score-test",
            query="dokumen test",
        )
        assert len(results) > 0
        assert results[0].score > 0.0

    @pytest.mark.asyncio
    async def test_top_k_limits_results(
        self, retriever: HybridRetriever
    ) -> None:
        """top_k harus membatasi jumlah hasil."""
        docs = [f"Dokumen nomor {i} tentang topik {i}" for i in range(10)]
        ids = [f"doc-{i}" for i in range(10)]
        await retriever.add_documents(
            collection_name="topk-test",
            documents=docs,
            ids=ids,
        )
        results = await retriever.search(
            collection_name="topk-test",
            query="dokumen topik",
            top_k=3,
        )
        assert len(results) <= 3


# ──────────────────────────────────────────────
#  Collection Management
# ──────────────────────────────────────────────
class TestCollectionManagement:
    """Test manajemen collection."""

    @pytest.mark.asyncio
    async def test_list_collections(
        self, retriever: HybridRetriever
    ) -> None:
        """list_collections harus mengembalikan daftar collection."""
        await retriever.add_documents(
            collection_name="store-a",
            documents=["Dokumen A"],
        )
        await retriever.add_documents(
            collection_name="store-b",
            documents=["Dokumen B"],
        )
        collections = await retriever.list_collections()
        assert len(collections) >= 2

    @pytest.mark.asyncio
    async def test_delete_collection(
        self, retriever: HybridRetriever
    ) -> None:
        """delete_collection harus menghapus collection."""
        await retriever.add_documents(
            collection_name="to-delete",
            documents=["Data yang akan dihapus"],
        )
        await retriever.delete_collection("to-delete")
        with pytest.raises(ValueError):
            await retriever.search(
                collection_name="to-delete",
                query="test",
            )
