"""Tests untuk DocumentProcessor."""

from __future__ import annotations

import asyncio

import pytest

from src.core.document_processor import DocumentProcessor, ProcessedChunk, RawDocument


# ──────────────────────────────────────────────
#  Inisialisasi
# ──────────────────────────────────────────────
class TestDocumentProcessorInit:
    """Test inisialisasi DocumentProcessor."""

    def test_default_values(self) -> None:
        """Harus bisa dibuat dengan default values."""
        processor = DocumentProcessor()
        assert processor.chunk_size == 500
        assert processor.chunk_overlap == 50

    def test_custom_values(self) -> None:
        """Harus bisa dikonfigurasi dengan custom values."""
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=20)
        assert processor.chunk_size == 200
        assert processor.chunk_overlap == 20

    def test_invalid_overlap_raises(self) -> None:
        """Overlap >= chunk_size harus raise ValueError."""
        with pytest.raises(ValueError, match="chunk_overlap"):
            DocumentProcessor(chunk_size=100, chunk_overlap=100)

        with pytest.raises(ValueError, match="chunk_overlap"):
            DocumentProcessor(chunk_size=100, chunk_overlap=150)


# ──────────────────────────────────────────────
#  split_text
# ──────────────────────────────────────────────
class TestSplitText:
    """Test metode split_text."""

    def test_short_text_single_chunk(self) -> None:
        """Teks pendek harus menghasilkan 1 chunk."""
        processor = DocumentProcessor(chunk_size=500)
        chunks = processor.split_text("Halo ini teks pendek.")
        assert len(chunks) == 1
        assert chunks[0] == "Halo ini teks pendek."

    def test_long_text_multiple_chunks(self, long_document: str) -> None:
        """Teks panjang harus menghasilkan banyak chunk."""
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=20)
        chunks = processor.split_text(long_document)
        assert len(chunks) > 1

    def test_empty_text_raises(self) -> None:
        """Teks kosong harus raise ValueError."""
        processor = DocumentProcessor()
        with pytest.raises(ValueError, match="kosong"):
            processor.split_text("")

    def test_whitespace_only_raises(self) -> None:
        """Teks hanya whitespace harus raise ValueError."""
        processor = DocumentProcessor()
        with pytest.raises(ValueError, match="kosong"):
            processor.split_text("   \n\t  ")

    def test_chunk_size_respected(self, long_document: str) -> None:
        """Setiap chunk tidak boleh melebihi chunk_size secara signifikan."""
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=20)
        chunks = processor.split_text(long_document)
        for chunk in chunks:
            # Tolerance: kadang sedikit melebihi karena separator logic
            assert len(chunk) <= 250, f"Chunk terlalu panjang: {len(chunk)}"


# ──────────────────────────────────────────────
#  process_documents
# ──────────────────────────────────────────────
class TestProcessDocuments:
    """Test metode process_documents."""

    @pytest.mark.asyncio
    async def test_single_document(self) -> None:
        """Proses satu dokumen harus menghasilkan chunk(s)."""
        processor = DocumentProcessor()
        raw_docs: list[RawDocument] = [
            RawDocument(
                id="doc-test",
                text="Ini adalah dokumen test untuk pemrosesan.",
                metadata={"source": "test"},
            )
        ]
        chunks = await processor.process_documents(raw_docs)
        assert len(chunks) >= 1
        assert all(isinstance(c, ProcessedChunk) for c in chunks)
        assert chunks[0].document_id == "doc-test"

    @pytest.mark.asyncio
    async def test_multiple_documents(
        self, sample_documents: list[dict[str, str]]
    ) -> None:
        """Proses banyak dokumen harus menghasilkan banyak chunk."""
        processor = DocumentProcessor()
        raw_docs: list[RawDocument] = [
            RawDocument(
                text=doc["text"],
                metadata=doc.get("metadata", {}),
            )
            for doc in sample_documents
        ]
        chunks = await processor.process_documents(raw_docs)
        assert len(chunks) >= len(sample_documents)

    @pytest.mark.asyncio
    async def test_empty_documents_raises(self) -> None:
        """Daftar dokumen kosong harus raise ValueError."""
        processor = DocumentProcessor()
        with pytest.raises(ValueError, match="kosong"):
            await processor.process_documents([])

    @pytest.mark.asyncio
    async def test_chunk_metadata_enriched(self) -> None:
        """Metadata chunk harus diperkaya dengan info chunking."""
        processor = DocumentProcessor()
        raw_docs: list[RawDocument] = [
            RawDocument(
                id="doc-meta",
                text="Dokumen dengan metadata kaya.",
                metadata={"source": "test", "author": "unit-test"},
            )
        ]
        chunks = await processor.process_documents(raw_docs)
        meta = chunks[0].metadata
        assert meta["document_id"] == "doc-meta"
        assert "chunk_index" in meta
        assert "total_chunks" in meta
        assert "char_count" in meta
        assert meta["source"] == "test"
        assert meta["author"] == "unit-test"

    @pytest.mark.asyncio
    async def test_auto_generate_doc_id(self) -> None:
        """Jika id tidak diberikan, harus auto-generate."""
        processor = DocumentProcessor()
        raw_docs: list[RawDocument] = [
            RawDocument(text="Dokumen tanpa id eksplisit.")
        ]
        chunks = await processor.process_documents(raw_docs)
        assert chunks[0].document_id.startswith("doc-")


# ──────────────────────────────────────────────
#  ID Generation
# ──────────────────────────────────────────────
class TestIDGeneration:
    """Test fungsi ID generation."""

    def test_doc_id_deterministic(self) -> None:
        """Hash yang sama harus menghasilkan ID yang sama."""
        id1 = DocumentProcessor._generate_doc_id("hello world")
        id2 = DocumentProcessor._generate_doc_id("hello world")
        assert id1 == id2

    def test_doc_id_different_for_different_text(self) -> None:
        """Teks berbeda harus menghasilkan ID berbeda."""
        id1 = DocumentProcessor._generate_doc_id("hello")
        id2 = DocumentProcessor._generate_doc_id("world")
        assert id1 != id2

    def test_chunk_id_includes_index(self) -> None:
        """Chunk ID harus mengandung index."""
        chunk_id = DocumentProcessor._generate_chunk_id(
            "doc-1", 0, "some text"
        )
        assert "chunk-0" in chunk_id
