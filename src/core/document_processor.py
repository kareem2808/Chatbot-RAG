"""Pemrosesan dan chunking dokumen untuk RAG pipeline.

Memotong dokumen teks menjadi chunk-chunk kecil menggunakan
RecursiveCharacterTextSplitter dengan overlap untuk menjaga
kontinuitas konteks antar chunk.

Contoh penggunaan:
    >>> from src.core.document_processor import DocumentProcessor
    >>> processor = DocumentProcessor()
    >>> chunks = processor.split_text("Teks panjang tentang toko...")
    >>> docs = await processor.process_documents([
    ...     {"id": "doc-1", "text": "...", "metadata": {"source": "faq"}}
    ... ])
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger: logging.Logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Type Definitions
# ──────────────────────────────────────────────
class RawDocument(TypedDict, total=False):
    """Struktur dokumen mentah sebelum diproses.

    Attributes:
        id: Identifier unik dokumen (opsional, di-generate jika kosong).
        text: Konten teks dokumen.
        metadata: Metadata tambahan (source, author, dsb.).
    """

    id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ProcessedChunk:
    """Satu chunk hasil pemrosesan dokumen.

    Attributes:
        chunk_id: ID unik chunk (hash dari konten + posisi).
        document_id: ID dokumen asal.
        text: Konten teks chunk.
        metadata: Metadata gabungan (dari dokumen + info chunking).
        chunk_index: Posisi urutan chunk dalam dokumen asli.
        total_chunks: Total chunk yang dihasilkan dari dokumen asli.
    """

    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any]
    chunk_index: int
    total_chunks: int


# ──────────────────────────────────────────────
#  Document Processor
# ──────────────────────────────────────────────
@dataclass
class DocumentProcessor:
    """Processor untuk memotong dokumen menjadi chunk-chunk optimal.

    Menggunakan RecursiveCharacterTextSplitter yang memecah teks
    secara hierarkis berdasarkan separator (paragraf → kalimat → kata)
    untuk menjaga keutuhan semantik.

    Args:
        chunk_size: Ukuran maksimum setiap chunk (karakter).
        chunk_overlap: Jumlah karakter overlap antar chunk.
        separators: Daftar separator untuk pemecahan hierarkis.
    """

    chunk_size: int = 500
    chunk_overlap: int = 50
    separators: list[str] = field(
        default_factory=lambda: ["\n\n", "\n", ". ", ", ", " ", ""]
    )

    def __post_init__(self) -> None:
        """Inisialisasi text splitter setelah dataclass terbentuk."""
        if self.chunk_overlap >= self.chunk_size:
            msg: str = (
                f"chunk_overlap ({self.chunk_overlap}) harus lebih kecil "
                f"dari chunk_size ({self.chunk_size})"
            )
            raise ValueError(msg)

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )
        logger.info(
            "DocumentProcessor diinisialisasi",
            extra={
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            },
        )

    # ── Public API ───────────────────────────
    def split_text(self, text: str) -> list[str]:
        """Pecah teks menjadi daftar chunk string.

        Args:
            text: Teks yang akan dipecah.

        Returns:
            Daftar string chunk.

        Raises:
            ValueError: Jika teks kosong atau hanya whitespace.
        """
        cleaned: str = text.strip()
        if not cleaned:
            msg = "Teks input tidak boleh kosong"
            raise ValueError(msg)

        chunks: list[str] = self._splitter.split_text(cleaned)
        logger.debug(
            "Teks dipecah menjadi chunks",
            extra={"input_length": len(cleaned), "num_chunks": len(chunks)},
        )
        return chunks

    async def process_documents(
        self,
        documents: list[RawDocument],
    ) -> list[ProcessedChunk]:
        """Proses batch dokumen secara asinkron.

        Setiap dokumen dipecah menjadi chunk-chunk dan diberi metadata
        yang diperkaya (chunk_index, total_chunks, document_id).

        Args:
            documents: Daftar dokumen mentah untuk diproses.

        Returns:
            Daftar seluruh chunk dari semua dokumen.

        Raises:
            ValueError: Jika daftar dokumen kosong.
        """
        if not documents:
            msg = "Daftar dokumen tidak boleh kosong"
            raise ValueError(msg)

        # Proses tiap dokumen di thread pool agar tidak blocking
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        tasks: list[asyncio.Future[list[ProcessedChunk]]] = [
            loop.run_in_executor(None, self._process_single_document, doc)
            for doc in documents
        ]
        results: list[list[ProcessedChunk]] = await asyncio.gather(*tasks)

        # Flatten hasil
        all_chunks: list[ProcessedChunk] = [
            chunk for doc_chunks in results for chunk in doc_chunks
        ]

        logger.info(
            "Batch dokumen diproses",
            extra={
                "num_documents": len(documents),
                "total_chunks": len(all_chunks),
            },
        )
        return all_chunks

    # ── Internal ─────────────────────────────
    def _process_single_document(
        self,
        document: RawDocument,
    ) -> list[ProcessedChunk]:
        """Proses satu dokumen menjadi daftar ProcessedChunk.

        Args:
            document: Dokumen mentah.

        Returns:
            Daftar chunk dari dokumen ini.
        """
        text: str = document.get("text", "").strip()
        if not text:
            logger.warning(
                "Dokumen dilewati karena teks kosong",
                extra={"document_id": document.get("id", "unknown")},
            )
            return []

        doc_id: str = document.get("id", "") or self._generate_doc_id(text)
        doc_metadata: dict[str, Any] = document.get("metadata", {})
        raw_chunks: list[str] = self._splitter.split_text(text)
        total: int = len(raw_chunks)

        processed: list[ProcessedChunk] = []
        for index, chunk_text in enumerate(raw_chunks):
            chunk_id: str = self._generate_chunk_id(doc_id, index, chunk_text)
            enriched_metadata: dict[str, Any] = {
                **doc_metadata,
                "document_id": doc_id,
                "chunk_index": index,
                "total_chunks": total,
                "char_count": len(chunk_text),
            }
            processed.append(
                ProcessedChunk(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    text=chunk_text,
                    metadata=enriched_metadata,
                    chunk_index=index,
                    total_chunks=total,
                )
            )

        logger.debug(
            "Dokumen diproses",
            extra={
                "document_id": doc_id,
                "num_chunks": total,
                "original_length": len(text),
            },
        )
        return processed

    @staticmethod
    def _generate_doc_id(text: str) -> str:
        """Generate ID dokumen dari hash SHA-256 konten.

        Args:
            text: Konten teks dokumen.

        Returns:
            String hex hash 12 karakter pertama.
        """
        return f"doc-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]}"

    @staticmethod
    def _generate_chunk_id(doc_id: str, index: int, text: str) -> str:
        """Generate ID unik untuk chunk.

        Args:
            doc_id: ID dokumen asal.
            index: Indeks chunk dalam dokumen.
            text: Konten teks chunk.

        Returns:
            String ID unik chunk.
        """
        content: str = f"{doc_id}:{index}:{text}"
        short_hash: str = hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()[:8]
        return f"{doc_id}_chunk-{index}_{short_hash}"
