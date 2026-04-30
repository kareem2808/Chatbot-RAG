"""Script untuk meng-ingest file dari direktori ke ChromaDB.

Mendukung semua format yang didukung FileLoader:
.txt, .pdf, .md, .csv, .json, .docx

Jalankan:
    python -m scripts.ingest_files                          # Ingest data/dummy/
    python -m scripts.ingest_files --dir data/custom/       # Direktori kustom
    python -m scripts.ingest_files --store-id toko-baru     # Store ID kustom
    python -m scripts.ingest_files --clear                  # Hapus data lama
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from src.core.document_processor import DocumentProcessor, RawDocument
from src.core.file_loader import FileLoader, LoadedDocument
from src.core.retriever import HybridRetriever
from src.utils.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Mapping nama file -> store_id
FILE_STORE_MAPPING: dict[str, str] = {
    "toko_sejahtera": "toko-demo",
    "kedai_kopi_nusantara": "toko-kopi",
    "techzone_elektronik": "toko-elektronik",
}


def resolve_store_id(file_path: Path, default_store_id: str | None) -> str:
    """Tentukan store_id dari nama file atau gunakan default.

    Args:
        file_path: Path ke file sumber.
        default_store_id: Store ID override dari CLI.

    Returns:
        Store ID yang akan digunakan.
    """
    if default_store_id:
        return default_store_id

    stem = file_path.stem.lower()
    for pattern, store_id in FILE_STORE_MAPPING.items():
        if pattern in stem:
            return store_id

    # Fallback: gunakan nama file sebagai store_id
    return stem.replace(" ", "-").replace("_", "-")


async def ingest_file(
    file_path: Path,
    store_id: str,
    loader: FileLoader,
    processor: DocumentProcessor,
    retriever: HybridRetriever,
) -> int:
    """Ingest satu file ke ChromaDB.

    Returns:
        Jumlah chunks yang ditambahkan.
    """
    # Load file
    documents: list[LoadedDocument] = loader.load_file(file_path)
    if not documents:
        logger.warning(f"  File kosong, skip: {file_path.name}")
        return 0

    # Konversi ke RawDocument
    raw_docs: list[RawDocument] = [
        RawDocument(text=doc.text, metadata=doc.metadata)
        for doc in documents
    ]

    # Chunking
    chunks = await processor.process_documents(raw_docs)
    if not chunks:
        logger.warning(f"  Tidak ada chunks yang dihasilkan dari: {file_path.name}")
        return 0

    # Upsert ke ChromaDB
    chunk_texts = [c.text for c in chunks]
    chunk_ids = [c.chunk_id for c in chunks]
    chunk_metadatas = [c.metadata for c in chunks]

    num_added = await retriever.add_documents(
        collection_name=store_id,
        documents=chunk_texts,
        metadatas=chunk_metadatas,
        ids=chunk_ids,
    )

    return num_added


async def ingest_directory(
    directory: Path,
    default_store_id: str | None,
    clear: bool,
) -> None:
    """Ingest semua file yang didukung dari direktori."""
    settings = get_settings()
    retriever = HybridRetriever(
        persist_directory=str(settings.chroma_db_path),
    )
    processor = DocumentProcessor()
    loader = FileLoader()

    # Kumpulkan semua file yang didukung
    from src.core.file_loader import SUPPORTED_EXTENSIONS
    all_files: list[Path] = sorted(
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not all_files:
        logger.error(f"Tidak ada file yang didukung di: {directory}")
        logger.info(f"Format yang didukung: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        sys.exit(1)

    logger.info(f"Ditemukan {len(all_files)} file untuk di-ingest:")
    for f in all_files:
        store = resolve_store_id(f, default_store_id)
        logger.info(f"  {f.name} -> store: {store}")

    # Proses setiap file
    total_chunks = 0
    for file_path in all_files:
        store_id = resolve_store_id(file_path, default_store_id)

        if clear:
            try:
                await retriever.delete_collection(store_id)
                logger.info(f"  Collection '{store_id}' dihapus")
            except Exception:
                pass  # Collection belum ada

        logger.info(f"\nIngesting: {file_path.name} -> {store_id}")
        try:
            num = await ingest_file(
                file_path, store_id, loader, processor, retriever
            )
            total_chunks += num
            logger.info(f"  ✅ {num} chunks ditambahkan ke '{store_id}'")
        except Exception as exc:
            logger.error(f"  ❌ Error: {exc}")

    # Verifikasi
    collections = await retriever.list_collections()
    logger.info(f"\nTotal collections: {len(collections)}")
    for col in collections:
        logger.info(f"  - {col}")
    logger.info(f"Total chunks di-ingest: {total_chunks}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest file ke sistem RAG UMKM Assistant",
    )
    parser.add_argument(
        "--dir",
        default="data/dummy",
        help="Direktori berisi file yang akan di-ingest (default: data/dummy)",
    )
    parser.add_argument(
        "--store-id",
        default=None,
        help="Override store_id (default: auto-detect dari nama file)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Hapus collection lama sebelum ingesting",
    )

    args = parser.parse_args()

    directory = Path(args.dir)
    if not directory.is_dir():
        logger.error(f"Direktori tidak ditemukan: {directory}")
        sys.exit(1)

    logger.info("=" * 50)
    logger.info("RAG UMKM Assistant — File Ingestion")
    logger.info("=" * 50)
    logger.info(f"Direktori: {directory}")
    logger.info(f"Store ID: {args.store_id or 'auto-detect'}")
    logger.info(f"Clear: {args.clear}")
    logger.info("")

    asyncio.run(
        ingest_directory(directory, args.store_id, args.clear)
    )

    logger.info("")
    logger.info("🎉 Ingestion selesai!")


if __name__ == "__main__":
    main()
