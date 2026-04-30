"""Seed script untuk mengisi data sampel ke dalam sistem RAG.

Mendukung dua mode:
1. Direct mode (default): Langsung menggunakan HybridRetriever + DocumentProcessor
2. API mode (--api): Mengirim data melalui endpoint /ingest

Jalankan dengan:
    python -m scripts.seed_data           # Direct mode
    python -m scripts.seed_data --api     # Via API endpoint
    python -m scripts.seed_data --clear   # Hapus data lama sebelum seed
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Any

import httpx

from src.core.document_processor import DocumentProcessor, RawDocument
from src.core.retriever import HybridRetriever
from src.utils.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Sample Data — 3 UMKM Stores
# ──────────────────────────────────────────────
SEED_STORES: dict[str, list[dict[str, Any]]] = {
    # ── Toko Demo: Toko Kelontong / Sembako ──
    "toko-demo": [
        {
            "text": (
                "Selamat datang di Toko Sejahtera! Kami adalah toko kelontong "
                "yang menyediakan berbagai kebutuhan sehari-hari dengan harga "
                "terjangkau. Berlokasi di Jl. Merdeka No. 45, Jakarta Selatan."
            ),
            "metadata": {"source": "info", "category": "about"},
        },
        {
            "text": (
                "Jam operasional Toko Sejahtera:\n"
                "- Senin-Jumat: 07.00 - 21.00 WIB\n"
                "- Sabtu: 07.00 - 22.00 WIB\n"
                "- Minggu: 08.00 - 20.00 WIB\n"
                "- Hari libur nasional: 09.00 - 18.00 WIB"
            ),
            "metadata": {"source": "info", "category": "jam_buka"},
        },
        {
            "text": (
                "Daftar harga beras:\n"
                "- Beras Premium 5kg: Rp 65.000\n"
                "- Beras Premium 10kg: Rp 125.000\n"
                "- Beras Medium 5kg: Rp 55.000\n"
                "- Beras Medium 10kg: Rp 105.000\n"
                "- Beras Organik 5kg: Rp 85.000\n"
                "Harga dapat berubah sewaktu-waktu."
            ),
            "metadata": {"source": "katalog", "category": "beras"},
        },
        {
            "text": (
                "Daftar harga minyak goreng:\n"
                "- Minyak Goreng Bimoli 1L: Rp 18.500\n"
                "- Minyak Goreng Bimoli 2L: Rp 35.000\n"
                "- Minyak Goreng Tropical 1L: Rp 17.000\n"
                "- Minyak Goreng Tropical 2L: Rp 32.000\n"
                "- Minyak Goreng Curah 1L: Rp 14.000"
            ),
            "metadata": {"source": "katalog", "category": "minyak_goreng"},
        },
        {
            "text": (
                "Daftar harga gula dan tepung:\n"
                "- Gula Pasir 1kg: Rp 15.500\n"
                "- Gula Merah 500g: Rp 12.000\n"
                "- Tepung Terigu Segitiga Biru 1kg: Rp 11.000\n"
                "- Tepung Terigu Cakra Kembar 1kg: Rp 12.500\n"
                "- Tepung Beras 500g: Rp 8.000\n"
                "- Tepung Maizena 150g: Rp 6.500"
            ),
            "metadata": {"source": "katalog", "category": "gula_tepung"},
        },
        {
            "text": (
                "Produk minuman yang tersedia:\n"
                "- Aqua 600ml: Rp 4.000\n"
                "- Aqua 1.5L: Rp 7.500\n"
                "- Teh Botol Sosro 450ml: Rp 5.000\n"
                "- Coca-Cola 390ml: Rp 6.500\n"
                "- Kopi Good Day Sachet: Rp 2.500\n"
                "- Susu Ultra 250ml: Rp 5.500\n"
                "- Susu Indomilk 1L: Rp 18.000"
            ),
            "metadata": {"source": "katalog", "category": "minuman"},
        },
        {
            "text": (
                "Produk snack dan makanan ringan:\n"
                "- Indomie Goreng: Rp 3.500\n"
                "- Indomie Kuah Soto: Rp 3.000\n"
                "- Chitato 68g: Rp 10.500\n"
                "- Taro 36g: Rp 5.000\n"
                "- Oreo 133g: Rp 9.500\n"
                "- Roti Tawar Sari Roti: Rp 16.000"
            ),
            "metadata": {"source": "katalog", "category": "snack"},
        },
        {
            "text": (
                "Layanan pengiriman Toko Sejahtera:\n"
                "- Gratis ongkir untuk pembelian minimal Rp 100.000 "
                "dalam radius 3 km.\n"
                "- Ongkir Rp 10.000 untuk radius 3-5 km.\n"
                "- Ongkir Rp 15.000 untuk radius 5-10 km.\n"
                "- Pengiriman di atas 10 km belum tersedia.\n"
                "- Waktu pengiriman: 1-3 jam setelah konfirmasi order."
            ),
            "metadata": {"source": "kebijakan", "category": "pengiriman"},
        },
        {
            "text": (
                "Metode pembayaran yang diterima:\n"
                "- Tunai\n"
                "- Transfer BCA, BRI, Mandiri\n"
                "- QRIS (GoPay, OVO, Dana, ShopeePay)\n"
                "- Tidak menerima kartu kredit."
            ),
            "metadata": {"source": "kebijakan", "category": "pembayaran"},
        },
        {
            "text": (
                "Promo bulan ini di Toko Sejahtera:\n"
                "- Beli 3 Indomie GRATIS 1 Indomie (berlaku semua varian)\n"
                "- Diskon 10% untuk pembelian beras 10kg\n"
                "- Bonus 1 sachet kopi untuk pembelian di atas Rp 50.000\n"
                "- Member card: setiap pembelian Rp 10.000 = 1 poin, "
                "100 poin bisa ditukar voucher Rp 10.000\n"
                "Promo berlaku hingga akhir bulan."
            ),
            "metadata": {"source": "promo", "category": "promo_bulanan"},
        },
    ],
    # ── Toko Kopi: Kedai Kopi Spesialti ──────
    "toko-kopi": [
        {
            "text": (
                "Kedai Kopi Nusantara — specialty coffee shop yang "
                "menghadirkan kopi terbaik dari seluruh Indonesia. "
                "Berlokasi di Jl. Sudirman No. 12, Bandung. "
                "Berdiri sejak 2020, kami berkomitmen menyajikan "
                "kopi single origin berkualitas tinggi."
            ),
            "metadata": {"source": "info", "category": "about"},
        },
        {
            "text": (
                "Jam operasional Kedai Kopi Nusantara:\n"
                "- Senin-Jumat: 08.00 - 22.00 WIB\n"
                "- Sabtu-Minggu: 09.00 - 23.00 WIB\n"
                "- Hari libur nasional: BUKA (jam normal)\n"
                "- Last order 30 menit sebelum tutup."
            ),
            "metadata": {"source": "info", "category": "jam_buka"},
        },
        {
            "text": (
                "Menu Espresso-Based:\n"
                "- Espresso Single: Rp 18.000\n"
                "- Espresso Double: Rp 24.000\n"
                "- Americano (Hot/Ice): Rp 22.000 / Rp 25.000\n"
                "- Cappuccino (Hot/Ice): Rp 28.000 / Rp 30.000\n"
                "- Cafe Latte (Hot/Ice): Rp 28.000 / Rp 30.000\n"
                "- Flat White: Rp 30.000\n"
                "- Mocha Latte (Hot/Ice): Rp 32.000 / Rp 35.000"
            ),
            "metadata": {"source": "menu", "category": "espresso"},
        },
        {
            "text": (
                "Menu Manual Brew (V60 / Chemex / Aeropress):\n"
                "- Aceh Gayo: Rp 30.000\n"
                "- Toraja Sapan: Rp 35.000\n"
                "- Flores Bajawa: Rp 32.000\n"
                "- Java Preanger: Rp 28.000\n"
                "- Papua Wamena: Rp 38.000\n"
                "- Bali Kintamani: Rp 33.000\n"
                "Semua manual brew menggunakan biji kopi single origin "
                "yang di-roast fresh setiap minggu."
            ),
            "metadata": {"source": "menu", "category": "manual_brew"},
        },
        {
            "text": (
                "Menu Non-Coffee:\n"
                "- Matcha Latte (Hot/Ice): Rp 30.000 / Rp 32.000\n"
                "- Chocolate (Hot/Ice): Rp 25.000 / Rp 28.000\n"
                "- Thai Tea (Ice): Rp 22.000\n"
                "- Fresh Juice (Jeruk/Mangga/Apel): Rp 20.000\n"
                "- Lemon Tea (Hot/Ice): Rp 18.000"
            ),
            "metadata": {"source": "menu", "category": "non_coffee"},
        },
        {
            "text": (
                "Menu Makanan Ringan:\n"
                "- Croissant Butter: Rp 25.000\n"
                "- Roti Bakar Cokelat: Rp 20.000\n"
                "- French Fries: Rp 22.000\n"
                "- Banana Bread: Rp 18.000\n"
                "- Cookies (3 pcs): Rp 15.000\n"
                "- Sandwich Tuna: Rp 30.000"
            ),
            "metadata": {"source": "menu", "category": "food"},
        },
        {
            "text": (
                "Penjualan biji kopi retail:\n"
                "- Aceh Gayo 200g: Rp 75.000\n"
                "- Toraja Sapan 200g: Rp 85.000\n"
                "- Flores Bajawa 200g: Rp 80.000\n"
                "- Java Preanger 200g: Rp 70.000\n"
                "- House Blend 200g: Rp 65.000\n"
                "- House Blend 500g: Rp 145.000\n"
                "Tersedia pilihan whole bean atau ground (fine/medium/coarse)."
            ),
            "metadata": {"source": "katalog", "category": "retail_beans"},
        },
        {
            "text": (
                "Promo Kedai Kopi Nusantara:\n"
                "- Happy Hour: Senin-Jumat jam 14.00-16.00, "
                "diskon 20% semua minuman espresso-based.\n"
                "- Member Card: beli 8 minuman GRATIS 1 minuman "
                "(berlaku semua menu minuman).\n"
                "- Student Discount: diskon 15% dengan menunjukkan "
                "kartu mahasiswa (berlaku Senin-Jumat).\n"
                "- Birthday Special: gratis 1 minuman di hari ulang tahun "
                "(tunjukkan KTP)."
            ),
            "metadata": {"source": "promo", "category": "promo"},
        },
        {
            "text": (
                "Layanan catering kopi untuk event:\n"
                "- Paket Small (30 cups): Rp 600.000\n"
                "- Paket Medium (50 cups): Rp 900.000\n"
                "- Paket Large (100 cups): Rp 1.500.000\n"
                "- Termasuk: barista, peralatan, dan biji kopi premium.\n"
                "- Booking minimal 3 hari sebelumnya.\n"
                "- Area layanan: Bandung dan sekitarnya."
            ),
            "metadata": {"source": "layanan", "category": "catering"},
        },
    ],
    # ── Toko Elektronik: Gadget & Accessories ─
    "toko-elektronik": [
        {
            "text": (
                "TechZone — toko elektronik dan aksesoris gadget terlengkap "
                "di Surabaya. Berlokasi di Jl. Basuki Rahmat No. 88, "
                "Surabaya. Menyediakan smartphone, laptop, tablet, dan "
                "aksesoris original dengan garansi resmi."
            ),
            "metadata": {"source": "info", "category": "about"},
        },
        {
            "text": (
                "Jam operasional TechZone:\n"
                "- Senin-Sabtu: 09.00 - 21.00 WIB\n"
                "- Minggu: 10.00 - 20.00 WIB\n"
                "- Hari libur nasional: TUTUP\n"
                "- Konsultasi online via WhatsApp 24 jam."
            ),
            "metadata": {"source": "info", "category": "jam_buka"},
        },
        {
            "text": (
                "Daftar harga smartphone:\n"
                "- Samsung Galaxy A15: Rp 2.499.000\n"
                "- Samsung Galaxy A55: Rp 5.999.000\n"
                "- Samsung Galaxy S24: Rp 12.999.000\n"
                "- iPhone 15: Rp 14.999.000\n"
                "- iPhone 15 Pro: Rp 19.999.000\n"
                "- Xiaomi Redmi Note 13: Rp 2.599.000\n"
                "- OPPO Reno 11: Rp 4.999.000\n"
                "Semua smartphone bergaransi resmi 1 tahun."
            ),
            "metadata": {"source": "katalog", "category": "smartphone"},
        },
        {
            "text": (
                "Daftar harga laptop:\n"
                "- ASUS VivoBook 14 (i3/8GB/256GB): Rp 6.999.000\n"
                "- ASUS VivoBook 14 (i5/8GB/512GB): Rp 8.999.000\n"
                "- Lenovo IdeaPad Slim 3 (i5/8GB/512GB): Rp 7.999.000\n"
                "- HP Pavilion 14 (i5/16GB/512GB): Rp 10.999.000\n"
                "- MacBook Air M2 (8GB/256GB): Rp 16.999.000\n"
                "- MacBook Air M3 (8GB/256GB): Rp 18.999.000\n"
                "Semua laptop bergaransi resmi 1 tahun."
            ),
            "metadata": {"source": "katalog", "category": "laptop"},
        },
        {
            "text": (
                "Aksesoris yang tersedia:\n"
                "- Case iPhone (berbagai model): Rp 50.000 - Rp 150.000\n"
                "- Case Samsung (berbagai model): Rp 40.000 - Rp 120.000\n"
                "- Tempered Glass: Rp 30.000 - Rp 80.000\n"
                "- Charger Fast Charging 25W: Rp 150.000\n"
                "- Charger Fast Charging 65W: Rp 250.000\n"
                "- Kabel USB-C to USB-C 1m: Rp 50.000\n"
                "- Power Bank 10.000mAh: Rp 200.000\n"
                "- Power Bank 20.000mAh: Rp 350.000"
            ),
            "metadata": {"source": "katalog", "category": "aksesoris"},
        },
        {
            "text": (
                "Audio dan wearable:\n"
                "- AirPods 3rd Gen: Rp 2.799.000\n"
                "- AirPods Pro 2: Rp 3.799.000\n"
                "- Samsung Galaxy Buds FE: Rp 1.299.000\n"
                "- JBL Tune 520BT: Rp 699.000\n"
                "- Apple Watch SE (2nd Gen): Rp 4.299.000\n"
                "- Samsung Galaxy Watch 6: Rp 3.999.000\n"
                "- Xiaomi Smart Band 8: Rp 499.000"
            ),
            "metadata": {"source": "katalog", "category": "audio_wearable"},
        },
        {
            "text": (
                "Kebijakan garansi TechZone:\n"
                "- Garansi resmi: sesuai ketentuan brand (1-2 tahun).\n"
                "- Garansi toko: 7 hari pengembalian jika produk cacat.\n"
                "- Klaim garansi: bawa nota pembelian + produk ke toko.\n"
                "- Garansi TIDAK berlaku untuk kerusakan akibat pengguna "
                "(jatuh, terkena air, modifikasi).\n"
                "- Produk aksesoris: garansi 30 hari."
            ),
            "metadata": {"source": "kebijakan", "category": "garansi"},
        },
        {
            "text": (
                "Metode pembayaran TechZone:\n"
                "- Tunai\n"
                "- Transfer Bank (BCA, BRI, Mandiri, BNI)\n"
                "- Kartu Kredit (Visa, Mastercard) — bisa cicilan 0% "
                "3/6/12 bulan untuk transaksi minimal Rp 3.000.000\n"
                "- QRIS (GoPay, OVO, Dana, ShopeePay)\n"
                "- Kredivo & Akulaku (cicilan online)"
            ),
            "metadata": {"source": "kebijakan", "category": "pembayaran"},
        },
        {
            "text": (
                "Promo TechZone bulan ini:\n"
                "- Diskon 5% untuk pembelian smartphone dengan kartu kredit.\n"
                "- Gratis tempered glass untuk pembelian smartphone apapun.\n"
                "- Diskon Rp 500.000 untuk laptop ASUS (stok terbatas).\n"
                "- Trade-in HP lama: dapatkan potongan hingga Rp 2.000.000 "
                "untuk pembelian HP baru.\n"
                "- Bundle hemat: beli laptop + mouse wireless Rp 99.000 saja."
            ),
            "metadata": {"source": "promo", "category": "promo_bulanan"},
        },
        {
            "text": (
                "Layanan pengiriman TechZone:\n"
                "- Gratis ongkir area Surabaya untuk pembelian di atas "
                "Rp 500.000.\n"
                "- Pengiriman luar kota via JNE/J&T/SiCepat.\n"
                "- Estimasi pengiriman: 1-2 hari (Jawa), 3-5 hari (luar Jawa).\n"
                "- Semua pengiriman diasuransikan.\n"
                "- COD tersedia untuk area Surabaya (minimal Rp 200.000)."
            ),
            "metadata": {"source": "kebijakan", "category": "pengiriman"},
        },
    ],
}


# ──────────────────────────────────────────────
#  Direct Seeding (menggunakan module langsung)
# ──────────────────────────────────────────────
async def seed_direct(*, clear: bool = False) -> None:
    """Seed data langsung menggunakan HybridRetriever + DocumentProcessor."""
    settings = get_settings()
    retriever = HybridRetriever(
        persist_directory=str(settings.chroma_db_path),
    )
    processor = DocumentProcessor()

    for store_id, documents in SEED_STORES.items():
        logger.info(f"Seeding store: {store_id} ({len(documents)} dokumen)")

        if clear:
            try:
                await retriever.delete_collection(store_id)
                logger.info(f"  Collection '{store_id}' dihapus")
            except Exception:
                logger.info(f"  Collection '{store_id}' belum ada, skip delete")

        # Konversi ke RawDocument
        raw_docs: list[RawDocument] = [
            RawDocument(text=doc["text"], metadata=doc.get("metadata", {}))
            for doc in documents
        ]

        # Proses chunking
        chunks = await processor.process_documents(raw_docs)
        logger.info(f"  Chunks dihasilkan: {len(chunks)}")

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
        logger.info(f"  ✅ {num_added} chunks berhasil ditambahkan ke '{store_id}'")

    # Verifikasi
    collections = await retriever.list_collections()
    logger.info(f"\nTotal collections: {len(collections)}")
    for col in collections:
        logger.info(f"  - {col}")


# ──────────────────────────────────────────────
#  API Seeding (via /ingest endpoint)
# ──────────────────────────────────────────────
async def seed_via_api(
    *,
    base_url: str = "http://localhost:8000/api/v1",
    api_key: str | None = None,
) -> None:
    """Seed data melalui endpoint /ingest (server harus berjalan)."""
    headers: dict[str, str] = {}
    if api_key:
        headers["X-API-Key"] = api_key

    async with httpx.AsyncClient(
        base_url=base_url, timeout=120.0, headers=headers
    ) as client:
        # Cek health dulu
        try:
            health_resp = await client.get("/health")
            health_resp.raise_for_status()
            logger.info(f"Server healthy: {health_resp.json()}")
        except Exception as exc:
            logger.error(f"Server tidak dapat dijangkau: {exc}")
            logger.error(f"Pastikan server berjalan di {base_url}")
            sys.exit(1)

        for store_id, documents in SEED_STORES.items():
            logger.info(f"Seeding store via API: {store_id}")

            payload = {
                "store_id": store_id,
                "documents": [
                    {"text": doc["text"], "metadata": doc.get("metadata", {})}
                    for doc in documents
                ],
            }

            try:
                response = await client.post("/ingest", json=payload)
                response.raise_for_status()
                result = response.json()
                logger.info(
                    f"  ✅ {result['chunks_created']} chunks created "
                    f"({result['processing_time_ms']:.0f}ms)"
                )
            except httpx.HTTPStatusError as exc:
                logger.error(
                    f"  ❌ Error: HTTP {exc.response.status_code} — "
                    f"{exc.response.text}"
                )
            except Exception as exc:
                logger.error(f"  ❌ Error: {exc}")


# ──────────────────────────────────────────────
#  CLI Entry Point
# ──────────────────────────────────────────────
def main() -> None:
    """Entry point CLI untuk seed script."""
    parser = argparse.ArgumentParser(
        description="Seed data sampel ke sistem RAG UMKM Assistant",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Kirim data via API /ingest (server harus running)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Hapus data lama sebelum seeding (hanya mode direct)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/api/v1",
        help="Base URL API (default: http://localhost:8000/api/v1)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key untuk autentikasi (jika diaktifkan)",
    )

    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("RAG UMKM Assistant — Seed Data Script")
    logger.info("=" * 50)
    logger.info(f"Mode: {'API' if args.api else 'Direct'}")
    logger.info(f"Stores: {', '.join(SEED_STORES.keys())}")
    logger.info(
        f"Total dokumen: "
        f"{sum(len(docs) for docs in SEED_STORES.values())}"
    )
    logger.info("")

    if args.api:
        asyncio.run(
            seed_via_api(base_url=args.base_url, api_key=args.api_key)
        )
    else:
        asyncio.run(seed_direct(clear=args.clear))

    logger.info("")
    logger.info("🎉 Seeding selesai!")


if __name__ == "__main__":
    main()
