"""Streamlit Chatbot UI — Dumb Client untuk RAG UMKM Assistant.

Antarmuka chat interaktif yang mengirim request ke backend FastAPI.
Tidak mengandung logika RAG — murni presentation layer.

Jalankan dengan:
    streamlit run src/ui/app.py
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import streamlit as st

# ──────────────────────────────────────────────
#  Konfigurasi
# ──────────────────────────────────────────────
API_BASE_URL: str = "http://localhost:8000/api/v1"
CHAT_ENDPOINT: str = f"{API_BASE_URL}/chat"
HEALTH_ENDPOINT: str = f"{API_BASE_URL}/health"
REQUEST_TIMEOUT: float = 60.0

# ──────────────────────────────────────────────
#  Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="RAG UMKM Assistant",
    page_icon="🏪",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
#  Custom CSS
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global ──────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Header ──────────────────────────── */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .main-header p {
        color: #8892a4;
        font-size: 0.95rem;
    }

    /* ── Status Badge ────────────────────── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
    }
    .status-online {
        background: rgba(16, 185, 129, 0.12);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }
    .status-offline {
        background: rgba(239, 68, 68, 0.12);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.25);
    }

    /* ── Sidebar ─────────────────────────── */
    .sidebar-section {
        background: linear-gradient(135deg, rgba(102,126,234,0.08), rgba(118,75,162,0.08));
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(102,126,234,0.15);
    }
    .sidebar-section h3 {
        font-size: 0.85rem;
        font-weight: 600;
        color: #667eea;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* ── Stats ────────────────────────────── */
    .stats-row {
        display: flex;
        gap: 0.75rem;
        margin-top: 0.5rem;
    }
    .stat-card {
        flex: 1;
        background: rgba(102,126,234,0.06);
        border-radius: 10px;
        padding: 0.6rem;
        text-align: center;
        border: 1px solid rgba(102,126,234,0.1);
    }
    .stat-card .value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #667eea;
    }
    .stat-card .label {
        font-size: 0.7rem;
        color: #8892a4;
        margin-top: 2px;
    }

    /* ── Chat Messages ───────────────────── */
    .stChatMessage {
        border-radius: 12px !important;
    }

    /* ── Processing info ─────────────────── */
    .processing-info {
        font-size: 0.75rem;
        color: #8892a4;
        text-align: right;
        margin-top: -0.5rem;
        padding-right: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
#  Session State Initialization
# ──────────────────────────────────────────────
def _init_session_state() -> None:
    """Inisialisasi session state jika belum ada."""
    defaults: dict[str, Any] = {
        "messages": [],
        "store_id": "toko-demo",
        "total_queries": 0,
        "total_tokens_used": 0,
        "backend_status": "unknown",
        "api_key": "",
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


_init_session_state()


def _get_auth_headers() -> dict[str, str]:
    """Build HTTP headers dengan API key jika tersedia."""
    headers: dict[str, str] = {}
    if st.session_state.api_key:
        headers["X-API-Key"] = st.session_state.api_key
    return headers


# ──────────────────────────────────────────────
#  API Client Functions
# ──────────────────────────────────────────────
def check_backend_health() -> dict[str, Any] | None:
    """Cek status backend FastAPI via /health endpoint.

    Returns:
        Dict respons health check, atau None jika gagal.
    """
    try:
        with httpx.Client(timeout=5.0) as client:
            response: httpx.Response = client.get(
                HEALTH_ENDPOINT, headers=_get_auth_headers()
            )
            response.raise_for_status()
            return response.json()
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
        return None


def send_chat_message(
    user_query: str,
    store_id: str,
) -> dict[str, Any]:
    """Kirim pesan ke backend FastAPI /chat endpoint.

    Args:
        user_query: Pertanyaan pengguna.
        store_id: ID toko/collection.

    Returns:
        Dict respons dari backend.

    Raises:
        httpx.ConnectError: Jika backend tidak dapat dijangkau.
        httpx.TimeoutException: Jika request timeout.
        httpx.HTTPStatusError: Jika backend mengembalikan error.
    """
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        response: httpx.Response = client.post(
            CHAT_ENDPOINT,
            json={
                "user_query": user_query,
                "store_id": store_id,
            },
            headers=_get_auth_headers(),
        )
        response.raise_for_status()
        return response.json()


# ──────────────────────────────────────────────
#  Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏪 RAG Assistant")
    st.markdown("---")

    # ── API Key ───────────────────────────────
    st.markdown(
        '<div class="sidebar-section">'
        "<h3>🔑 API Key</h3>"
        "</div>",
        unsafe_allow_html=True,
    )

    api_key_input: str = st.text_input(
        "API Key",
        value=st.session_state.api_key,
        type="password",
        placeholder="Kosongkan jika auth nonaktif",
        help="Masukkan API key jika autentikasi diaktifkan di server.",
    )
    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input

    st.markdown("---")

    # ── Store ID Selection ───────────────────
    st.markdown(
        '<div class="sidebar-section">'
        "<h3>🔗 Pilih Toko</h3>"
        "</div>",
        unsafe_allow_html=True,
    )

    store_options: list[str] = [
        "toko-demo",
        "toko-kopi",
        "toko-elektronik",
    ]

    selected_store: str = st.selectbox(
        "ID Toko (Collection)",
        options=store_options,
        index=store_options.index(st.session_state.store_id)
        if st.session_state.store_id in store_options
        else 0,
        help="Pilih toko UMKM yang ingin ditanyakan",
    )

    # Jika store berubah, reset percakapan
    if selected_store != st.session_state.store_id:
        st.session_state.store_id = selected_store
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.rerun()

    # Opsi manual input store_id
    custom_store: str = st.text_input(
        "Atau masukkan ID Toko kustom",
        placeholder="contoh: toko-anda",
        help="Alfanumerik, underscore, dan dash saja",
    )
    if custom_store and custom_store != st.session_state.store_id:
        st.session_state.store_id = custom_store
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.rerun()

    st.markdown("---")

    # ── Backend Status ───────────────────────
    st.markdown(
        '<div class="sidebar-section">'
        "<h3>📡 Status Backend</h3>"
        "</div>",
        unsafe_allow_html=True,
    )

    if st.button("🔄 Cek Koneksi", use_container_width=True):
        with st.spinner("Memeriksa koneksi..."):
            health: dict[str, Any] | None = check_backend_health()
            if health:
                st.session_state.backend_status = "online"
                st.success(f"✅ Server: {health.get('status', 'unknown')}")
                st.caption(f"ChromaDB: {health.get('chroma_db', 'N/A')}")
                st.caption(f"Environment: {health.get('environment', 'N/A')}")
            else:
                st.session_state.backend_status = "offline"
                st.error("❌ Backend tidak dapat dijangkau")

    # Status badge
    if st.session_state.backend_status == "online":
        st.markdown(
            '<span class="status-badge status-online">● Online</span>',
            unsafe_allow_html=True,
        )
    elif st.session_state.backend_status == "offline":
        st.markdown(
            '<span class="status-badge status-offline">● Offline</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Session Stats ────────────────────────
    st.markdown(
        '<div class="sidebar-section">'
        "<h3>📊 Statistik Sesi</h3>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="stats-row">
            <div class="stat-card">
                <div class="value">{st.session_state.total_queries}</div>
                <div class="label">Pertanyaan</div>
            </div>
            <div class="stat-card">
                <div class="value">{len(st.session_state.messages) // 2}</div>
                <div class="label">Percakapan</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Clear Chat ───────────────────────────
    if st.button("🗑️ Hapus Riwayat Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.rerun()

    # ── Footer ───────────────────────────────
    st.markdown("---")
    st.caption("RAG UMKM Assistant v1.0.0")
    st.caption(f"Toko aktif: `{st.session_state.store_id}`")


# ──────────────────────────────────────────────
#  Main Chat Area
# ──────────────────────────────────────────────

# Header
st.markdown(
    """
    <div class="main-header">
        <h1>🏪 RAG UMKM Assistant</h1>
        <p>Asisten AI untuk toko Anda — didukung oleh Gemini 1.5 Flash</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Info bar
col1, col2 = st.columns([3, 1])
with col1:
    st.caption(f"📍 Toko aktif: **{st.session_state.store_id}**")
with col2:
    if st.session_state.backend_status == "online":
        st.caption("🟢 Backend aktif")
    elif st.session_state.backend_status == "offline":
        st.caption("🔴 Backend mati")
    else:
        st.caption("⚪ Belum dicek")

st.markdown("---")

# ── Render Chat History ──────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        st.markdown(msg["content"])
        if "processing_time" in msg:
            st.markdown(
                f'<div class="processing-info">'
                f'⏱ {msg["processing_time"]:.0f}ms · '
                f'{msg.get("num_sources", 0)} sumber</div>',
                unsafe_allow_html=True,
            )

# ── Welcome Message ──────────────────────────
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(
            f"Halo! 👋 Saya asisten AI untuk **{st.session_state.store_id}**.\n\n"
            "Silakan tanyakan apa saja seputar produk, harga, jam buka, "
            "promo, atau layanan toko.\n\n"
            "_Contoh: \"Berapa harga kopi arabika?\" atau \"Jam buka toko kapan?\"_"
        )

# ── Chat Input ───────────────────────────────
user_input: str | None = st.chat_input(
    placeholder="Ketik pertanyaan Anda di sini...",
)

if user_input:
    # Tampilkan pesan pengguna
    st.session_state.messages.append(
        {"role": "user", "content": user_input, "avatar": "👤"}
    )
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Kirim ke backend
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Sedang berpikir... 🧠"):
            try:
                start_time: float = time.time()
                response_data: dict[str, Any] = send_chat_message(
                    user_query=user_input,
                    store_id=st.session_state.store_id,
                )
                elapsed_ms: float = (time.time() - start_time) * 1000

                answer: str = response_data.get("answer", "Tidak ada jawaban.")
                num_sources: int = response_data.get("num_sources", 0)
                intent: str = response_data.get("intent", "unknown")
                api_time: float = response_data.get(
                    "processing_time_ms", elapsed_ms
                )

                st.markdown(answer)
                st.markdown(
                    f'<div class="processing-info">'
                    f"⏱ {api_time:.0f}ms · {num_sources} sumber · "
                    f"intent: {intent}</div>",
                    unsafe_allow_html=True,
                )

                # Simpan ke riwayat
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "avatar": "🤖",
                        "processing_time": api_time,
                        "num_sources": num_sources,
                        "intent": intent,
                    }
                )
                st.session_state.total_queries += 1
                st.session_state.backend_status = "online"

            except httpx.ConnectError:
                st.session_state.backend_status = "offline"
                error_msg: str = (
                    "⚠️ **Tidak dapat terhubung ke server backend.**\n\n"
                    "Pastikan server FastAPI berjalan di "
                    f"`{API_BASE_URL}`.\n\n"
                    "```bash\n"
                    "uvicorn src.api.main:app --reload\n"
                    "```"
                )
                st.error(error_msg)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "❌ Gagal terhubung ke server.",
                        "avatar": "🤖",
                    }
                )

            except httpx.TimeoutException:
                error_msg = (
                    "⏳ **Request timeout.**\n\n"
                    "Server membutuhkan waktu terlalu lama untuk merespons. "
                    "Silakan coba lagi."
                )
                st.warning(error_msg)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "⏳ Request timeout. Silakan coba lagi.",
                        "avatar": "🤖",
                    }
                )

            except httpx.HTTPStatusError as exc:
                status_code: int = exc.response.status_code
                try:
                    error_detail: dict[str, Any] = exc.response.json().get(
                        "detail", {}
                    )
                    error_message: str = (
                        error_detail.get("message", str(exc))
                        if isinstance(error_detail, dict)
                        else str(error_detail)
                    )
                except Exception:
                    error_message = str(exc)

                if status_code == 429:
                    st.warning(
                        f"🚫 **Rate limit tercapai.**\n\n{error_message}"
                    )
                elif status_code == 400:
                    st.error(
                        f"❌ **Request tidak valid.**\n\n{error_message}"
                    )
                else:
                    st.error(
                        f"💥 **Error server (HTTP {status_code}).**\n\n"
                        f"{error_message}"
                    )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"❌ Error: {error_message}",
                        "avatar": "🤖",
                    }
                )

            except Exception as exc:
                st.error(
                    f"💥 **Error tidak terduga:**\n\n`{exc!s}`\n\n"
                    "Silakan coba lagi atau hubungi administrator."
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"❌ Error: {exc!s}",
                        "avatar": "🤖",
                    }
                )
