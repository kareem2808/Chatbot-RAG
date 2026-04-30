"""Logging produksi dengan rotasi file asinkron dan format JSON.

Menyediakan logger terpusat dengan output JSON terstruktur yang
mudah di-parse oleh ELK/Datadog/CloudWatch, serta rotasi file
otomatis untuk mencegah disk penuh di production.

Contoh penggunaan:
    >>> from src.utils.logger import get_logger
    >>> logger = get_logger("api.routes")
    >>> logger.info("Request masuk", extra={"user_id": "u-123"})
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, ClassVar

# ──────────────────────────────────────────────
#  Konstanta default
# ──────────────────────────────────────────────
_DEFAULT_LOG_DIR: str = "./logs"
_DEFAULT_LOG_LEVEL: str = "INFO"
_DEFAULT_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB
_DEFAULT_BACKUP_COUNT: int = 5
_DEFAULT_APP_NAME: str = "rag-system"


# ──────────────────────────────────────────────
#  JSON Formatter
# ──────────────────────────────────────────────
class JSONFormatter(logging.Formatter):
    """Formatter yang menghasilkan output log dalam format JSON terstruktur.

    Setiap baris log menjadi satu JSON object yang berisi timestamp,
    level, logger name, message, dan metadata tambahan dari ``extra``.
    """

    # Field bawaan LogRecord yang TIDAK perlu dimasukkan ke JSON
    _RESERVED_ATTRS: ClassVar[frozenset[str]] = frozenset(
        {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
            "taskName",
        }
    )

    def __init__(self, app_name: str = _DEFAULT_APP_NAME) -> None:
        super().__init__()
        self._app_name: str = app_name

    def format(self, record: logging.LogRecord) -> str:
        """Konversi LogRecord menjadi string JSON satu baris.

        Args:
            record: LogRecord dari Python logging.

        Returns:
            String JSON terstruktur (satu baris).
        """
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "app": self._app_name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process_id": record.process,
            "thread_id": record.thread,
        }

        # Tambahkan field `extra` yang dikirim pengguna
        for key, value in record.__dict__.items():
            if key not in self._RESERVED_ATTRS and not key.startswith("_"):
                log_entry[key] = value

        # Sertakan traceback jika ada exception
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        if record.stack_info:
            log_entry["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(log_entry, default=str, ensure_ascii=False)


# ──────────────────────────────────────────────
#  Async Rotating File Handler
# ──────────────────────────────────────────────
class AsyncRotatingFileHandler(RotatingFileHandler):
    """RotatingFileHandler yang mendelegasikan I/O ke thread pool.

    Pada saat ``emit()`` dipanggil dan event loop aktif, penulisan log
    akan di-offload ke thread executor sehingga tidak memblokir coroutine.
    Jika tidak ada event loop (misalnya saat startup), fallback ke
    penulisan sinkron biasa.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Tulis log record, asinkron jika event loop tersedia.

        Args:
            record: LogRecord yang akan ditulis.
        """
        try:
            loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
            loop.run_in_executor(None, super().emit, record)
        except RuntimeError:
            # Tidak ada event loop yang berjalan → tulis sinkron
            super().emit(record)


# ──────────────────────────────────────────────
#  Registry & Factory
# ──────────────────────────────────────────────
_loggers: dict[str, logging.Logger] = {}
_root_configured: bool = False


def _configure_root_logger(
    log_dir: str = _DEFAULT_LOG_DIR,
    log_level: str = _DEFAULT_LOG_LEVEL,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    backup_count: int = _DEFAULT_BACKUP_COUNT,
    app_name: str = _DEFAULT_APP_NAME,
) -> None:
    """Konfigurasi root logger dengan console handler dan file handler.

    Dipanggil sekali saat ``get_logger`` pertama kali digunakan.

    Args:
        log_dir: Direktori penyimpanan file log.
        log_level: Level logging minimum.
        max_bytes: Ukuran maksimum file log sebelum rotasi.
        backup_count: Jumlah file backup yang disimpan.
        app_name: Nama aplikasi untuk ditampilkan di log.
    """
    global _root_configured  # noqa: PLW0603

    if _root_configured:
        return

    # Pastikan direktori log ada
    log_path: Path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    root_logger: logging.Logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    json_formatter = JSONFormatter(app_name=app_name)

    # ── Console handler (stderr) ─────────────
    console_handler = logging.StreamHandler(stream=sys.stderr)
    console_handler.setFormatter(json_formatter)
    root_logger.addHandler(console_handler)

    # ── Async rotating file handler ──────────
    file_handler = AsyncRotatingFileHandler(
        filename=str(log_path / "app.log"),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(json_formatter)
    root_logger.addHandler(file_handler)

    # ── Error-only file handler ──────────────
    error_handler = AsyncRotatingFileHandler(
        filename=str(log_path / "error.log"),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(json_formatter)
    root_logger.addHandler(error_handler)

    _root_configured = True


def get_logger(
    name: str,
    *,
    log_dir: str = _DEFAULT_LOG_DIR,
    log_level: str = _DEFAULT_LOG_LEVEL,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    backup_count: int = _DEFAULT_BACKUP_COUNT,
    app_name: str = _DEFAULT_APP_NAME,
) -> logging.Logger:
    """Dapatkan logger bernama dengan konfigurasi JSON + rotasi file.

    Logger di-cache berdasarkan nama; parameter konfigurasi hanya
    berpengaruh pada pemanggilan pertama (saat root logger dikonfigurasi).

    Args:
        name: Nama logger (biasanya ``__name__`` atau domain logis).
        log_dir: Direktori penyimpanan file log.
        log_level: Level logging minimum.
        max_bytes: Ukuran maksimum file log sebelum rotasi (byte).
        backup_count: Jumlah file backup rotasi yang disimpan.
        app_name: Nama aplikasi untuk label di log JSON.

    Returns:
        Instance ``logging.Logger`` yang telah terkonfigurasi.
    """
    if name in _loggers:
        return _loggers[name]

    _configure_root_logger(
        log_dir=log_dir,
        log_level=log_level,
        max_bytes=max_bytes,
        backup_count=backup_count,
        app_name=app_name,
    )

    logger: logging.Logger = logging.getLogger(name)
    _loggers[name] = logger
    return logger
