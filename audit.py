# audit.py
"""
Audit logging module.

Records accountability events for AssessedIT / NSW DoE compliance.
Logs are written to:
  1. Python's standard logging (stdout/stderr) — captured by Streamlit Cloud,
     Docker, and any log aggregation service connected to the host.
  2. A rotating local log file (audit.log) — for self-hosted deployments.
     On ephemeral platforms (e.g. Streamlit Community Cloud) the file will
     not persist across restarts; use log (1) for durable records there.

No student data content is ever written to the log. Events record only:
  - UTC timestamp
  - Session ID (random UUID, generated once per browser session)
  - Authenticated user email (DoE accountability requirement)
  - Event type
  - Safe metadata (file size, export format, active filters — never names
    or nomination content)
"""

import logging
import logging.handlers
import uuid
from datetime import datetime, timezone

import streamlit as st

# ── Logger setup ───────────────────────────────────────────────────────────────

_LOG_FILE = "audit.log"
_MAX_BYTES = 5 * 1024 * 1024   # 5 MB per file
_BACKUP_COUNT = 5               # keep up to 5 rotated files (~25 MB total)

_logger = logging.getLogger("sociogram.audit")
_logger.setLevel(logging.INFO)

if not _logger.handlers:
    # stdout handler — always present
    _stream_handler = logging.StreamHandler()
    _stream_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    _logger.addHandler(_stream_handler)

    # rotating file handler — best-effort (fails silently on read-only filesystems)
    try:
        _file_handler = logging.handlers.RotatingFileHandler(
            _LOG_FILE,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        _file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        _logger.addHandler(_file_handler)
    except OSError:
        pass  # read-only filesystem (e.g. Streamlit Community Cloud sandbox)


# ── Session ID ─────────────────────────────────────────────────────────────────

def _session_id() -> str:
    """Returns a stable UUID for the current browser session."""
    if "audit_session_id" not in st.session_state:
        st.session_state["audit_session_id"] = str(uuid.uuid4())
    return st.session_state["audit_session_id"]


def _user_email() -> str:
    """Returns the authenticated user's email, or 'anonymous' in dev mode."""
    user = st.session_state.get("auth_user")
    if user:
        return user.get("email", "unknown")
    return "dev@det.nsw.edu.au"  # dev bypass — AUTH_ENABLED=false


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _emit(event: str, **meta):
    """Core emit — formats and writes one audit record."""
    meta_str = " ".join(f"{k}={v}" for k, v in meta.items()) if meta else ""
    _logger.info(
        "[AUDIT] ts=%s session=%s user=%s event=%s %s",
        _now_utc(),
        _session_id(),
        _user_email(),
        event,
        meta_str,
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def session_start():
    """
    Call once at app startup (after auth).
    Logs only once per session using a session_state guard.
    """
    if st.session_state.get("audit_session_started"):
        return
    st.session_state["audit_session_started"] = True
    _emit("SESSION_START")


def file_uploaded(file_size_bytes: int):
    """Log that a CSV was uploaded. Records file size only — no filename or content."""
    _emit("FILE_UPLOAD", size_bytes=file_size_bytes)


def sample_data_loaded():
    """Log that the built-in example dataset was loaded."""
    _emit("SAMPLE_DATA_LOAD")


def pdf_exported(student_count: int, categories_shown: list[str]):
    """Log a PDF export. Records aggregate counts and active filters only."""
    _emit(
        "PDF_EXPORT",
        students=student_count,
        categories="+".join(categories_shown) if categories_shown else "none",
    )


def csv_exported(student_count: int, categories_shown: list[str]):
    """Log a CSV export. Records aggregate counts and active filters only."""
    _emit(
        "CSV_EXPORT",
        students=student_count,
        categories="+".join(categories_shown) if categories_shown else "none",
    )
