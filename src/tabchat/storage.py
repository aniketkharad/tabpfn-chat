"""Disk-backed session storage engine for tabchat."""

from __future__ import annotations

import asyncio
import io
import json
import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from tabchat.config import get_settings


class SessionNotFoundError(KeyError):
    """Raised when a requested session directory does not exist."""


class SessionStore:
    """Manages disk-backed session persistence under data/sessions/<session_id>/."""

    def __init__(self, data_dir: Path | str | None = None) -> None:
        if data_dir is None:
            self.data_dir = get_settings().DATA_DIR
        else:
            self.data_dir = Path(data_dir).resolve()

        self.sessions_dir = self.data_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        # In-memory session locks to prevent concurrent write collisions
        self._locks: dict[str, asyncio.Lock] = {}

    def get_lock(self, session_id: str) -> asyncio.Lock:
        """Return or create an asyncio.Lock for the specified session_id."""
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]

    def _session_dir(self, session_id: str) -> Path:
        """Get the directory path for a session ID."""
        return self.sessions_dir / session_id

    def create_session(self) -> str:
        """Generate a URL-safe 32-character token and create the session directory."""
        # 16 bytes hex-encoded produces exactly 32 URL-safe characters
        session_id = secrets.token_hex(16)
        s_dir = self._session_dir(session_id)
        s_dir.mkdir(parents=True, exist_ok=True)
        return session_id

    def session_exists(self, session_id: str) -> bool:
        """Return True if session directory exists."""
        return self._session_dir(session_id).is_dir()

    def _require_session(self, session_id: str) -> Path:
        """Ensure session directory exists or raise SessionNotFoundError."""
        s_dir = self._session_dir(session_id)
        if not s_dir.is_dir():
            raise SessionNotFoundError(f"Session '{session_id}' not found.")
        return s_dir

    def save_dataset(self, session_id: str, file_bytes: bytes, card: dict[str, Any]) -> None:
        """Save uploaded CSV bytes and generated dataset card JSON."""
        s_dir = self._require_session(session_id)
        raw_path = s_dir / "raw.csv"
        card_path = s_dir / "dataset_card.json"

        raw_path.write_bytes(file_bytes)
        card_path.write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")

    def get_dataset(self, session_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Retrieve dataset as (DataFrame, dataset_card_dict)."""
        s_dir = self._require_session(session_id)
        raw_path = s_dir / "raw.csv"
        card_path = s_dir / "dataset_card.json"

        if not raw_path.exists():
            raise FileNotFoundError(f"Dataset raw.csv not found for session '{session_id}'.")
        if not card_path.exists():
            raise FileNotFoundError(f"dataset_card.json not found for session '{session_id}'.")

        df = pd.read_csv(io.BytesIO(raw_path.read_bytes()), encoding="utf-8-sig")
        card = json.loads(card_path.read_text(encoding="utf-8"))
        return df, card

    def append_history(self, session_id: str, role: str, message: str) -> None:
        """Append a chat turn to history.jsonl with timestamp."""
        s_dir = self._require_session(session_id)
        history_path = s_dir / "history.jsonl"
        record = {
            "role": role,
            "content": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def get_history(self, session_id: str) -> list[dict[str, Any]]:
        """Return list of historical chat messages."""
        s_dir = self._require_session(session_id)
        history_path = s_dir / "history.jsonl"
        if not history_path.exists():
            return []

        messages: list[dict[str, Any]] = []
        with open(history_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    messages.append(json.loads(line))
        return messages

    def save_job_spec(self, session_id: str, spec: dict[str, Any]) -> None:
        """Persist the validated job specification JSON."""
        s_dir = self._require_session(session_id)
        spec_path = s_dir / "job_spec.json"
        spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")

    def get_job_spec(self, session_id: str) -> dict[str, Any] | None:
        """Retrieve job spec JSON if present, else None."""
        s_dir = self._require_session(session_id)
        spec_path = s_dir / "job_spec.json"
        if not spec_path.exists():
            return None
        return json.loads(spec_path.read_text(encoding="utf-8"))

    def save_results(self, session_id: str, results: dict[str, Any]) -> None:
        """Persist execution results JSON."""
        s_dir = self._require_session(session_id)
        results_path = s_dir / "results.json"
        results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    def get_results(self, session_id: str) -> dict[str, Any] | None:
        """Retrieve results JSON if present, else None."""
        s_dir = self._require_session(session_id)
        results_path = s_dir / "results.json"
        if not results_path.exists():
            return None
        return json.loads(results_path.read_text(encoding="utf-8"))

    def delete_session(self, session_id: str) -> None:
        """Recursively remove session directory and its lock."""
        s_dir = self._session_dir(session_id)
        if s_dir.exists():
            shutil.rmtree(s_dir)
        self._locks.pop(session_id, None)
