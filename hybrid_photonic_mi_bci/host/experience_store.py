"""SQLite-backed experience-library group management for the host app."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4

from .models import utc_now_iso


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ExperienceGroupRecord:
    group_id: str
    name: str
    description: str
    device: str
    channel_set: tuple[str, ...]
    task_type: str
    preprocessing_config: dict[str, Any]
    fbcsp_config: dict[str, Any]
    encoder_version: str
    is_active: bool
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ExperienceEntryRecord:
    entry_id: str
    group_id: str
    subject_id: str
    session_id: str
    source: str
    artifact_path: str
    metrics: dict[str, Any]
    metadata: dict[str, Any]
    created_at: str


class ExperienceStore:
    """Manage experience-library groups and entries.

    The SQLite database stores metadata and points to artifact files. Actual
    arrays such as CSP filters, selected feature indices, or candidate heads can
    be stored as NPZ files under the same artifact root.
    """

    def __init__(self, db_path: str | Path = "artifacts/host/cyton_experience.sqlite"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS groups (
                    group_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    device TEXT NOT NULL,
                    channel_set_json TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    preprocessing_json TEXT NOT NULL,
                    fbcsp_json TEXT NOT NULL,
                    encoder_version TEXT NOT NULL,
                    is_active INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entries (
                    entry_id TEXT PRIMARY KEY,
                    group_id TEXT NOT NULL,
                    subject_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    artifact_path TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(group_id) REFERENCES groups(group_id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                ("schema_version", str(SCHEMA_VERSION)),
            )

    def create_group(
        self,
        name: str,
        description: str = "",
        device: str = "OpenBCI Cyton",
        channel_set: tuple[str, ...] = ("C3", "C4", "Cz", "FC3", "FC4", "CP3", "CP4", "CPz"),
        task_type: str = "left/right/foot/reject",
        preprocessing_config: dict[str, Any] | None = None,
        fbcsp_config: dict[str, Any] | None = None,
        encoder_version: str = "fbcsp-mlp-v1",
        make_active: bool = True,
    ) -> ExperienceGroupRecord:
        self.initialize()
        group_id = f"group_{uuid4().hex[:12]}"
        now = utc_now_iso()
        preprocessing = preprocessing_config or {
            "reference": "CAR",
            "window": [1.0, 4.0],
            "sampling_rate": 250.0,
        }
        fbcsp = fbcsp_config or {
            "bands": [[8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 32]],
            "csp_components": 2,
            "selected_features": 32,
        }
        with self._connect() as conn:
            if make_active:
                conn.execute("UPDATE groups SET is_active = 0")
            conn.execute(
                """
                INSERT INTO groups VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    group_id,
                    name,
                    description,
                    device,
                    json.dumps(list(channel_set), ensure_ascii=False),
                    task_type,
                    json.dumps(preprocessing, ensure_ascii=False),
                    json.dumps(fbcsp, ensure_ascii=False),
                    encoder_version,
                    1 if make_active else 0,
                    now,
                    now,
                ),
            )
        return self.get_group(group_id)

    def list_groups(self) -> list[ExperienceGroupRecord]:
        self.initialize()
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM groups ORDER BY created_at DESC").fetchall()
        return [_group_from_row(row) for row in rows]

    def get_group(self, group_id: str) -> ExperienceGroupRecord:
        self.initialize()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM groups WHERE group_id = ?", (group_id,)).fetchone()
        if row is None:
            raise KeyError(f"experience group {group_id!r} not found")
        return _group_from_row(row)

    def get_active_group(self) -> ExperienceGroupRecord | None:
        self.initialize()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM groups WHERE is_active = 1 LIMIT 1").fetchone()
        return _group_from_row(row) if row is not None else None

    def set_active_group(self, group_id: str) -> None:
        self.initialize()
        now = utc_now_iso()
        with self._connect() as conn:
            exists = conn.execute("SELECT 1 FROM groups WHERE group_id = ?", (group_id,)).fetchone()
            if exists is None:
                raise KeyError(f"experience group {group_id!r} not found")
            conn.execute("UPDATE groups SET is_active = 0")
            conn.execute(
                "UPDATE groups SET is_active = 1, updated_at = ? WHERE group_id = ?",
                (now, group_id),
            )

    def delete_group(self, group_id: str) -> None:
        self.initialize()
        with self._connect() as conn:
            conn.execute("DELETE FROM groups WHERE group_id = ?", (group_id,))

    def add_entry(
        self,
        group_id: str,
        subject_id: str,
        session_id: str,
        source: str,
        artifact_path: str | Path,
        metrics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExperienceEntryRecord:
        self.initialize()
        _ = self.get_group(group_id)
        entry_id = f"entry_{uuid4().hex[:12]}"
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO entries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry_id,
                    group_id,
                    subject_id,
                    session_id,
                    source,
                    str(artifact_path),
                    json.dumps(metrics or {}, ensure_ascii=False),
                    json.dumps(metadata or {}, ensure_ascii=False),
                    now,
                ),
            )
        return self.get_entry(entry_id)

    def get_entry(self, entry_id: str) -> ExperienceEntryRecord:
        self.initialize()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM entries WHERE entry_id = ?", (entry_id,)).fetchone()
        if row is None:
            raise KeyError(f"experience entry {entry_id!r} not found")
        return _entry_from_row(row)

    def list_entries(self, group_id: str | None = None) -> list[ExperienceEntryRecord]:
        self.initialize()
        with self._connect() as conn:
            if group_id is None:
                rows = conn.execute("SELECT * FROM entries ORDER BY created_at DESC").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM entries WHERE group_id = ? ORDER BY created_at DESC",
                    (group_id,),
                ).fetchall()
        return [_entry_from_row(row) for row in rows]

    def delete_entry(self, entry_id: str) -> None:
        self.initialize()
        with self._connect() as conn:
            conn.execute("DELETE FROM entries WHERE entry_id = ?", (entry_id,))

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def _group_from_row(row: sqlite3.Row) -> ExperienceGroupRecord:
    return ExperienceGroupRecord(
        group_id=str(row["group_id"]),
        name=str(row["name"]),
        description=str(row["description"]),
        device=str(row["device"]),
        channel_set=tuple(json.loads(row["channel_set_json"])),
        task_type=str(row["task_type"]),
        preprocessing_config=dict(json.loads(row["preprocessing_json"])),
        fbcsp_config=dict(json.loads(row["fbcsp_json"])),
        encoder_version=str(row["encoder_version"]),
        is_active=bool(row["is_active"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _entry_from_row(row: sqlite3.Row) -> ExperienceEntryRecord:
    return ExperienceEntryRecord(
        entry_id=str(row["entry_id"]),
        group_id=str(row["group_id"]),
        subject_id=str(row["subject_id"]),
        session_id=str(row["session_id"]),
        source=str(row["source"]),
        artifact_path=str(row["artifact_path"]),
        metrics=dict(json.loads(row["metrics_json"])),
        metadata=dict(json.loads(row["metadata_json"])),
        created_at=str(row["created_at"]),
    )
