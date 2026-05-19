from __future__ import annotations

import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Persona:
    id: str
    name: str
    image_path: str
    voice: str
    speaker_wav: str | None = None
    system_prompt: str | None = None
    created_at: float = field(default_factory=time.time)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS personas (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    image_path    TEXT NOT NULL,
    voice         TEXT NOT NULL,
    speaker_wav   TEXT,
    system_prompt TEXT,
    created_at    REAL NOT NULL
);
"""


class PersonaStore:
    """SQLite-backed persona registry. Synchronous; trivially fast.

    Used by both the API and the orchestrator. Safe to call from multiple
    threads since each call opens/closes its own connection.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, isolation_level=None, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def upsert(self, persona: Persona) -> Persona:
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO personas(id, name, image_path, voice, speaker_wav, system_prompt, created_at)
                VALUES (:id, :name, :image_path, :voice, :speaker_wav, :system_prompt, :created_at)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    image_path=excluded.image_path,
                    voice=excluded.voice,
                    speaker_wav=excluded.speaker_wav,
                    system_prompt=excluded.system_prompt
                """,
                asdict(persona),
            )
        return persona

    def get(self, persona_id: str) -> Persona | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM personas WHERE id = ?", (persona_id,)).fetchone()
            return _row_to_persona(row) if row else None

    def list(self) -> list[Persona]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM personas ORDER BY created_at DESC").fetchall()
            return [_row_to_persona(r) for r in rows]

    def delete(self, persona_id: str) -> bool:
        with self._conn() as c:
            cur = c.execute("DELETE FROM personas WHERE id = ?", (persona_id,))
            return cur.rowcount > 0


def _row_to_persona(row: sqlite3.Row) -> Persona:
    return Persona(
        id=row["id"],
        name=row["name"],
        image_path=row["image_path"],
        voice=row["voice"],
        speaker_wav=row["speaker_wav"],
        system_prompt=row["system_prompt"],
        created_at=row["created_at"],
    )
