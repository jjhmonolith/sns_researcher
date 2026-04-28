"""Session model and manager — multi-topic research session support."""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

SESSIONS_FILE = DATA_DIR / "sessions.json"
SESSIONS_DIR = DATA_DIR / "sessions"


class SessionConfig(BaseModel):
    """Configuration for a single research session."""

    id: str = ""
    name: str = ""
    topic_description: str = ""
    keywords: list[str] = Field(default_factory=list)
    platforms: list[str] = Field(default_factory=lambda: ["linkedin"])
    status: str = "stopped"  # stopped, running, error
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    @property
    def base_dir(self) -> Path:
        return SESSIONS_DIR / self.id

    @property
    def knowledge_dir(self) -> Path:
        return self.base_dir / "knowledge"

    @property
    def cookies_linkedin_path(self) -> Path:
        return self.base_dir / "cookies_linkedin.json"

    @property
    def cookies_x_path(self) -> Path:
        return self.base_dir / "cookies_x.json"

    @property
    def cookies_fb_path(self) -> Path:
        return self.base_dir / "cookies_fb.json"

    @property
    def queue_path(self) -> Path:
        return self.base_dir / "queue.json"

    @property
    def followed_authors_path(self) -> Path:
        return self.base_dir / "followed_authors.json"

    @property
    def stats_path(self) -> Path:
        return self.base_dir / "stats.json"

    def ensure_dirs(self) -> None:
        """Create the full directory structure for this session."""
        dirs = [
            self.base_dir,
            self.knowledge_dir,
            self.knowledge_dir / "raw",
            self.knowledge_dir / "atoms",
            self.knowledge_dir / "maps" / "weekly",
            self.knowledge_dir / "maps" / "monthly",
            self.knowledge_dir / "insights" / "people",
            self.knowledge_dir / "insights" / "digests",
            self.knowledge_dir / "archive" / "legacy_topics",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


class SessionManager:
    """CRUD for research sessions, backed by sessions.json."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    def _load(self) -> list[dict]:
        if SESSIONS_FILE.exists():
            try:
                return json.loads(SESSIONS_FILE.read_text())
            except Exception:
                return []
        return []

    def _save(self, data: list[dict]) -> None:
        SESSIONS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    @staticmethod
    def _slugify(name: str) -> str:
        slug = re.sub(r"[^\w\s-]", "", name.lower())
        slug = re.sub(r"[\s_]+", "-", slug).strip("-")
        return slug[:40] or "session"

    def create_session(
        self,
        name: str,
        topic_description: str,
        keywords: list[str],
        platforms: list[str],
    ) -> SessionConfig:
        with self._lock:
            data = self._load()
            slug = self._slugify(name)

            # Deduplicate slug
            existing_ids = {d["id"] for d in data}
            base_slug = slug
            counter = 1
            while slug in existing_ids:
                slug = f"{base_slug}-{counter}"
                counter += 1

            config = SessionConfig(
                id=slug,
                name=name,
                topic_description=topic_description,
                keywords=keywords,
                platforms=platforms,
            )
            config.ensure_dirs()

            data.append(config.model_dump())
            self._save(data)
            logger.info(f"Created session: {slug} ({name})")
            return config

    def list_sessions(self) -> list[SessionConfig]:
        with self._lock:
            data = self._load()
            return [SessionConfig(**d) for d in data]

    def get_session(self, session_id: str) -> SessionConfig | None:
        with self._lock:
            data = self._load()
            for d in data:
                if d["id"] == session_id:
                    return SessionConfig(**d)
            return None

    def update_session(self, session_id: str, **fields) -> bool:
        with self._lock:
            data = self._load()
            for d in data:
                if d["id"] == session_id:
                    d.update(fields)
                    self._save(data)
                    return True
            return False

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            data = self._load()
            new_data = [d for d in data if d["id"] != session_id]
            if len(new_data) == len(data):
                return False
            self._save(new_data)
            logger.info(f"Deleted session: {session_id}")
            return True
