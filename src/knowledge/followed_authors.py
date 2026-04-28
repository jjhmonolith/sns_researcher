"""Followed authors management - periodic re-visiting of high-value authors."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from src.config import KNOWLEDGE_DIR

logger = logging.getLogger(__name__)

AUTHORS_FILE = KNOWLEDGE_DIR / "followed_authors.json"
MAX_AUTHORS = 50


class FollowedAuthors:
    """Tracks high-value LinkedIn authors for periodic re-visiting.

    Completely separate from ExplorationQueue — bypasses visited_urls.
    """

    def __init__(self, file_path: Path | None = None) -> None:
        self._file_path = file_path or (KNOWLEDGE_DIR / "followed_authors.json")
        self._authors: list[dict] = []
        self._load()

    def _load(self) -> None:
        if self._file_path.exists():
            try:
                data = json.loads(self._file_path.read_text())
                self._authors = data.get("authors", [])
                logger.info(f"Loaded {len(self._authors)} followed authors.")
            except Exception as e:
                logger.error(f"Error loading followed authors: {e}")
                self._authors = []
        else:
            self._authors = []

    def _save(self) -> None:
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"authors": self._authors}
        self._file_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def _normalize_url(self, url: str) -> str:
        return url.split("?")[0].rstrip("/")

    def _find(self, profile_url: str) -> dict | None:
        normalized = self._normalize_url(profile_url)
        for author in self._authors:
            if self._normalize_url(author["profile_url"]) == normalized:
                return author
        return None

    def add(self, profile_url: str, name: str = "", headline: str = "") -> bool:
        """Add an author to the followed list.

        If already present, increments relevant_post_count.
        Returns True if newly added, False if already existed.
        """
        if not profile_url:
            return False
        # Accept LinkedIn (/in/) and X (x.com/ or twitter.com/) profiles
        is_linkedin = "/in/" in profile_url
        is_x = "x.com/" in profile_url or "twitter.com/" in profile_url
        if not is_linkedin and not is_x:
            return False

        existing = self._find(profile_url)
        if existing:
            existing["relevant_post_count"] = existing.get("relevant_post_count", 0) + 1
            if name and not existing.get("name"):
                existing["name"] = name
            if headline and not existing.get("headline"):
                existing["headline"] = headline
            self._save()
            return False

        # Evict weakest if at cap
        if len(self._authors) >= MAX_AUTHORS:
            self._evict_weakest()

        author = {
            "profile_url": profile_url,
            "name": name,
            "headline": headline,
            "first_seen": datetime.now().isoformat(),
            "last_visited": "",
            "visit_count": 0,
            "relevant_post_count": 1,
            "platform_followed": False,
        }
        self._authors.append(author)
        self._save()
        logger.info(f"Now following: {name or profile_url[:40]} ({len(self._authors)} total)")
        return True

    def _evict_weakest(self) -> None:
        """Remove the least valuable author to make room."""
        if not self._authors:
            return
        # Score: relevant_post_count (lower = weaker), break ties by oldest last_visited
        self._authors.sort(
            key=lambda a: (a.get("relevant_post_count", 0), a.get("last_visited", "")),
        )
        evicted = self._authors.pop(0)
        logger.info(f"Evicted author: {evicted.get('name', evicted['profile_url'][:40])}")

    def pick_for_visit(self, count: int = 2) -> list[dict]:
        """Select authors to re-visit this cycle.

        Prioritizes by: relevant_post_count / (hours_since_last_visit + 1)
        Skips authors visited within the last 2 hours.
        """
        if not self._authors:
            return []

        now = datetime.now()
        candidates = []

        for author in self._authors:
            last = author.get("last_visited", "")
            if last:
                try:
                    last_dt = datetime.fromisoformat(last)
                    hours_since = (now - last_dt).total_seconds() / 3600
                    if hours_since < 2:
                        continue  # Skip recently visited
                except (ValueError, TypeError):
                    hours_since = 999
            else:
                hours_since = 999  # Never visited → high priority

            score = author.get("relevant_post_count", 1) / (hours_since + 1)
            candidates.append((score, author))

        # Sort by score descending, pick top N
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [c[1] for c in candidates[:count]]

    def record_visit(self, profile_url: str) -> None:
        """Record that an author was visited."""
        author = self._find(profile_url)
        if author:
            author["last_visited"] = datetime.now().isoformat()
            author["visit_count"] = author.get("visit_count", 0) + 1
            self._save()

    def mark_platform_followed(self, profile_url: str) -> None:
        """Mark that the user was followed on the actual platform."""
        author = self._find(profile_url)
        if author:
            author["platform_followed"] = True
            self._save()

    def needs_platform_follow(self, profile_url: str) -> bool:
        """Check if this author needs to be followed on the platform."""
        author = self._find(profile_url)
        if not author:
            return False
        return not author.get("platform_followed", False)

    def get_stats(self) -> dict:
        return {
            "total": len(self._authors),
            "total_visits": sum(a.get("visit_count", 0) for a in self._authors),
            "top_authors": [
                {"name": a.get("name", "?"), "posts": a.get("relevant_post_count", 0)}
                for a in sorted(
                    self._authors,
                    key=lambda a: a.get("relevant_post_count", 0),
                    reverse=True,
                )[:5]
            ],
        }
