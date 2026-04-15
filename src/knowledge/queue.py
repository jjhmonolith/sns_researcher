"""Exploration queue management - prioritized URL queue with persistence."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.config import KNOWLEDGE_DIR
from src.knowledge.models import QueueItem, QueueItemStatus, QueueItemType

logger = logging.getLogger(__name__)

QUEUE_FILE = KNOWLEDGE_DIR / "queue.json"


class ExplorationQueue:
    """Priority queue for URLs to explore, persisted to disk."""

    def __init__(self) -> None:
        self._items: list[QueueItem] = []
        self._visited_urls: set[str] = set()
        self._load()

    def _load(self) -> None:
        """Load queue state from disk."""
        if QUEUE_FILE.exists():
            try:
                data = json.loads(QUEUE_FILE.read_text())
                self._items = [QueueItem(**item) for item in data.get("items", [])]
                self._visited_urls = set(data.get("visited_urls", []))
                logger.info(
                    f"Loaded queue: {len(self.pending_items)} pending, "
                    f"{len(self._visited_urls)} visited"
                )
            except Exception as e:
                logger.error(f"Error loading queue: {e}")
                self._items = []
                self._visited_urls = set()
        else:
            self._items = []
            self._visited_urls = set()

    def _save(self) -> None:
        """Persist queue state to disk."""
        QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "items": [item.model_dump() for item in self._items[-1000:]],  # Keep last 1000
            "visited_urls": list(self._visited_urls)[-5000:],  # Keep last 5000
        }
        QUEUE_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    @property
    def pending_items(self) -> list[QueueItem]:
        """Get all pending items sorted by priority (highest first)."""
        return sorted(
            [i for i in self._items if i.status == QueueItemStatus.PENDING],
            key=lambda x: x.priority,
            reverse=True,
        )

    @property
    def size(self) -> int:
        """Number of pending items in the queue."""
        return len(self.pending_items)

    def add(
        self,
        url: str,
        item_type: QueueItemType,
        priority: int = 50,
        source_post_id: str = "",
        reason: str = "",
    ) -> bool:
        """Add a URL to the exploration queue.
        
        Returns True if added, False if already visited or queued.
        """
        # Normalize URL
        normalized = url.split("?")[0].rstrip("/")

        if normalized in self._visited_urls:
            return False

        # Check if already in queue
        for item in self._items:
            existing_normalized = item.url.split("?")[0].rstrip("/")
            if existing_normalized == normalized and item.status == QueueItemStatus.PENDING:
                # Update priority if new priority is higher
                if priority > item.priority:
                    item.priority = priority
                return False

        item = QueueItem(
            url=url,
            item_type=item_type,
            priority=min(priority, 100),
            source_post_id=source_post_id,
            reason=reason,
        )
        self._items.append(item)
        self._save()
        logger.debug(f"Queue add: [{item_type.value}] {url[:60]}... (priority={priority})")
        return True

    def pop(self) -> QueueItem | None:
        """Get the highest-priority pending item and mark it in-progress."""
        pending = self.pending_items
        if not pending:
            return None

        item = pending[0]
        item.status = QueueItemStatus.IN_PROGRESS
        self._save()
        return item

    def mark_completed(self, url: str) -> None:
        """Mark a URL as completed/visited."""
        normalized = url.split("?")[0].rstrip("/")
        self._visited_urls.add(normalized)

        for item in self._items:
            if item.url.split("?")[0].rstrip("/") == normalized:
                item.status = QueueItemStatus.COMPLETED
                break

        self._save()

    def mark_failed(self, url: str) -> None:
        """Mark a URL as failed."""
        for item in self._items:
            if item.url.split("?")[0].rstrip("/") == url.split("?")[0].rstrip("/"):
                item.status = QueueItemStatus.FAILED
                break
        self._save()

    def mark_visited(self, url: str) -> None:
        """Mark a URL as already visited without adding to queue."""
        normalized = url.split("?")[0].rstrip("/")
        self._visited_urls.add(normalized)

    def is_visited(self, url: str) -> bool:
        """Check if a URL has already been visited."""
        normalized = url.split("?")[0].rstrip("/")
        return normalized in self._visited_urls

    def add_profile(self, url: str, priority: int = 50, source_post_id: str = "", reason: str = "") -> bool:
        """Shortcut to add a profile URL."""
        return self.add(url, QueueItemType.PROFILE_URL, priority, source_post_id, reason)

    def add_post(self, url: str, priority: int = 50, source_post_id: str = "", reason: str = "") -> bool:
        """Shortcut to add a post URL."""
        return self.add(url, QueueItemType.POST_URL, priority, source_post_id, reason)

    def get_stats(self) -> dict:
        """Get queue statistics."""
        statuses = {}
        for item in self._items:
            statuses[item.status.value] = statuses.get(item.status.value, 0) + 1

        return {
            "total_items": len(self._items),
            "pending": statuses.get("pending", 0),
            "completed": statuses.get("completed", 0),
            "failed": statuses.get("failed", 0),
            "visited_urls": len(self._visited_urls),
        }
