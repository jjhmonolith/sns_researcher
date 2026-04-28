"""Persistent stats — saves/restores cumulative stats across restarts."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.config import DATA_DIR
from src.knowledge.models import AgentStats, TokenUsage

logger = logging.getLogger(__name__)

STATS_FILE = DATA_DIR / "stats.json"


def save_stats(stats: AgentStats, platform: str = "linkedin", stats_file: Path | None = None) -> None:
    """Save cumulative stats to disk."""
    file = stats_file or STATS_FILE
    all_stats = _load_all(stats_file=stats_file)
    all_stats[platform] = {
        "total_posts_scanned": stats.total_posts_scanned,
        "relevant_posts_found": stats.relevant_posts_found,
        "total_sessions": stats.total_sessions,
        "first_started_at": stats.first_started_at,
        "last_synthesis_at": stats.last_synthesis_at,
        "nano_input_tokens": stats.token_usage.nano_input_tokens,
        "nano_output_tokens": stats.token_usage.nano_output_tokens,
        "powerful_input_tokens": stats.token_usage.powerful_input_tokens,
        "powerful_output_tokens": stats.token_usage.powerful_output_tokens,
    }
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(all_stats, indent=2, ensure_ascii=False))


def restore_stats(stats: AgentStats, token_usage: TokenUsage, platform: str = "linkedin", stats_file: Path | None = None) -> None:
    """Restore cumulative stats from disk into the given objects."""
    all_stats = _load_all(stats_file=stats_file)
    saved = all_stats.get(platform, {})
    if not saved:
        return

    stats.total_posts_scanned = saved.get("total_posts_scanned", 0)
    stats.relevant_posts_found = saved.get("relevant_posts_found", 0)
    stats.total_sessions = saved.get("total_sessions", 0)
    stats.first_started_at = saved.get("first_started_at", "")
    stats.last_synthesis_at = saved.get("last_synthesis_at", "")
    token_usage.nano_input_tokens = saved.get("nano_input_tokens", 0)
    token_usage.nano_output_tokens = saved.get("nano_output_tokens", 0)
    token_usage.powerful_input_tokens = saved.get("powerful_input_tokens", 0)
    token_usage.powerful_output_tokens = saved.get("powerful_output_tokens", 0)

    logger.info(
        f"[{platform}] Restored stats: scanned={stats.total_posts_scanned}, "
        f"relevant={stats.relevant_posts_found}, sessions={stats.total_sessions}"
    )


def _load_all(stats_file: Path | None = None) -> dict:
    file = stats_file or STATS_FILE
    if file.exists():
        try:
            return json.loads(file.read_text())
        except Exception:
            pass
    return {}
