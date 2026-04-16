"""Main crawling loop - orchestrates the entire agent lifecycle."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta

from src.browser.session import BrowserSession
from src.browser.navigator import Navigator
from src.browser.extractor import ContentExtractor
from src.agent.relevance import RelevanceJudge
from src.agent.synthesizer import KnowledgeSynthesizer
from src.agent.weekly_synthesizer import WeeklySynthesizer
from src.agent.monthly_synthesizer import MonthlySynthesizer
from src.knowledge.store import KnowledgeStore
from src.knowledge.queue import ExplorationQueue
from src.knowledge.models import (
    AgentStats,
    AgentStatus,
    ActivityLog,
    LinkedInPost,
    PersistentStats,
    QueueItemType,
    TokenUsage,
)
from src.config import get_settings, ensure_dirs, STATS_FILE, HEARTBEAT_FILE

logger = logging.getLogger(__name__)


class LinkedInCrawler:
    """Main agent that continuously crawls LinkedIn for AX-related content."""

    def __init__(self) -> None:
        ensure_dirs()
        self.settings = get_settings()
        self.session = BrowserSession()
        self.store = KnowledgeStore()
        self.queue = ExplorationQueue()
        self.token_usage = TokenUsage()
        self.relevance = RelevanceJudge(token_usage=self.token_usage)
        self.synthesizer = KnowledgeSynthesizer(
            store=self.store, token_usage=self.token_usage
        )
        self.weekly_synthesizer = WeeklySynthesizer(
            store=self.store, token_usage=self.token_usage
        )
        self.monthly_synthesizer = MonthlySynthesizer(
            store=self.store, token_usage=self.token_usage
        )

        # Session state (resets each run)
        self.stats = AgentStats()
        self.activity_log: list[ActivityLog] = []
        self._pending_relevant_posts: list[LinkedInPost] = []
        self._saved_post_ids: set[str] = set()
        self._stop_requested = False
        self._pause_requested = False
        self._last_synthesis_time: datetime | None = None
        self._last_weekly_time: datetime | None = None
        self._last_monthly_time: datetime | None = None
        self._current_cycle: int = 0
        self._session_posts_scanned: int = 0
        self._session_relevant_found: int = 0

        # Load persistent cumulative stats
        self._persistent = self._load_persistent_stats()

    # ── Persistent Stats ──────────────────────────────────────────────

    def _load_persistent_stats(self) -> PersistentStats:
        """Load cumulative stats from disk, or create fresh."""
        if STATS_FILE.exists():
            try:
                data = json.loads(STATS_FILE.read_text(encoding="utf-8"))
                return PersistentStats(**data)
            except Exception as e:
                logger.warning(f"Could not load stats.json: {e}")
        return PersistentStats()

    def _save_persistent_stats(self) -> None:
        """Flush cumulative stats to disk."""
        try:
            STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
            # Merge current token usage into persistent
            self._persistent.nano_input_tokens += self.token_usage.nano_input_tokens
            self._persistent.nano_output_tokens += self.token_usage.nano_output_tokens
            self._persistent.powerful_input_tokens += (
                self.token_usage.powerful_input_tokens
            )
            self._persistent.powerful_output_tokens += (
                self.token_usage.powerful_output_tokens
            )
            # Reset session token counters so we don't double-add next save
            self.token_usage.nano_input_tokens = 0
            self.token_usage.nano_output_tokens = 0
            self.token_usage.powerful_input_tokens = 0
            self.token_usage.powerful_output_tokens = 0

            STATS_FILE.write_text(
                json.dumps(self._persistent.model_dump(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"Could not save stats.json: {e}")

    def _write_heartbeat(self) -> None:
        """Write a heartbeat timestamp so the dashboard can verify liveness."""
        try:
            HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "timestamp": datetime.now().isoformat(),
                "status": self.stats.status.value,
                "cycle": self._current_cycle,
                "pid": os.getpid(),
                "current_action": self.stats.current_action,
            }
            HEARTBEAT_FILE.write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as e:
            logger.warning(f"Could not write heartbeat: {e}")

    # ── Control ───────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self.stats.status == AgentStatus.RUNNING

    def request_stop(self) -> None:
        self._stop_requested = True
        self._log("info", "stop_requested", "Graceful stop requested")

    def request_pause(self) -> None:
        self._pause_requested = not self._pause_requested
        state = "paused" if self._pause_requested else "resumed"
        self._log("info", state, f"Agent {state}")

    # ── Main Loop ─────────────────────────────────────────────────────

    async def run(self) -> None:
        """Main entry point - runs the full crawling lifecycle."""
        now = datetime.now()
        self.stats.started_at = now.isoformat()
        self.stats.status = AgentStatus.INITIALIZING
        self._saved_post_ids = self.store.get_all_post_ids()

        # Update persistent session tracking
        self._persistent.total_sessions += 1
        if not self._persistent.first_started_at:
            self._persistent.first_started_at = now.isoformat()

        try:
            self._log("info", "starting", "Starting browser session...")
            self.stats.status = AgentStatus.WAITING_LOGIN
            self._write_heartbeat()
            await self.session.start(headless=True)

            self.stats.status = AgentStatus.RUNNING
            self._log(
                "info", "logged_in", "Successfully logged in. Starting crawl loop."
            )

            navigator = Navigator(self.session.page)
            extractor = ContentExtractor(self.session.page)

            cycle = 0
            while not self._stop_requested:
                # Handle pause
                while self._pause_requested and not self._stop_requested:
                    self.stats.status = AgentStatus.PAUSED
                    self._write_heartbeat()
                    await asyncio.sleep(5)

                if self._stop_requested:
                    break

                self.stats.status = AgentStatus.RUNNING
                cycle += 1
                self._current_cycle = cycle
                self._persistent.total_cycles += 1
                self._log("info", "cycle_start", f"Starting cycle {cycle}")
                self._write_heartbeat()

                try:
                    if await navigator.is_rate_limited():
                        self._log("warning", "rate_limited", "Rate limit detected!")
                        await navigator.handle_rate_limit()
                        continue

                    await self._scan_feed(navigator, extractor)
                    keyword_idx = (cycle - 1) % len(self.settings.keywords_list)
                    keyword = self.settings.keywords_list[keyword_idx]
                    await self._scan_search(navigator, extractor, keyword)

                    if cycle % 3 == 0:
                        await self._scan_global_hashtag(navigator, extractor)

                    await self._process_queue(navigator, extractor, max_items=3)
                    await self._check_synthesis()

                    if cycle % 5 == 0:
                        await self.session.refresh_cookies()

                    self.stats.queue_size = self.queue.size
                    self.stats.token_usage = self.token_usage

                    self._log(
                        "info", "cycle_end", f"Cycle {cycle} complete. Waiting..."
                    )
                    self._write_heartbeat()

                    # Save stats every cycle
                    self._persistent.total_posts_scanned = (
                        self._load_persistent_stats().total_posts_scanned
                        + self._session_posts_scanned
                    )
                    self._persistent.relevant_posts_found = (
                        self._load_persistent_stats().relevant_posts_found
                        + self._session_relevant_found
                    )
                    self._session_posts_scanned = 0
                    self._session_relevant_found = 0
                    self._save_persistent_stats()

                    await navigator.random_delay()

                except Exception as e:
                    self._log("error", "cycle_error", f"Error in cycle {cycle}: {e}")
                    self.stats.errors.append(f"[{datetime.now().isoformat()}] {str(e)}")
                    self._write_heartbeat()
                    await asyncio.sleep(60)

        except Exception as e:
            self.stats.status = AgentStatus.ERROR
            self._log("error", "fatal_error", f"Fatal error: {e}")
            self._write_heartbeat()
            raise
        finally:
            self.stats.status = AgentStatus.STOPPED
            self._write_heartbeat()
            # Final stats save
            self._persistent.total_posts_scanned += self._session_posts_scanned
            self._persistent.relevant_posts_found += self._session_relevant_found
            self._save_persistent_stats()
            await self.session.stop()
            self._log("info", "stopped", "Agent stopped.")

    # ── Scan Methods ──────────────────────────────────────────────────

    async def _scan_feed(
        self, navigator: Navigator, extractor: ContentExtractor
    ) -> None:
        self.stats.current_action = "Scanning home feed"
        self._log("info", "feed_scan", "Scanning home feed...")
        await navigator.go_to_feed()
        await navigator.scroll_feed(scroll_count=5)
        await navigator.expand_all_posts()
        posts = await extractor.extract_feed_posts()
        await self._process_posts(posts)
        await navigator.short_delay()

    async def _scan_global_hashtag(
        self, navigator: Navigator, extractor: ContentExtractor
    ) -> None:
        url = await navigator.go_to_global_hashtag_feed()
        self.stats.current_action = (
            f"Global hashtag: {url.split('hashtag/')[-1].rstrip('/')}"
        )
        self._log("info", "global_scan", f"Scanning global hashtag: {url}")
        await navigator.scroll_feed(scroll_count=4)
        await navigator.expand_all_posts()
        posts = await extractor.extract_feed_posts()
        if not posts:
            posts = await extractor.extract_search_results()
        await self._process_posts(posts)
        await navigator.short_delay()

    async def _scan_search(
        self, navigator: Navigator, extractor: ContentExtractor, keyword: str
    ) -> None:
        self.stats.current_action = f"Searching: {keyword}"
        self._log("info", "search_scan", f"Searching for: {keyword}")
        await navigator.search_posts(keyword)
        await asyncio.sleep(3)
        await navigator.scroll_feed(scroll_count=3)
        posts = await extractor.extract_search_results()
        if not posts:
            posts = await extractor.extract_feed_posts()
        await self._process_posts(posts)
        await navigator.short_delay()

    async def _process_queue(
        self, navigator: Navigator, extractor: ContentExtractor, max_items: int = 3
    ) -> None:
        for _ in range(max_items):
            if self._stop_requested:
                break
            item = self.queue.pop()
            if not item:
                break
            self.stats.current_action = f"Queue: {item.url[:50]}..."
            self.stats.current_url = item.url
            self._log("info", "queue_process", f"Processing: {item.url[:60]}")
            try:
                if item.item_type == QueueItemType.PROFILE_URL:
                    await navigator.go_to_profile(item.url)
                    posts = await extractor.extract_profile_posts()
                    await self._process_posts(posts)
                elif item.item_type == QueueItemType.POST_URL:
                    await navigator.go_to_post(item.url)
                    post = await extractor.extract_post_page(item.url)
                    if post:
                        await self._process_posts([post])
                self.queue.mark_completed(item.url)
            except Exception as e:
                self._log("error", "queue_error", f"Failed to process {item.url}: {e}")
                self.queue.mark_failed(item.url)
            await navigator.short_delay()

    async def _process_posts(self, posts: list[LinkedInPost]) -> None:
        for post in posts:
            if self._stop_requested:
                break
            if post.post_id in self._saved_post_ids:
                continue

            self.stats.total_posts_scanned += 1
            self._session_posts_scanned += 1

            post = await self.relevance.judge(post)

            if post.is_relevant:
                self.stats.relevant_posts_found += 1
                self._session_relevant_found += 1
                self.stats.posts_since_last_synthesis += 1

                self.store.save_post(post)
                self._saved_post_ids.add(post.post_id)
                self._pending_relevant_posts.append(post)

                self._log(
                    "info",
                    "relevant_post",
                    f"[{post.relevance_score}] {post.author.name}: {post.summary[:60]}",
                )
                self._enqueue_follow_targets(post)

                if post.url:
                    self.queue.mark_visited(post.url)
            else:
                if post.url:
                    self.queue.mark_visited(post.url)

    def _enqueue_follow_targets(self, post: LinkedInPost) -> None:
        if not post.should_follow_links:
            return
        for target in post.follow_targets:
            if "/in/" in target:
                self.queue.add_profile(
                    url=target,
                    priority=post.relevance_score,
                    source_post_id=post.post_id,
                    reason=f"Mentioned in relevant post (score={post.relevance_score})",
                )
            elif "linkedin.com" in target:
                self.queue.add_post(
                    url=target,
                    priority=post.relevance_score,
                    source_post_id=post.post_id,
                    reason=f"Linked from relevant post (score={post.relevance_score})",
                )
        if post.author.profile_url and post.relevance_score >= 60:
            self.queue.add_profile(
                url=post.author.profile_url,
                priority=min(post.relevance_score + 10, 100),
                source_post_id=post.post_id,
                reason=f"Author of highly relevant post (score={post.relevance_score})",
            )
        for profile_url in post.mentioned_profiles[:3]:
            self.queue.add_profile(
                url=profile_url,
                priority=max(post.relevance_score - 10, 30),
                source_post_id=post.post_id,
                reason="Mentioned in relevant post",
            )
        for post_url in post.linked_posts[:3]:
            self.queue.add_post(
                url=post_url,
                priority=max(post.relevance_score - 5, 30),
                source_post_id=post.post_id,
                reason="Linked from relevant post",
            )

    async def _check_synthesis(self) -> None:
        should_synthesize = False
        reason = ""

        if (
            self.stats.posts_since_last_synthesis
            >= self.settings.synthesis_interval_posts
        ):
            should_synthesize = True
            reason = f"Post threshold reached ({self.stats.posts_since_last_synthesis} posts)"

        if self._last_synthesis_time:
            hours_since = (
                datetime.now() - self._last_synthesis_time
            ).total_seconds() / 3600
            if (
                hours_since >= self.settings.synthesis_interval_hours
                and self._pending_relevant_posts
            ):
                should_synthesize = True
                reason = f"Time threshold reached ({hours_since:.1f} hours)"
        elif self._pending_relevant_posts and len(self._pending_relevant_posts) >= 5:
            should_synthesize = True
            reason = "Initial synthesis"

        if should_synthesize:
            self._log("info", "synthesis_start", f"Starting synthesis: {reason}")
            self.stats.status = AgentStatus.SYNTHESIZING
            self.stats.current_action = "Synthesizing knowledge..."
            self._write_heartbeat()
            try:
                results = await self.synthesizer.synthesize(
                    self._pending_relevant_posts
                )
                self._last_synthesis_time = datetime.now()
                self.stats.last_synthesis_at = self._last_synthesis_time.isoformat()
                self.stats.posts_since_last_synthesis = 0
                self._pending_relevant_posts = []
                self._log("info", "synthesis_done", f"Synthesis complete: {results}")
            except Exception as e:
                self._log("error", "synthesis_error", f"Synthesis failed: {e}")
            self.stats.status = AgentStatus.RUNNING

        now = datetime.now()
        if self._last_weekly_time is None:
            atoms = self.store.get_all_atoms()
            if len(atoms) >= 5 and now.weekday() == 0:
                should_weekly = True
            else:
                should_weekly = False
        elif (now - self._last_weekly_time).days >= 7:
            should_weekly = True
        else:
            should_weekly = False

        if should_weekly:
            self._log("info", "weekly_synthesis_start", "Starting weekly synthesis...")
            self.stats.status = AgentStatus.SYNTHESIZING
            self.stats.current_action = "Weekly knowledge map..."
            self._write_heartbeat()
            try:
                await self.weekly_synthesizer.run()
                self._last_weekly_time = now
                self._log("info", "weekly_synthesis_done", "Weekly map complete.")
            except Exception as e:
                self._log(
                    "error", "weekly_synthesis_error", f"Weekly synthesis failed: {e}"
                )
            self.stats.status = AgentStatus.RUNNING

        if self._last_monthly_time is None:
            atoms = self.store.get_all_atoms()
            if len(atoms) >= 10 and now.day == 1:
                should_monthly = True
            else:
                should_monthly = False
        elif (now - self._last_monthly_time).days >= 30:
            should_monthly = True
        else:
            should_monthly = False

        if should_monthly:
            self._log(
                "info", "monthly_synthesis_start", "Starting monthly synthesis..."
            )
            self.stats.status = AgentStatus.SYNTHESIZING
            self.stats.current_action = "Monthly knowledge map..."
            self._write_heartbeat()
            try:
                await self.monthly_synthesizer.run()
                self._last_monthly_time = now
                self._log("info", "monthly_synthesis_done", "Monthly map complete.")
            except Exception as e:
                self._log(
                    "error", "monthly_synthesis_error", f"Monthly synthesis failed: {e}"
                )
            self.stats.status = AgentStatus.RUNNING

    def _log(self, level: str, action: str, detail: str) -> None:
        entry = ActivityLog(action=action, detail=detail, level=level)
        self.activity_log.append(entry)
        if len(self.activity_log) > 500:
            self.activity_log = self.activity_log[-500:]
        if level == "error":
            logger.error(f"[{action}] {detail}")
        elif level == "warning":
            logger.warning(f"[{action}] {detail}")
        else:
            logger.info(f"[{action}] {detail}")

    def get_status_dict(self) -> dict:
        """Get full agent status as a dictionary for the dashboard."""
        now = datetime.now()

        # Uptime
        uptime_seconds = 0
        if self.stats.started_at:
            try:
                started = datetime.fromisoformat(self.stats.started_at)
                uptime_seconds = int((now - started).total_seconds())
            except Exception:
                pass

        # Cumulative totals from persistent stats + current session
        p = self._persistent
        cum_scanned = p.total_posts_scanned + self._session_posts_scanned
        cum_relevant = p.relevant_posts_found + self._session_relevant_found

        # Cumulative token usage (persistent + current session)
        cum_nano_in = p.nano_input_tokens + self.token_usage.nano_input_tokens
        cum_nano_out = p.nano_output_tokens + self.token_usage.nano_output_tokens
        cum_pow_in = p.powerful_input_tokens + self.token_usage.powerful_input_tokens
        cum_pow_out = p.powerful_output_tokens + self.token_usage.powerful_output_tokens

        nano_cost = (cum_nano_in * 0.20 + cum_nano_out * 1.25) / 1_000_000
        pow_cost = (cum_pow_in * 2.50 + cum_pow_out * 15.00) / 1_000_000
        total_cost = nano_cost + pow_cost

        return {
            "status": self.stats.status.value,
            "started_at": self.stats.started_at,
            "uptime_seconds": uptime_seconds,
            "current_cycle": self._current_cycle,
            "pid": os.getpid(),
            "total_sessions": p.total_sessions,
            "first_started_at": p.first_started_at,
            # Cumulative (across all sessions)
            "total_posts_scanned": cum_scanned,
            "relevant_posts_found": cum_relevant,
            "relevance_rate": (
                f"{(cum_relevant / cum_scanned * 100):.1f}%"
                if cum_scanned > 0
                else "N/A"
            ),
            # Session only
            "session_posts_scanned": self._session_posts_scanned,
            "session_relevant_found": self._session_relevant_found,
            "posts_since_last_synthesis": self.stats.posts_since_last_synthesis,
            "last_synthesis_at": self.stats.last_synthesis_at or "Not yet",
            "atom_count": len(self.store.get_all_atoms()),
            "queue_size": self.queue.size,
            "queue_stats": self.queue.get_stats(),
            "current_action": self.stats.current_action,
            "current_url": self.stats.current_url,
            "token_usage": {
                "nano_input": cum_nano_in,
                "nano_output": cum_nano_out,
                "powerful_input": cum_pow_in,
                "powerful_output": cum_pow_out,
                "nano_cost": f"${nano_cost:.4f}",
                "powerful_cost": f"${pow_cost:.4f}",
                "total_cost": f"${total_cost:.4f}",
                # Session-only for reference
                "session_nano_input": self.token_usage.nano_input_tokens,
                "session_nano_output": self.token_usage.nano_output_tokens,
                "session_powerful_input": self.token_usage.powerful_input_tokens,
                "session_powerful_output": self.token_usage.powerful_output_tokens,
            },
            "recent_errors": self.stats.errors[-10:],
            "posts_saved_today": self.store.count_posts_today(),
            "activity_log": [
                {
                    "timestamp": entry.timestamp,
                    "action": entry.action,
                    "detail": entry.detail,
                    "level": entry.level,
                }
                for entry in self.activity_log[-50:]
            ],
        }
