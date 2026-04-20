"""X (Twitter) crawling loop - mirrors LinkedIn crawler architecture."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from src.browser.x_session import XBrowserSession
from src.browser.x_navigator import XNavigator
from src.browser.x_extractor import XContentExtractor
from src.agent.relevance import RelevanceJudge
from src.agent.synthesizer import KnowledgeSynthesizer
from src.agent.weekly_synthesizer import WeeklySynthesizer
from src.agent.monthly_synthesizer import MonthlySynthesizer
from src.knowledge.store import KnowledgeStore
from src.knowledge.queue import ExplorationQueue
from src.knowledge.followed_authors import FollowedAuthors
from src.knowledge.models import (
    AgentStats,
    AgentStatus,
    ActivityLog,
    LinkedInPost,
    QueueItemType,
    TokenUsage,
)
from src.config import get_settings, ensure_dirs
from src.knowledge.persistent_stats import save_stats, restore_stats

logger = logging.getLogger(__name__)

# X-specific search keywords (can overlap with LinkedIn keywords)
X_SEARCH_KEYWORDS = [
    "AI transformation enterprise",
    "enterprise AI adoption",
    "agentic AI",
    "기업 AI 도입",
    "AI agent 업무자동화",
    "generative AI business",
]


class XCrawler:
    """Crawls X/Twitter for AX-related content. Shares knowledge base with LinkedIn crawler."""

    def __init__(
        self,
        store: KnowledgeStore,
        queue: ExplorationQueue,
        followed_authors: FollowedAuthors,
        token_usage: TokenUsage,
    ) -> None:
        ensure_dirs()
        self.settings = get_settings()
        self.session = XBrowserSession()
        self.store = store
        self.queue = queue
        self.followed_authors = followed_authors
        self.token_usage = token_usage
        self.relevance = RelevanceJudge(token_usage=self.token_usage, store=self.store)

        # State
        self.stats = AgentStats()
        self.activity_log: list[ActivityLog] = []
        self._pending_relevant_posts: list[LinkedInPost] = []
        self._saved_post_ids: set[str] = set()
        self._stop_requested = False
        self._pause_requested = False

    @property
    def is_running(self) -> bool:
        return self.stats.status == AgentStatus.RUNNING

    def request_stop(self) -> None:
        self._stop_requested = True
        self._log("info", "stop_requested", "[X] Graceful stop requested")

    def request_pause(self) -> None:
        self._pause_requested = not self._pause_requested
        state = "paused" if self._pause_requested else "resumed"
        self._log("info", state, f"[X] Agent {state}")

    async def run(self) -> None:
        """Main entry point — runs the X crawling lifecycle."""
        restore_stats(self.stats, self.token_usage, platform="x")
        self.stats.started_at = datetime.now().isoformat()
        if not self.stats.first_started_at:
            self.stats.first_started_at = self.stats.started_at
        self.stats.total_sessions += 1
        self.stats.status = AgentStatus.INITIALIZING
        self._saved_post_ids = self.store.get_all_post_ids()

        try:
            self._log("info", "starting", "[X] Starting browser session...")
            self.stats.status = AgentStatus.WAITING_LOGIN
            await self.session.start(headless=True)

            self.stats.status = AgentStatus.RUNNING
            self._log("info", "logged_in", "[X] Logged in. Starting crawl loop.")

            navigator = XNavigator(self.session.page)
            extractor = XContentExtractor(self.session.page)

            cycle = 0
            while not self._stop_requested:
                while self._pause_requested and not self._stop_requested:
                    self.stats.status = AgentStatus.PAUSED
                    await asyncio.sleep(5)

                if self._stop_requested:
                    break

                self.stats.status = AgentStatus.RUNNING
                cycle += 1
                self._log("info", "cycle_start", f"[X] Cycle {cycle}")

                try:
                    if await navigator.is_rate_limited():
                        self._log("warning", "rate_limited", "[X] Rate limit detected!")
                        await navigator.handle_rate_limit()
                        continue

                    # Phase 1: Home feed
                    await self._scan_feed(navigator, extractor)

                    # Phase 2: Keyword search (rotate)
                    keyword_idx = (cycle - 1) % len(X_SEARCH_KEYWORDS)
                    keyword = X_SEARCH_KEYWORDS[keyword_idx]
                    await self._scan_search(navigator, extractor, keyword)

                    # Phase 3: Followed authors (shared with LinkedIn)
                    await self._visit_followed_authors(navigator, extractor)

                    # Refresh cookies
                    if cycle % 5 == 0:
                        await self.session.refresh_cookies()

                    self.stats.token_usage = self.token_usage

                    if cycle % 5 == 0:
                        save_stats(self.stats, platform="x")

                    self._log("info", "cycle_end", f"[X] Cycle {cycle} complete.")
                    await navigator.random_delay()

                except Exception as e:
                    self._log("error", "cycle_error", f"[X] Error in cycle {cycle}: {e}")
                    await asyncio.sleep(60)

        except Exception as e:
            self.stats.status = AgentStatus.ERROR
            self._log("error", "fatal_error", f"[X] Fatal error: {e}")
            raise
        finally:
            save_stats(self.stats, platform="x")
            self.stats.status = AgentStatus.STOPPED
            await self.session.stop()
            self._log("info", "stopped", "[X] Agent stopped.")

    async def _scan_feed(self, navigator: XNavigator, extractor: XContentExtractor) -> None:
        self.stats.current_action = "[X] Scanning home feed"
        self._log("info", "x_feed_scan", "[X] Scanning home feed...")
        await navigator.go_to_feed()
        await navigator.scroll_feed(scroll_count=4)
        posts = await extractor.extract_feed_posts()
        await self._process_posts(posts)
        await navigator.short_delay()

    async def _scan_search(
        self, navigator: XNavigator, extractor: XContentExtractor, keyword: str
    ) -> None:
        self.stats.current_action = f"[X] Searching: {keyword}"
        self._log("info", "x_search_scan", f"[X] Searching: {keyword}")
        await navigator.search_posts(keyword)
        await asyncio.sleep(3)
        await navigator.scroll_feed(scroll_count=3)
        posts = await extractor.extract_search_results()
        await self._process_posts(posts)
        await navigator.short_delay()

    async def _visit_followed_authors(
        self, navigator: XNavigator, extractor: XContentExtractor
    ) -> None:
        """Visit followed authors that have X profiles."""
        authors = self.followed_authors.pick_for_visit(count=1)
        for author_info in authors:
            if self._stop_requested:
                break
            url = author_info.get("profile_url", "")
            # Only visit X profiles (skip LinkedIn profiles)
            if "x.com" not in url and "twitter.com" not in url:
                continue
            name = author_info.get("name", url[:40])
            self.stats.current_action = f"[X] Following: {name}"
            self._log("info", "x_follow_visit", f"[X] Visiting: {name}")
            try:
                await navigator.go_to_profile(url)
                posts = await extractor.extract_profile_posts()
                await self._process_posts(posts)
                self.followed_authors.record_visit(url)
            except Exception as e:
                self._log("error", "x_follow_error", f"[X] Failed: {e}")
            await navigator.short_delay()

    async def _process_posts(self, posts: list[LinkedInPost]) -> None:
        """Judge novelty and save relevant posts."""
        for post in posts:
            if self._stop_requested:
                break
            if post.post_id in self._saved_post_ids:
                continue

            self.stats.total_posts_scanned += 1
            post = await self.relevance.judge(post)

            if post.is_relevant:
                self.stats.relevant_posts_found += 1
                self.store.save_post(post)
                self._saved_post_ids.add(post.post_id)
                self._pending_relevant_posts.append(post)
                self._log(
                    "info", "x_relevant_post",
                    f"[X] [N:{post.novelty_score}] {post.author.name}: {post.summary[:60]}",
                )
                self._enqueue_follow_targets(post)
                if post.url:
                    self.queue.mark_visited(post.url)
            else:
                if post.url:
                    self.queue.mark_visited(post.url)

    def _enqueue_follow_targets(self, post: LinkedInPost) -> None:
        """Add X profiles from relevant posts to followed authors."""
        if post.author.profile_url and ("x.com" in post.author.profile_url or "twitter.com" in post.author.profile_url):
            self.followed_authors.add(
                profile_url=post.author.profile_url,
                name=post.author.name,
                headline=post.author.headline,
            )
        for profile_url in post.mentioned_profiles[:5]:
            if "x.com" in profile_url or "twitter.com" in profile_url:
                self.queue.add_profile(
                    url=profile_url,
                    priority=max(post.relevance_score - 10, 25),
                    source_post_id=post.post_id,
                    reason="[X] Mentioned in relevant tweet",
                )

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
        return {
            "platform": "x",
            "status": self.stats.status.value,
            "started_at": self.stats.started_at,
            "total_posts_scanned": self.stats.total_posts_scanned,
            "relevant_posts_found": self.stats.relevant_posts_found,
            "current_action": self.stats.current_action,
            "activity_log": [
                {"timestamp": e.timestamp, "action": e.action, "detail": e.detail, "level": e.level}
                for e in self.activity_log[-30:]
            ],
        }
