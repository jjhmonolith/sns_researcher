"""Facebook crawling loop — mbasic.facebook.com with stealth."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from src.browser.fb_session import FBBrowserSession
from src.browser.fb_navigator import FBNavigator
from src.browser.fb_extractor import FBContentExtractor
from src.agent.relevance import RelevanceJudge
from src.knowledge.store import KnowledgeStore
from src.knowledge.queue import ExplorationQueue
from src.knowledge.followed_authors import FollowedAuthors
from src.knowledge.models import (
    AgentStats, AgentStatus, ActivityLog, LinkedInPost, TokenUsage,
)
from src.knowledge.persistent_stats import save_stats, restore_stats
from src.config import get_settings, ensure_dirs

logger = logging.getLogger(__name__)

FB_SEARCH_KEYWORDS = [
    "AI transformation enterprise",
    "enterprise AI adoption",
    "generative AI business",
]


class FBCrawler:
    """Crawls Facebook for research content. Uses mbasic for stealth."""

    def __init__(
        self,
        store: KnowledgeStore | None = None,
        queue: ExplorationQueue | None = None,
        followed_authors: FollowedAuthors | None = None,
        token_usage: TokenUsage | None = None,
        session_config=None,
    ) -> None:
        self._session_config = session_config
        self.settings = get_settings()

        if session_config:
            session_config.ensure_dirs()
            self.session = FBBrowserSession(cookies_path=session_config.cookies_fb_path)
            self.store = store or KnowledgeStore(base_dir=session_config.knowledge_dir)
            self.queue = queue or ExplorationQueue(file_path=session_config.queue_path)
            self.followed_authors = followed_authors or FollowedAuthors(file_path=session_config.followed_authors_path)
            self.token_usage = token_usage or TokenUsage()
            self._keywords = session_config.keywords
            self._stats_file = session_config.stats_path
            topic = session_config.topic_description
        else:
            ensure_dirs()
            self.session = FBBrowserSession()
            self.store = store or KnowledgeStore()
            self.queue = queue or ExplorationQueue()
            self.followed_authors = followed_authors or FollowedAuthors()
            self.token_usage = token_usage or TokenUsage()
            self._keywords = list(FB_SEARCH_KEYWORDS)
            self._stats_file = None
            topic = ""

        self.relevance = RelevanceJudge(
            topic_description=topic,
            keywords=self._keywords,
            token_usage=self.token_usage,
            store=self.store,
        )

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
        self._log("info", "stop_requested", "[FB] Graceful stop requested")

    def request_pause(self) -> None:
        self._pause_requested = not self._pause_requested
        state = "paused" if self._pause_requested else "resumed"
        self._log("info", state, f"[FB] Agent {state}")

    async def run(self) -> None:
        restore_stats(self.stats, self.token_usage, platform="facebook", stats_file=self._stats_file)
        self.stats.started_at = datetime.now().isoformat()
        if not self.stats.first_started_at:
            self.stats.first_started_at = self.stats.started_at
        self.stats.total_sessions += 1
        self.stats.status = AgentStatus.INITIALIZING
        self._saved_post_ids = self.store.get_all_post_ids()

        try:
            self._log("info", "starting", "[FB] Starting browser session...")
            self.stats.status = AgentStatus.WAITING_LOGIN
            await self.session.start(headless=True)

            self.stats.status = AgentStatus.RUNNING
            self._log("info", "logged_in", "[FB] Logged in. Starting crawl loop.")

            navigator = FBNavigator(self.session.page)
            extractor = FBContentExtractor(self.session.page)

            cycle = 0
            while not self._stop_requested:
                while self._pause_requested and not self._stop_requested:
                    self.stats.status = AgentStatus.PAUSED
                    await asyncio.sleep(5)

                if self._stop_requested:
                    break

                self.stats.status = AgentStatus.RUNNING
                cycle += 1
                self._log("info", "cycle_start", f"[FB] Cycle {cycle}")

                try:
                    if await navigator.is_rate_limited():
                        self._log("warning", "rate_limited", "[FB] Rate limit detected!")
                        await navigator.handle_rate_limit()
                        continue

                    # Phase 1: News feed
                    await self._scan_feed(navigator, extractor)

                    # Phase 2: Keyword search (rotate)
                    if self._keywords:
                        keyword_idx = (cycle - 1) % len(self._keywords)
                        keyword = self._keywords[keyword_idx]
                        await self._scan_search(navigator, extractor, keyword)

                    # Refresh cookies
                    if cycle % 3 == 0:
                        await self.session.refresh_cookies()

                    if cycle % 5 == 0:
                        save_stats(self.stats, platform="facebook", stats_file=self._stats_file)

                    self._log("info", "cycle_end", f"[FB] Cycle {cycle} complete.")
                    # Extra long delay for Facebook
                    await navigator.random_delay()

                except Exception as e:
                    self._log("error", "cycle_error", f"[FB] Error in cycle {cycle}: {e}")
                    await asyncio.sleep(120)

        except Exception as e:
            self.stats.status = AgentStatus.ERROR
            self._log("error", "fatal_error", f"[FB] Fatal error: {e}")
            raise
        finally:
            save_stats(self.stats, platform="facebook", stats_file=self._stats_file)
            self.stats.status = AgentStatus.STOPPED
            await self.session.stop()
            self._log("info", "stopped", "[FB] Agent stopped.")

    async def _scan_feed(self, navigator: FBNavigator, extractor: FBContentExtractor) -> None:
        self.stats.current_action = "[FB] Scanning news feed"
        self._log("info", "fb_feed_scan", "[FB] Scanning news feed...")
        await navigator.go_to_feed()
        await navigator.scroll_feed(scroll_count=3)
        await navigator.expand_all_posts()
        posts = await extractor.extract_feed_posts()
        await self._process_posts(posts)
        await navigator.short_delay()

    async def _scan_search(
        self, navigator: FBNavigator, extractor: FBContentExtractor, keyword: str
    ) -> None:
        self.stats.current_action = f"[FB] Searching: {keyword}"
        self._log("info", "fb_search_scan", f"[FB] Searching: {keyword}")
        await navigator.search_posts(keyword)
        await navigator.scroll_feed(scroll_count=2)
        await navigator.expand_all_posts()
        posts = await extractor.extract_search_results()
        await self._process_posts(posts)
        await navigator.short_delay()

    async def _process_posts(self, posts: list[LinkedInPost]) -> None:
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
                self._log(
                    "info", "fb_relevant_post",
                    f"[FB] [N:{post.novelty_score}] {post.author.name}: {post.summary[:60]}",
                )

                # Register author for following
                if post.author.profile_url:
                    self.followed_authors.add(
                        profile_url=post.author.profile_url,
                        name=post.author.name,
                        headline="",
                    )

                if post.url:
                    self.queue.mark_visited(post.url)
            else:
                if post.url:
                    self.queue.mark_visited(post.url)

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
            "platform": "facebook",
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
