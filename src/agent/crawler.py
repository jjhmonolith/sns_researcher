"""Main crawling loop - orchestrates the entire agent lifecycle."""

from __future__ import annotations

import asyncio
import logging
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


class LinkedInCrawler:
    """Main agent that continuously crawls LinkedIn for AX-related content."""

    def __init__(self) -> None:
        ensure_dirs()
        self.settings = get_settings()
        self.session = BrowserSession()
        self.store = KnowledgeStore()
        self.queue = ExplorationQueue()
        self.followed_authors = FollowedAuthors()
        self.token_usage = TokenUsage()
        self.relevance = RelevanceJudge(token_usage=self.token_usage, store=self.store)
        self.synthesizer = KnowledgeSynthesizer(store=self.store, token_usage=self.token_usage)
        self.weekly_synthesizer = WeeklySynthesizer(store=self.store, token_usage=self.token_usage)
        self.monthly_synthesizer = MonthlySynthesizer(store=self.store, token_usage=self.token_usage)

        # State
        self.stats = AgentStats()
        self.activity_log: list[ActivityLog] = []
        self._pending_relevant_posts: list[LinkedInPost] = []
        self._saved_post_ids: set[str] = set()
        self._stop_requested = False
        self._pause_requested = False
        self._last_synthesis_time: datetime | None = None
        self._last_weekly_time: datetime | None = None
        self._last_monthly_time: datetime | None = None

    @property
    def is_running(self) -> bool:
        return self.stats.status == AgentStatus.RUNNING

    def request_stop(self) -> None:
        """Request the crawler to stop gracefully."""
        self._stop_requested = True
        self._log("info", "stop_requested", "Graceful stop requested")

    def request_pause(self) -> None:
        """Toggle pause state."""
        self._pause_requested = not self._pause_requested
        state = "paused" if self._pause_requested else "resumed"
        self._log("info", state, f"Agent {state}")

    async def run(self) -> None:
        """Main entry point - runs the full crawling lifecycle."""
        # Restore cumulative stats from disk
        restore_stats(self.stats, self.token_usage, platform="linkedin")
        self.stats.started_at = datetime.now().isoformat()
        if not self.stats.first_started_at:
            self.stats.first_started_at = self.stats.started_at
        self.stats.total_sessions += 1
        self.stats.status = AgentStatus.INITIALIZING
        self._saved_post_ids = self.store.get_all_post_ids()

        try:
            # Start browser and log in
            self._log("info", "starting", "Starting browser session...")
            self.stats.status = AgentStatus.WAITING_LOGIN
            await self.session.start(headless=True)

            self.stats.status = AgentStatus.RUNNING
            self._log("info", "logged_in", "Successfully logged in. Starting crawl loop.")

            navigator = Navigator(self.session.page)
            extractor = ContentExtractor(self.session.page)

            # Main loop
            cycle = 0
            while not self._stop_requested:
                # Handle pause
                while self._pause_requested and not self._stop_requested:
                    self.stats.status = AgentStatus.PAUSED
                    await asyncio.sleep(5)

                if self._stop_requested:
                    break

                self.stats.status = AgentStatus.RUNNING
                cycle += 1
                self._log("info", "cycle_start", f"Starting cycle {cycle}")

                try:
                    # Check for rate limiting
                    if await navigator.is_rate_limited():
                        self._log("warning", "rate_limited", "Rate limit detected!")
                        await navigator.handle_rate_limit()
                        continue

                    # Phase 1: Scan home feed (Korean personalized)
                    await self._scan_feed(navigator, extractor)

                    # Phase 1.5: Re-visit followed authors
                    await self._visit_followed_authors(navigator, extractor)

                    # Phase 2: Search with keywords (rotate Korean + English)
                    keyword_idx = (cycle - 1) % len(self.settings.keywords_list)
                    keyword = self.settings.keywords_list[keyword_idx]
                    await self._scan_search(navigator, extractor, keyword)

                    # Phase 3: Global hashtag feed (every 3rd cycle)
                    if cycle % 3 == 0:
                        await self._scan_global_hashtag(navigator, extractor)

                    # Phase 4: Process exploration queue
                    await self._process_queue(navigator, extractor, max_items=3)

                    # Phase 4: Check synthesis trigger
                    await self._check_synthesis()

                    # Refresh cookies periodically
                    if cycle % 5 == 0:
                        await self.session.refresh_cookies()

                    # Update stats
                    self.stats.queue_size = self.queue.size
                    self.stats.token_usage = self.token_usage

                    # Save stats periodically
                    if cycle % 5 == 0:
                        save_stats(self.stats, platform="linkedin")

                    # Long delay between cycles
                    self._log("info", "cycle_end", f"Cycle {cycle} complete. Waiting...")
                    await navigator.random_delay()

                except Exception as e:
                    self._log("error", "cycle_error", f"Error in cycle {cycle}: {e}")
                    self.stats.errors.append(f"[{datetime.now().isoformat()}] {str(e)}")
                    await asyncio.sleep(60)

        except Exception as e:
            self.stats.status = AgentStatus.ERROR
            self._log("error", "fatal_error", f"Fatal error: {e}")
            raise
        finally:
            save_stats(self.stats, platform="linkedin")
            self.stats.status = AgentStatus.STOPPED
            await self.session.stop()
            self._log("info", "stopped", "Agent stopped.")

    async def _visit_followed_authors(
        self, navigator: Navigator, extractor: ContentExtractor
    ) -> None:
        """Re-visit 1-2 followed authors to check for new posts."""
        authors_to_visit = self.followed_authors.pick_for_visit(count=2)
        if not authors_to_visit:
            return

        for author_info in authors_to_visit:
            if self._stop_requested:
                break
            url = author_info["profile_url"]
            name = author_info.get("name", url[:40])
            self.stats.current_action = f"Following: {name}"
            self.stats.current_url = url
            self._log("info", "follow_visit", f"Re-visiting followed author: {name}")

            try:
                # Follow on LinkedIn if not yet followed
                if self.followed_authors.needs_platform_follow(url):
                    followed = await navigator.follow_user(url)
                    if followed:
                        self.followed_authors.mark_platform_followed(url)
                        self._log("info", "platform_follow", f"Followed on LinkedIn: {name}")
                    await navigator.short_delay()

                await navigator.go_to_profile(url)
                posts = await extractor.extract_profile_posts()
                await self._process_posts(posts)
                self.followed_authors.record_visit(url)
            except Exception as e:
                self._log("error", "follow_error", f"Failed to visit {name}: {e}")

            await navigator.short_delay()

    async def _scan_feed(self, navigator: Navigator, extractor: ContentExtractor) -> None:
        """Scan the home feed for relevant posts."""
        self.stats.current_action = "Scanning home feed"
        self._log("info", "feed_scan", "Scanning home feed...")

        await navigator.go_to_feed()
        await navigator.scroll_feed(scroll_count=5)
        await navigator.expand_all_posts()

        posts = await extractor.extract_feed_posts()
        await self._process_posts(posts)

        await navigator.short_delay()

    async def _scan_global_hashtag(self, navigator: Navigator, extractor: ContentExtractor) -> None:
        """Scan a global hashtag feed for international AX content."""
        url = await navigator.go_to_global_hashtag_feed()
        self.stats.current_action = f"Global hashtag: {url.split('hashtag/')[-1].rstrip('/')}"
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
        """Search LinkedIn and process results."""
        self.stats.current_action = f"Searching: {keyword}"
        self._log("info", "search_scan", f"Searching for: {keyword}")

        await navigator.search_posts(keyword)
        await asyncio.sleep(3)
        await navigator.scroll_feed(scroll_count=3)

        posts = await extractor.extract_search_results()
        if not posts:
            # Fallback: try extracting as feed posts
            posts = await extractor.extract_feed_posts()

        await self._process_posts(posts)

        await navigator.short_delay()

    async def _process_queue(
        self, navigator: Navigator, extractor: ContentExtractor, max_items: int = 3
    ) -> None:
        """Process items from the exploration queue."""
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

                        # Extract comment authors for relevant posts
                        if post.is_relevant and post.comments_count > 0:
                            await navigator.expand_comments(max_clicks=2)
                            comment_authors = await extractor.extract_comment_authors()
                            self._enqueue_comment_authors(comment_authors, post)

                self.queue.mark_completed(item.url)

            except Exception as e:
                self._log("error", "queue_error", f"Failed to process {item.url}: {e}")
                self.queue.mark_failed(item.url)

            await navigator.short_delay()

    async def _process_posts(self, posts: list[LinkedInPost]) -> None:
        """Judge relevance and save relevant posts, add links to queue."""
        for post in posts:
            if self._stop_requested:
                break

            # Skip already-processed posts
            if post.post_id in self._saved_post_ids:
                continue

            self.stats.total_posts_scanned += 1

            # Judge relevance
            post = await self.relevance.judge(post)

            if post.is_relevant:
                self.stats.relevant_posts_found += 1
                self.stats.posts_since_last_synthesis += 1

                # Save to knowledge store
                self.store.save_post(post)
                self._saved_post_ids.add(post.post_id)
                self._pending_relevant_posts.append(post)

                self._log(
                    "info",
                    "relevant_post",
                    f"[N:{post.novelty_score}] {post.author.name}: {post.summary[:60]}",
                )

                # Add follow targets to queue
                self._enqueue_follow_targets(post)

                # Mark post URL as visited
                if post.url:
                    self.queue.mark_visited(post.url)
            else:
                # Still mark URL as visited to avoid re-processing
                if post.url:
                    self.queue.mark_visited(post.url)

    def _enqueue_comment_authors(
        self, comment_authors: list[dict], source_post: LinkedInPost
    ) -> None:
        """Add comment authors to the exploration queue."""
        added = 0
        for author in comment_authors[:10]:
            profile_url = author.get("profile_url", "")
            if not profile_url:
                continue
            if self.queue.add_profile(
                url=profile_url,
                priority=max(source_post.relevance_score - 15, 25),
                source_post_id=source_post.post_id,
                reason=f"Commented on relevant post",
            ):
                added += 1
            self.queue.record_mention(profile_url)
        if added:
            self._log("info", "comment_authors", f"Queued {added} comment authors from {source_post.post_id}")

    def _enqueue_follow_targets(self, post: LinkedInPost) -> None:
        """Add links and profiles from a relevant post to the exploration queue."""
        # Follow targets suggested by GPT
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

        # Register author for periodic following
        if post.author.profile_url:
            self.followed_authors.add(
                profile_url=post.author.profile_url,
                name=post.author.name,
                headline=post.author.headline,
            )

        # Author profile (high priority for highly relevant posts)
        if post.author.profile_url and post.relevance_score >= self.settings.relevance_threshold:
            self.queue.add_profile(
                url=post.author.profile_url,
                priority=min(post.relevance_score + 10, 100),
                source_post_id=post.post_id,
                reason=f"Author of highly relevant post (score={post.relevance_score})",
            )

        # Mentioned profiles
        for profile_url in post.mentioned_profiles[:5]:
            self.queue.add_profile(
                url=profile_url,
                priority=max(post.relevance_score - 10, 30),
                source_post_id=post.post_id,
                reason="Mentioned in relevant post",
            )

        # Linked posts
        for post_url in post.linked_posts[:5]:
            self.queue.add_post(
                url=post_url,
                priority=max(post.relevance_score - 5, 30),
                source_post_id=post.post_id,
                reason="Linked from relevant post",
            )

        # Track mention frequency for priority boosting
        profile_urls_to_track = []
        if post.author.profile_url:
            profile_urls_to_track.append(post.author.profile_url)
        profile_urls_to_track.extend(post.mentioned_profiles[:5])
        for url in profile_urls_to_track:
            self.queue.record_mention(url)

    async def _check_synthesis(self) -> None:
        """Check if it's time to run synthesis and do it if needed."""
        should_synthesize = False
        reason = ""

        # Check post count trigger
        if self.stats.posts_since_last_synthesis >= self.settings.synthesis_interval_posts:
            should_synthesize = True
            reason = f"Post threshold reached ({self.stats.posts_since_last_synthesis} posts)"

        # Check time trigger
        if self._last_synthesis_time:
            hours_since = (datetime.now() - self._last_synthesis_time).total_seconds() / 3600
            if hours_since >= self.settings.synthesis_interval_hours and self._pending_relevant_posts:
                should_synthesize = True
                reason = f"Time threshold reached ({hours_since:.1f} hours)"
        elif self._pending_relevant_posts and len(self._pending_relevant_posts) >= 5:
            # First synthesis after collecting some posts
            should_synthesize = True
            reason = "Initial synthesis"

        if should_synthesize:
            self._log("info", "synthesis_start", f"Starting synthesis: {reason}")
            self.stats.status = AgentStatus.SYNTHESIZING
            self.stats.current_action = "Synthesizing knowledge..."

            try:
                results = await self.synthesizer.synthesize(self._pending_relevant_posts)
                self._last_synthesis_time = datetime.now()
                self.stats.last_synthesis_at = self._last_synthesis_time.isoformat()
                self.stats.posts_since_last_synthesis = 0
                self._pending_relevant_posts = []
                self._log("info", "synthesis_done", f"Synthesis complete: {results}")
            except Exception as e:
                self._log("error", "synthesis_error", f"Synthesis failed: {e}")

            self.stats.status = AgentStatus.RUNNING

        # Weekly synthesis — every Monday or 7 days since last
        now = datetime.now()
        should_weekly = False
        if self._last_weekly_time is None:
            # First run: trigger on Monday or after 7 days of data
            atoms = self.store.get_all_atoms()
            if len(atoms) >= 5 and now.weekday() == 0:  # Monday
                should_weekly = True
        elif (now - self._last_weekly_time).days >= 7:
            should_weekly = True

        if should_weekly:
            self._log("info", "weekly_synthesis_start", "Starting weekly synthesis...")
            self.stats.status = AgentStatus.SYNTHESIZING
            self.stats.current_action = "Weekly knowledge map..."
            try:
                await self.weekly_synthesizer.run()
                self._last_weekly_time = now
                self._log("info", "weekly_synthesis_done", "Weekly map complete.")
            except Exception as e:
                self._log("error", "weekly_synthesis_error", f"Weekly synthesis failed: {e}")
            self.stats.status = AgentStatus.RUNNING

        # Monthly synthesis — 1st of month or 30+ days since last
        should_monthly = False
        if self._last_monthly_time is None:
            atoms = self.store.get_all_atoms()
            if len(atoms) >= 10 and now.day == 1:
                should_monthly = True
        elif (now - self._last_monthly_time).days >= 30:
            should_monthly = True

        if should_monthly:
            self._log("info", "monthly_synthesis_start", "Starting monthly synthesis...")
            self.stats.status = AgentStatus.SYNTHESIZING
            self.stats.current_action = "Monthly knowledge map..."
            try:
                await self.monthly_synthesizer.run()
                self._last_monthly_time = now
                self._log("info", "monthly_synthesis_done", "Monthly map complete.")
            except Exception as e:
                self._log("error", "monthly_synthesis_error", f"Monthly synthesis failed: {e}")
            self.stats.status = AgentStatus.RUNNING

    def _log(self, level: str, action: str, detail: str) -> None:
        """Add an activity log entry."""
        entry = ActivityLog(
            action=action,
            detail=detail,
            level=level,
        )
        self.activity_log.append(entry)
        # Keep only last 500 entries in memory
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
        return {
            "status": self.stats.status.value,
            "started_at": self.stats.started_at,
            "total_posts_scanned": self.stats.total_posts_scanned,
            "relevant_posts_found": self.stats.relevant_posts_found,
            "relevance_rate": (
                f"{(self.stats.relevant_posts_found / self.stats.total_posts_scanned * 100):.1f}%"
                if self.stats.total_posts_scanned > 0
                else "N/A"
            ),
            "posts_since_last_synthesis": self.stats.posts_since_last_synthesis,
            "last_synthesis_at": self.stats.last_synthesis_at or "Not yet",
            "atom_count": len(self.store.get_all_atoms()),
            "queue_size": self.queue.size,
            "queue_stats": self.queue.get_stats(),
            "current_action": self.stats.current_action,
            "current_url": self.stats.current_url,
            "token_usage": {
                "nano_input": self.token_usage.nano_input_tokens,
                "nano_output": self.token_usage.nano_output_tokens,
                "powerful_input": self.token_usage.powerful_input_tokens,
                "powerful_output": self.token_usage.powerful_output_tokens,
                "nano_cost": f"${self.token_usage.nano_cost:.4f}",
                "powerful_cost": f"${self.token_usage.powerful_cost:.4f}",
                "total_cost": f"${self.token_usage.total_cost:.4f}",
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
