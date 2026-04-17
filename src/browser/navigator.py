"""Browser navigation - feed scrolling, search, profile visits, rate limiting."""

from __future__ import annotations

import asyncio
import logging
import random
from urllib.parse import quote_plus

from playwright.async_api import Page

from src.config import get_settings

logger = logging.getLogger(__name__)

# Global LinkedIn search URLs by language/region
# LinkedIn personalizes feed by account language setting;
# to see global/English content we use search with English keywords + geo filters
GLOBAL_SEARCH_SORT = "date_posted"

# Curated hashtag feeds that surface international AX content
GLOBAL_HASHTAG_FEEDS = [
    # English - highest volume
    "https://www.linkedin.com/feed/hashtag/aitransformation/",
    "https://www.linkedin.com/feed/hashtag/enterpriseai/",
    "https://www.linkedin.com/feed/hashtag/digitaltransformation/",
    "https://www.linkedin.com/feed/hashtag/aiagents/",
    "https://www.linkedin.com/feed/hashtag/generativeai/",
    # Japanese
    "https://www.linkedin.com/feed/hashtag/ai変革/",
    "https://www.linkedin.com/feed/hashtag/デジタルトランスフォーメーション/",
    # Chinese
    "https://www.linkedin.com/feed/hashtag/ai转型/",
]


class Navigator:
    """Handles LinkedIn page navigation with human-like behavior."""

    def __init__(self, page: Page) -> None:
        self.page = page
        self.settings = get_settings()
        self._hashtag_feed_index = 0

    async def go_to_feed(self) -> None:
        """Navigate to LinkedIn home feed."""
        logger.info("Navigating to home feed...")
        await self.page.goto(
            "https://www.linkedin.com/feed/",
            wait_until="domcontentloaded",
            timeout=30000,
        )
        await self._wait_for_page_load()

    async def scroll_feed(self, scroll_count: int = 5) -> None:
        """Scroll the feed with human-like behavior to load more posts.
        
        Args:
            scroll_count: Number of scroll actions to perform.
        """
        for i in range(scroll_count):
            # Randomize scroll distance
            scroll_distance = random.randint(600, 1200)
            await self.page.evaluate(f"window.scrollBy(0, {scroll_distance})")
            logger.debug(f"Scroll {i+1}/{scroll_count}: {scroll_distance}px")

            # Random short pause between scrolls (human-like)
            pause = random.uniform(1.5, 4.0)
            await asyncio.sleep(pause)

            # Occasionally pause longer (reading behavior)
            if random.random() < 0.2:
                long_pause = random.uniform(3.0, 8.0)
                logger.debug(f"Simulating reading pause: {long_pause:.1f}s")
                await asyncio.sleep(long_pause)

        # Wait for any lazy-loaded content
        await asyncio.sleep(2)

    async def expand_all_posts(self) -> None:
        """Click 'see more' buttons on truncated posts to get full content."""
        try:
            see_more_selectors = [
                "button.feed-shared-inline-show-more-text__see-more-less-toggle",
                "button[aria-label*='더 보기']",
                "button[aria-label*='see more']",
                "button.see-more",
            ]
            for selector in see_more_selectors:
                buttons = await self.page.query_selector_all(selector)
                for button in buttons[:15]:  # Limit to avoid too many clicks
                    try:
                        visible = await button.is_visible()
                        if visible:
                            await button.click()
                            await asyncio.sleep(random.uniform(0.3, 0.8))
                    except Exception:
                        continue
        except Exception as e:
            logger.debug(f"Error expanding posts: {e}")

    async def expand_comments(self, max_clicks: int = 2) -> None:
        """Click to show/load comments on a post page."""
        try:
            # Click the comments button to open the comment section
            comment_btn_selectors = [
                "button[aria-label*='comment']",
                "button[aria-label*='댓글']",
            ]
            for sel in comment_btn_selectors:
                btn = await self.page.query_selector(sel)
                if btn and await btn.is_visible():
                    await btn.click()
                    await asyncio.sleep(random.uniform(1.5, 3.0))
                    break

            # Click "show more comments" buttons
            more_selectors = [
                "button.comments-comments-list__load-more-comments-button",
                "button[aria-label*='more comments']",
                "button[aria-label*='이전 댓글']",
            ]
            for _ in range(max_clicks):
                clicked = False
                for sel in more_selectors:
                    btn = await self.page.query_selector(sel)
                    if btn and await btn.is_visible():
                        await btn.click()
                        await asyncio.sleep(random.uniform(1.0, 2.5))
                        clicked = True
                        break
                if not clicked:
                    break
        except Exception as e:
            logger.debug(f"Error expanding comments: {e}")

    async def search_posts(self, keyword: str, date_filter: str = "past-week") -> None:
        """Search LinkedIn for posts matching a keyword.
        
        Args:
            keyword: Search term to use.
            date_filter: One of 'past-24h', 'past-week', 'past-month'.
        """
        encoded = quote_plus(keyword)
        date_param = {
            "past-24h": "&datePosted=%22past-24h%22",
            "past-week": "&datePosted=%22past-week%22",
            "past-month": "&datePosted=%22past-month%22",
        }.get(date_filter, "")
        search_url = (
            f"https://www.linkedin.com/search/results/content/"
            f"?keywords={encoded}&origin=GLOBAL_SEARCH_HEADER"
            f"&sortBy=%22{GLOBAL_SEARCH_SORT}%22{date_param}"
        )
        logger.info(f"Searching for: {keyword}")
        await self.page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
        await self._wait_for_page_load()

    async def go_to_global_hashtag_feed(self) -> str:
        """Navigate to the next hashtag feed in rotation (for international content).
        
        Returns:
            The hashtag URL visited.
        """
        url = GLOBAL_HASHTAG_FEEDS[self._hashtag_feed_index % len(GLOBAL_HASHTAG_FEEDS)]
        self._hashtag_feed_index += 1
        logger.info(f"Navigating to hashtag feed: {url}")
        await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await self._wait_for_page_load()
        return url

    async def go_to_post(self, url: str) -> None:
        """Navigate to a specific post URL.
        
        Args:
            url: Full LinkedIn post URL.
        """
        logger.info(f"Navigating to post: {url[:80]}...")
        await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await self._wait_for_page_load()

    async def go_to_profile(self, url: str) -> None:
        """Navigate to a LinkedIn profile page.
        
        Args:
            url: Full LinkedIn profile URL.
        """
        # Ensure we go to the "recent activity" section
        if "/recent-activity/" not in url and "/detail/" not in url:
            url = url.rstrip("/") + "/recent-activity/all/"

        logger.info(f"Navigating to profile: {url[:80]}...")
        await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await self._wait_for_page_load()

    async def random_delay(self, min_override: int | None = None, max_override: int | None = None) -> None:
        """Wait a random amount of time to avoid detection.
        
        Args:
            min_override: Override minimum delay in seconds.
            max_override: Override maximum delay in seconds.
        """
        min_delay = min_override if min_override is not None else self.settings.scroll_delay_min
        max_delay = max_override if max_override is not None else self.settings.scroll_delay_max
        delay = random.uniform(min_delay, max_delay)
        logger.info(f"Waiting {delay:.0f}s before next action...")
        await asyncio.sleep(delay)

    async def short_delay(self) -> None:
        """Short delay for between-action pauses (3-10s)."""
        delay = random.uniform(3, 10)
        await asyncio.sleep(delay)

    async def is_rate_limited(self) -> bool:
        """Check if LinkedIn is showing a rate limit or challenge page."""
        try:
            url = self.page.url
            if "checkpoint" in url or "challenge" in url:
                return True

            # Check for common rate limit indicators
            body_text = await self.page.inner_text("body")
            rate_limit_phrases = [
                "unusual activity",
                "비정상적인 활동",
                "verify your identity",
                "security verification",
            ]
            for phrase in rate_limit_phrases:
                if phrase.lower() in body_text.lower():
                    return True
        except Exception:
            pass
        return False

    async def handle_rate_limit(self) -> None:
        """Handle rate limiting - wait a long time and retry."""
        logger.warning("Rate limit detected! Waiting 10 minutes before retrying...")
        await asyncio.sleep(600)  # 10 minutes

    async def _wait_for_page_load(self) -> None:
        """Wait for page to finish loading key elements."""
        try:
            await self.page.wait_for_load_state("domcontentloaded", timeout=15000)
        except Exception:
            pass
        # Additional wait for dynamic content
        await asyncio.sleep(random.uniform(2, 4))

    async def get_current_url(self) -> str:
        """Get the current page URL."""
        return self.page.url
