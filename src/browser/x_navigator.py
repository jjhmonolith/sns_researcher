"""X (Twitter) navigation - feed, search, profile, rate limiting."""

from __future__ import annotations

import asyncio
import logging
import random
from urllib.parse import quote_plus

from playwright.async_api import Page

from src.config import get_settings

logger = logging.getLogger(__name__)


class XNavigator:
    """Handles X/Twitter page navigation with human-like behavior."""

    def __init__(self, page: Page) -> None:
        self.page = page
        self.settings = get_settings()

    async def go_to_feed(self) -> None:
        """Navigate to X home timeline."""
        logger.info("[X] Navigating to home feed...")
        await self.page.goto("https://x.com/home", wait_until="domcontentloaded", timeout=30000)
        await self._wait_for_page_load()

    async def scroll_feed(self, scroll_count: int = 5) -> None:
        """Scroll the feed with human-like behavior."""
        for i in range(scroll_count):
            scroll_distance = random.randint(500, 1000)
            await self.page.evaluate(f"window.scrollBy(0, {scroll_distance})")
            pause = random.uniform(2.0, 5.0)
            await asyncio.sleep(pause)
            if random.random() < 0.2:
                await asyncio.sleep(random.uniform(3.0, 8.0))
        await asyncio.sleep(2)

    async def search_posts(self, keyword: str) -> None:
        """Search X for posts matching a keyword (latest tab)."""
        encoded = quote_plus(keyword)
        search_url = f"https://x.com/search?q={encoded}&src=typed_query&f=live"
        logger.info(f"[X] Searching for: {keyword}")
        await self.page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
        await self._wait_for_page_load()

    async def go_to_profile(self, handle_or_url: str) -> None:
        """Navigate to a user's profile page."""
        if handle_or_url.startswith("http"):
            url = handle_or_url
        elif handle_or_url.startswith("@"):
            url = f"https://x.com/{handle_or_url[1:]}"
        else:
            url = f"https://x.com/{handle_or_url}"

        logger.info(f"[X] Navigating to profile: {url[:60]}...")
        await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await self._wait_for_page_load()

    async def go_to_post(self, url: str) -> None:
        """Navigate to a specific post/tweet URL."""
        logger.info(f"[X] Navigating to post: {url[:80]}...")
        await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await self._wait_for_page_load()

    async def is_rate_limited(self) -> bool:
        """Check if X is showing a rate limit or challenge page."""
        try:
            url = self.page.url
            if "suspended" in url or "/i/flow/" in url:
                return True
            body_text = await self.page.inner_text("body")
            rate_phrases = [
                "rate limit",
                "something went wrong",
                "try again",
                "비정상적인 활동",
            ]
            for phrase in rate_phrases:
                if phrase.lower() in body_text.lower()[:500]:
                    return True
        except Exception:
            pass
        return False

    async def handle_rate_limit(self) -> None:
        """Handle rate limiting — wait longer than LinkedIn."""
        logger.warning("[X] Rate limit detected! Waiting 15 minutes...")
        await asyncio.sleep(900)

    async def random_delay(self, min_s: int | None = None, max_s: int | None = None) -> None:
        """Wait a random amount of time to avoid detection."""
        mn = min_s if min_s is not None else self.settings.scroll_delay_min
        mx = max_s if max_s is not None else self.settings.scroll_delay_max
        delay = random.uniform(mn, mx)
        logger.info(f"[X] Waiting {delay:.0f}s...")
        await asyncio.sleep(delay)

    async def short_delay(self) -> None:
        """Short delay between actions (3-10s)."""
        await asyncio.sleep(random.uniform(3, 10))

    async def _wait_for_page_load(self) -> None:
        """Wait for page to finish loading."""
        try:
            await self.page.wait_for_load_state("domcontentloaded", timeout=15000)
        except Exception:
            pass
        await asyncio.sleep(random.uniform(2, 4))
