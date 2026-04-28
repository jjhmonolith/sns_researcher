"""Facebook navigation — www.facebook.com for reliability."""

from __future__ import annotations

import asyncio
import logging
import random
from urllib.parse import quote_plus

from playwright.async_api import Page

from src.config import get_settings

logger = logging.getLogger(__name__)


class FBNavigator:
    """Handles Facebook navigation using mbasic (simple HTML) version.

    www.facebook.com has minimal JS, simpler DOM, and weaker bot detection
    compared to www.facebook.com. Trade-off: no infinite scroll, paginated instead.
    """

    def __init__(self, page: Page) -> None:
        self.page = page
        self.settings = get_settings()

    async def go_to_feed(self) -> None:
        """Navigate to Facebook news feed."""
        logger.info("[FB] Navigating to news feed...")
        await self.page.goto("https://www.facebook.com/", wait_until="domcontentloaded", timeout=30000)
        await self._wait_for_page_load()

    async def search_posts(self, keyword: str) -> None:
        """Search Facebook for posts matching a keyword."""
        encoded = quote_plus(keyword)
        url = f"https://www.facebook.com/search/posts/?q={encoded}&source=filter&isTrending=0"
        logger.info(f"[FB] Searching for: {keyword}")
        await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await self._wait_for_page_load()

    async def go_to_group(self, group_url: str) -> None:
        """Navigate to a Facebook group page."""
        # Convert www URL to mbasic
        url = group_url.replace("www.facebook.com", "www.facebook.com")
        url = url.replace("www.facebook.com", "www.facebook.com")
        logger.info(f"[FB] Navigating to group: {url[:60]}...")
        await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await self._wait_for_page_load()

    async def go_to_profile(self, profile_url: str) -> None:
        """Navigate to a user's profile page."""
        url = profile_url.replace("www.facebook.com", "www.facebook.com")
        url = url.replace("www.facebook.com", "www.facebook.com")
        logger.info(f"[FB] Navigating to profile: {url[:60]}...")
        await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await self._wait_for_page_load()

    async def go_to_post(self, url: str) -> None:
        """Navigate to a specific post."""
        url = url.replace("www.facebook.com", "www.facebook.com")
        url = url.replace("www.facebook.com", "www.facebook.com")
        logger.info(f"[FB] Navigating to post: {url[:80]}...")
        await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await self._wait_for_page_load()

    async def expand_all_posts(self) -> None:
        """Click 'See more' / '더 보기' on truncated posts to get full text."""
        try:
            see_more_selectors = [
                "div[role='button']:has-text('See more')",
                "div[role='button']:has-text('더 보기')",
                "div[dir='auto'] div[role='button']",
            ]
            for sel in see_more_selectors:
                buttons = await self.page.query_selector_all(sel)
                for btn in buttons[:15]:
                    try:
                        visible = await btn.is_visible()
                        text = (await btn.inner_text()).strip().lower()
                        if visible and text in ("see more", "더 보기"):
                            await btn.click()
                            await asyncio.sleep(random.uniform(0.5, 1.5))
                    except Exception:
                        continue
        except Exception as e:
            logger.debug(f"[FB] Error expanding posts: {e}")

    async def scroll_feed(self, scroll_count: int = 3) -> None:
        """Scroll the feed to load more posts."""
        for i in range(scroll_count):
            await self.page.evaluate(f"window.scrollBy(0, {random.randint(600, 1200)})")
            await asyncio.sleep(random.uniform(2.0, 5.0))
            if random.random() < 0.2:
                await asyncio.sleep(random.uniform(3.0, 8.0))
        await asyncio.sleep(2)

    async def load_more_posts(self, max_pages: int = 2) -> None:
        """Scroll + expand posts on www.facebook.com."""
        await self.scroll_feed(scroll_count=max_pages * 2)
        await self.expand_all_posts()

    async def is_rate_limited(self) -> bool:
        """Check if Facebook is showing a checkpoint/challenge."""
        try:
            url = self.page.url
            if "checkpoint" in url or "security" in url:
                return True
            body = await self.page.inner_text("body")
            for phrase in ["suspicious activity", "비정상적인 활동", "confirm your identity", "보안 확인"]:
                if phrase.lower() in body.lower()[:500]:
                    return True
        except Exception:
            pass
        return False

    async def handle_rate_limit(self) -> None:
        """Wait extra long — Facebook rate limits are harsh."""
        logger.warning("[FB] Rate limit/checkpoint detected! Waiting 20 minutes...")
        await asyncio.sleep(1200)

    async def random_delay(self, min_s: int | None = None, max_s: int | None = None) -> None:
        """Longer delays than LinkedIn/X to avoid Facebook detection."""
        mn = min_s if min_s is not None else max(self.settings.scroll_delay_min, 45)
        mx = max_s if max_s is not None else max(self.settings.scroll_delay_max, 180)
        delay = random.uniform(mn, mx)
        logger.info(f"[FB] Waiting {delay:.0f}s...")
        await asyncio.sleep(delay)

    async def short_delay(self) -> None:
        """Short delay — still longer than other platforms."""
        await asyncio.sleep(random.uniform(5, 15))

    async def _wait_for_page_load(self) -> None:
        try:
            await self.page.wait_for_load_state("domcontentloaded", timeout=15000)
        except Exception:
            pass
        await asyncio.sleep(random.uniform(2, 5))
