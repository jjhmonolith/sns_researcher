"""X (Twitter) browser session - login, cookie persistence, session restoration."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Playwright

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

X_COOKIES_PATH = DATA_DIR / "cookies_x.json"


class XBrowserSession:
    """Manages Playwright browser for X/Twitter."""

    def __init__(self) -> None:
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._is_logged_in: bool = False

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("X browser session not started. Call start() first.")
        return self._page

    @property
    def is_logged_in(self) -> bool:
        return self._is_logged_in

    async def start(self, headless: bool = True) -> None:
        """Start browser. Restore cookies or open headed for manual login."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        has_cookies = X_COOKIES_PATH.exists() and X_COOKIES_PATH.stat().st_size > 10

        if not has_cookies:
            logger.info("[X] No saved cookies. Starting headed browser for manual login...")
            await self._start_browser(headless=False)
            await self._wait_for_manual_login()
        else:
            logger.info("[X] Restoring session from saved cookies...")
            await self._start_browser(headless=headless)
            await self._restore_cookies()

            if not await self._verify_session():
                logger.warning("[X] Session expired. Restarting for manual login...")
                await self.stop()
                await self._start_browser(headless=False)
                await self._wait_for_manual_login()

        self._is_logged_in = True
        logger.info("[X] Session is active.")

    async def _start_browser(self, headless: bool = True) -> None:
        """Launch Playwright browser with anti-detection for X."""
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        )
        self._context = await self._browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            locale="ko-KR",
            timezone_id="Asia/Seoul",
        )
        await self._context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        """)
        self._page = await self._context.new_page()

    async def _wait_for_manual_login(self) -> None:
        """Navigate to X login and wait for user to complete login."""
        page = self.page
        await page.goto("https://x.com/i/flow/login")
        logger.info("=" * 60)
        logger.info("Please log in to X (Twitter) in the browser window.")
        logger.info("The agent will detect login automatically.")
        logger.info("=" * 60)

        max_wait = 300
        elapsed = 0
        while elapsed < max_wait:
            try:
                url = page.url
                if "/home" in url:
                    logger.info("[X] Login detected via URL redirect to /home.")
                    break
                # Check for logged-in nav (X uses aria-label on nav)
                nav = await page.query_selector("nav[aria-label='Primary']")
                if not nav:
                    nav = await page.query_selector("a[aria-label='Home']")
                if nav:
                    logger.info("[X] Login detected via navigation element.")
                    break
            except Exception:
                pass
            await asyncio.sleep(2)
            elapsed += 2

        if elapsed >= max_wait:
            raise TimeoutError("[X] Login timed out after 5 minutes.")

        await asyncio.sleep(3)
        await self._save_cookies()

    async def _save_cookies(self) -> None:
        if self._context is None:
            return
        cookies = await self._context.cookies()
        X_COOKIES_PATH.write_text(json.dumps(cookies, indent=2, ensure_ascii=False))
        logger.info(f"[X] Saved {len(cookies)} cookies.")

    async def _restore_cookies(self) -> None:
        if self._context is None or not X_COOKIES_PATH.exists():
            return
        cookies = json.loads(X_COOKIES_PATH.read_text())
        await self._context.add_cookies(cookies)
        logger.info(f"[X] Restored {len(cookies)} cookies.")

    async def _verify_session(self) -> bool:
        """Navigate to X home and check if logged in."""
        page = self.page
        try:
            await page.goto("https://x.com/home", wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(3)

            url = page.url
            if "/login" in url or "/i/flow/login" in url:
                return False

            # Check for home timeline indicator
            nav = await page.query_selector("a[aria-label='Home']")
            if nav:
                return True

            timeline = await page.query_selector("div[data-testid='primaryColumn']")
            if timeline:
                return True

            return False
        except Exception as e:
            logger.error(f"[X] Session verification error: {e}")
            return False

    async def refresh_cookies(self) -> None:
        await self._save_cookies()

    async def stop(self) -> None:
        try:
            if self._context:
                await self._save_cookies()
        except Exception:
            pass
        try:
            if self._browser:
                await self._browser.close()
        except Exception:
            pass
        try:
            if self._playwright:
                await self._playwright.stop()
        except Exception:
            pass
        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None
        self._is_logged_in = False
        logger.info("[X] Browser session closed.")
