"""Browser session management - login, cookie persistence, session restoration."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Playwright

from src.config import COOKIES_PATH, DATA_DIR, get_settings

logger = logging.getLogger(__name__)


class BrowserSession:
    """Manages Playwright browser lifecycle, login, and cookie persistence."""

    def __init__(self, cookies_path: Path | None = None) -> None:
        self._cookies_path = cookies_path or COOKIES_PATH
        self._global_cookies_path = COOKIES_PATH
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._is_logged_in: bool = False

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("Browser session not started. Call start() first.")
        return self._page

    @property
    def is_logged_in(self) -> bool:
        return self._is_logged_in

    async def start(self, headless: bool = True) -> None:
        """Start the browser. If cookies exist, restore session. Otherwise open headed for login."""
        self._cookies_path.parent.mkdir(parents=True, exist_ok=True)

        has_cookies = self._cookies_path.exists() and self._cookies_path.stat().st_size > 10
        if not has_cookies and self._global_cookies_path != self._cookies_path:
            if self._global_cookies_path.exists() and self._global_cookies_path.stat().st_size > 10:
                import shutil
                shutil.copy2(self._global_cookies_path, self._cookies_path)
                logger.info("Copied global LinkedIn cookies to session.")
                has_cookies = True

        if not has_cookies:
            logger.info("No saved cookies found. Starting headed browser for manual login...")
            await self._start_browser(headless=False)
            await self._wait_for_manual_login()
        else:
            logger.info("Restoring session from saved cookies...")
            await self._start_browser(headless=headless)
            await self._restore_cookies()

            # Verify session is still valid
            if not await self._verify_session():
                logger.warning("Saved session expired. Restarting for manual login...")
                await self.stop()
                await self._start_browser(headless=False)
                await self._wait_for_manual_login()

        self._is_logged_in = True
        logger.info("LinkedIn session is active.")

    async def _start_browser(self, headless: bool = True) -> None:
        """Launch Playwright browser with anti-detection settings."""
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
        # Mask webdriver detection
        await self._context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        """)
        self._page = await self._context.new_page()

    async def _wait_for_manual_login(self) -> None:
        """Navigate to LinkedIn login and wait for user to complete login manually."""
        page = self.page
        await page.goto("https://www.linkedin.com/login")
        logger.info("=" * 60)
        logger.info("Please log in to LinkedIn manually in the browser window.")
        logger.info("The agent will detect login automatically and continue.")
        logger.info("=" * 60)

        # Poll until we detect successful login (feed page or nav element)
        max_wait = 300  # 5 minutes
        elapsed = 0
        while elapsed < max_wait:
            try:
                url = page.url
                # Check if redirected to feed or if the global nav is present
                if "/feed" in url or "/mynetwork" in url:
                    logger.info("Login detected via URL redirect.")
                    break

                # Check for the presence of the global navigation bar (logged-in indicator)
                nav = await page.query_selector("div.global-nav__content")
                if nav:
                    logger.info("Login detected via navigation bar.")
                    break
            except Exception:
                pass

            await asyncio.sleep(2)
            elapsed += 2

        if elapsed >= max_wait:
            raise TimeoutError("Login timed out after 5 minutes. Please restart.")

        # Small delay to let the page fully load
        await asyncio.sleep(3)
        await self._save_cookies()
        logger.info("Cookies saved successfully.")

    async def _save_cookies(self) -> None:
        """Save current browser cookies to disk (session + global)."""
        if self._context is None:
            return
        cookies = await self._context.cookies()
        data = json.dumps(cookies, indent=2, ensure_ascii=False)
        self._cookies_path.write_text(data)
        if self._global_cookies_path != self._cookies_path:
            self._global_cookies_path.parent.mkdir(parents=True, exist_ok=True)
            self._global_cookies_path.write_text(data)
        logger.info(f"Saved {len(cookies)} cookies.")

    async def _restore_cookies(self) -> None:
        """Restore cookies from disk into the browser context."""
        if self._context is None or not self._cookies_path.exists():
            return
        cookies = json.loads(self._cookies_path.read_text())
        await self._context.add_cookies(cookies)
        logger.info(f"Restored {len(cookies)} cookies.")

    async def _verify_session(self) -> bool:
        """Navigate to LinkedIn feed and check if we're still logged in."""
        page = self.page
        try:
            await page.goto("https://www.linkedin.com/feed/", wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(3)

            url = page.url
            if "/login" in url or "/authwall" in url or "checkpoint" in url:
                logger.warning("Session verification failed - redirected to login.")
                return False

            # Check for logged-in indicator
            nav = await page.query_selector("div.global-nav__content")
            if nav:
                return True

            # Fallback: check for feed content
            feed = await page.query_selector("div.scaffold-layout__main")
            if feed:
                return True

            return False
        except Exception as e:
            logger.error(f"Session verification error: {e}")
            return False

    async def refresh_cookies(self) -> None:
        """Refresh and save current cookies (call periodically)."""
        await self._save_cookies()

    async def stop(self) -> None:
        """Close the browser and clean up resources."""
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
        logger.info("Browser session closed.")
