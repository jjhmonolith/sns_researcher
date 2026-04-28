"""Facebook browser session — stealth mode with mbasic.facebook.com."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Playwright

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

FB_COOKIES_PATH = DATA_DIR / "cookies_fb.json"


class FBBrowserSession:
    """Manages Playwright browser for Facebook with stealth patches.

    Cookies are saved to the session-specific path AND a shared global path.
    On start, if session cookies don't exist, the global cookies are tried first.
    This allows logging in once and reusing across sessions.
    """

    def __init__(self, cookies_path: Path | None = None) -> None:
        self._cookies_path = cookies_path or FB_COOKIES_PATH
        self._global_cookies_path = FB_COOKIES_PATH  # shared fallback
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._is_logged_in: bool = False

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("FB browser session not started.")
        return self._page

    @property
    def is_logged_in(self) -> bool:
        return self._is_logged_in

    async def start(self, headless: bool = True) -> None:
        self._cookies_path.parent.mkdir(parents=True, exist_ok=True)

        # Try session cookies first, then global shared cookies
        has_cookies = self._cookies_path.exists() and self._cookies_path.stat().st_size > 10
        if not has_cookies and self._global_cookies_path != self._cookies_path:
            if self._global_cookies_path.exists() and self._global_cookies_path.stat().st_size > 10:
                import shutil
                shutil.copy2(self._global_cookies_path, self._cookies_path)
                logger.info("[FB] Copied global cookies to session.")
                has_cookies = True

        if not has_cookies:
            logger.info("[FB] No saved cookies. Starting headed browser for login...")
            await self._start_browser(headless=False)
            await self._wait_for_manual_login()
        else:
            logger.info("[FB] Restoring session from cookies...")
            await self._start_browser(headless=headless)
            await self._restore_cookies()

            if not await self._verify_session():
                logger.warning("[FB] Session expired. Restarting for login...")
                await self.stop()
                await self._start_browser(headless=False)
                await self._wait_for_manual_login()

        self._is_logged_in = True
        logger.info("[FB] Session is active.")

    async def _start_browser(self, headless: bool = True) -> None:
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=headless,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
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
        # Apply stealth patches
        try:
            from playwright_stealth import stealth_async
            await stealth_async(self._context)
            logger.info("[FB] Stealth patches applied.")
        except ImportError:
            logger.warning("[FB] playwright-stealth not installed. Running without stealth.")
            await self._context.add_init_script(
                "Object.defineProperty(navigator, 'webdriver', { get: () => undefined });"
            )

        self._page = await self._context.new_page()

    async def _wait_for_manual_login(self) -> None:
        page = self.page
        # Use m.facebook.com for login — better cookie support than mbasic
        await page.goto("https://www.facebook.com/login/")
        logger.info("=" * 60)
        logger.info("Please log in to Facebook in the browser window.")
        logger.info("The agent will detect login automatically.")
        logger.info("=" * 60)

        max_wait = 300
        elapsed = 0
        while elapsed < max_wait:
            try:
                # Check if c_user cookie exists — this is the definitive login signal
                cookies = await self._context.cookies("https://www.facebook.com")
                cookie_names = {c["name"] for c in cookies}
                if "c_user" in cookie_names and "xs" in cookie_names:
                    logger.info(f"[FB] Login confirmed — c_user + xs cookies present ({len(cookies)} total).")
                    break
            except Exception:
                pass
            await asyncio.sleep(2)
            elapsed += 2

        if elapsed >= max_wait:
            raise TimeoutError("[FB] Login timed out.")

        # Wait for all redirects to complete and cookies to settle
        await asyncio.sleep(5)
        await self._save_cookies()

    async def _save_cookies(self) -> None:
        if self._context is None:
            return
        cookies = await self._context.cookies()
        data = json.dumps(cookies, indent=2, ensure_ascii=False)
        self._cookies_path.write_text(data)
        # Also save to global path so other sessions can reuse
        if self._global_cookies_path != self._cookies_path:
            self._global_cookies_path.parent.mkdir(parents=True, exist_ok=True)
            self._global_cookies_path.write_text(data)
        logger.info(f"[FB] Saved {len(cookies)} cookies.")

    async def _restore_cookies(self) -> None:
        if self._context is None or not self._cookies_path.exists():
            return
        cookies = json.loads(self._cookies_path.read_text())
        await self._context.add_cookies(cookies)
        logger.info(f"[FB] Restored {len(cookies)} cookies.")

    async def _verify_session(self) -> bool:
        page = self.page
        try:
            # First check: do we have the essential cookies?
            cookies = await self._context.cookies("https://www.facebook.com")
            cookie_names = {c["name"] for c in cookies}
            if "c_user" not in cookie_names:
                logger.warning("[FB] No c_user cookie — session invalid.")
                return False

            await page.goto("https://www.facebook.com/", wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(4)
            url = page.url
            if "login" in url or "checkpoint" in url:
                return False
            # Double check: no login form visible
            login_form = await page.query_selector("input[name='email']")
            if login_form:
                return False
            return True
        except Exception as e:
            logger.error(f"[FB] Session verification error: {e}")
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
        logger.info("[FB] Browser session closed.")
