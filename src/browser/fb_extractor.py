"""Facebook content extraction — optimized for www.facebook.com HTML."""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime
from urllib.parse import urlparse

from playwright.async_api import Page, ElementHandle

from src.knowledge.models import LinkedInPost, Author, PostType, CrawlSource

logger = logging.getLogger(__name__)


def _generate_fb_post_id(url: str, content: str) -> str:
    """Generate deterministic ID — strip tracking params to avoid duplicates."""
    # Strip tracking params (__cft__, __tn__, etc.) before hashing
    clean_url = re.split(r"[?&]__cft__", url)[0]
    clean_url = re.split(r"[?&]__tn__", clean_url)[0]

    match = re.search(r"/story\.php\?story_fbid=(\d+)", clean_url)
    if match:
        return f"fb_{match.group(1)}"
    # New format: /posts/pfbid0XXXX
    match = re.search(r"/posts/(pfbid\w+)", clean_url)
    if match:
        return f"fb_{match.group(1)[:20]}"
    match = re.search(r"/posts/(\d+)", clean_url)
    if match:
        return f"fb_{match.group(1)}"
    match = re.search(r"fbid=(\d+)", clean_url)
    if match:
        return f"fb_{match.group(1)}"
    # Fallback: hash cleaned URL + content
    h = hashlib.md5((clean_url + content[:200]).encode()).hexdigest()[:12]
    return f"fb_{h}"


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    lines = text.split("\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
    text = "\n".join(line for line in lines if line)
    return text.strip()


class FBContentExtractor:
    """Extracts posts from www.facebook.com pages.

    mbasic uses simple server-rendered HTML — no React, no complex JS.
    DOM structure is much simpler than www.facebook.com.
    """

    def __init__(self, page: Page) -> None:
        self.page = page

    async def extract_feed_posts(self) -> list[LinkedInPost]:
        """Extract posts from the mbasic news feed."""
        return await self._extract_posts(CrawlSource.FB_FEED)

    async def extract_search_results(self) -> list[LinkedInPost]:
        """Extract posts from mbasic search results."""
        return await self._extract_posts(CrawlSource.FB_SEARCH)

    async def extract_group_posts(self) -> list[LinkedInPost]:
        """Extract posts from a group page."""
        return await self._extract_posts(CrawlSource.FB_GROUP)

    async def extract_profile_posts(self) -> list[LinkedInPost]:
        """Extract posts from a user's profile/timeline."""
        return await self._extract_posts(CrawlSource.FB_PROFILE, limit=10)

    async def _extract_posts(
        self, source: CrawlSource, limit: int = 20
    ) -> list[LinkedInPost]:
        """Core extraction for m.facebook.com pages."""
        posts = []

        # m.facebook.com uses div[role='article'] or div[data-ft] for posts
        post_selectors = [
            "div[role='article']",
            "article",
            "div[data-ft]",
            "div[data-store]",
        ]

        elements = []
        for sel in post_selectors:
            elements = await self.page.query_selector_all(sel)
            if elements:
                break

        if not elements:
            logger.warning("[FB] No post elements found.")
            return posts

        for element in elements[:limit]:
            try:
                post = await self._extract_single_post(element, source)
                if post and post.content and len(post.content) > 15:
                    posts.append(post)
            except Exception as e:
                logger.debug(f"[FB] Failed to extract post: {e}")
                continue

        logger.info(f"[FB] Extracted {len(posts)} posts.")
        return posts

    async def _extract_single_post(
        self, element: ElementHandle, source: CrawlSource
    ) -> LinkedInPost | None:
        """Extract data from a single mbasic post element."""
        author = await self._extract_author(element)
        content = await self._extract_content(element)
        if not content:
            return None

        url = await self._extract_post_url(element)
        date = await self._extract_date(element)
        reactions, comments = await self._extract_metrics(element)
        profile_urls, post_urls, external_urls = self._extract_urls(content)

        post_id = _generate_fb_post_id(url or "", content)

        return LinkedInPost(
            post_id=post_id,
            url=url,
            author=author,
            content=_clean_text(content),
            post_type=PostType.ORIGINAL,
            platform="facebook",
            published_date=date,
            reactions_count=reactions,
            comments_count=comments,
            mentioned_profiles=profile_urls,
            external_links=external_urls,
            linked_posts=post_urls,
            crawl_source=source,
        )

    async def _extract_author(self, element: ElementHandle) -> Author:
        author = Author()
        try:
            # m.facebook.com: author name in various containers
            name_selectors = [
                "h3 a strong",
                "h3 a",
                "h4 a",
                "strong > a",
                "a[data-hovercard]",
                "header a",
                "span > a[href*='facebook.com']",
            ]
            for sel in name_selectors:
                el = await element.query_selector(sel)
                if el:
                    author.name = _clean_text(await el.inner_text())
                    href = await el.get_attribute("href")
                    if href:
                        if href.startswith("/"):
                            author.profile_url = f"https://www.facebook.com{href.split('?')[0]}"
                        elif "facebook.com" in href:
                            author.profile_url = href.split("?")[0]
                    if author.name:
                        break
        except Exception as e:
            logger.debug(f"[FB] Error extracting author: {e}")
        return author

    async def _extract_content(self, element: ElementHandle) -> str:
        """Extract post text content."""
        content_selectors = [
            "div[data-ft] > div > span",  # mbasic content span
            "div.bq > div",  # common mbasic content class
            "p",
        ]
        for sel in content_selectors:
            els = await element.query_selector_all(sel)
            if els:
                texts = []
                for el in els:
                    text = await el.inner_text()
                    if text and len(text.strip()) > 3:
                        texts.append(text.strip())
                combined = "\n".join(texts)
                if len(combined) > 10:
                    return combined

        # Fallback: get all text from the element
        try:
            full_text = await element.inner_text()
            # Remove common mbasic noise (Like, Comment, Share buttons text)
            noise_patterns = [
                r"(Like|좋아요|Comment|댓글|Share|공유)\s*$",
                r"^\d+ (likes?|comments?|shares?|좋아요|댓글|공유)",
            ]
            lines = full_text.split("\n")
            clean_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped and not any(re.match(p, stripped, re.IGNORECASE) for p in noise_patterns):
                    clean_lines.append(stripped)
            return "\n".join(clean_lines[:20])  # Cap to avoid huge blocks
        except Exception:
            pass
        return ""

    async def _extract_post_url(self, element: ElementHandle) -> str:
        try:
            link_selectors = [
                "a[href*='/posts/']",
                "a[href*='story.php']",
                "a[href*='permalink']",
            ]
            for sel in link_selectors:
                el = await element.query_selector(sel)
                if el:
                    href = await el.get_attribute("href")
                    if href:
                        # Strip tracking params
                        href = re.split(r"[?&]__cft__", href)[0]
                        href = re.split(r"[?&]__tn__", href)[0]
                        if href.startswith("/"):
                            return f"https://www.facebook.com{href}"
                        return href
        except Exception:
            pass
        return ""

    async def _extract_date(self, element: ElementHandle) -> str:
        try:
            # mbasic: date is usually in an <abbr> tag or small text near the author
            abbr = await element.query_selector("abbr")
            if abbr:
                title = await abbr.get_attribute("title")
                if title:
                    iso = re.search(r"\d{4}-\d{2}-\d{2}", title)
                    if iso:
                        return iso.group(0)
                text = await abbr.inner_text()
                if text:
                    return self._parse_fb_date(text)

            # Fallback: look for timestamp-like text
            time_selectors = ["span.bt", "td.bn abbr"]
            for sel in time_selectors:
                el = await element.query_selector(sel)
                if el:
                    text = await el.inner_text()
                    return self._parse_fb_date(text)
        except Exception:
            pass
        return ""

    @staticmethod
    def _parse_fb_date(text: str) -> str:
        """Parse Facebook date strings into YYYY-MM-DD."""
        if not text:
            return ""
        from datetime import timedelta
        now = datetime.now()
        text = text.strip()

        patterns = [
            (r"(\d+)\s*(hr|hour|시간)", "hours"),
            (r"(\d+)\s*(min|분)", "minutes"),
            (r"(\d+)\s*(d|day|일)", "days"),
            (r"Just now|방금", "now"),
            (r"Yesterday|어제", "yesterday"),
        ]
        for pattern, unit in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                if unit == "now":
                    return now.strftime("%Y-%m-%d")
                if unit == "yesterday":
                    return (now - timedelta(days=1)).strftime("%Y-%m-%d")
                n = int(m.group(1))
                if unit == "minutes":
                    return (now - timedelta(minutes=n)).strftime("%Y-%m-%d")
                if unit == "hours":
                    return (now - timedelta(hours=n)).strftime("%Y-%m-%d")
                if unit == "days":
                    return (now - timedelta(days=n)).strftime("%Y-%m-%d")

        iso = re.search(r"\d{4}-\d{2}-\d{2}", text)
        if iso:
            return iso.group(0)
        return ""

    async def _extract_metrics(self, element: ElementHandle) -> tuple[int, int]:
        reactions = comments = 0
        try:
            # mbasic shows reactions/comments as text like "12 Likes" or "3 Comments"
            text = await element.inner_text()
            react_match = re.search(r"(\d+)\s*(like|좋아요|reaction)", text, re.IGNORECASE)
            if react_match:
                reactions = int(react_match.group(1))
            comment_match = re.search(r"(\d+)\s*(comment|댓글)", text, re.IGNORECASE)
            if comment_match:
                comments = int(comment_match.group(1))
        except Exception:
            pass
        return reactions, comments

    @staticmethod
    def _extract_urls(text: str) -> tuple[list[str], list[str], list[str]]:
        profiles: list[str] = []
        posts: list[str] = []
        externals: list[str] = []

        url_pattern = re.compile(r"https?://[^\s<>\"')\]]+")
        for url in url_pattern.findall(text):
            url = url.rstrip(".,;:!?)")
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if "facebook.com" in domain:
                if "/posts/" in url or "story.php" in url or "permalink" in url:
                    posts.append(url)
                elif "/profile.php" in url or re.match(r"^/[a-zA-Z0-9.]+/?$", parsed.path):
                    profiles.append(url)
            elif domain:
                externals.append(url)

        return profiles, posts, externals
