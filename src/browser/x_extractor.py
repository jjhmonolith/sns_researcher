"""X (Twitter) content extraction - tweets, authors, metrics."""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime
from urllib.parse import urlparse

from playwright.async_api import Page, ElementHandle

from src.knowledge.models import LinkedInPost, Author, PostType, CrawlSource

logger = logging.getLogger(__name__)


def _generate_tweet_id(url: str, content: str) -> str:
    """Generate a deterministic ID from tweet URL or content."""
    match = re.search(r"/status/(\d+)", url)
    if match:
        return f"x_{match.group(1)}"
    h = hashlib.md5((url + content[:200]).encode()).hexdigest()[:12]
    return f"x_{h}"


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    lines = text.split("\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
    text = "\n".join(line for line in lines if line)
    return text.strip()


def _parse_relative_date(text: str) -> str:
    """Parse X relative dates like '2h', '3d', '1시간' into YYYY-MM-DD."""
    if not text:
        return ""
    text = text.strip()
    now = datetime.now()
    patterns = [
        (r"(\d+)\s*시간", "hours"), (r"(\d+)\s*분", "minutes"),
        (r"(\d+)\s*일", "days"), (r"(\d+)\s*h\b", "hours"),
        (r"(\d+)\s*m\b", "minutes"), (r"(\d+)\s*d\b", "days"),
        (r"(\d+)\s*s\b", "seconds"),
    ]
    from datetime import timedelta
    for pattern, unit in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            n = int(m.group(1))
            delta = {"seconds": timedelta(seconds=n), "minutes": timedelta(minutes=n),
                     "hours": timedelta(hours=n), "days": timedelta(days=n)}.get(unit)
            if delta:
                return (now - delta).strftime("%Y-%m-%d")
    # ISO date in datetime attr
    iso = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if iso:
        return iso.group(0)
    return ""


class XContentExtractor:
    """Extracts structured content from X/Twitter pages."""

    def __init__(self, page: Page) -> None:
        self.page = page

    async def extract_feed_posts(self) -> list[LinkedInPost]:
        """Extract tweets from the home timeline or search results."""
        return await self._extract_tweets(CrawlSource.X_FEED)

    async def extract_search_results(self) -> list[LinkedInPost]:
        """Extract tweets from search results page."""
        return await self._extract_tweets(CrawlSource.X_SEARCH)

    async def extract_profile_posts(self) -> list[LinkedInPost]:
        """Extract tweets from a user's profile page."""
        return await self._extract_tweets(CrawlSource.X_PROFILE, limit=10)

    async def _extract_tweets(
        self, source: CrawlSource, limit: int = 30
    ) -> list[LinkedInPost]:
        """Core extraction: find tweet elements and parse each one."""
        posts = []

        # X uses data-testid="tweet" for tweet containers
        tweet_selectors = [
            "article[data-testid='tweet']",
            "article[role='article']",
        ]

        elements = []
        for sel in tweet_selectors:
            elements = await self.page.query_selector_all(sel)
            if elements:
                break

        if not elements:
            logger.warning("[X] No tweet elements found.")
            return posts

        for element in elements[:limit]:
            try:
                post = await self._extract_single_tweet(element, source)
                if post and post.content:
                    posts.append(post)
            except Exception as e:
                logger.debug(f"[X] Failed to extract tweet: {e}")
                continue

        logger.info(f"[X] Extracted {len(posts)} tweets.")
        return posts

    async def _extract_single_tweet(
        self, element: ElementHandle, source: CrawlSource
    ) -> LinkedInPost | None:
        """Extract data from a single tweet element."""
        author = await self._extract_author(element)
        content = await self._extract_content(element)
        if not content:
            return None

        url = await self._extract_tweet_url(element)
        published_date = await self._extract_date(element)
        likes, replies, retweets, views = await self._extract_metrics(element)

        # Extract URLs from content
        profile_urls, post_urls, external_urls = self._extract_urls(content)

        post_id = _generate_tweet_id(url or "", content)

        return LinkedInPost(
            post_id=post_id,
            url=url,
            author=author,
            content=_clean_text(content),
            post_type=PostType.ORIGINAL,
            platform="x",
            published_date=published_date,
            reactions_count=likes,
            comments_count=replies,
            reposts_count=retweets,
            mentioned_profiles=profile_urls,
            external_links=external_urls,
            linked_posts=post_urls,
            crawl_source=source,
        )

    async def _extract_author(self, element: ElementHandle) -> Author:
        """Extract tweet author info."""
        author = Author()
        try:
            # Display name — data-testid="User-Name" contains name + handle
            user_name_el = await element.query_selector("div[data-testid='User-Name']")
            if user_name_el:
                # Name is usually the first link's text
                name_link = await user_name_el.query_selector("a span")
                if name_link:
                    author.name = _clean_text(await name_link.inner_text())

                # Handle (@xxx)
                handle_el = await user_name_el.query_selector("a[href^='/'] span")
                spans = await user_name_el.query_selector_all("a[href^='/']")
                for span in spans:
                    href = await span.get_attribute("href")
                    if href and href.startswith("/") and not href.startswith("/i/"):
                        author.profile_url = f"https://x.com{href}"
                        author.linkedin_id = href.strip("/")  # reuse field for handle
                        break

            # Fallback: headline from user name block
            if not author.name:
                fallback_selectors = [
                    "a[role='link'] span",
                    "span[data-testid='tweetText'] a",
                ]
                for sel in fallback_selectors:
                    el = await element.query_selector(sel)
                    if el:
                        author.name = _clean_text(await el.inner_text())
                        if author.name:
                            break
        except Exception as e:
            logger.debug(f"[X] Error extracting author: {e}")
        return author

    async def _extract_content(self, element: ElementHandle) -> str:
        """Extract tweet text content."""
        content_selectors = [
            "div[data-testid='tweetText']",
            "div[lang] span",
        ]
        for sel in content_selectors:
            el = await element.query_selector(sel)
            if el:
                text = await el.inner_text()
                if text and len(text.strip()) > 5:
                    return text
        return ""

    async def _extract_tweet_url(self, element: ElementHandle) -> str:
        """Extract the permalink URL of a tweet."""
        try:
            # Timestamp link contains the tweet URL
            time_link = await element.query_selector("a[href*='/status/'] time")
            if time_link:
                parent = await time_link.evaluate("el => el.parentElement.getAttribute('href')")
                if parent:
                    return f"https://x.com{parent}"

            # Fallback: find any /status/ link
            links = await element.query_selector_all("a[href*='/status/']")
            for link in links:
                href = await link.get_attribute("href")
                if href and "/status/" in href:
                    if href.startswith("/"):
                        return f"https://x.com{href}"
                    return href
        except Exception:
            pass
        return ""

    async def _extract_date(self, element: ElementHandle) -> str:
        """Extract published date from tweet."""
        try:
            time_el = await element.query_selector("time[datetime]")
            if time_el:
                dt_attr = await time_el.get_attribute("datetime")
                if dt_attr:
                    iso = re.search(r"\d{4}-\d{2}-\d{2}", dt_attr)
                    if iso:
                        return iso.group(0)

            # Fallback: relative time text
            time_text_el = await element.query_selector("time")
            if time_text_el:
                text = await time_text_el.inner_text()
                return _parse_relative_date(text)
        except Exception:
            pass
        return ""

    async def _extract_metrics(
        self, element: ElementHandle
    ) -> tuple[int, int, int, int]:
        """Extract likes, replies, retweets, views."""
        likes = replies = retweets = views = 0
        try:
            # X uses data-testid for metric buttons/groups
            metric_map = {
                "reply": "replies",
                "retweet": "retweets",
                "like": "likes",
            }
            group = await element.query_selector("div[role='group']")
            if group:
                buttons = await group.query_selector_all("button")
                for btn in buttons:
                    label = await btn.get_attribute("aria-label") or ""
                    label_lower = label.lower()
                    # Parse "123 Likes", "45 replies", etc.
                    count_match = re.search(r"(\d[\d,]*)", label)
                    count = int(count_match.group(1).replace(",", "")) if count_match else 0

                    if "repl" in label_lower or "답글" in label_lower:
                        replies = count
                    elif "retweet" in label_lower or "리트윗" in label_lower or "repost" in label_lower:
                        retweets = count
                    elif "like" in label_lower or "마음" in label_lower or "좋아요" in label_lower:
                        likes = count
                    elif "view" in label_lower or "조회" in label_lower:
                        views = count
        except Exception as e:
            logger.debug(f"[X] Error extracting metrics: {e}")
        return likes, replies, retweets, views

    @staticmethod
    def _extract_urls(text: str) -> tuple[list[str], list[str], list[str]]:
        """Extract profile URLs, post URLs, and external URLs from text."""
        profiles: list[str] = []
        posts: list[str] = []
        externals: list[str] = []

        url_pattern = re.compile(r"https?://[^\s<>\"')\]]+")
        for url in url_pattern.findall(text):
            url = url.rstrip(".,;:!?)")
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            if "x.com" in domain or "twitter.com" in domain:
                if "/status/" in parsed.path:
                    posts.append(url)
                elif re.match(r"^/\w+/?$", parsed.path):
                    profiles.append(url)
            elif domain and "x.com" not in domain and "twitter.com" not in domain:
                externals.append(url)

        # Also extract @mentions
        mentions = re.findall(r"@(\w{1,15})", text)
        for handle in mentions:
            profile_url = f"https://x.com/{handle}"
            if profile_url not in profiles:
                profiles.append(profile_url)

        return profiles, posts, externals
