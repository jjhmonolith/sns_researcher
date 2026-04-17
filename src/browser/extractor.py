"""Content extraction from LinkedIn pages - posts, profiles, links."""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timedelta
from urllib.parse import urlparse, urljoin

from playwright.async_api import Page, ElementHandle

from src.knowledge.models import LinkedInPost, Author, PostType, CrawlSource

logger = logging.getLogger(__name__)


def _generate_post_id(url: str, content: str) -> str:
    """Generate a deterministic post ID from URL or content."""
    # Try to extract LinkedIn's activity ID from URL
    match = re.search(r"activity[:-](\d+)", url)
    if match:
        return f"li_{match.group(1)}"
    # Fallback: hash of content
    h = hashlib.md5((url + content[:200]).encode()).hexdigest()[:12]
    return f"li_{h}"


def _clean_text(text: str) -> str:
    """Clean extracted text - normalize whitespace, remove invisible chars."""
    if not text:
        return ""
    # Remove zero-width characters
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    # Normalize whitespace but preserve newlines for readability
    lines = text.split("\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
    text = "\n".join(line for line in lines if line)
    return text.strip()


def _parse_relative_date(text: str) -> str:
    """Parse a relative date string (Korean/English) into YYYY-MM-DD.

    Handles formats like '2h', '3d', '1w', '2mo',
    '1시간 전', '3일 전', '2주 전', '1개월 전', etc.
    Returns empty string if unparseable.
    """
    if not text:
        return ""
    # Clean common LinkedIn suffixes like "1일 •", "2h •", "1일 • 수정됨"
    text = text.strip().split("•")[0].strip().split("·")[0].strip()
    if not text:
        return ""
    now = datetime.now()

    patterns: list[tuple[str, str]] = [
        # Korean (with and without "전")
        (r"(\d+)\s*분\s*전?", "minutes"),
        (r"(\d+)\s*시간\s*전?", "hours"),
        (r"(\d+)\s*일\s*전?", "days"),
        (r"(\d+)\s*주\s*전?", "weeks"),
        (r"(\d+)\s*개월\s*전?", "months"),
        (r"(\d+)\s*년\s*전?", "years"),
        # English abbreviated
        (r"(\d+)\s*min", "minutes"),
        (r"(\d+)\s*h\b", "hours"),
        (r"(\d+)\s*d\b", "days"),
        (r"(\d+)\s*w\b", "weeks"),
        (r"(\d+)\s*mo", "months"),
        (r"(\d+)\s*y\b", "years"),
        # English verbose
        (r"(\d+)\s*minute", "minutes"),
        (r"(\d+)\s*hour", "hours"),
        (r"(\d+)\s*day", "days"),
        (r"(\d+)\s*week", "weeks"),
        (r"(\d+)\s*month", "months"),
        (r"(\d+)\s*year", "years"),
    ]

    for pattern, unit in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            n = int(m.group(1))
            if unit == "minutes":
                dt = now - timedelta(minutes=n)
            elif unit == "hours":
                dt = now - timedelta(hours=n)
            elif unit == "days":
                dt = now - timedelta(days=n)
            elif unit == "weeks":
                dt = now - timedelta(weeks=n)
            elif unit == "months":
                dt = now - timedelta(days=n * 30)
            elif unit == "years":
                dt = now - timedelta(days=n * 365)
            else:
                continue
            return dt.strftime("%Y-%m-%d")

    # Try ISO-like date (e.g. "2026-04-15" or "Apr 15, 2026")
    iso_match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if iso_match:
        return iso_match.group(0)

    return ""


def _extract_linkedin_urls(text: str) -> tuple[list[str], list[str], list[str]]:
    """Extract LinkedIn profile URLs, post URLs, and external URLs from text.
    
    Returns:
        (profile_urls, post_urls, external_urls)
    """
    profile_urls = []
    post_urls = []
    external_urls = []

    url_pattern = re.compile(r'https?://[^\s<>"\')\]]+')
    urls = url_pattern.findall(text)

    for url in urls:
        url = url.rstrip(".,;:!?)")
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        if "linkedin.com" in domain:
            path = parsed.path.lower()
            if "/in/" in path:
                profile_urls.append(url)
            elif "/posts/" in path or "activity" in path or "/pulse/" in path:
                post_urls.append(url)
        elif domain and "linkedin" not in domain:
            external_urls.append(url)

    return profile_urls, post_urls, external_urls


class ContentExtractor:
    """Extracts structured content from LinkedIn pages."""

    def __init__(self, page: Page) -> None:
        self.page = page

    async def extract_feed_posts(self) -> list[LinkedInPost]:
        """Extract all visible posts from the current feed page."""
        posts = []
        # LinkedIn feed post containers
        selectors = [
            "div.feed-shared-update-v2",
            "div[data-urn*='activity']",
            "div.occludable-update",
        ]

        post_elements = []
        for selector in selectors:
            elements = await self.page.query_selector_all(selector)
            if elements:
                post_elements = elements
                break

        if not post_elements:
            logger.warning("No post elements found on page.")
            return posts

        for element in post_elements:
            try:
                post = await self._extract_single_post(element, CrawlSource.HOME_FEED)
                if post and post.content:
                    posts.append(post)
            except Exception as e:
                logger.debug(f"Failed to extract post: {e}")
                continue

        logger.info(f"Extracted {len(posts)} posts from feed.")
        return posts

    async def extract_post_page(self, url: str, source: CrawlSource = CrawlSource.LINKED_POST) -> LinkedInPost | None:
        """Extract a single post from its dedicated page URL."""
        try:
            # The main post content on a detail page
            article = await self.page.query_selector("article.feed-shared-update-v2")
            if not article:
                article = await self.page.query_selector("div.feed-shared-update-v2")
            if not article:
                article = await self.page.query_selector("div.scaffold-layout__main")

            if article:
                post = await self._extract_single_post(article, source)
                if post:
                    post.url = url
                    return post

            # Fallback: extract from page body
            return await self._extract_from_page_body(url, source)
        except Exception as e:
            logger.error(f"Error extracting post page {url}: {e}")
            return None

    async def extract_profile_posts(self) -> list[LinkedInPost]:
        """Extract recent posts from a profile page."""
        posts = []
        # Profile page activity section
        activity_selectors = [
            "div.profile-creator-shared-feed-update__container",
            "div.feed-shared-update-v2",
            "li.profile-creator-shared-content-view__list-item",
        ]

        post_elements = []
        for selector in activity_selectors:
            elements = await self.page.query_selector_all(selector)
            if elements:
                post_elements = elements
                break

        for element in post_elements[:10]:  # Limit to recent 10
            try:
                post = await self._extract_single_post(element, CrawlSource.PROFILE)
                if post and post.content:
                    posts.append(post)
            except Exception as e:
                logger.debug(f"Failed to extract profile post: {e}")
                continue

        logger.info(f"Extracted {len(posts)} posts from profile.")
        return posts

    async def extract_search_results(self) -> list[LinkedInPost]:
        """Extract posts from LinkedIn search results page."""
        posts = []
        result_selectors = [
            "div.search-results-container li.reusable-search__result-container",
            "div.feed-shared-update-v2",
            "ul.reusable-search__entity-result-list li",
        ]

        result_elements = []
        for selector in result_selectors:
            elements = await self.page.query_selector_all(selector)
            if elements:
                result_elements = elements
                break

        for element in result_elements:
            try:
                post = await self._extract_single_post(element, CrawlSource.SEARCH)
                if post and post.content:
                    posts.append(post)
            except Exception as e:
                logger.debug(f"Failed to extract search result: {e}")
                continue

        logger.info(f"Extracted {len(posts)} posts from search results.")
        return posts

    async def _extract_single_post(
        self, element: ElementHandle, source: CrawlSource
    ) -> LinkedInPost | None:
        """Extract data from a single post element."""
        # Extract author info
        author = await self._extract_author(element)

        # Extract content text
        content = await self._extract_content_text(element)
        if not content:
            return None

        # Extract post URL
        url = await self._extract_post_url(element)

        # Extract engagement metrics
        reactions, comments, reposts = await self._extract_metrics(element)

        # Extract published date
        published_date = await self._extract_published_date(element)

        # Extract links from content
        full_text = content + " " + (await self._extract_all_hrefs(element))
        profile_urls, post_urls, external_urls = _extract_linkedin_urls(full_text)

        post_id = _generate_post_id(url or "", content)

        # Determine post type
        post_type = PostType.ORIGINAL
        shared_indicator = await element.query_selector("span.feed-shared-header__text")
        if shared_indicator:
            post_type = PostType.SHARED

        return LinkedInPost(
            post_id=post_id,
            url=url,
            author=author,
            content=_clean_text(content),
            post_type=post_type,
            published_date=published_date,
            reactions_count=reactions,
            comments_count=comments,
            reposts_count=reposts,
            mentioned_profiles=profile_urls,
            external_links=external_urls,
            linked_posts=post_urls,
            crawl_source=source,
        )

    async def _extract_author(self, element: ElementHandle) -> Author:
        """Extract author information from a post element."""
        author = Author()
        try:
            # Author name (2026-04 LinkedIn DOM)
            name_selectors = [
                "a.update-components-actor__meta-link span.t-bold span[aria-hidden='true']",
                "span.t-bold span[aria-hidden='true']",
                "span.feed-shared-actor__name span[aria-hidden='true']",
                "span.feed-shared-actor__name",
                "a.feed-shared-actor__container span.t-bold span",
                "span.update-components-actor__name span",
            ]
            for sel in name_selectors:
                name_el = await element.query_selector(sel)
                if name_el:
                    author.name = _clean_text(await name_el.inner_text())
                    if author.name:
                        break

            # Author headline (2026-04 LinkedIn DOM)
            headline_selectors = [
                "span.update-components-actor__description span[aria-hidden='true']",
                "span.update-components-actor__description",
                "span.feed-shared-actor__description span[aria-hidden='true']",
                "span.feed-shared-actor__sub-description",
            ]
            for sel in headline_selectors:
                hl_el = await element.query_selector(sel)
                if hl_el:
                    author.headline = _clean_text(await hl_el.inner_text())
                    if author.headline:
                        break

            # Author profile URL (2026-04 LinkedIn DOM)
            link_selectors = [
                "a.update-components-actor__meta-link[href*='/in/']",
                "a.feed-shared-actor__container",
                "a.update-components-actor__container",
                "a.app-aware-link[href*='/in/']",
            ]
            for sel in link_selectors:
                link_el = await element.query_selector(sel)
                if link_el:
                    href = await link_el.get_attribute("href")
                    if href and "/in/" in href:
                        author.profile_url = href.split("?")[0]
                        # Extract ID from URL
                        match = re.search(r"/in/([^/]+)", href)
                        if match:
                            author.linkedin_id = match.group(1)
                        break
        except Exception as e:
            logger.debug(f"Error extracting author: {e}")

        return author

    async def _extract_content_text(self, element: ElementHandle) -> str:
        """Extract the main text content from a post element."""
        content_selectors = [
            "div.feed-shared-update-v2__description-wrapper span[dir='ltr']",
            "div.feed-shared-update-v2__description span.break-words",
            "div.feed-shared-inline-show-more-text span[dir='ltr']",
            "span.break-words span[dir='ltr']",
            "div.update-components-text span.break-words",
            "div.feed-shared-text span[dir='ltr']",
        ]

        for sel in content_selectors:
            content_el = await element.query_selector(sel)
            if content_el:
                text = await content_el.inner_text()
                if text and len(text.strip()) > 10:
                    return text

        # Broad fallback: try getting all text in the post body area
        try:
            body = await element.query_selector("div.feed-shared-update-v2__description-wrapper")
            if body:
                return await body.inner_text()
        except Exception:
            pass

        return ""

    async def _extract_post_url(self, element: ElementHandle) -> str:
        """Extract the permalink URL of a post."""
        try:
            # Look for the timestamp link which usually contains the post URL
            link_selectors = [
                "a.feed-shared-actor__sub-description-link",
                "a[data-urn*='activity']",
                "a[href*='activity']",
                "a.app-aware-link[href*='/feed/update/']",
            ]
            for sel in link_selectors:
                link_el = await element.query_selector(sel)
                if link_el:
                    href = await link_el.get_attribute("href")
                    if href:
                        return href.split("?")[0]

            # Try data-urn attribute
            urn = await element.get_attribute("data-urn")
            if urn:
                match = re.search(r"activity:(\d+)", urn)
                if match:
                    return f"https://www.linkedin.com/feed/update/urn:li:activity:{match.group(1)}/"
        except Exception:
            pass
        return ""

    async def _extract_published_date(self, element: ElementHandle) -> str:
        """Extract the published date of a post element."""
        try:
            # 1. Try <time datetime="..."> attribute (most precise)
            time_el = await element.query_selector("time[datetime]")
            if time_el:
                dt_attr = await time_el.get_attribute("datetime")
                if dt_attr:
                    # LinkedIn sometimes uses ISO format like "2026-04-15T10:30:00.000Z"
                    iso_match = re.search(r"\d{4}-\d{2}-\d{2}", dt_attr)
                    if iso_match:
                        return iso_match.group(0)

            # 2. Try timestamp link text (relative dates like "2h", "3일 전")
            date_selectors = [
                "span.update-components-actor__sub-description span[aria-hidden='true']",
                "a.feed-shared-actor__sub-description-link span",
                "a.feed-shared-actor__sub-description-link",
                "span.feed-shared-actor__sub-description span[aria-hidden='true']",
            ]
            for sel in date_selectors:
                el = await element.query_selector(sel)
                if el:
                    text = await el.inner_text()
                    text = _clean_text(text)
                    if text:
                        parsed = _parse_relative_date(text)
                        if parsed:
                            return parsed
        except Exception as e:
            logger.debug(f"Error extracting published date: {e}")

        return ""

    async def _extract_metrics(self, element: ElementHandle) -> tuple[int, int, int]:
        """Extract reaction, comment, and repost counts."""
        reactions = 0
        comments = 0
        reposts = 0

        try:
            # Reactions
            reaction_selectors = [
                "span.social-details-social-counts__reactions-count",
                "button.social-details-social-counts__count-value",
            ]
            for sel in reaction_selectors:
                el = await element.query_selector(sel)
                if el:
                    text = await el.inner_text()
                    reactions = self._parse_count(text)
                    if reactions > 0:
                        break

            # Comments count
            comment_selectors = [
                "button[aria-label*='comment'] span",
                "li.social-details-social-counts__comments button span",
            ]
            for sel in comment_selectors:
                el = await element.query_selector(sel)
                if el:
                    text = await el.inner_text()
                    comments = self._parse_count(text)
                    if comments > 0:
                        break

            # Reposts
            repost_selectors = [
                "button[aria-label*='repost'] span",
                "li.social-details-social-counts__item--with-social-proof button span",
            ]
            for sel in repost_selectors:
                el = await element.query_selector(sel)
                if el:
                    text = await el.inner_text()
                    reposts = self._parse_count(text)
                    if reposts > 0:
                        break
        except Exception as e:
            logger.debug(f"Error extracting metrics: {e}")

        return reactions, comments, reposts

    async def _extract_all_hrefs(self, element: ElementHandle) -> str:
        """Extract all href values from links within the element as a space-joined string."""
        try:
            links = await element.query_selector_all("a[href]")
            hrefs = []
            for link in links:
                href = await link.get_attribute("href")
                if href:
                    hrefs.append(href)
            return " ".join(hrefs)
        except Exception:
            return ""

    @staticmethod
    def _parse_count(text: str) -> int:
        """Parse a count string like '1,234' or '1.2K' into an integer."""
        if not text:
            return 0
        # Take only the first line — LinkedIn often appends "손해성님 외 11명" etc.
        text = text.strip().split("\n")[0].strip()
        text = text.replace(",", "").replace(" ", "")
        text = re.sub(r"[^0-9.kKmM만천]", "", text)

        if not text:
            return 0

        try:
            if "k" in text.lower():
                return int(float(text.lower().replace("k", "")) * 1000)
            if "m" in text.lower():
                return int(float(text.lower().replace("m", "")) * 1_000_000)
            if "만" in text:
                return int(float(text.replace("만", "")) * 10000)
            if "천" in text:
                return int(float(text.replace("천", "")) * 1000)
            return int(float(text))
        except ValueError:
            return 0
