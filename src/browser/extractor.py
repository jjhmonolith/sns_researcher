"""Content extraction from LinkedIn pages - posts, profiles, links.

Updated 2026-04-22 for LinkedIn's new hashed-class DOM structure.
Posts are now inside div[role="list"] > div children, with div[role="listitem"]
marking actual posts. Class names are obfuscated CSS modules, so we rely on
semantic HTML attributes (role, aria-label, href patterns, tag structure).
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timedelta
from urllib.parse import urlparse

from playwright.async_api import Page, ElementHandle

from src.knowledge.models import LinkedInPost, Author, PostType, CrawlSource

logger = logging.getLogger(__name__)


def _generate_post_id(url: str, content: str) -> str:
    """Generate a deterministic post ID from URL or content."""
    match = re.search(r"activity[:-](\d+)", url)
    if match:
        return f"li_{match.group(1)}"
    h = hashlib.md5((url + content[:200]).encode()).hexdigest()[:12]
    return f"li_{h}"


def _clean_text(text: str) -> str:
    """Clean extracted text - normalize whitespace, remove invisible chars."""
    if not text:
        return ""
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    lines = text.split("\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
    text = "\n".join(line for line in lines if line)
    return text.strip()


def _parse_relative_date(text: str) -> str:
    """Parse a relative date string (Korean/English) into YYYY-MM-DD."""
    if not text:
        return ""
    text = text.strip().split("•")[0].strip().split("·")[0].strip()
    if not text:
        return ""
    now = datetime.now()

    patterns: list[tuple[str, str]] = [
        (r"(\d+)\s*분\s*전?", "minutes"),
        (r"(\d+)\s*시간\s*전?", "hours"),
        (r"(\d+)\s*일\s*전?", "days"),
        (r"(\d+)\s*주\s*전?", "weeks"),
        (r"(\d+)\s*개월\s*전?", "months"),
        (r"(\d+)\s*년\s*전?", "years"),
        (r"(\d+)\s*min", "minutes"),
        (r"(\d+)\s*h\b", "hours"),
        (r"(\d+)\s*d\b", "days"),
        (r"(\d+)\s*w\b", "weeks"),
        (r"(\d+)\s*mo", "months"),
        (r"(\d+)\s*y\b", "years"),
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

    iso_match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if iso_match:
        return iso_match.group(0)
    return ""


def _extract_linkedin_urls(text: str) -> tuple[list[str], list[str], list[str]]:
    """Extract LinkedIn profile URLs, post URLs, and external URLs from text."""
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
    """Extracts structured content from LinkedIn pages.

    LinkedIn (as of 2026-04) uses hashed CSS-module class names, making
    class-based selectors unreliable. This extractor relies on:
    - Semantic roles: role="list", role="listitem", role="button"
    - ARIA attributes: aria-label, aria-hidden
    - Tag structure: a[href*="/in/"], a[href*="/posts/"]
    - Text patterns: "피드 게시물", "반응 N", "댓글 N", "퍼감 N"
    """

    def __init__(self, page: Page) -> None:
        self.page = page

    # ------------------------------------------------------------------
    # Public extraction methods
    # ------------------------------------------------------------------

    async def extract_feed_posts(self) -> list[LinkedInPost]:
        """Extract all visible posts from the current feed page."""
        posts = []
        post_elements = await self._get_feed_post_elements()

        if not post_elements:
            logger.warning("No post elements found on page.")
            return posts

        for element in post_elements:
            try:
                post = await self._extract_single_post(element, CrawlSource.HOME_FEED)
                if post and post.content and len(post.content.strip()) > 20:
                    posts.append(post)
            except Exception as e:
                logger.debug(f"Failed to extract post: {e}")
                continue

        logger.info(f"Extracted {len(posts)} posts from feed.")
        return posts

    async def extract_post_page(self, url: str, source: CrawlSource = CrawlSource.LINKED_POST) -> LinkedInPost | None:
        """Extract a single post from its dedicated page URL."""
        try:
            # Try new DOM: role=listitem
            article = await self.page.query_selector("div[role='listitem']")
            # Fallback: legacy selectors
            if not article:
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

            return await self._extract_from_page_body(url, source)
        except Exception as e:
            logger.error(f"Error extracting post page {url}: {e}")
            return None

    async def extract_profile_posts(self) -> list[LinkedInPost]:
        """Extract recent posts from a profile page."""
        posts = []
        # Try new DOM first, then legacy
        post_elements = await self._get_feed_post_elements()
        if not post_elements:
            for sel in [
                "div.profile-creator-shared-feed-update__container",
                "div.feed-shared-update-v2",
                "li.profile-creator-shared-content-view__list-item",
            ]:
                elements = await self.page.query_selector_all(sel)
                if elements:
                    post_elements = elements
                    break

        for element in post_elements[:10]:
            try:
                post = await self._extract_single_post(element, CrawlSource.PROFILE)
                if post and post.content and len(post.content.strip()) > 20:
                    posts.append(post)
            except Exception as e:
                logger.debug(f"Failed to extract profile post: {e}")
                continue

        logger.info(f"Extracted {len(posts)} posts from profile.")
        return posts

    async def extract_search_results(self) -> list[LinkedInPost]:
        """Extract posts from LinkedIn search results page."""
        posts = []
        # Try new DOM first
        post_elements = await self._get_feed_post_elements()
        if not post_elements:
            for sel in [
                "div.search-results-container li.reusable-search__result-container",
                "div.feed-shared-update-v2",
                "ul.reusable-search__entity-result-list li",
            ]:
                elements = await self.page.query_selector_all(sel)
                if elements:
                    post_elements = elements
                    break

        for element in post_elements:
            try:
                post = await self._extract_single_post(element, CrawlSource.SEARCH)
                if post and post.content and len(post.content.strip()) > 20:
                    posts.append(post)
            except Exception as e:
                logger.debug(f"Failed to extract search result: {e}")
                continue

        logger.info(f"Extracted {len(posts)} posts from search results.")
        return posts

    async def extract_comment_authors(self) -> list[dict]:
        """Extract comment author profiles from the current page."""
        results = []
        seen_urls: set[str] = set()

        # Try new DOM: buttons with aria-label containing comment-related text
        # Also try legacy selectors
        comment_selectors = [
            "article.comments-comment-entity",
            "div.comments-comment-item",
            "article.comments-comment-item",
            "div.comments-comment-entity",
        ]

        elements = []
        for selector in comment_selectors:
            elements = await self.page.query_selector_all(selector)
            if elements:
                break

        for element in elements[:20]:
            try:
                author_info = await self._extract_comment_author_info(element, seen_urls)
                if author_info:
                    results.append(author_info)
            except Exception as e:
                logger.debug(f"Failed to extract comment author: {e}")
                continue

        logger.info(f"Extracted {len(results)} comment authors.")
        return results

    # ------------------------------------------------------------------
    # Private: find post container elements
    # ------------------------------------------------------------------

    async def _get_feed_post_elements(self) -> list[ElementHandle]:
        """Find feed post elements using LinkedIn's 2026-04 DOM structure.

        Posts live inside div[role="list"] > div children.
        Real posts contain div[role="listitem"] and have substantial text.
        UI widgets (profile prompts, sort buttons) are filtered out.
        """
        # Primary: new LinkedIn DOM (role=list > children)
        post_elements = await self.page.evaluate_handle("""() => {
            const listEl = document.querySelector('[role="list"]');
            if (!listEl) return [];

            const items = Array.from(listEl.children);
            // Filter: real posts have role=listitem and text > 100 chars
            return items.filter(item => {
                const hasListItem = item.querySelector('[role="listitem"]');
                const textLen = (item.textContent || '').length;
                // Skip short UI widgets and empty items
                return hasListItem && textLen > 100;
            });
        }""")

        elements = []
        if post_elements:
            props = await post_elements.get_properties()
            for prop in props.values():
                el = prop.as_element()
                if el:
                    elements.append(el)

        if elements:
            logger.debug(f"Found {len(elements)} post elements via role=list")
            return elements

        # Fallback: legacy selectors
        for sel in [
            "div.feed-shared-update-v2",
            "div[data-urn*='activity']",
            "div.occludable-update",
        ]:
            legacy = await self.page.query_selector_all(sel)
            if legacy:
                logger.debug(f"Found {len(legacy)} posts via legacy selector: {sel}")
                return legacy

        return []

    # ------------------------------------------------------------------
    # Private: extract data from a single post element
    # ------------------------------------------------------------------

    async def _extract_single_post(
        self, element: ElementHandle, source: CrawlSource
    ) -> LinkedInPost | None:
        """Extract data from a single post element."""
        author = await self._extract_author(element)
        content = await self._extract_content_text(element)
        if not content:
            return None

        url = await self._extract_post_url(element)
        reactions, comments, reposts = await self._extract_metrics(element)
        published_date = await self._extract_published_date(element)

        full_text = content + " " + (await self._extract_all_hrefs(element))
        profile_urls, post_urls, external_urls = _extract_linkedin_urls(full_text)

        post_id = _generate_post_id(url or "", content)

        # Detect shared/reposted content
        post_type = PostType.ORIGINAL
        full_post_text = await element.inner_text()
        if "님이 퍼감" in full_post_text or "reposted" in full_post_text.lower():
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
        """Extract author information using new DOM structure."""
        author = Author()
        try:
            # New DOM: author name is in the first <a> link's aria-label or text
            # Pattern: a[href*="/company/"] or a[href*="/in/"] with author text
            author_data = await element.evaluate("""(el) => {
                const result = { name: '', headline: '', profileUrl: '', linkedinId: '' };

                // Find the first link to a profile or company (author link)
                const authorLinks = el.querySelectorAll('a[href*="/in/"], a[href*="/company/"]');
                for (const link of authorLinks) {
                    const href = link.getAttribute('href') || '';
                    const text = (link.textContent || '').trim();
                    // Skip empty or very long text links (likely not author name)
                    if (!text || text.length > 100 || text.length < 2) continue;
                    // Skip hashtag/search links
                    if (href.includes('/search/') || href.includes('/hashtag/')) continue;

                    result.profileUrl = href.split('?')[0];

                    // Extract name from the link's text or aria-label
                    const ariaLabel = link.querySelector('[aria-label]');
                    if (ariaLabel) {
                        result.name = ariaLabel.getAttribute('aria-label').split('인증됨')[0].trim();
                    }
                    if (!result.name) {
                        // Get the first meaningful text (usually author/org name)
                        const pEl = link.querySelector('p');
                        if (pEl) result.name = pEl.textContent.trim();
                        else result.name = text.split('\\n')[0].trim();
                    }

                    // Extract linkedin ID
                    const inMatch = href.match(/\\/in\\/([^/]+)/);
                    if (inMatch) result.linkedinId = inMatch[1];

                    break;
                }

                // Headline: look for text near the relative date (e.g., "2주 •")
                // In new DOM, the second text block after author is usually the headline
                const allPs = el.querySelectorAll('p');
                for (let i = 0; i < Math.min(allPs.length, 5); i++) {
                    const pText = (allPs[i].textContent || '').trim();
                    // Look for relative date patterns (indicates this is the date/headline area)
                    if (/\\d+[시분일주개년hmdwy]/.test(pText) && pText.includes('•')) {
                        // The text before the date might be headline
                        const parts = pText.split('•').map(s => s.trim());
                        if (parts.length >= 1) {
                            // Check if date is standalone or with headline
                            const dateMatch = parts[parts.length-1].match(/\\d+[시분일주개년hmdwy]/);
                            if (dateMatch && parts.length > 1) {
                                result.headline = parts.slice(0, -1).join(' ').trim();
                            }
                        }
                    }
                }

                return result;
            }""")

            if author_data:
                author.name = author_data.get("name", "")
                author.headline = author_data.get("headline", "")
                author.profile_url = author_data.get("profileUrl", "")
                author.linkedin_id = author_data.get("linkedinId", "")

        except Exception as e:
            logger.debug(f"Error extracting author: {e}")

        return author

    async def _extract_content_text(self, element: ElementHandle) -> str:
        """Extract the main text content from a post element.

        New DOM: post body text is in <p><span>...</span></p> inside the post.
        We skip the author name, date, and engagement text blocks.
        """
        try:
            content = await element.evaluate("""(el) => {
                // Strategy: find the longest <p> element that contains
                // the actual post body (not author name, not engagement counts)
                const allPs = el.querySelectorAll('p');
                let bestText = '';
                let bestLen = 0;

                for (const p of allPs) {
                    const text = (p.textContent || '').trim();
                    // Skip short text (buttons, counts, names)
                    if (text.length < 30) continue;
                    // Skip engagement text
                    if (/^(반응|댓글|퍼감|추천|좋아요|축하해요)\s*\\d*/.test(text)) continue;
                    // Skip "피드 게시물" header
                    if (text === '피드 게시물') continue;

                    if (text.length > bestLen) {
                        bestLen = text.length;
                        bestText = text;
                    }
                }

                // If no good <p> found, try getting all text from role=listitem
                // minus the known UI sections
                if (!bestText || bestLen < 30) {
                    const listItem = el.querySelector('[role="listitem"]');
                    if (listItem) {
                        const fullText = listItem.textContent || '';
                        // Remove common UI strings
                        bestText = fullText
                            .replace(/피드 게시물/g, '')
                            .replace(/번역 표시/g, '')
                            .replace(/…\\s*더보기/g, '')
                            .replace(/추천|댓글|퍼가기|보내기/g, '')
                            .trim();
                    }
                }

                return bestText || '';
            }""")
            return content
        except Exception as e:
            logger.debug(f"Error extracting content: {e}")

        return ""

    async def _extract_post_url(self, element: ElementHandle) -> str:
        """Extract the permalink URL of a post."""
        try:
            url = await element.evaluate("""(el) => {
                // Look for links containing /posts/ or activity
                const links = el.querySelectorAll('a[href]');
                for (const link of links) {
                    const href = link.getAttribute('href') || '';
                    if (href.includes('/posts/') || href.includes('activity:')) {
                        return href.split('?')[0];
                    }
                }
                // Try company posts links
                for (const link of links) {
                    const href = link.getAttribute('href') || '';
                    if (href.includes('/company/') && href.includes('/posts/')) {
                        return href.split('?')[0];
                    }
                }
                // Try data-urn (legacy)
                const urnEl = el.querySelector('[data-urn*="activity"]');
                if (urnEl) {
                    const urn = urnEl.getAttribute('data-urn');
                    const match = urn.match(/activity:(\\d+)/);
                    if (match) return 'https://www.linkedin.com/feed/update/urn:li:activity:' + match[1] + '/';
                }
                return '';
            }""")
            return url
        except Exception:
            pass
        return ""

    async def _extract_published_date(self, element: ElementHandle) -> str:
        """Extract the published date of a post element."""
        try:
            date_text = await element.evaluate("""(el) => {
                // New DOM: relative date is often the first link text like "2주 •"
                // or inside a <time> element
                const timeEl = el.querySelector('time[datetime]');
                if (timeEl) return timeEl.getAttribute('datetime');

                // Look for text with relative date pattern near the top of the post
                const allText = el.querySelectorAll('p, span');
                for (const node of allText) {
                    const text = (node.textContent || '').trim();
                    // Match patterns like "2주", "1주 •", "3일", "2시간", "1d", "2w", "3h"
                    if (/^\\d+[시분일주개년hmdwy]/.test(text) || /\\d+[시분일주개년hmdwy]\\s*•/.test(text)) {
                        return text;
                    }
                    // Match "N주 전" or "Nw ago" style
                    if (/\\d+\\s*(분|시간|일|주|개월|년|min|hour|day|week|month|year)/.test(text) && text.length < 30) {
                        return text;
                    }
                }

                // Try first link text that looks like a date
                const firstLinks = el.querySelectorAll('a');
                for (const link of firstLinks) {
                    const text = (link.textContent || '').trim();
                    if (/\\d+[주일시분]/.test(text) && text.length < 30) return text;
                    if (/\\d+[hdwmy]\\b/.test(text) && text.length < 20) return text;
                }
                return '';
            }""")

            if date_text:
                # Try ISO format first
                iso_match = re.search(r"\d{4}-\d{2}-\d{2}", date_text)
                if iso_match:
                    return iso_match.group(0)
                return _parse_relative_date(date_text)
        except Exception as e:
            logger.debug(f"Error extracting published date: {e}")
        return ""

    async def _extract_metrics(self, element: ElementHandle) -> tuple[int, int, int]:
        """Extract reaction, comment, and repost counts.

        New DOM uses text patterns like "반응 842", "댓글 4", "퍼감 30".
        """
        try:
            metrics = await element.evaluate("""(el) => {
                const result = { reactions: 0, comments: 0, reposts: 0 };
                const fullText = el.textContent || '';

                // Korean patterns
                let m = fullText.match(/반응\\s*([\\d,.]+[kKmM만천]?)/);
                if (m) result.reactions = m[1];

                m = fullText.match(/댓글\\s*([\\d,.]+[kKmM만천]?)/);
                if (m) result.comments = m[1];

                m = fullText.match(/퍼감\\s*([\\d,.]+[kKmM만천]?)/);
                if (m) result.reposts = m[1];

                // English fallback
                if (!result.reactions) {
                    m = fullText.match(/(\\d[\\d,.]*[kKmM]?)\\s*reactions?/i);
                    if (m) result.reactions = m[1];
                }
                if (!result.comments) {
                    m = fullText.match(/(\\d[\\d,.]*[kKmM]?)\\s*comments?/i);
                    if (m) result.comments = m[1];
                }
                if (!result.reposts) {
                    m = fullText.match(/(\\d[\\d,.]*[kKmM]?)\\s*reposts?/i);
                    if (m) result.reposts = m[1];
                }

                return result;
            }""")

            reactions = self._parse_count(str(metrics.get("reactions", "0")))
            comments = self._parse_count(str(metrics.get("comments", "0")))
            reposts = self._parse_count(str(metrics.get("reposts", "0")))
            return reactions, comments, reposts

        except Exception as e:
            logger.debug(f"Error extracting metrics: {e}")
        return 0, 0, 0

    async def _extract_all_hrefs(self, element: ElementHandle) -> str:
        """Extract all href values from links within the element."""
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

    async def _extract_from_page_body(self, url: str, source: CrawlSource) -> LinkedInPost | None:
        """Fallback extraction from the entire page body."""
        try:
            body_text = await self.page.inner_text("body")
            if body_text and len(body_text.strip()) > 50:
                content = _clean_text(body_text[:3000])
                post_id = _generate_post_id(url, content)
                return LinkedInPost(
                    post_id=post_id,
                    url=url,
                    content=content,
                    crawl_source=source,
                )
        except Exception:
            pass
        return None

    async def _extract_comment_author_info(self, element: ElementHandle, seen_urls: set) -> dict | None:
        """Extract a comment author's info from a comment element."""
        author_info: dict[str, str] = {}

        link_selectors = [
            "a[data-tracking-control-name*='comment'][href*='/in/']",
            "a.comments-post-meta__name-text[href*='/in/']",
            "a.comments-post-meta__profile-link[href*='/in/']",
            "a[href*='/in/']",
        ]
        for sel in link_selectors:
            link_el = await element.query_selector(sel)
            if link_el:
                href = await link_el.get_attribute("href")
                if href and "/in/" in href:
                    normalized = href.split("?")[0].rstrip("/")
                    if normalized not in seen_urls:
                        seen_urls.add(normalized)
                        author_info["profile_url"] = normalized
                    break

        if not author_info.get("profile_url"):
            return None

        name_selectors = [
            "span.comments-post-meta__name-text",
            "span.hoverable-link-text span",
            "a[href*='/in/'] span",
        ]
        for sel in name_selectors:
            name_el = await element.query_selector(sel)
            if name_el:
                name = _clean_text(await name_el.inner_text())
                if name:
                    author_info["name"] = name
                    break

        text_selectors = [
            "span.comments-comment-item__main-content",
            "span[dir='ltr']",
        ]
        for sel in text_selectors:
            text_el = await element.query_selector(sel)
            if text_el:
                text = _clean_text(await text_el.inner_text())
                if text and len(text) > 5:
                    author_info["comment_text"] = text[:500]
                    break

        return author_info

    @staticmethod
    def _parse_count(text: str) -> int:
        """Parse a count string like '1,234' or '1.2K' into an integer."""
        if not text:
            return 0
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
