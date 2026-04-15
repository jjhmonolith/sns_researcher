"""Knowledge store - Markdown file-based knowledge management."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import frontmatter

from src.config import KNOWLEDGE_DIR
from src.knowledge.models import LinkedInPost, Author

logger = logging.getLogger(__name__)


class KnowledgeStore:
    """Manages the Markdown-based knowledge repository."""

    def __init__(self) -> None:
        self.base_dir = KNOWLEDGE_DIR
        self._ensure_structure()

    def _ensure_structure(self) -> None:
        """Ensure all knowledge directories exist."""
        dirs = [
            self.base_dir / "raw",
            self.base_dir / "insights" / "topics",
            self.base_dir / "insights" / "people",
            self.base_dir / "insights" / "digests",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

        # Create index.md if not exists
        index_path = self.base_dir / "index.md"
        if not index_path.exists():
            index_path.write_text(
                "---\ntitle: LinkedIn AX Research Knowledge Base\n"
                f"created: {datetime.now().isoformat()}\n---\n\n"
                "# LinkedIn AX Research Knowledge Base\n\n"
                "기업과 조직의 AX (AI Transformation) 관련 LinkedIn 리서치 결과\n\n"
                "## 구조\n\n"
                "- `raw/` - 수집된 원문 포스트\n"
                "- `insights/topics/` - 주제별 정리된 인사이트\n"
                "- `insights/people/` - 주요 인물 프로필\n"
                "- `insights/digests/` - 정기 다이제스트\n",
                encoding="utf-8",
            )

    def save_post(self, post: LinkedInPost) -> Path:
        """Save a post as a Markdown file with YAML frontmatter.
        
        Returns:
            Path to the saved file.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        day_dir = self.base_dir / "raw" / today
        day_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{post.post_id}.md"
        filepath = day_dir / filename

        # Build frontmatter metadata
        metadata = {
            "post_id": post.post_id,
            "url": post.url,
            "author_name": post.author.name,
            "author_headline": post.author.headline,
            "author_profile": post.author.profile_url,
            "relevance_score": post.relevance_score,
            "topics": post.relevance_topics,
            "crawl_source": post.crawl_source.value,
            "crawled_at": post.crawled_at,
            "reactions": post.reactions_count,
            "comments": post.comments_count,
            "post_type": post.post_type.value,
        }

        # Build content
        content_parts = []
        content_parts.append(f"# {post.author.name}: {post.summary or '(no summary)'}\n")

        if post.summary:
            content_parts.append(f"**AI 요약**: {post.summary}\n")

        content_parts.append(f"**관련성 점수**: {post.relevance_score}/100\n")
        content_parts.append(f"**토픽**: {', '.join(post.relevance_topics) if post.relevance_topics else 'N/A'}\n")

        content_parts.append("\n## 원문\n")
        content_parts.append(post.content)

        if post.external_links:
            content_parts.append("\n\n## 외부 링크\n")
            for link in post.external_links:
                content_parts.append(f"- {link}")

        if post.mentioned_profiles:
            content_parts.append("\n\n## 언급된 프로필\n")
            for profile in post.mentioned_profiles:
                content_parts.append(f"- {profile}")

        md_content = "\n".join(content_parts)

        # Write using python-frontmatter
        fm_post = frontmatter.Post(md_content, **metadata)
        filepath.write_text(frontmatter.dumps(fm_post), encoding="utf-8")

        logger.info(f"Saved post: {filepath}")
        return filepath

    def save_insight(self, topic_slug: str, title: str, content: str, metadata: dict | None = None) -> Path:
        """Save or update a topic insight document.
        
        Args:
            topic_slug: URL-safe name for the topic file.
            title: Human-readable title.
            content: Markdown content.
            metadata: Additional frontmatter metadata.
            
        Returns:
            Path to the saved file.
        """
        filepath = self.base_dir / "insights" / "topics" / f"{topic_slug}.md"

        meta = {
            "title": title,
            "updated_at": datetime.now().isoformat(),
        }
        if metadata:
            meta.update(metadata)

        fm_post = frontmatter.Post(content, **meta)
        filepath.write_text(frontmatter.dumps(fm_post), encoding="utf-8")

        logger.info(f"Saved insight: {filepath}")
        return filepath

    def save_person(self, person_slug: str, name: str, content: str, metadata: dict | None = None) -> Path:
        """Save or update a person profile document.
        
        Args:
            person_slug: URL-safe name for the person file.
            name: Full name.
            content: Markdown content about the person.
            metadata: Additional frontmatter metadata.
            
        Returns:
            Path to the saved file.
        """
        filepath = self.base_dir / "insights" / "people" / f"{person_slug}.md"

        meta = {
            "name": name,
            "updated_at": datetime.now().isoformat(),
        }
        if metadata:
            meta.update(metadata)

        fm_post = frontmatter.Post(content, **meta)
        filepath.write_text(frontmatter.dumps(fm_post), encoding="utf-8")

        logger.info(f"Saved person profile: {filepath}")
        return filepath

    def save_digest(self, content: str, metadata: dict | None = None) -> Path:
        """Save a periodic digest.
        
        Returns:
            Path to the saved file.
        """
        now = datetime.now()
        filename = f"{now.strftime('%Y-%m-%d_%H%M')}.md"
        filepath = self.base_dir / "insights" / "digests" / filename

        meta = {
            "title": f"Digest - {now.strftime('%Y-%m-%d %H:%M')}",
            "created_at": now.isoformat(),
        }
        if metadata:
            meta.update(metadata)

        fm_post = frontmatter.Post(content, **meta)
        filepath.write_text(frontmatter.dumps(fm_post), encoding="utf-8")

        logger.info(f"Saved digest: {filepath}")
        return filepath

    def get_recent_posts(self, days: int = 3, limit: int = 50) -> list[dict]:
        """Get recently saved posts with their metadata.
        
        Args:
            days: Number of recent days to look at.
            limit: Maximum number of posts to return.
            
        Returns:
            List of dicts with post metadata and preview.
        """
        raw_dir = self.base_dir / "raw"
        posts = []

        if not raw_dir.exists():
            return posts

        # Get date directories, sorted newest first
        date_dirs = sorted(raw_dir.iterdir(), reverse=True)

        for date_dir in date_dirs[:days]:
            if not date_dir.is_dir():
                continue
            for md_file in sorted(date_dir.glob("*.md"), reverse=True):
                try:
                    fm = frontmatter.load(str(md_file))
                    posts.append({
                        "path": str(md_file),
                        "metadata": dict(fm.metadata),
                        "content_preview": fm.content[:300],
                    })
                    if len(posts) >= limit:
                        return posts
                except Exception as e:
                    logger.debug(f"Error reading {md_file}: {e}")

        return posts

    def get_existing_insights(self) -> str:
        """Read all existing topic insights as a single string for context.
        
        Returns:
            Concatenated content of all topic insight files.
        """
        insights_dir = self.base_dir / "insights" / "topics"
        parts = []

        if not insights_dir.exists():
            return ""

        for md_file in sorted(insights_dir.glob("*.md")):
            try:
                fm = frontmatter.load(str(md_file))
                parts.append(f"## {fm.metadata.get('title', md_file.stem)}\n\n{fm.content}")
            except Exception:
                continue

        return "\n\n---\n\n".join(parts)

    def get_existing_people(self) -> str:
        """Read all existing people profiles as a single string.
        
        Returns:
            Concatenated content of all people profile files.
        """
        people_dir = self.base_dir / "insights" / "people"
        parts = []

        if not people_dir.exists():
            return ""

        for md_file in sorted(people_dir.glob("*.md")):
            try:
                fm = frontmatter.load(str(md_file))
                parts.append(f"## {fm.metadata.get('name', md_file.stem)}\n\n{fm.content}")
            except Exception:
                continue

        return "\n\n---\n\n".join(parts)

    def count_posts_today(self) -> int:
        """Count how many posts were saved today."""
        today = datetime.now().strftime("%Y-%m-%d")
        day_dir = self.base_dir / "raw" / today
        if not day_dir.exists():
            return 0
        return len(list(day_dir.glob("*.md")))

    def get_all_post_ids(self) -> set[str]:
        """Get all saved post IDs to avoid duplicates."""
        raw_dir = self.base_dir / "raw"
        ids = set()
        if not raw_dir.exists():
            return ids

        for date_dir in raw_dir.iterdir():
            if date_dir.is_dir():
                for md_file in date_dir.glob("*.md"):
                    ids.add(md_file.stem)

        return ids
