"""Knowledge store - Markdown file-based knowledge management (atoms/ structure)."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

import frontmatter

from src.config import KNOWLEDGE_DIR
from src.knowledge.models import LinkedInPost

logger = logging.getLogger(__name__)


class KnowledgeStore:
    """Manages the Markdown-based knowledge repository."""

    def __init__(self) -> None:
        self.base_dir = KNOWLEDGE_DIR
        self._ensure_structure()

    def _ensure_structure(self) -> None:
        dirs = [
            self.base_dir / "raw",
            self.base_dir / "atoms",
            self.base_dir / "maps" / "weekly",
            self.base_dir / "maps" / "monthly",
            self.base_dir / "insights" / "people",
            self.base_dir / "insights" / "digests",
            self.base_dir / "archive" / "legacy_topics",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        self._refresh_index()

    # ── Raw Posts ────────────────────────────────────────────────────

    def save_post(self, post: LinkedInPost) -> Path:
        """Save a raw LinkedIn post as Markdown."""
        today = datetime.now().strftime("%Y-%m-%d")
        day_dir = self.base_dir / "raw" / today
        day_dir.mkdir(parents=True, exist_ok=True)
        filepath = day_dir / f"{post.post_id}.md"

        metadata = {
            "post_id": post.post_id,
            "url": post.url,
            "author_name": post.author.name,
            "author_headline": post.author.headline,
            "author_profile": post.author.profile_url,
            "relevance_score": post.relevance_score,
            "novelty_score": post.novelty_score,
            "novelty_reason": post.novelty_reason,
            "topics": post.relevance_topics,
            "crawl_source": post.crawl_source.value,
            "published_date": post.published_date,
            "crawled_at": post.crawled_at,
            "reactions": post.reactions_count,
            "comments": post.comments_count,
            "post_type": post.post_type.value,
        }

        content_parts = [f"# {post.author.name or '(Unknown)'}: {post.summary or '(no summary)'}\n"]
        if post.summary:
            content_parts.append(f"**AI 요약**: {post.summary}\n")
        if post.published_date:
            content_parts.append(f"**게시일**: {post.published_date}\n")
        content_parts.append(f"**참신성 점수**: {post.novelty_score}/100\n")
        if post.novelty_reason:
            content_parts.append(f"**참신성 이유**: {post.novelty_reason}\n")
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

        fm_post = frontmatter.Post("\n".join(content_parts), **metadata)
        filepath.write_text(frontmatter.dumps(fm_post), encoding="utf-8")
        logger.info(f"Saved post: {filepath.name}")
        return filepath

    # ── Atom Notes ───────────────────────────────────────────────────

    def save_atom(
        self,
        atom_id: str,
        concept: str,
        body: str,
        maturity: str = "seedling",
        ttl_days: int = 365,
        sources: list[dict] | None = None,
        related: list[str] | None = None,
    ) -> Path:
        """Save or update an atomic concept note.

        If the file already exists, new content is APPENDED (not overwritten),
        preserving the full history.
        """
        filepath = self.base_dir / "atoms" / f"{atom_id}.md"
        now = datetime.now().strftime("%Y-%m-%d")

        if filepath.exists():
            # Append mode — keep existing metadata, add new section
            fm = frontmatter.load(str(filepath))
            existing_body = fm.content
            existing_sources = fm.metadata.get("sources", [])
            existing_related = fm.metadata.get("related", [])

            # Merge sources & related (dedup)
            merged_sources = existing_sources + (sources or [])
            seen_urls = set()
            deduped_sources = []
            for s in merged_sources:
                url = s.get("url", "")
                if url not in seen_urls:
                    seen_urls.add(url)
                    deduped_sources.append(s)

            merged_related = list(dict.fromkeys(existing_related + (related or [])))

            # Bump maturity if higher
            maturity_rank = {"seedling": 0, "budding": 1, "evergreen": 2}
            current_rank = maturity_rank.get(fm.metadata.get("maturity", "seedling"), 0)
            new_rank = maturity_rank.get(maturity, 0)
            final_maturity = maturity if new_rank > current_rank else fm.metadata.get("maturity", "seedling")

            new_body = f"{existing_body}\n\n---\n\n### 업데이트 {now}\n\n{body}"

            meta = dict(fm.metadata)
            meta.update({
                "updated": now,
                "maturity": final_maturity,
                "sources": deduped_sources,
                "related": merged_related,
            })
            fm_post = frontmatter.Post(new_body, **meta)
        else:
            # New atom
            meta = {
                "id": atom_id,
                "concept": concept,
                "maturity": maturity,
                "ttl_days": ttl_days,
                "created": now,
                "updated": now,
                "sources": sources or [],
                "related": related or [],
            }
            fm_post = frontmatter.Post(body, **meta)

        filepath.write_text(frontmatter.dumps(fm_post), encoding="utf-8")
        logger.info(f"Saved atom: {filepath.name} (maturity={fm_post.metadata.get('maturity')})")
        return filepath

    def update_atom_links(self, atom_id: str, related: list[str]) -> None:
        """Update the related links field of an existing atom."""
        filepath = self.base_dir / "atoms" / f"{atom_id}.md"
        if not filepath.exists():
            return
        try:
            fm = frontmatter.load(str(filepath))
            existing = fm.metadata.get("related", [])
            merged = list(dict.fromkeys(existing + related))
            fm.metadata["related"] = merged
            filepath.write_text(frontmatter.dumps(fm), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not update links for {atom_id}: {e}")

    def get_atoms_context(self, max_chars: int = 6000) -> str:
        """Return a concise summary of existing atoms for LLM context.

        Instead of dumping full content (which grows unboundedly), returns
        concept titles + first 300 chars of each atom, sorted by recency.
        Caps total output at max_chars.
        """
        atoms_dir = self.base_dir / "atoms"
        if not atoms_dir.exists():
            return ""

        # Sort by modification time, newest first
        files = sorted(atoms_dir.glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True)
        parts = []
        total = 0
        for f in files:
            try:
                fm = frontmatter.load(str(f))
                concept = fm.metadata.get("concept", f.stem)
                maturity = fm.metadata.get("maturity", "?")
                preview = fm.content[:300].replace("\n", " ").strip()
                entry = f"**[{maturity}] {concept}**\n{preview}\n"
                if total + len(entry) > max_chars:
                    break
                parts.append(entry)
                total += len(entry)
            except Exception:
                continue
        return "\n---\n".join(parts)

    def get_atom_by_id(self, atom_id: str) -> dict | None:
        """Load a single atom file, returns metadata + content dict."""
        filepath = self.base_dir / "atoms" / f"{atom_id}.md"
        if not filepath.exists():
            return None
        try:
            fm = frontmatter.load(str(filepath))
            return {"metadata": dict(fm.metadata), "content": fm.content}
        except Exception:
            return None

    def get_all_atoms(self) -> list[dict]:
        """Return lightweight list of all atoms (metadata only, no full content)."""
        atoms_dir = self.base_dir / "atoms"
        if not atoms_dir.exists():
            return []
        result = []
        for f in sorted(atoms_dir.glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True):
            try:
                fm = frontmatter.load(str(f))
                result.append({
                    "id": fm.metadata.get("id", f.stem),
                    "concept": fm.metadata.get("concept", f.stem),
                    "maturity": fm.metadata.get("maturity", "seedling"),
                    "ttl_days": fm.metadata.get("ttl_days", 365),
                    "updated": fm.metadata.get("updated", ""),
                    "related_count": len(fm.metadata.get("related", [])),
                    "source_count": len(fm.metadata.get("sources", [])),
                    "path": str(f),
                })
            except Exception:
                continue
        return result

    def get_expired_atoms(self) -> list[str]:
        """Return atom IDs whose TTL has expired."""
        expired = []
        atoms_dir = self.base_dir / "atoms"
        if not atoms_dir.exists():
            return expired
        today = datetime.now()
        for f in atoms_dir.glob("*.md"):
            try:
                fm = frontmatter.load(str(f))
                updated_str = fm.metadata.get("updated", "")
                ttl = int(fm.metadata.get("ttl_days", 365))
                if updated_str:
                    updated = datetime.strptime(updated_str, "%Y-%m-%d")
                    days_old = (today - updated).days
                    if days_old > ttl:
                        expired.append(fm.metadata.get("id", f.stem))
            except Exception:
                continue
        return expired

    # ── People (append mode) ─────────────────────────────────────────

    def save_person(self, person_slug: str, name: str, content: str, metadata: dict | None = None) -> Path:
        """Save or APPEND to a person profile (preserves history)."""
        filepath = self.base_dir / "insights" / "people" / f"{person_slug}.md"
        now = datetime.now().strftime("%Y-%m-%d")

        if filepath.exists():
            fm = frontmatter.load(str(filepath))
            existing_body = fm.content
            new_body = f"{existing_body}\n\n---\n\n### 업데이트 {now}\n\n{content}"
            meta = dict(fm.metadata)
            meta["updated_at"] = datetime.now().isoformat()
            if metadata:
                # Merge but don't overwrite non-empty fields
                for k, v in metadata.items():
                    if v and not meta.get(k):
                        meta[k] = v
        else:
            new_body = content
            meta = {"name": name, "created_at": datetime.now().isoformat(), "updated_at": datetime.now().isoformat()}
            if metadata:
                meta.update(metadata)

        fm_post = frontmatter.Post(new_body, **meta)
        filepath.write_text(frontmatter.dumps(fm_post), encoding="utf-8")
        logger.info(f"Saved person: {filepath.name}")
        return filepath

    # ── Maps (weekly / monthly) ──────────────────────────────────────

    def save_weekly_map(self, content: str, week_label: str | None = None) -> Path:
        """Save a weekly knowledge map."""
        if not week_label:
            now = datetime.now()
            week_label = f"{now.year}-W{now.isocalendar()[1]:02d}"
        filepath = self.base_dir / "maps" / "weekly" / f"{week_label}.md"
        meta = {
            "title": f"Weekly Map — {week_label}",
            "week": week_label,
            "created_at": datetime.now().isoformat(),
        }
        fm_post = frontmatter.Post(content, **meta)
        filepath.write_text(frontmatter.dumps(fm_post), encoding="utf-8")
        logger.info(f"Saved weekly map: {filepath.name}")
        return filepath

    def save_monthly_map(self, content: str, month_label: str | None = None) -> Path:
        """Save a monthly knowledge map."""
        if not month_label:
            month_label = datetime.now().strftime("%Y-%m")
        filepath = self.base_dir / "maps" / "monthly" / f"{month_label}.md"
        meta = {
            "title": f"Monthly Map — {month_label}",
            "month": month_label,
            "created_at": datetime.now().isoformat(),
        }
        fm_post = frontmatter.Post(content, **meta)
        filepath.write_text(frontmatter.dumps(fm_post), encoding="utf-8")
        logger.info(f"Saved monthly map: {filepath.name}")
        return filepath

    # ── Digests (unchanged) ──────────────────────────────────────────

    def save_digest(self, content: str, metadata: dict | None = None) -> Path:
        now = datetime.now()
        filename = f"{now.strftime('%Y-%m-%d_%H%M')}.md"
        filepath = self.base_dir / "insights" / "digests" / filename
        meta = {"title": f"Digest - {now.strftime('%Y-%m-%d %H:%M')}", "created_at": now.isoformat()}
        if metadata:
            meta.update(metadata)
        fm_post = frontmatter.Post(content, **meta)
        filepath.write_text(frontmatter.dumps(fm_post), encoding="utf-8")
        logger.info(f"Saved digest: {filepath.name}")
        return filepath

    # ── Queries ──────────────────────────────────────────────────────

    def get_recent_posts(self, days: int = 3, limit: int = 50) -> list[dict]:
        raw_dir = self.base_dir / "raw"
        posts = []
        if not raw_dir.exists():
            return posts
        date_dirs = sorted(raw_dir.iterdir(), reverse=True)
        for date_dir in date_dirs[:days]:
            if not date_dir.is_dir():
                continue
            for md_file in sorted(date_dir.glob("*.md"), reverse=True):
                try:
                    fm = frontmatter.load(str(md_file))
                    posts.append({
                        "path": str(md_file.relative_to(self.base_dir)),
                        "metadata": dict(fm.metadata),
                        "content_preview": fm.content[:500],
                    })
                    if len(posts) >= limit:
                        return posts
                except Exception:
                    continue
        return posts

    def get_existing_people(self) -> str:
        people_dir = self.base_dir / "insights" / "people"
        parts = []
        if not people_dir.exists():
            return ""
        for md_file in sorted(people_dir.glob("*.md")):
            try:
                fm = frontmatter.load(str(md_file))
                # Only first 600 chars per person to limit token use
                preview = fm.content[:600]
                parts.append(f"## {fm.metadata.get('name', md_file.stem)}\n\n{preview}")
            except Exception:
                continue
        return "\n\n---\n\n".join(parts)

    def count_posts_today(self) -> int:
        today = datetime.now().strftime("%Y-%m-%d")
        day_dir = self.base_dir / "raw" / today
        if not day_dir.exists():
            return 0
        return len(list(day_dir.glob("*.md")))

    def get_all_post_ids(self) -> set[str]:
        raw_dir = self.base_dir / "raw"
        ids = set()
        if not raw_dir.exists():
            return ids
        for date_dir in raw_dir.iterdir():
            if date_dir.is_dir():
                for md_file in date_dir.glob("*.md"):
                    ids.add(md_file.stem)
        return ids

    # ── Index ────────────────────────────────────────────────────────

    def _refresh_index(self) -> None:
        """Auto-update index.md to reflect current knowledge structure."""
        atoms = self.get_all_atoms()
        people_dir = self.base_dir / "insights" / "people"
        people_count = len(list(people_dir.glob("*.md"))) if people_dir.exists() else 0
        raw_count = sum(
            len(list(d.glob("*.md")))
            for d in (self.base_dir / "raw").iterdir()
            if d.is_dir()
        ) if (self.base_dir / "raw").exists() else 0

        maturity_counts = {"seedling": 0, "budding": 0, "evergreen": 0}
        for a in atoms:
            maturity_counts[a.get("maturity", "seedling")] = maturity_counts.get(a.get("maturity", "seedling"), 0) + 1

        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [
            f"---\ntitle: LinkedIn AX Research Knowledge Base\nupdated: '{now}'\n---\n",
            "# LinkedIn AX Research Knowledge Base\n",
            "기업과 조직의 AX (AI Transformation) 관련 LinkedIn 리서치 지식베이스\n",
            "## 현황\n",
            f"| 항목 | 수 |",
            f"|------|---|",
            f"| 원문 포스트 | {raw_count} |",
            f"| 원자 노트 (atoms) | {len(atoms)} |",
            f"| 인물 프로필 | {people_count} |",
            f"| Seedling 🌱 | {maturity_counts['seedling']} |",
            f"| Budding 🌿 | {maturity_counts['budding']} |",
            f"| Evergreen 🌳 | {maturity_counts['evergreen']} |",
            f"\n_Last updated: {now}_\n",
            "## 구조\n",
            "- `raw/` — 수집된 원문 포스트 (날짜별)",
            "- `atoms/` — 원자적 개념 노트 (Zettelkasten 방식)",
            "- `maps/weekly/` — 주간 통합 인사이트",
            "- `maps/monthly/` — 월간 트렌드 분석",
            "- `insights/people/` — 주요 인물 프로필",
            "- `insights/digests/` — 단기 합성 다이제스트",
            "\n## 원자 노트 목록\n",
        ]

        for a in atoms[:50]:
            maturity_emoji = {"seedling": "🌱", "budding": "🌿", "evergreen": "🌳"}.get(a["maturity"], "📄")
            lines.append(f"- {maturity_emoji} **{a['concept']}** `{a['id']}`")

        (self.base_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")
