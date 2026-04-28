"""Knowledge synthesizer - extracts atomic notes and updates knowledge base."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime

from src.config import get_settings
from src.agent.llm import call_claude
from src.knowledge.models import LinkedInPost, TokenUsage
from src.knowledge.store import KnowledgeStore
from src.knowledge.git_sync import GitSync

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """당신은 {topic} 분야의 전문 리서치 분석가입니다.
소셜 미디어 포스트들을 분석하여 원자적 개념 노트(Atomic Notes)로 추출하는 것이 임무입니다.

원칙:
1. 원자성: 노트 하나 = 개념 하나. 여러 개념이 섞이면 분리
2. 사실과 의견 구분 (사실: 검증 가능, 의견: 해석/주장)
3. 출처와 날짜 명시
4. 기존 지식과 연결점 명시
5. 한국어 작성 (영문 고유명사는 원문 유지)"""

ATOM_EXTRACTION_PROMPT = """아래 LinkedIn 포스트들에서 핵심 개념 단위의 원자 노트를 추출해주세요.

## 기존 지식 베이스 (컨텍스트)
{existing_context}

## 새 포스트들
{posts_content}

## 요청
각 포스트에서 독립적인 개념/인사이트를 원자 노트로 추출하세요.
하나의 포스트에서 여러 원자 노트가 나올 수 있습니다 (최대 3개/포스트).
중복되거나 기존 지식 베이스에 이미 있는 내용은 건너뛰세요.

반드시 아래 JSON 형식으로만 응답하세요:
{{
  "atoms": [
    {{
      "concept": "개념을 명사구로 한 줄 (예: '현대차 Claude Code 도입의 변화관리 패턴')",
      "slug": "영문-소문자-하이픈 (예: 'hyundai-claude-code-change-pattern')",
      "maturity": "seedling",
      "ttl_days": 365,
      "body": "## 핵심 내용\\n\\n(사실과 의견 구분)\\n\\n## 시사점\\n\\n(AX 관점)\\n\\n## 출처\\n- 작성자: ...\\n- URL: ...\\n- 날짜: {today}",
      "source_url": "포스트 URL",
      "source_author": "작성자 이름"
    }}
  ]
}}

ttl_days 기준: AI 트렌드/사례=180, 방법론/전략=365, 원칙/구조=730"""

DIGEST_PROMPT = """아래 새로 수집된 포스트들을 분석하여 다이제스트를 작성해주세요.

## 기존 지식 베이스 요약
{existing_context}

## 새 포스트들
{posts_content}

# 다이제스트 - {date}

## 핵심 발견
- (3-5개 bullet, 가장 중요한 인사이트)

## 주제별 정리
### [주제]
- 사실: ...
- 의견: ...
- AX 시사점: ...

## 주목할 인물/조직

## 트렌드 & 시사점

## 후속 탐색 제안"""

PEOPLE_PROMPT = """아래 포스트들에서 AX 관련 주요 인물을 파악하고 프로필을 작성하세요.

## 기존 인물 프로필
{existing_profiles}

## 포스트들
{posts_content}

반드시 아래 JSON만 응답하세요 (다른 텍스트 없이):
{{
  "people": [
    {{
      "name": "이름 (불명이면 직함+소속으로)",
      "slug": "영문-소문자-하이픈",
      "headline": "직함/소속",
      "profile_url": "LinkedIn URL 또는 빈 문자열",
      "key_views": ["주요 관점/주장 1-3개"],
      "notable_posts": ["주목할 포스트 한줄 요약"],
      "relevance": "AX 분야 관련성 한 줄"
    }}
  ]
}}"""


class KnowledgeSynthesizer:
    """Synthesizes collected posts into atomic knowledge notes."""

    def __init__(self, store: KnowledgeStore, token_usage: TokenUsage | None = None, topic_description: str = "") -> None:
        self.store = store
        self.token_usage = token_usage or TokenUsage()
        self.topic = topic_description or "기업 AX (AI Transformation)"
        self._system_prompt = SYSTEM_PROMPT_TEMPLATE.format(topic=self.topic)
        self.git = GitSync()
        self.git.ensure_git_config()

    async def synthesize(self, posts: list[LinkedInPost]) -> dict:
        """Run a synthesis cycle: extract atoms, update people, generate digest."""
        if not posts:
            return {}

        logger.info(f"Synthesis started: {len(posts)} posts")
        results: dict = {}

        existing_context = self.store.get_atoms_context(max_chars=4000)

        # 1. Extract atomic notes
        try:
            atom_paths = await self._extract_atoms(posts, existing_context)
            results["atoms"] = [str(p) for p in atom_paths]
        except Exception as e:
            logger.error(f"Atom extraction failed: {e}")

        # 2. Update people profiles
        try:
            people_paths = await self._update_people(posts)
            results["people"] = [str(p) for p in people_paths]
        except Exception as e:
            logger.error(f"People update failed: {e}")

        # 3. Generate digest
        try:
            digest_path = await self._generate_digest(posts, existing_context)
            results["digest"] = str(digest_path)
        except Exception as e:
            logger.error(f"Digest generation failed: {e}")

        # 4. Auto-link atoms via embeddings
        try:
            await self._link_atoms(results.get("atoms", []))
        except Exception as e:
            logger.warning(f"Auto-linking failed (non-critical): {e}")

        # 5. Refresh index
        self.store._refresh_index()

        logger.info(f"Synthesis complete: {results}")

        # 6. Git commit & push
        try:
            pushed = await self.git.commit_and_push(results, post_count=len(posts))
            if pushed:
                logger.info("Git: pushed to GitHub.")
        except Exception as e:
            logger.error(f"Git sync error: {e}")

        return results

    async def _extract_atoms(self, posts: list[LinkedInPost], existing_context: str) -> list[str]:
        """Extract atomic concept notes from posts."""
        posts_content = self._format_posts(posts)
        today = datetime.now().strftime("%Y-%m-%d")

        prompt = ATOM_EXTRACTION_PROMPT.format(
            existing_context=existing_context or "(아직 없음)",
            posts_content=posts_content,
            today=today,
        )

        response = await self._call_llm(prompt, max_tokens=4000)
        if not response:
            return []

        data = self._extract_json(response)
        if not data:
            logger.warning("No JSON in atom extraction response")
            return []

        paths = []
        today_compact = datetime.now().strftime("%Y%m%d")

        for atom in data.get("atoms", []):
            concept = atom.get("concept", "").strip()
            slug = atom.get("slug", "") or self._slugify(concept)
            if not concept or not slug:
                continue

            atom_id = f"{today_compact}-{slug[:50]}"
            source_url = atom.get("source_url", "")
            source_author = atom.get("source_author", "")

            sources = []
            if source_url:
                sources.append({
                    "url": source_url,
                    "author": source_author,
                    "date": today,
                })

            path = self.store.save_atom(
                atom_id=atom_id,
                concept=concept,
                body=atom.get("body", ""),
                maturity=atom.get("maturity", "seedling"),
                ttl_days=int(atom.get("ttl_days", 365)),
                sources=sources,
            )
            paths.append(str(path))

        logger.info(f"Extracted {len(paths)} atoms")
        return paths

    async def _update_people(self, posts: list[LinkedInPost]) -> list[str]:
        """Update person profiles from posts."""
        existing = self.store.get_existing_people()
        posts_content = self._format_posts(posts)

        prompt = PEOPLE_PROMPT.format(
            existing_profiles=existing[:3000] or "(없음)",
            posts_content=posts_content,
        )

        response = await self._call_llm(prompt, max_tokens=3000)
        if not response:
            return []

        data = self._extract_json(response)
        if not data:
            logger.warning(f"People JSON parse failed. Raw: {response[:200]}")
            return []

        paths = []
        today = datetime.now().strftime("%Y-%m-%d")

        for person in data.get("people", []):
            name = person.get("name", "").strip()
            if not name:
                continue
            slug = person.get("slug", "") or self._slugify(name)

            lines = [f"# {name}\n"]
            if person.get("headline"):
                lines.append(f"**소속/직함**: {person['headline']}")
            if person.get("profile_url"):
                lines.append(f"**LinkedIn**: {person['profile_url']}")
            lines.append(f"**업데이트**: {today}\n")
            if person.get("relevance"):
                lines.append(f"\n## AX 관련성\n{person['relevance']}")
            if person.get("key_views"):
                lines.append("\n## 주요 관점")
                for v in person["key_views"]:
                    lines.append(f"- {v}")
            if person.get("notable_posts"):
                lines.append("\n## 주목할 포스트")
                for p in person["notable_posts"]:
                    lines.append(f"- {p}")

            path = self.store.save_person(
                person_slug=slug,
                name=name,
                content="\n".join(lines),
                metadata={
                    "headline": person.get("headline", ""),
                    "profile_url": person.get("profile_url", ""),
                },
            )
            paths.append(str(path))
            logger.info(f"Saved person: {name}")

        return paths

    async def _generate_digest(self, posts: list[LinkedInPost], existing_context: str) -> str:
        """Generate a short-cycle digest."""
        prompt = DIGEST_PROMPT.format(
            existing_context=existing_context[:3000] or "(없음)",
            posts_content=self._format_posts(posts),
            date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        )
        response = await self._call_llm(prompt, max_tokens=4000)
        if not response:
            return ""
        path = self.store.save_digest(
            content=response,
            metadata={"post_count": len(posts), "model": "claude-code-cli"},
        )
        return str(path)

    async def _link_atoms(self, new_atom_paths: list[str]) -> None:
        """Auto-link new atoms to related existing atoms via embeddings."""
        if not new_atom_paths:
            return
        try:
            from src.knowledge.embedder import Embedder
            embedder = Embedder()
            import frontmatter as fm_lib
            for path_str in new_atom_paths:
                from pathlib import Path
                atom_file = Path(path_str)
                if not atom_file.exists():
                    continue
                fm = fm_lib.load(str(atom_file))
                atom_id = fm.metadata.get("id", atom_file.stem)
                concept = fm.metadata.get("concept", "")
                related = await embedder.find_related(atom_id, concept, fm.content)
                if related:
                    self.store.update_atom_links(atom_id, related)
                    logger.info(f"Linked {atom_id} → {related}")
        except Exception as e:
            logger.warning(f"Embedding link error: {e}")

    async def _call_llm(self, prompt: str, max_tokens: int = 4000) -> str:
        return await call_claude(prompt, system_prompt=self._system_prompt)

    def _format_posts(self, posts: list[LinkedInPost]) -> str:
        parts = []
        for i, post in enumerate(posts, 1):
            part = f"### 포스트 {i}\n"
            if post.author.name:
                part += f"- 작성자: {post.author.name}"
                if post.author.headline:
                    part += f" ({post.author.headline})"
                part += "\n"
            if post.url:
                part += f"- URL: {post.url}\n"
            part += f"- 관련성: {post.relevance_score}/100 | 토픽: {', '.join(post.relevance_topics)}\n"
            part += f"- 요약: {post.summary}\n"
            part += f"- 본문:\n{post.content[:1500]}"
            if len(post.content) > 1500:
                part += "...(truncated)"
            parts.append(part)
        return "\n\n".join(parts)

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        if not text:
            return None
        for pattern in [
            lambda t: json.loads(t.strip()),
            lambda t: json.loads(re.search(r"```json\s*(.*?)\s*```", t, re.DOTALL).group(1)),
            lambda t: json.loads(re.search(r"```\s*(.*?)\s*```", t, re.DOTALL).group(1)),
            lambda t: json.loads(re.search(r"\{.*\}", t, re.DOTALL).group(0)),
        ]:
            try:
                return pattern(text)
            except Exception:
                continue
        return None

    @staticmethod
    def _slugify(text: str) -> str:
        slug = re.sub(r"[^\w\s-]", "", text.lower())
        slug = re.sub(r"[\s_]+", "-", slug)
        return slug.strip("-")[:60] or "general"
