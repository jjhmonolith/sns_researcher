"""Knowledge synthesizer using GPT-5.4 - periodic deep analysis and knowledge organization."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime

from openai import AsyncOpenAI

from src.config import get_settings
from src.knowledge.models import LinkedInPost, TokenUsage
from src.knowledge.store import KnowledgeStore
from src.knowledge.git_sync import GitSync

logger = logging.getLogger(__name__)

SYNTHESIS_SYSTEM_PROMPT = """당신은 기업 AX (AI Transformation) 분야의 전문 리서치 분석가입니다.
LinkedIn에서 수집된 포스트들을 분석하여 체계적인 지식으로 정리하는 것이 임무입니다.

분석 원칙:
1. 사실과 의견을 구분하여 기록
2. 트렌드와 패턴을 식별
3. 핵심 인물/조직의 관점을 정리
4. 실행 가능한 인사이트를 도출
5. 기존 지식과의 연결점을 찾아 통합
6. 한국어로 작성 (영문 고유명사는 원문 유지)

출력 형식은 요청에 따라 다릅니다."""

DIGEST_PROMPT_TEMPLATE = """아래는 최근 수집된 LinkedIn 포스트들입니다. 이 포스트들을 분석하여 다이제스트를 작성해주세요.

## 기존 지식 베이스 (컨텍스트)
{existing_knowledge}

## 새로 수집된 포스트들
{posts_content}

## 요청
위 포스트들을 분석하여 아래 형식의 다이제스트를 작성해주세요:

# 다이제스트 - {date}

## 핵심 발견 (Key Findings)
- 가장 중요한 인사이트 3-5개를 bullet point로

## 주제별 정리
### [주제1]
- 관련 포스트 요약과 인사이트
### [주제2]
- ...

## 주목할 인물/조직
- 이번 기간 주목할만한 발언을 한 인물/조직과 그 내용

## 트렌드 & 시사점
- 포스트들에서 관찰되는 트렌드
- 기업 AX에 대한 시사점

## 후속 탐색 제안
- 더 깊이 살펴볼 만한 주제나 인물"""

TOPIC_UPDATE_PROMPT_TEMPLATE = """아래 새 포스트들의 인사이트를 기존 주제 문서에 통합하여, 주제별로 정리된 문서를 작성해주세요.

## 기존 주제 문서 (있는 경우)
{existing_content}

## 새로운 포스트들
{new_insights}

## 요청
새로운 포스트 내용을 바탕으로, 아래 형식의 주제별 인사이트 문서를 작성해주세요.
기존 내용이 있으면 통합하고, 없으면 새로 작성하세요.

# {topic} 인사이트

## 핵심 트렌드
- (주요 트렌드 bullet)

## 사례 & 데이터
- (구체적 사례, 수치, 사실 위주)

## 시사점
- (기업 AX 관점의 실행 가능한 시사점)

## 출처 포스트
- (날짜, 작성자, 핵심 주장 한 줄)

날짜: {date}"""

PERSON_PROFILE_PROMPT_TEMPLATE = """아래 포스트들에서 언급된 주요 인물의 프로필을 작성/업데이트해주세요.

## 기존 인물 프로필
{existing_profiles}

## 새로 수집된 포스트들
{posts_content}

## 요청
포스트에서 AX 관련 의미있는 발언/활동을 한 인물들을 파악하고,
반드시 아래 JSON 형식으로만 응답하세요 (다른 텍스트 없이 JSON만):

{{
  "people": [
    {{
      "name": "이름",
      "slug": "url-safe-slug",
      "headline": "직함/소속",
      "profile_url": "LinkedIn URL 또는 빈 문자열",
      "key_views": ["이 인물의 주요 관점/주장"],
      "notable_posts": ["주목할 포스트 요약"],
      "relevance": "이 인물이 AX 분야에서 중요한 이유"
    }}
  ]
}}"""


class KnowledgeSynthesizer:
    """Synthesizes collected posts into organized knowledge using GPT-5.4."""

    def __init__(self, store: KnowledgeStore, token_usage: TokenUsage | None = None) -> None:
        settings = get_settings()
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.model_powerful
        self.store = store
        self.token_usage = token_usage or TokenUsage()
        self.git = GitSync()
        self.git.ensure_git_config()

    async def synthesize(self, posts: list[LinkedInPost]) -> dict:
        """Run full synthesis cycle on collected posts.
        
        This generates:
        1. A digest summarizing the new posts
        2. Updated topic documents
        3. Updated person profiles
        
        Args:
            posts: List of relevant posts to synthesize.
            
        Returns:
            Dict with paths to generated files.
        """
        if not posts:
            logger.info("No posts to synthesize.")
            return {}

        logger.info(f"Starting synthesis of {len(posts)} posts...")
        results = {}

        # 1. Generate digest
        try:
            digest_path = await self._generate_digest(posts)
            results["digest"] = str(digest_path)
        except Exception as e:
            logger.error(f"Digest generation failed: {e}")

        # 2. Update topic documents
        try:
            topic_paths = await self._update_topics(posts)
            results["topics"] = [str(p) for p in topic_paths]
        except Exception as e:
            logger.error(f"Topic update failed: {e}")

        # 3. Update person profiles
        try:
            people_paths = await self._update_people(posts)
            results["people"] = [str(p) for p in people_paths]
        except Exception as e:
            logger.error(f"People update failed: {e}")

        logger.info(f"Synthesis complete. Generated: {results}")

        # 4. Auto commit & push to GitHub
        try:
            pushed = await self.git.commit_and_push(results, post_count=len(posts))
            if pushed:
                logger.info("Git: knowledge base pushed to GitHub.")
            else:
                logger.warning("Git: push failed or nothing to commit.")
        except Exception as e:
            logger.error(f"Git sync error: {e}")

        return results

    async def _generate_digest(self, posts: list[LinkedInPost]) -> str:
        """Generate a digest document from collected posts."""
        existing_knowledge = self.store.get_existing_insights()
        if len(existing_knowledge) > 5000:
            existing_knowledge = existing_knowledge[:5000] + "\n\n...(truncated)"

        posts_content = self._format_posts_for_prompt(posts)

        prompt = DIGEST_PROMPT_TEMPLATE.format(
            existing_knowledge=existing_knowledge or "(아직 축적된 지식 없음)",
            posts_content=posts_content,
            date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        )

        response = await self._call_llm(prompt)

        if response:
            path = self.store.save_digest(
                content=response,
                metadata={
                    "post_count": len(posts),
                    "model": self.model,
                },
            )
            return str(path)

        return ""

    async def _update_topics(self, posts: list[LinkedInPost]) -> list[str]:
        """Update or create topic insight documents based on new posts.
        
        Groups posts by topic and updates one document per major topic cluster.
        At most 3 topics are processed per synthesis to limit LLM calls.
        """
        # Collect all topics mentioned across posts, count frequency
        topic_count: dict[str, int] = {}
        topic_posts: dict[str, list[LinkedInPost]] = {}
        for post in posts:
            for topic in post.relevance_topics:
                topic_count[topic] = topic_count.get(topic, 0) + 1
                topic_posts.setdefault(topic, []).append(post)

        if not topic_count:
            # Default to a general AX topic
            topic_posts["enterprise_ax"] = posts
            topic_count["enterprise_ax"] = len(posts)

        # Process top 3 most frequent topics only
        top_topics = sorted(topic_count, key=lambda t: topic_count[t], reverse=True)[:3]

        paths = []
        for topic in top_topics:
            try:
                slug = self._slugify(topic)
                t_posts = topic_posts[topic]

                # Load existing content for this topic
                import frontmatter as fm_lib
                topic_file = self.store.base_dir / "insights" / "topics" / f"{slug}.md"
                existing_content = ""
                if topic_file.exists():
                    existing_fm = fm_lib.load(str(topic_file))
                    existing_content = existing_fm.content

                new_insights = self._format_posts_for_prompt(t_posts)

                prompt = TOPIC_UPDATE_PROMPT_TEMPLATE.format(
                    existing_content=existing_content or f"# {topic}\n\n(새로운 주제)",
                    new_insights=new_insights,
                    topic=topic,
                    date=datetime.now().strftime("%Y-%m-%d"),
                )

                response = await self._call_llm(prompt)
                if response:
                    path = self.store.save_insight(
                        topic_slug=slug,
                        title=topic,
                        content=response,
                        metadata={"post_count": len(t_posts)},
                    )
                    paths.append(str(path))
                    logger.info(f"Updated topic: {topic} -> {path}")
            except Exception as e:
                logger.error(f"Error updating topic '{topic}': {e}")

        return paths

    async def _update_people(self, posts: list[LinkedInPost]) -> list[str]:
        """Extract and update notable person profiles."""
        existing_profiles = self.store.get_existing_people()
        if len(existing_profiles) > 3000:
            existing_profiles = existing_profiles[:3000] + "\n\n...(truncated)"

        posts_content = self._format_posts_for_prompt(posts)

        prompt = PERSON_PROFILE_PROMPT_TEMPLATE.format(
            existing_profiles=existing_profiles or "(아직 등록된 인물 없음)",
            posts_content=posts_content,
        )

        response = await self._call_llm(prompt)
        if not response:
            return []

        paths = []
        try:
            # Try multiple JSON extraction strategies
            data = self._extract_json(response)
            people = data.get("people", []) if data else []

            if not people:
                logger.warning("No people extracted from synthesis response.")
                logger.debug(f"Raw people response: {response[:300]}")
                return paths

            for person in people:
                name = person.get("name", "").strip()
                if not name:
                    continue
                slug = person.get("slug", "") or self._slugify(name)

                content_parts = [f"# {name}\n"]
                if person.get("headline"):
                    content_parts.append(f"**소속/직함**: {person['headline']}\n")
                if person.get("profile_url"):
                    content_parts.append(f"**LinkedIn**: {person['profile_url']}\n")
                content_parts.append(f"**업데이트**: {datetime.now().strftime('%Y-%m-%d')}\n")
                if person.get("relevance"):
                    content_parts.append(f"\n## AX 분야 관련성\n{person['relevance']}\n")
                if person.get("key_views"):
                    content_parts.append("\n## 주요 관점\n")
                    for view in person["key_views"]:
                        content_parts.append(f"- {view}")
                if person.get("notable_posts"):
                    content_parts.append("\n\n## 주목할 포스트\n")
                    for post_summary in person["notable_posts"]:
                        content_parts.append(f"- {post_summary}")

                content = "\n".join(content_parts)
                path = self.store.save_person(
                    person_slug=slug,
                    name=name,
                    content=content,
                    metadata={
                        "headline": person.get("headline", ""),
                        "profile_url": person.get("profile_url", ""),
                    },
                )
                paths.append(str(path))
                logger.info(f"Saved person profile: {name} -> {path}")

        except Exception as e:
            logger.error(f"Error parsing people response: {e}")
            logger.debug(f"Raw response: {response[:500]}")

        return paths

    async def _call_llm(self, user_prompt: str) -> str:
        """Make a call to GPT-5.4 for synthesis.
        
        Args:
            user_prompt: The user message to send.
            
        Returns:
            The model's response text.
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_completion_tokens=4000,
            )

            # Track token usage
            if response.usage:
                self.token_usage.powerful_input_tokens += response.usage.prompt_tokens
                self.token_usage.powerful_output_tokens += response.usage.completion_tokens

            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""

    def _format_posts_for_prompt(self, posts: list[LinkedInPost]) -> str:
        """Format posts into a prompt-friendly string."""
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
            part += f"- 관련성: {post.relevance_score}/100\n"
            part += f"- 토픽: {', '.join(post.relevance_topics)}\n"
            part += f"- 요약: {post.summary}\n"
            # Limit content to avoid token explosion
            content = post.content[:1500]
            if len(post.content) > 1500:
                content += "...(truncated)"
            part += f"- 본문:\n{content}\n"
            parts.append(part)

        return "\n".join(parts)

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        """Robustly extract JSON from LLM response text.
        
        Tries multiple strategies:
        1. Direct JSON parse
        2. Extract from ```json ... ``` code block
        3. Find first { ... } block
        """
        if not text:
            return None

        # Strategy 1: direct parse
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Strategy 2: extract from ```json block
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Strategy 3: extract from ``` block (no lang specifier)
        match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Strategy 4: find first { ... } in the text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    @staticmethod
    def _slugify(text: str) -> str:
        """Convert text to a URL-safe slug."""
        # Replace Korean/special chars with underscores
        slug = re.sub(r"[^\w\s-]", "", text.lower())
        slug = re.sub(r"[\s-]+", "_", slug)
        slug = slug.strip("_")
        return slug or "general"
