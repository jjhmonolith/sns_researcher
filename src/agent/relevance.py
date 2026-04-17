"""Relevance & novelty judgment using GPT-5.4 nano."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

from src.config import get_settings
from src.knowledge.models import LinkedInPost, TokenUsage

if TYPE_CHECKING:
    from src.knowledge.store import KnowledgeStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """당신은 LinkedIn 포스트의 관련성 및 참신성(Novelty) 분석 전문가입니다.

## 1단계: 관련성 판별 (AX 주제)
포스트가 "기업과 조직의 AX (AI Transformation)" 범위에 속하는지 판단합니다.

관련 주제: 기업 AI 도입/전환 전략, AI 업무 프로세스 혁신, 기업용 AI 솔루션/도구/플랫폼,
AI 역량 구축/인재 양성, AI 거버넌스/윤리/규제, AI 산업/비즈니스 모델 변화,
경영진 AI 비전, AI 스타트업 B2B, 생성형 AI 기업 활용, AI Agent 업무 자동화.

비관련: 기업 적용 무관한 순수 기술 논문, AI 무관 개인 일상, 단순 채용 공고,
AI 무관 광고, 교육/EdTech (기업 AX와 무관), 개인 브랜딩/자기PR.

## 2단계: 참신성 판별 (Novelty)
관련 포스트라면, 이 내용이 얼마나 새롭고 가치 있는 인사이트인지 판단합니다.

높은 참신성 (70-100):
- 구체적 사례 연구 (기업명, 수치, 결과 포함)
- 새로운 데이터/통계/조사 결과 발표
- 독창적 프레임워크나 방법론 제시
- 실패 사례와 교훈 공유
- 최신 도구/플랫폼의 실사용 리뷰

낮은 참신성 (0-30):
- "AI가 비즈니스를 바꾸고 있다" 류의 일반론
- 이미 잘 알려진 사실의 반복
- 구체적 데이터 없는 추상적 주장
- 유행어 나열 (buzzword-heavy) 콘텐츠

기존 지식 베이스에 이미 유사한 인사이트가 있다면 참신성을 낮게 평가하세요.

응답은 반드시 아래 JSON 형식으로만 답하세요:
{
  "is_relevant": true/false,
  "novelty_score": 0-100 사이 정수,
  "novelty_reason": "이 포스트가 참신하거나 참신하지 않은 이유 한 줄",
  "topics": ["관련 토픽1", "토픽2"],
  "summary": "1-2문장 요약",
  "should_follow_links": true/false,
  "follow_targets": ["따라가볼 URL이나 프로필"]
}"""


class RelevanceJudge:
    """Uses GPT-5.4 nano to judge post relevance and novelty."""

    def __init__(
        self,
        token_usage: TokenUsage | None = None,
        store: KnowledgeStore | None = None,
    ) -> None:
        settings = get_settings()
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.model_fast
        self.threshold = settings.relevance_threshold
        self.token_usage = token_usage or TokenUsage()
        self.store = store

    async def judge(self, post: LinkedInPost) -> LinkedInPost:
        """Analyze a post's relevance and novelty, update it with results."""
        if not post.content or len(post.content.strip()) < 20:
            post.relevance_score = 0
            post.novelty_score = 0
            post.is_relevant = False
            post.is_novel = False
            post.summary = "Content too short to analyze"
            return post

        user_message = self._build_prompt(post)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.1,
                max_completion_tokens=500,
                response_format={"type": "json_object"},
            )

            if response.usage:
                self.token_usage.nano_input_tokens += response.usage.prompt_tokens
                self.token_usage.nano_output_tokens += response.usage.completion_tokens

            content = response.choices[0].message.content
            if content:
                result = json.loads(content)

                is_ax_relevant = result.get("is_relevant", False)
                novelty_score = min(100, max(0, int(result.get("novelty_score", 0))))

                post.novelty_score = novelty_score
                post.novelty_reason = result.get("novelty_reason", "")
                post.relevance_score = novelty_score  # backward compat
                post.is_novel = novelty_score >= self.threshold
                post.is_relevant = is_ax_relevant and post.is_novel
                post.relevance_topics = result.get("topics", [])
                post.summary = result.get("summary", "")
                post.should_follow_links = result.get("should_follow_links", False)
                post.follow_targets = result.get("follow_targets", [])

                logger.info(
                    f"Novelty: {novelty_score}/100 "
                    f"{'[RELEVANT+NOVEL]' if post.is_relevant else '[skip]'} "
                    f"(AX={is_ax_relevant}) "
                    f"- {post.summary[:60]}..."
                )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse relevance response: {e}")
            post.relevance_score = 0
            post.novelty_score = 0
            post.is_relevant = False
            post.is_novel = False
        except Exception as e:
            logger.error(f"Relevance judgment error: {e}")
            post.relevance_score = 0
            post.novelty_score = 0
            post.is_relevant = False
            post.is_novel = False

        return post

    def _build_prompt(self, post: LinkedInPost) -> str:
        """Build the user prompt from post data."""
        parts = []

        if post.author.name:
            parts.append(f"작성자: {post.author.name}")
        if post.author.headline:
            parts.append(f"직함: {post.author.headline}")

        parts.append(f"\n본문:\n{post.content[:2000]}")

        if post.reactions_count > 0:
            parts.append(f"\n반응: {post.reactions_count}개, 댓글: {post.comments_count}개")

        if post.external_links:
            parts.append(f"\n외부 링크: {', '.join(post.external_links[:5])}")

        if post.mentioned_profiles:
            parts.append(f"\n언급된 프로필: {', '.join(post.mentioned_profiles[:5])}")

        # Inject existing knowledge for novelty comparison
        if self.store:
            atoms_context = self.store.get_atoms_context(max_chars=4000)
            if atoms_context:
                parts.append(
                    f"\n\n--- 기존 지식 베이스 (참신성 판단 참고용) ---\n{atoms_context}"
                )

        return "\n".join(parts)

    async def batch_judge(self, posts: list[LinkedInPost]) -> list[LinkedInPost]:
        """Judge multiple posts sequentially."""
        results = []
        for post in posts:
            judged = await self.judge(post)
            results.append(judged)
        return results
