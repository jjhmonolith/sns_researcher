"""Relevance judgment using GPT-5.4 nano - determines if a post is related to the topic."""

from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI

from src.config import get_settings
from src.knowledge.models import LinkedInPost, TokenUsage

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """당신은 LinkedIn 포스트 관련성 분석 전문가입니다.
주어진 포스트가 "기업과 조직의 AX (AI Transformation)" 주제와 관련이 있는지 판단합니다.

관련 주제 범위:
- 기업의 AI 도입/전환 전략 (AX, AI Transformation, Digital Transformation)
- AI를 활용한 업무 프로세스 혁신
- 기업용 AI 솔루션, 도구, 플랫폼
- 조직 내 AI 역량 구축, AI 인재 양성
- AI 거버넌스, 윤리, 규제가 기업에 미치는 영향
- AI로 인한 산업/비즈니스 모델 변화
- 기업 경영진의 AI 비전과 전략
- AI 스타트업의 B2B 솔루션
- 생성형 AI의 기업 활용 사례
- AI Agent를 활용한 업무 자동화

관련 없는 주제:
- 순수 기술 논문/연구 (기업 적용과 무관한)
- 개인 일상, 자기계발 (AI와 무관한)
- 채용 공고 단순 게시
- 제품 광고/마케팅 (AI와 무관한)

응답은 반드시 아래 JSON 형식으로만 답하세요:
{
  "relevance_score": 0-100 사이 정수,
  "is_relevant": true/false,
  "topics": ["관련 토픽1", "토픽2"],
  "summary": "1-2문장 요약",
  "should_follow_links": true/false,
  "follow_targets": ["따라가볼 URL이나 프로필"],
  "reason": "판단 근거 한 줄"
}"""


class RelevanceJudge:
    """Uses GPT-5.4 nano to judge post relevance to AX topic."""

    def __init__(self, token_usage: TokenUsage | None = None) -> None:
        settings = get_settings()
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.model_fast
        self.threshold = settings.relevance_threshold
        self.token_usage = token_usage or TokenUsage()

    async def judge(self, post: LinkedInPost) -> LinkedInPost:
        """Analyze a post's relevance and update it with results.
        
        Args:
            post: The LinkedIn post to analyze.
            
        Returns:
            The same post object with relevance fields populated.
        """
        if not post.content or len(post.content.strip()) < 20:
            post.relevance_score = 0
            post.is_relevant = False
            post.summary = "Content too short to analyze"
            return post

        # Build the user message with post info
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

            # Track token usage
            if response.usage:
                self.token_usage.nano_input_tokens += response.usage.prompt_tokens
                self.token_usage.nano_output_tokens += response.usage.completion_tokens

            # Parse response
            content = response.choices[0].message.content
            if content:
                result = json.loads(content)
                post.relevance_score = min(100, max(0, int(result.get("relevance_score", 0))))
                post.is_relevant = post.relevance_score >= self.threshold
                post.relevance_topics = result.get("topics", [])
                post.summary = result.get("summary", "")
                post.should_follow_links = result.get("should_follow_links", False)
                post.follow_targets = result.get("follow_targets", [])

                logger.info(
                    f"Relevance: {post.relevance_score}/100 "
                    f"{'[RELEVANT]' if post.is_relevant else '[skip]'} "
                    f"- {post.summary[:60]}..."
                )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse relevance response: {e}")
            post.relevance_score = 0
            post.is_relevant = False
        except Exception as e:
            logger.error(f"Relevance judgment error: {e}")
            post.relevance_score = 0
            post.is_relevant = False

        return post

    def _build_prompt(self, post: LinkedInPost) -> str:
        """Build the user prompt from post data."""
        parts = []

        if post.author.name:
            parts.append(f"작성자: {post.author.name}")
        if post.author.headline:
            parts.append(f"직함: {post.author.headline}")

        parts.append(f"\n본문:\n{post.content[:2000]}")  # Limit content length

        if post.reactions_count > 0:
            parts.append(f"\n반응: {post.reactions_count}개, 댓글: {post.comments_count}개")

        if post.external_links:
            parts.append(f"\n외부 링크: {', '.join(post.external_links[:5])}")

        if post.mentioned_profiles:
            parts.append(f"\n언급된 프로필: {', '.join(post.mentioned_profiles[:5])}")

        return "\n".join(parts)

    async def batch_judge(self, posts: list[LinkedInPost]) -> list[LinkedInPost]:
        """Judge multiple posts sequentially (to respect rate limits).
        
        Args:
            posts: List of posts to analyze.
            
        Returns:
            The same posts with relevance fields populated.
        """
        results = []
        for post in posts:
            judged = await self.judge(post)
            results.append(judged)
        return results
