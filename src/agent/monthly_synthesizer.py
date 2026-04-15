"""Monthly synthesizer - deep trend analysis, TTL review, knowledge map."""

from __future__ import annotations

import logging
from datetime import datetime

from openai import AsyncOpenAI

from src.config import get_settings
from src.knowledge.models import TokenUsage
from src.knowledge.store import KnowledgeStore
from src.knowledge.git_sync import GitSync

logger = logging.getLogger(__name__)

MONTHLY_PROMPT = """당신은 기업 AX 분야의 시니어 리서치 애널리스트입니다.
지난 한 달의 지식 베이스 전체를 분석하여 월간 지식맵을 작성하세요.

## 전체 원자 노트 목록
{atoms_summary}

## 이번 달 주간 맵 요약
{weekly_summaries}

# 월간 지식맵 — {month_label}

## 이달의 핵심 트렌드 (Top 5)
1. ...

## 지식 클러스터 맵
### 클러스터 A: [주제]
- 핵심 원자: ...
- 성숙도 분포: 🌱n 🌿n 🌳n
- 클러스터 인사이트: ...

## 시간적 변화 분석
- 이번 달 새로 등장한 개념:
- 이전 달 대비 강화된 개념:
- 약해지거나 사라진 신호:

## 만료 검토 필요 노트 (TTL 근접)
{expiring_atoms}

## 지식 격차 (Gap Analysis)
- 충분히 탐색되지 않은 영역:
- 서로 연결됐어야 하는데 고립된 개념:

## 다음 달 리서치 방향
- 우선 탐색 주제:
- 검증 필요 가설:

## 누적 인사이트 요약
- 한 달간 쌓인 가장 확실한 인사이트 (🌳 evergreen 수준)"""

TTL_REVIEW_PROMPT = """아래 만료된 또는 만료 임박 원자 노트들을 검토해주세요.
각 노트를 갱신해야 할지, 폐기할지, 유지할지 판단하세요.

## 노트 목록
{atoms}

반드시 아래 JSON만 응답하세요:
{{
  "reviews": [
    {{
      "id": "atom-id",
      "action": "renew 또는 archive 또는 keep",
      "reason": "한 줄 이유",
      "new_ttl_days": 숫자 (renew일 때만)
    }}
  ]
}}"""


class MonthlySynthesizer:
    """Produces monthly knowledge maps with trend analysis and TTL review."""

    def __init__(self, store: KnowledgeStore, token_usage: TokenUsage | None = None) -> None:
        settings = get_settings()
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.model_powerful
        self.store = store
        self.token_usage = token_usage or TokenUsage()
        self.git = GitSync()

    async def run(self) -> dict:
        """Run the monthly synthesis cycle."""
        now = datetime.now()
        month_label = now.strftime("%Y-%m")
        logger.info(f"Monthly synthesis started: {month_label}")
        results: dict = {}

        atoms = self.store.get_all_atoms()
        if not atoms:
            logger.info("No atoms yet — skipping monthly synthesis.")
            return {}

        # Build atoms summary
        atoms_summary_lines = []
        for a in atoms:
            emoji = {"seedling": "🌱", "budding": "🌿", "evergreen": "🌳"}.get(a["maturity"], "📄")
            atoms_summary_lines.append(
                f"- {emoji} `{a['id']}` {a['concept']} "
                f"(sources:{a['source_count']}, related:{a['related_count']}, updated:{a['updated']})"
            )
        atoms_summary = "\n".join(atoms_summary_lines)

        # Load weekly maps from this month
        weekly_dir = self.store.base_dir / "maps" / "weekly"
        weekly_summaries = ""
        if weekly_dir.exists():
            import frontmatter
            weekly_files = sorted(weekly_dir.glob(f"{now.year}-W*.md"), reverse=True)[:5]
            summaries = []
            for wf in weekly_files:
                try:
                    fm = frontmatter.load(str(wf))
                    summaries.append(f"### {fm.metadata.get('week', wf.stem)}\n{fm.content[:800]}")
                except Exception:
                    continue
            weekly_summaries = "\n\n---\n\n".join(summaries)

        # Check expiring atoms
        expired_ids = self.store.get_expired_atoms()
        expiring_lines = []
        for atom_id in expired_ids[:10]:
            a = next((x for x in atoms if x["id"] == atom_id), None)
            if a:
                expiring_lines.append(f"- `{atom_id}` {a['concept']} (updated: {a['updated']}, ttl: {a['ttl_days']}d)")
        expiring_str = "\n".join(expiring_lines) or "(없음)"

        prompt = MONTHLY_PROMPT.format(
            atoms_summary=atoms_summary,
            weekly_summaries=weekly_summaries or "(주간 맵 없음)",
            month_label=month_label,
            expiring_atoms=expiring_str,
        )

        response = await self._call_llm(prompt, max_tokens=5000)
        if response:
            path = self.store.save_monthly_map(response, month_label=month_label)
            results["monthly_map"] = str(path)
            logger.info(f"Monthly map saved: {path.name}")

        # TTL review for expired atoms
        if expired_ids:
            try:
                ttl_results = await self._ttl_review(expired_ids, atoms)
                results["ttl_reviews"] = ttl_results
            except Exception as e:
                logger.warning(f"TTL review failed: {e}")

        # Refresh index & push
        self.store._refresh_index()
        try:
            await self.git.commit_and_push(results, post_count=0)
        except Exception as e:
            logger.error(f"Git sync error: {e}")

        logger.info(f"Monthly synthesis complete: {results}")
        return results

    async def _ttl_review(self, expired_ids: list[str], all_atoms: list[dict]) -> list[dict]:
        """Review expired atoms and decide renew/archive/keep."""
        lines = []
        for atom_id in expired_ids[:15]:
            a = next((x for x in all_atoms if x["id"] == atom_id), None)
            atom_data = self.store.get_atom_by_id(atom_id)
            if a and atom_data:
                lines.append(
                    f"id: {atom_id}\n"
                    f"concept: {a['concept']}\n"
                    f"maturity: {a['maturity']}\n"
                    f"ttl_days: {a['ttl_days']}\n"
                    f"updated: {a['updated']}\n"
                    f"preview: {atom_data['content'][:400]}\n"
                )

        if not lines:
            return []

        prompt = TTL_REVIEW_PROMPT.format(atoms="\n---\n".join(lines))
        response = await self._call_llm(prompt, max_tokens=1000)
        if not response:
            return []

        from src.agent.synthesizer import KnowledgeSynthesizer
        data = KnowledgeSynthesizer._extract_json(response)
        if not data:
            return []

        applied = []
        archive_dir = self.store.base_dir / "archive"
        archive_dir.mkdir(exist_ok=True)

        for review in data.get("reviews", []):
            atom_id = review.get("id", "")
            action = review.get("action", "keep")

            if action == "archive":
                src = self.store.base_dir / "atoms" / f"{atom_id}.md"
                dst = archive_dir / f"{atom_id}.md"
                if src.exists():
                    src.rename(dst)
                    logger.info(f"Archived expired atom: {atom_id}")

            elif action == "renew":
                atom_data = self.store.get_atom_by_id(atom_id)
                a = next((x for x in all_atoms if x["id"] == atom_id), None)
                if atom_data and a:
                    new_ttl = int(review.get("new_ttl_days", a["ttl_days"]))
                    self.store.save_atom(
                        atom_id=atom_id,
                        concept=atom_data["metadata"].get("concept", ""),
                        body=f"_TTL 갱신 ({review.get('reason', '')})_",
                        maturity=a["maturity"],
                        ttl_days=new_ttl,
                    )
                    logger.info(f"Renewed atom TTL: {atom_id} → {new_ttl}d")

            applied.append(review)

        return applied

    async def _call_llm(self, prompt: str, max_tokens: int = 4000) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_completion_tokens=max_tokens,
            )
            if response.usage:
                self.token_usage.powerful_input_tokens += response.usage.prompt_tokens
                self.token_usage.powerful_output_tokens += response.usage.completion_tokens
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""
