"""Weekly synthesizer - integrates atoms into a cohesive weekly knowledge map."""

from __future__ import annotations

import logging
from datetime import datetime

from src.config import get_settings
from src.agent.llm import call_claude
from src.knowledge.models import TokenUsage
from src.knowledge.store import KnowledgeStore
from src.knowledge.git_sync import GitSync

logger = logging.getLogger(__name__)

WEEKLY_PROMPT = """당신은 기업 AX 분야 전문 애널리스트입니다.
지난 한 주 동안 수집된 원자 노트들을 통합하여 주간 지식맵을 작성해주세요.

## 이번 주 원자 노트 목록
{atoms_summary}

## 기존 월간 맵 (컨텍스트)
{monthly_context}

# 주간 지식맵 — {week_label}

## 이번 주 핵심 시그널
- (3-5개: 새롭게 등장하거나 강화된 신호)

## 주제 클러스터
### [클러스터1]
- 관련 원자: `atom-id-1`, `atom-id-2`
- 통합 인사이트: ...
- 트렌드 방향: ...

## 지식 성숙도 변화
- 이번 주 새로 등장한 개념 (🌱 seedling):
- 반복 확인되어 강화된 개념 (🌿 budding→🌳 evergreen 후보):

## 고아 노트 (연결 필요)
{orphan_list}

## 다음 주 탐색 우선순위
- (이번 주 인사이트를 바탕으로 더 깊이 탐색할 주제)

## 기존 지식과의 차이/발전
- (지난주 대비 달라진 관점이나 새로운 연결)"""

MATURITY_UPGRADE_PROMPT = """아래 원자 노트들이 여러 번 확인된 개념입니다.
성숙도를 올려야 할 것들을 판단해주세요.

## 노트 목록
{atoms}

반드시 아래 JSON만 응답하세요:
{{
  "upgrades": [
    {{"id": "atom-id", "from": "seedling", "to": "budding", "reason": "한 줄 이유"}},
    {{"id": "atom-id", "from": "budding", "to": "evergreen", "reason": "한 줄 이유"}}
  ]
}}"""


class WeeklySynthesizer:
    """Produces weekly knowledge maps from accumulated atoms."""

    def __init__(self, store: KnowledgeStore, token_usage: TokenUsage | None = None) -> None:
        self.store = store
        self.token_usage = token_usage or TokenUsage()
        self.git = GitSync()

    async def run(self) -> dict:
        """Run the weekly synthesis cycle."""
        now = datetime.now()
        week_label = f"{now.year}-W{now.isocalendar()[1]:02d}"
        logger.info(f"Weekly synthesis started: {week_label}")
        results: dict = {}

        atoms = self.store.get_all_atoms()
        if not atoms:
            logger.info("No atoms yet — skipping weekly synthesis.")
            return {}

        # Build atoms summary for prompt
        atoms_summary_lines = []
        for a in atoms:
            emoji = {"seedling": "🌱", "budding": "🌿", "evergreen": "🌳"}.get(a["maturity"], "📄")
            related_str = f"(related: {a['related_count']})" if a["related_count"] else "(고아)"
            atoms_summary_lines.append(
                f"- {emoji} `{a['id']}` **{a['concept']}** {related_str} | 최종갱신: {a['updated']}"
            )
        atoms_summary = "\n".join(atoms_summary_lines)

        # Load latest monthly map for context
        monthly_dir = self.store.base_dir / "maps" / "monthly"
        monthly_context = ""
        if monthly_dir.exists():
            monthly_files = sorted(monthly_dir.glob("*.md"), reverse=True)
            if monthly_files:
                try:
                    import frontmatter
                    fm = frontmatter.load(str(monthly_files[0]))
                    monthly_context = fm.content[:2000]
                except Exception:
                    pass

        # Orphan atoms
        try:
            from src.knowledge.embedder import Embedder
            orphans = Embedder().get_orphan_atoms()
        except Exception:
            orphans = [a["id"] for a in atoms if a["related_count"] == 0]

        orphan_list = "\n".join(f"- `{o}`" for o in orphans[:10]) or "(없음)"

        prompt = WEEKLY_PROMPT.format(
            atoms_summary=atoms_summary,
            monthly_context=monthly_context or "(없음)",
            week_label=week_label,
            orphan_list=orphan_list,
        )

        response = await self._call_llm(prompt)
        if response:
            path = self.store.save_weekly_map(response, week_label=week_label)
            results["weekly_map"] = str(path)
            logger.info(f"Weekly map saved: {path.name}")

        # Maturity upgrades
        try:
            upgrade_results = await self._check_maturity_upgrades(atoms)
            results["maturity_upgrades"] = upgrade_results
        except Exception as e:
            logger.warning(f"Maturity upgrade check failed: {e}")

        # Refresh index
        self.store._refresh_index()

        # Git push
        try:
            await self.git.commit_and_push(results, post_count=0)
        except Exception as e:
            logger.error(f"Git sync error: {e}")

        logger.info(f"Weekly synthesis complete: {results}")
        return results

    async def _check_maturity_upgrades(self, atoms: list[dict]) -> list[dict]:
        """Check which atoms should be upgraded in maturity."""
        # Only consider atoms with multiple sources or high related count
        candidates = [
            a for a in atoms
            if a["source_count"] >= 2 or a["related_count"] >= 2
        ]
        if not candidates:
            return []

        # Load full content of candidates
        import frontmatter
        lines = []
        for a in candidates[:20]:
            atom_data = self.store.get_atom_by_id(a["id"])
            if atom_data:
                lines.append(
                    f"id: {a['id']}\n"
                    f"concept: {a['concept']}\n"
                    f"maturity: {a['maturity']}\n"
                    f"sources: {a['source_count']}개, related: {a['related_count']}개\n"
                    f"preview: {atom_data['content'][:300]}\n"
                )

        if not lines:
            return []

        prompt = MATURITY_UPGRADE_PROMPT.format(atoms="\n---\n".join(lines))
        response = await self._call_llm(prompt, max_tokens=1000)
        if not response:
            return []

        from src.agent.synthesizer import KnowledgeSynthesizer
        data = KnowledgeSynthesizer._extract_json(response)
        if not data:
            return []

        applied = []
        for upgrade in data.get("upgrades", []):
            atom_id = upgrade.get("id", "")
            new_maturity = upgrade.get("to", "")
            if atom_id and new_maturity in ("budding", "evergreen"):
                atom_data = self.store.get_atom_by_id(atom_id)
                if atom_data:
                    self.store.save_atom(
                        atom_id=atom_id,
                        concept=atom_data["metadata"].get("concept", ""),
                        body=f"_성숙도 업그레이드: {upgrade.get('from')} → {new_maturity}_\n_{upgrade.get('reason', '')}_",
                        maturity=new_maturity,
                        ttl_days=atom_data["metadata"].get("ttl_days", 365),
                    )
                    applied.append(upgrade)
                    logger.info(f"Maturity upgrade: {atom_id} → {new_maturity}")

        return applied

    async def _call_llm(self, prompt: str, max_tokens: int = 4000) -> str:
        return await call_claude(prompt)
