"""One-time migration: merge duplicate topics/ files into atoms/ format."""

from __future__ import annotations

import asyncio
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import frontmatter
from openai import AsyncOpenAI

from src.config import ROOT_DIR, get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TOPICS_DIR = ROOT_DIR / "knowledge" / "insights" / "topics"
ATOMS_DIR = ROOT_DIR / "knowledge" / "atoms"
ARCHIVE_DIR = ROOT_DIR / "knowledge" / "archive" / "legacy_topics"
PEOPLE_DIR = ROOT_DIR / "knowledge" / "insights" / "people"

CONSOLIDATION_PROMPT = """아래 여러 개의 주제 문서들은 동일하거나 유사한 주제를 다루는 중복/파편화된 파일들입니다.
이를 하나의 통합된 원자 노트(Atomic Note)로 합쳐주세요.

## 입력 문서들
{documents}

## 요청
위 문서들의 내용을 통합하여 아래 JSON 형식으로 응답해주세요:

{{
  "concept": "핵심 개념을 명사구로 한 줄 (예: '기업 AX 도입 전략과 변화관리')",
  "slug": "영문-소문자-하이픈-구분 (예: 'enterprise-ax-strategy-change-mgmt')",
  "maturity": "seedling 또는 budding 또는 evergreen",
  "ttl_days": 숫자 (AI트렌드=180, 방법론=365, 원칙=730),
  "body": "통합된 마크다운 본문 (## 섹션 구조 유지, 중복 제거, 출처 이력 포함)"
}}

원칙:
- 사실과 의견을 구분하여 유지
- 중복 내용은 하나로 통합
- 출처 이력(날짜, 원본 URL)은 ## 출처 이력 섹션에 모두 보존
- body는 한국어로 작성"""


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text.strip("-") or "general"


def _cluster_topics(files: list[Path]) -> list[list[Path]]:
    """Simple keyword-based clustering of topic files into related groups."""
    # Define canonical cluster keywords
    clusters: dict[str, list[str]] = {
        "ax-strategy": ["도입전환_전략", "ax", "transformation", "digital"],
        "capability-building": ["역량_구축", "인재_양성", "교육", "워크플로우"],
        "process-innovation": ["프로세스_혁신", "업무_혁신", "자동화"],
        "governance": ["거버넌스", "윤리", "규제", "락인", "소유권"],
        "generative-ai-usage": ["생성형_ai", "활용_사례", "활용영향"],
        "agent-automation": ["agent", "에이전트", "자동화"],
        "talent-org": ["인재역량", "창업", "시대_인재"],
        "b2b-solutions": ["스타트업", "b2b", "솔루션"],
        "industry-change": ["산업", "비즈니스_모델"],
        "education-ecosystem": ["교육에코", "에코시스템"],
    }

    assigned: dict[str, str] = {}  # filename -> cluster_key
    for f in files:
        stem = f.stem.lower()
        best = "misc"
        for cluster_key, keywords in clusters.items():
            if any(kw in stem for kw in keywords):
                best = cluster_key
                break
        assigned[f.name] = best

    # Group by cluster
    groups: dict[str, list[Path]] = {}
    for f in files:
        key = assigned[f.name]
        groups.setdefault(key, []).append(f)

    return list(groups.values())


async def consolidate_group(client: AsyncOpenAI, model: str, group: list[Path]) -> dict | None:
    """Call GPT to consolidate a group of topic files into one atom."""
    docs_parts = []
    for i, f in enumerate(group, 1):
        try:
            fm = frontmatter.load(str(f))
            docs_parts.append(f"### 문서 {i}: {f.name}\n\n{fm.content[:3000]}")
        except Exception as e:
            logger.warning(f"Could not read {f}: {e}")

    if not docs_parts:
        return None

    prompt = CONSOLIDATION_PROMPT.format(documents="\n\n---\n\n".join(docs_parts))

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_completion_tokens=4000,
            response_format={"type": "json_object"},
        )
        return __import__("json").loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Consolidation failed: {e}")
        return None


async def main() -> None:
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    ATOMS_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    topic_files = list(TOPICS_DIR.glob("*.md"))
    if not topic_files:
        logger.info("No topic files found. Nothing to migrate.")
        return

    logger.info(f"Found {len(topic_files)} topic files to migrate.")

    # Cluster into related groups
    groups = _cluster_topics(topic_files)
    logger.info(f"Grouped into {len(groups)} clusters.")

    today = datetime.now().strftime("%Y%m%d")
    migrated = 0

    for group in groups:
        if not group:
            continue

        group_names = [f.name for f in group]
        logger.info(f"Consolidating {len(group)} files: {group_names}")

        if len(group) == 1:
            # Single file — just convert format
            f = group[0]
            try:
                fm = frontmatter.load(str(f))
                result = {
                    "concept": fm.metadata.get("title", f.stem.replace("_", " ")),
                    "slug": _slugify(fm.metadata.get("title", f.stem)),
                    "maturity": "budding",
                    "ttl_days": 365,
                    "body": fm.content,
                }
            except Exception as e:
                logger.warning(f"Failed to read {f}: {e}")
                continue
        else:
            result = await consolidate_group(client, settings.model_powerful, group)
            if not result:
                continue

        # Write atom file
        atom_id = f"{today}-{_slugify(result.get('slug', 'general'))}"
        atom_path = ATOMS_DIR / f"{atom_id}.md"

        # Avoid overwriting if already exists
        if atom_path.exists():
            atom_id = f"{atom_id}-2"
            atom_path = ATOMS_DIR / f"{atom_id}.md"

        meta = {
            "id": atom_id,
            "concept": result.get("concept", ""),
            "maturity": result.get("maturity", "budding"),
            "ttl_days": result.get("ttl_days", 365),
            "created": today,
            "updated": today,
            "sources": [],
            "related": [],
            "migrated_from": [f.name for f in group],
        }
        fm_post = frontmatter.Post(result.get("body", ""), **meta)
        atom_path.write_text(frontmatter.dumps(fm_post), encoding="utf-8")
        logger.info(f"  → Created atom: {atom_path.name}")
        migrated += 1

        # Archive originals
        for f in group:
            dest = ARCHIVE_DIR / f.name
            f.rename(dest)
            logger.info(f"  → Archived: {f.name}")

    logger.info(f"Migration complete: {migrated} atoms created, {len(topic_files)} files archived.")


if __name__ == "__main__":
    asyncio.run(main())
