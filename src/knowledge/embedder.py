"""Embedder - generates embeddings for atom notes and auto-links related notes."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import frontmatter
from openai import AsyncOpenAI

from src.config import ROOT_DIR, get_settings

logger = logging.getLogger(__name__)

EMBEDDINGS_DIR = ROOT_DIR / ".embeddings"
ATOMS_DIR = ROOT_DIR / "knowledge" / "atoms"
SIMILARITY_THRESHOLD = 0.75
EMBEDDING_MODEL = "text-embedding-3-small"


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class Embedder:
    """Manages embeddings for atom notes and resolves related links."""

    def __init__(self) -> None:
        settings = get_settings()
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    def _embed_path(self, atom_id: str) -> Path:
        return EMBEDDINGS_DIR / f"{atom_id}.json"

    def _load_embedding(self, atom_id: str) -> list[float] | None:
        path = self._embed_path(atom_id)
        if path.exists():
            return json.loads(path.read_text())
        return None

    def _save_embedding(self, atom_id: str, vector: list[float]) -> None:
        self._embed_path(atom_id).write_text(json.dumps(vector))

    async def embed_text(self, text: str) -> list[float]:
        """Get embedding vector for a text string."""
        response = await self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text[:8000],  # token limit safety
        )
        return response.data[0].embedding

    async def embed_atom(self, atom_id: str, text: str) -> list[float]:
        """Get or create embedding for an atom note."""
        cached = self._load_embedding(atom_id)
        if cached:
            return cached
        vector = await self.embed_text(text)
        self._save_embedding(atom_id, vector)
        return vector

    async def find_related(self, atom_id: str, concept: str, body: str, top_k: int = 5) -> list[str]:
        """Find related atom notes by embedding similarity.

        Args:
            atom_id: ID of the source atom (to exclude self).
            concept: Concept title of the atom.
            body: Body text of the atom.
            top_k: Max number of related notes to return.

        Returns:
            List of related atom IDs sorted by similarity (highest first).
        """
        if not ATOMS_DIR.exists():
            return []

        all_atoms = list(ATOMS_DIR.glob("*.md"))
        if not all_atoms:
            return []

        # Embed the source note
        source_text = f"{concept}\n\n{body[:2000]}"
        source_vec = await self.embed_atom(atom_id, source_text)

        scores: list[tuple[float, str]] = []
        for atom_file in all_atoms:
            candidate_id = atom_file.stem
            if candidate_id == atom_id:
                continue

            # Load or compute candidate embedding
            candidate_vec = self._load_embedding(candidate_id)
            if candidate_vec is None:
                try:
                    fm = frontmatter.load(str(atom_file))
                    candidate_text = f"{fm.metadata.get('concept', '')}\n\n{fm.content[:2000]}"
                    candidate_vec = await self.embed_atom(candidate_id, candidate_text)
                except Exception as e:
                    logger.debug(f"Could not embed {candidate_id}: {e}")
                    continue

            sim = _cosine_similarity(source_vec, candidate_vec)
            if sim >= SIMILARITY_THRESHOLD:
                scores.append((sim, candidate_id))

        scores.sort(reverse=True)
        return [atom_id for _, atom_id in scores[:top_k]]

    async def refresh_all_embeddings(self) -> int:
        """Ensure all existing atoms have embeddings cached. Returns count updated."""
        count = 0
        if not ATOMS_DIR.exists():
            return 0
        for atom_file in ATOMS_DIR.glob("*.md"):
            atom_id = atom_file.stem
            if not self._embed_path(atom_id).exists():
                try:
                    fm = frontmatter.load(str(atom_file))
                    text = f"{fm.metadata.get('concept', '')}\n\n{fm.content[:2000]}"
                    await self.embed_atom(atom_id, text)
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to embed {atom_id}: {e}")
        return count

    def get_orphan_atoms(self) -> list[str]:
        """Return atom IDs that have no related links."""
        orphans = []
        if not ATOMS_DIR.exists():
            return orphans
        for atom_file in ATOMS_DIR.glob("*.md"):
            try:
                fm = frontmatter.load(str(atom_file))
                related = fm.metadata.get("related", [])
                if not related:
                    orphans.append(atom_file.stem)
            except Exception:
                continue
        return orphans
