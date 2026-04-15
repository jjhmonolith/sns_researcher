"""Git sync - automatically commit and push knowledge base after each synthesis."""

from __future__ import annotations

import asyncio
import logging
import subprocess
from datetime import datetime
from pathlib import Path

from src.config import ROOT_DIR

logger = logging.getLogger(__name__)


class GitSync:
    """Handles automatic git commit and push after synthesis cycles."""

    def __init__(self, repo_dir: Path = ROOT_DIR) -> None:
        self.repo_dir = repo_dir

    def _run(self, args: list[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run a git command synchronously."""
        return subprocess.run(
            ["git"] + args,
            cwd=self.repo_dir,
            capture_output=True,
            text=True,
            check=check,
        )

    async def commit_and_push(self, synthesis_results: dict, post_count: int) -> bool:
        """Stage knowledge base changes and push to remote.

        Args:
            synthesis_results: Dict returned by KnowledgeSynthesizer.synthesize()
            post_count: Number of posts included in this synthesis.

        Returns:
            True if push succeeded, False otherwise.
        """
        try:
            # Run in thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                None, self._sync, synthesis_results, post_count
            )
            return success
        except Exception as e:
            logger.error(f"Git sync failed: {e}")
            return False

    def _sync(self, synthesis_results: dict, post_count: int) -> bool:
        """Blocking git sync logic (runs in executor)."""
        try:
            # 1. Stage knowledge base (raw posts + insights only, not secrets)
            self._run(["add", "knowledge/"])
            self._run(["add", "src/", "pyproject.toml", ".gitignore", ".env.example"], check=False)

            # 2. Check if there's anything to commit
            status = self._run(["status", "--porcelain"])
            if not status.stdout.strip():
                logger.info("Git sync: nothing to commit.")
                return True

            # 3. Build commit message
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            digest_count = 1 if synthesis_results.get("digest") else 0
            topic_count = len(synthesis_results.get("topics", []))
            people_count = len(synthesis_results.get("people", []))

            msg = (
                f"[auto] synthesis {now}\n\n"
                f"- Posts analyzed: {post_count}\n"
                f"- Digests: {digest_count}\n"
                f"- Topics updated: {topic_count}\n"
                f"- People profiles: {people_count}\n"
            )

            # List changed files
            changed_files = [
                line.strip() for line in status.stdout.strip().splitlines()
            ]
            if changed_files:
                msg += "\nChanged files:\n"
                msg += "\n".join(f"  {f}" for f in changed_files[:20])
                if len(changed_files) > 20:
                    msg += f"\n  ... and {len(changed_files) - 20} more"

            # 4. Commit
            self._run(["commit", "-m", msg])
            logger.info(f"Git committed: {post_count} posts, {topic_count} topics, {people_count} people")

            # 5. Push (with retry on failure)
            for attempt in range(3):
                try:
                    result = self._run(["push", "origin", "main"])
                    logger.info(f"Git pushed to origin/main successfully.")
                    return True
                except subprocess.CalledProcessError as e:
                    # Try to set upstream on first push
                    if "has no upstream" in e.stderr or "set-upstream" in e.stderr:
                        self._run(["push", "--set-upstream", "origin", "main"])
                        logger.info("Git: set upstream and pushed.")
                        return True
                    # Pull & rebase if remote has diverged
                    if attempt < 2 and ("rejected" in e.stderr or "fetch first" in e.stderr):
                        logger.warning(f"Push rejected, pulling first (attempt {attempt+1})...")
                        self._run(["pull", "--rebase", "origin", "main"], check=False)
                        continue
                    logger.error(f"Git push failed: {e.stderr}")
                    return False

            return False

        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {e.cmd} -> {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Git sync error: {e}")
            return False

    def ensure_git_config(self) -> None:
        """Ensure git user config is set (required for commits)."""
        try:
            name = self._run(["config", "user.name"], check=False)
            if not name.stdout.strip():
                self._run(["config", "user.name", "LinkedIn Researcher Agent"])
                self._run(["config", "user.email", "agent@linkedin-researcher.local"])
                logger.info("Git user config set.")
        except Exception as e:
            logger.warning(f"Could not set git config: {e}")

    def is_git_repo(self) -> bool:
        """Check if the directory is a git repository."""
        try:
            result = self._run(["rev-parse", "--git-dir"], check=False)
            return result.returncode == 0
        except Exception:
            return False
