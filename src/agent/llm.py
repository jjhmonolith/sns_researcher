"""LLM abstraction — calls Claude Code CLI for synthesis tasks."""

from __future__ import annotations

import asyncio
import logging
import shutil

logger = logging.getLogger(__name__)

# Claude Code CLI path discovery
_CLAUDE_BIN: str | None = None


def _find_claude_bin() -> str | None:
    """Find the Claude Code CLI binary."""
    global _CLAUDE_BIN
    if _CLAUDE_BIN:
        return _CLAUDE_BIN

    # 1. Check PATH
    found = shutil.which("claude")
    if found:
        _CLAUDE_BIN = found
        return _CLAUDE_BIN

    # 2. Check known macOS install location
    import glob
    patterns = [
        "/Users/*/Library/Application Support/Claude/claude-code/*/claude.app/Contents/MacOS/claude",
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern), reverse=True)  # newest version first
        if matches:
            _CLAUDE_BIN = matches[0]
            return _CLAUDE_BIN

    return None


async def call_claude(
    prompt: str,
    system_prompt: str = "",
    max_turns: int = 1,
    timeout_seconds: int = 120,
) -> str:
    """Call Claude Code CLI with a prompt and return the text response.

    Args:
        prompt: The user prompt to send.
        system_prompt: Optional system prompt (prepended to user prompt).
        max_turns: Max agentic turns (default 1 = single response).
        timeout_seconds: Timeout for the CLI process.

    Returns:
        The text response from Claude, or empty string on failure.
    """
    claude_bin = _find_claude_bin()
    if not claude_bin:
        logger.error("Claude Code CLI not found. Install with: npm install -g @anthropic-ai/claude-code")
        return ""

    # Combine system + user prompt
    full_prompt = prompt
    if system_prompt:
        full_prompt = f"{system_prompt}\n\n---\n\n{prompt}"

    try:
        proc = await asyncio.create_subprocess_exec(
            claude_bin,
            "-p", full_prompt,
            "--output-format", "text",
            "--max-turns", str(max_turns),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout_seconds,
        )

        if proc.returncode != 0:
            err_msg = stderr.decode("utf-8", errors="replace")[:200]
            logger.error(f"Claude CLI exited with code {proc.returncode}: {err_msg}")
            return ""

        result = stdout.decode("utf-8", errors="replace").strip()
        logger.info(f"Claude CLI responded ({len(result)} chars)")
        return result

    except asyncio.TimeoutError:
        logger.error(f"Claude CLI timed out after {timeout_seconds}s")
        if proc:
            proc.kill()
        return ""
    except Exception as e:
        logger.error(f"Claude CLI call failed: {e}")
        return ""
