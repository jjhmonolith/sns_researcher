"""LLM abstraction — calls Claude Code CLI for synthesis tasks."""

from __future__ import annotations

import asyncio
import logging
import shutil

logger = logging.getLogger(__name__)

_CLAUDE_BIN: str | None = None


def _find_claude_bin() -> str | None:
    """Find the Claude Code CLI binary."""
    global _CLAUDE_BIN
    if _CLAUDE_BIN:
        return _CLAUDE_BIN

    found = shutil.which("claude")
    if found:
        _CLAUDE_BIN = found
        return _CLAUDE_BIN

    import glob
    patterns = [
        "/Users/*/Library/Application Support/Claude/claude-code/*/claude.app/Contents/MacOS/claude",
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern), reverse=True)
        if matches:
            _CLAUDE_BIN = matches[0]
            return _CLAUDE_BIN

    return None


async def call_claude(
    prompt: str,
    system_prompt: str = "",
    timeout_seconds: int = 180,
) -> str:
    """Call Claude Code CLI with prompt via stdin pipe."""
    claude_bin = _find_claude_bin()
    if not claude_bin:
        logger.error("Claude Code CLI not found.")
        return ""

    full_prompt = prompt
    if system_prompt:
        full_prompt = f"{system_prompt}\n\n---\n\n{prompt}"

    try:
        # Pipe prompt via stdin using shell echo
        proc = await asyncio.create_subprocess_shell(
            f'echo {_shell_quote(full_prompt)} | "{claude_bin}" --output-format text --max-turns 1',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout_seconds,
        )

        if proc.returncode != 0:
            err_msg = stderr.decode("utf-8", errors="replace")[:500]
            stdout_msg = stdout.decode("utf-8", errors="replace")[:200]
            logger.error(f"Claude CLI exit {proc.returncode}. stderr: {err_msg}. stdout: {stdout_msg}")
            return ""

        result = stdout.decode("utf-8", errors="replace").strip()
        logger.info(f"Claude CLI responded ({len(result)} chars)")
        return result

    except asyncio.TimeoutError:
        logger.error(f"Claude CLI timed out after {timeout_seconds}s")
        try:
            proc.kill()
        except Exception:
            pass
        return ""
    except Exception as e:
        logger.error(f"Claude CLI call failed: {e}")
        return ""


def _shell_quote(s: str) -> str:
    """Quote a string for safe shell use with $'...' syntax."""
    import shlex
    return shlex.quote(s)
