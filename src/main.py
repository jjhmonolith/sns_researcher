"""Entry point — multi-session research agent with dashboard."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import signal
import sys
import threading

import uvicorn

from src.config import ensure_dirs, get_settings


# Global orchestrator for dashboard access
_orchestrator = None


def get_orchestrator():
    return _orchestrator


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("researcher.log", encoding="utf-8"),
        ],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("playwright").setLevel(logging.WARNING)


def run_dashboard(host: str, port: int) -> None:
    from src.dashboard.app import create_app
    app = create_app()
    config = uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    server.run()


class SessionOrchestrator:
    """Manages multiple research sessions, each with independent crawlers."""

    def __init__(self) -> None:
        from src.session import SessionManager
        self.manager = SessionManager()
        self._running: dict[str, dict] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set the main event loop reference (called from the async entry point)."""
        self._loop = loop

    def start_session_threadsafe(self, session_id: str) -> bool:
        """Start a session from any thread (e.g., dashboard thread)."""
        if not self._loop:
            return False
        future = asyncio.run_coroutine_threadsafe(
            self._start_session(session_id), self._loop
        )
        try:
            return future.result(timeout=10)
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to start session: {e}")
            return False

    def stop_session_threadsafe(self, session_id: str) -> bool:
        """Stop a session from any thread."""
        if not self._loop:
            return False
        # Request stop immediately (thread-safe flag set)
        entry = self._running.get(session_id)
        if not entry:
            return False
        for crawler in entry["crawlers"].values():
            if hasattr(crawler, "request_stop"):
                crawler.request_stop()
        self.manager.update_session(session_id, status="stopped")
        return True

    async def _start_session(self, session_id: str) -> bool:
        """Start crawlers for a session (must run in main loop)."""
        if session_id in self._running:
            return False

        config = self.manager.get_session(session_id)
        if not config:
            return False

        config.ensure_dirs()
        logger = logging.getLogger(__name__)
        logger.info(f"Starting session: {config.name} ({session_id})")

        from src.knowledge.store import KnowledgeStore
        from src.knowledge.queue import ExplorationQueue
        from src.knowledge.followed_authors import FollowedAuthors
        from src.knowledge.models import TokenUsage

        store = KnowledgeStore(base_dir=config.knowledge_dir)
        queue = ExplorationQueue(file_path=config.queue_path)
        followed_authors = FollowedAuthors(file_path=config.followed_authors_path)
        token_usage = TokenUsage()

        crawlers = {}
        tasks = []

        if "linkedin" in config.platforms:
            from src.agent.crawler import LinkedInCrawler
            li = LinkedInCrawler(session_config=config)
            crawlers["linkedin"] = li
            tasks.append(asyncio.create_task(li.run(), name=f"{session_id}-linkedin"))

        if "x" in config.platforms:
            from src.agent.x_crawler import XCrawler
            x = XCrawler(
                store=store, queue=queue,
                followed_authors=followed_authors,
                token_usage=token_usage,
                session_config=config,
            )
            crawlers["x"] = x
            tasks.append(asyncio.create_task(x.run(), name=f"{session_id}-x"))

        if "facebook" in config.platforms:
            from src.agent.fb_crawler import FBCrawler
            fb = FBCrawler(
                store=store, queue=queue,
                followed_authors=followed_authors,
                token_usage=token_usage,
                session_config=config,
            )
            crawlers["facebook"] = fb
            tasks.append(asyncio.create_task(fb.run(), name=f"{session_id}-facebook"))

        self._running[session_id] = {"crawlers": crawlers, "tasks": tasks}
        self.manager.update_session(session_id, status="running")
        return True

    def get_session_crawlers(self, session_id: str) -> dict:
        entry = self._running.get(session_id)
        return entry["crawlers"] if entry else {}

    def is_session_running(self, session_id: str) -> bool:
        return session_id in self._running

    def get_all_running_ids(self) -> list[str]:
        return list(self._running.keys())


async def run_orchestrator() -> None:
    """Main async entry — start orchestrator and auto-resume sessions."""
    global _orchestrator
    _orchestrator = SessionOrchestrator()
    _orchestrator.set_loop(asyncio.get_event_loop())

    logger = logging.getLogger(__name__)
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Shutdown signal received...")
        for session_id in list(_orchestrator._running.keys()):
            entry = _orchestrator._running[session_id]
            for crawler in entry["crawlers"].values():
                if hasattr(crawler, "request_stop"):
                    crawler.request_stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    # Auto-start sessions that were previously running
    sessions = _orchestrator.manager.list_sessions()
    for s in sessions:
        if s.status == "running":
            logger.info(f"Auto-resuming session: {s.name} ({s.id})")
            await _orchestrator._start_session(s.id)

    if not _orchestrator._running:
        logger.info("No active sessions. Use the dashboard to create and start sessions.")

    # Keep running — clean up completed tasks
    while True:
        await asyncio.sleep(5)
        for session_id in list(_orchestrator._running.keys()):
            entry = _orchestrator._running[session_id]
            all_done = all(t.done() for t in entry["tasks"])
            if all_done:
                logger.info(f"Session {session_id} tasks completed.")
                del _orchestrator._running[session_id]
                _orchestrator.manager.update_session(session_id, status="stopped")


def main() -> None:
    setup_logging()
    ensure_dirs()

    settings = get_settings()
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Research Agent — Multi-Session Service")
    logger.info("=" * 60)
    logger.info(f"Dashboard: http://{settings.dashboard_host}:{settings.dashboard_port}")
    logger.info("=" * 60)

    dashboard_thread = threading.Thread(
        target=run_dashboard,
        args=(settings.dashboard_host, settings.dashboard_port),
        daemon=True,
    )
    dashboard_thread.start()
    logger.info(f"Dashboard started at http://localhost:{settings.dashboard_port}")

    try:
        asyncio.run(run_orchestrator())
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        logger.info("Agent shut down complete.")


if __name__ == "__main__":
    main()
