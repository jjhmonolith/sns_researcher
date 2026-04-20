"""Entry point - starts crawlers and dashboard concurrently."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import threading

import uvicorn

from src.config import ensure_dirs, get_settings
from src.agent.crawler import LinkedInCrawler


# Global crawler references for dashboard access
_crawlers: dict[str, object] = {}


def get_crawler(platform: str = "linkedin"):
    """Get a specific crawler by platform name."""
    return _crawlers.get(platform)


def get_crawlers() -> dict[str, object]:
    """Get all active crawlers."""
    return _crawlers


def setup_logging() -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("linkedin_researcher.log", encoding="utf-8"),
        ],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("playwright").setLevel(logging.WARNING)


def run_dashboard(host: str, port: int) -> None:
    """Run the FastAPI dashboard in a separate thread."""
    from src.dashboard.app import create_app

    app = create_app()
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    server.run()


async def run_crawlers() -> None:
    """Run all enabled crawlers concurrently."""
    logger = logging.getLogger(__name__)

    # Shared resources
    from src.knowledge.store import KnowledgeStore
    from src.knowledge.queue import ExplorationQueue
    from src.knowledge.followed_authors import FollowedAuthors
    from src.knowledge.models import TokenUsage

    store = KnowledgeStore()
    queue = ExplorationQueue()
    followed_authors = FollowedAuthors()
    token_usage = TokenUsage()

    # LinkedIn crawler (always enabled)
    linkedin_crawler = LinkedInCrawler()
    _crawlers["linkedin"] = linkedin_crawler

    # X crawler (enabled via ENABLE_X env var)
    x_crawler = None
    enable_x = os.environ.get("ENABLE_X", "").lower() in ("true", "1", "yes")
    if enable_x:
        from src.agent.x_crawler import XCrawler
        x_crawler = XCrawler(
            store=store,
            queue=queue,
            followed_authors=followed_authors,
            token_usage=token_usage,
        )
        _crawlers["x"] = x_crawler
        logger.info("X (Twitter) crawler enabled.")

    # Handle graceful shutdown
    loop = asyncio.get_event_loop()

    def signal_handler():
        logging.info("Shutdown signal received...")
        for name, crawler in _crawlers.items():
            if hasattr(crawler, "request_stop"):
                crawler.request_stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    # Run crawlers concurrently
    tasks = [linkedin_crawler.run()]
    if x_crawler:
        tasks.append(x_crawler.run())

    await asyncio.gather(*tasks, return_exceptions=True)


def main() -> None:
    """Main entry point."""
    setup_logging()
    ensure_dirs()

    settings = get_settings()
    logger = logging.getLogger(__name__)

    enable_x = os.environ.get("ENABLE_X", "").lower() in ("true", "1", "yes")

    logger.info("=" * 60)
    logger.info("AX Research Agent")
    logger.info("=" * 60)
    logger.info(f"Topic: 기업과 조직의 AX (AI Transformation)")
    logger.info(f"Platforms: LinkedIn" + (" + X" if enable_x else ""))
    logger.info(f"Keywords: {settings.keywords_list}")
    logger.info(f"Dashboard: http://{settings.dashboard_host}:{settings.dashboard_port}")
    logger.info("=" * 60)

    # Start dashboard in a background thread
    dashboard_thread = threading.Thread(
        target=run_dashboard,
        args=(settings.dashboard_host, settings.dashboard_port),
        daemon=True,
    )
    dashboard_thread.start()
    logger.info(f"Dashboard started at http://localhost:{settings.dashboard_port}")

    # Run crawlers in the main async loop
    try:
        asyncio.run(run_crawlers())
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        logger.info("Agent shut down complete.")


if __name__ == "__main__":
    main()
