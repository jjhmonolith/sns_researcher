"""Entry point - starts the crawler and dashboard concurrently."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import threading

import uvicorn

from src.config import ensure_dirs, get_settings
from src.agent.crawler import LinkedInCrawler


# Global crawler reference for dashboard access
_crawler: LinkedInCrawler | None = None


def get_crawler() -> LinkedInCrawler | None:
    return _crawler


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
    # Reduce noise from libraries
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


async def run_crawler() -> None:
    """Run the main crawler loop."""
    global _crawler
    _crawler = LinkedInCrawler()

    # Handle graceful shutdown
    loop = asyncio.get_event_loop()

    def signal_handler():
        logging.info("Shutdown signal received...")
        if _crawler:
            _crawler.request_stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    await _crawler.run()


def main() -> None:
    """Main entry point."""
    setup_logging()
    ensure_dirs()

    settings = get_settings()
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("LinkedIn AX Research Agent")
    logger.info("=" * 60)
    logger.info(f"Topic: 기업과 조직의 AX (AI Transformation)")
    logger.info(f"Keywords: {settings.keywords_list}")
    logger.info(f"Models: fast={settings.model_fast}, powerful={settings.model_powerful}")
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

    # Run the crawler in the main async loop
    try:
        asyncio.run(run_crawler())
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        logger.info("Agent shut down complete.")


if __name__ == "__main__":
    main()
