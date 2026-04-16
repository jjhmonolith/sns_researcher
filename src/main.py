"""Entry point - starts the crawler and dashboard concurrently."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import threading

import uvicorn

from src.config import ensure_dirs, get_settings, PID_FILE, HEARTBEAT_FILE, STATS_FILE
from src.agent.state import set_crawler, get_crawler


def setup_logging() -> None:
    """Configure logging for the application."""
    from src.config import LOGS_DIR

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(LOGS_DIR / "agent.log"), encoding="utf-8"),
        ],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("playwright").setLevel(logging.WARNING)


def cleanup_stale_process() -> None:
    """Kill any previously running agent (by PID file) and free the dashboard port."""
    logger = logging.getLogger(__name__)

    # Kill by PID file
    if PID_FILE.exists():
        try:
            old_pid = int(PID_FILE.read_text().strip())
            if old_pid != os.getpid():
                try:
                    os.kill(old_pid, signal.SIGTERM)
                    logger.info(f"Sent SIGTERM to old agent PID {old_pid}")
                    import time

                    time.sleep(2)
                    # Force kill if still alive
                    try:
                        os.kill(old_pid, signal.SIGKILL)
                        logger.info(f"Force killed old agent PID {old_pid}")
                    except ProcessLookupError:
                        pass  # Already dead
                except ProcessLookupError:
                    logger.info(f"Old PID {old_pid} already gone.")
        except Exception as e:
            logger.warning(f"Could not clean up old PID file: {e}")
        PID_FILE.unlink(missing_ok=True)

    # Kill anything holding the dashboard port
    settings = get_settings()
    port = settings.dashboard_port
    try:
        import subprocess

        result = subprocess.run(
            ["lsof", "-ti", f":{port}"], capture_output=True, text=True
        )
        for pid_str in result.stdout.strip().splitlines():
            pid = int(pid_str.strip())
            if pid != os.getpid():
                try:
                    os.kill(pid, signal.SIGKILL)
                    logger.info(f"Killed PID {pid} holding port {port}")
                except ProcessLookupError:
                    pass
    except Exception as e:
        logger.warning(f"Port cleanup error: {e}")


def write_pid() -> None:
    """Write current PID to PID file."""
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))


def cleanup_pid() -> None:
    """Remove PID file on clean shutdown."""
    PID_FILE.unlink(missing_ok=True)


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
    from src.agent.crawler import LinkedInCrawler

    crawler = LinkedInCrawler()
    set_crawler(crawler)

    loop = asyncio.get_event_loop()

    def signal_handler():
        logging.info("Shutdown signal received...")
        c = get_crawler()
        if c:
            c.request_stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    await crawler.run()


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
    logger.info(
        f"Models: fast={settings.model_fast}, powerful={settings.model_powerful}"
    )
    logger.info(
        f"Dashboard: http://{settings.dashboard_host}:{settings.dashboard_port}"
    )
    logger.info("=" * 60)

    # Clean up any stale agent and free the port
    cleanup_stale_process()

    # Write our PID
    write_pid()
    logger.info(f"PID: {os.getpid()} written to {PID_FILE}")

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
        cleanup_pid()
        logger.info("Agent shut down complete.")


if __name__ == "__main__":
    main()
