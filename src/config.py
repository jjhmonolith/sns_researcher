"""Configuration management for LinkedIn Researcher Agent."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic import Field


# Project root directory
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
KNOWLEDGE_DIR = ROOT_DIR / "knowledge"
COOKIES_PATH = DATA_DIR / "cookies.json"


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    # OpenAI
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    model_fast: str = Field(default="gpt-5.4-nano", alias="MODEL_FAST")
    model_powerful: str = Field(default="gpt-5.4", alias="MODEL_POWERFUL")

    # LinkedIn Search
    search_keywords: str = Field(
        default="기업 AX,AI 트랜스포메이션,AI Transformation,enterprise AI,디지털 전환,AX 전략",
        alias="SEARCH_KEYWORDS",
    )

    # Relevance
    relevance_threshold: int = Field(default=40, alias="RELEVANCE_THRESHOLD")

    # Crawling
    scroll_delay_min: int = Field(default=30, alias="SCROLL_DELAY_MIN")
    scroll_delay_max: int = Field(default=120, alias="SCROLL_DELAY_MAX")
    max_posts_per_session: int = Field(default=200, alias="MAX_POSTS_PER_SESSION")
    synthesis_interval_posts: int = Field(default=20, alias="SYNTHESIS_INTERVAL_POSTS")
    synthesis_interval_hours: int = Field(default=6, alias="SYNTHESIS_INTERVAL_HOURS")

    # Dashboard
    dashboard_port: int = Field(default=8501, alias="DASHBOARD_PORT")
    dashboard_host: str = Field(default="0.0.0.0", alias="DASHBOARD_HOST")

    @property
    def keywords_list(self) -> list[str]:
        """Parse comma-separated keywords into a list."""
        return [kw.strip() for kw in self.search_keywords.split(",") if kw.strip()]

    model_config = {
        "env_file": str(ROOT_DIR / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


def ensure_dirs() -> None:
    """Ensure all required directories exist."""
    dirs = [
        DATA_DIR,
        KNOWLEDGE_DIR,
        KNOWLEDGE_DIR / "raw",
        KNOWLEDGE_DIR / "insights" / "topics",
        KNOWLEDGE_DIR / "insights" / "people",
        KNOWLEDGE_DIR / "insights" / "digests",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# Singleton settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create the singleton settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
