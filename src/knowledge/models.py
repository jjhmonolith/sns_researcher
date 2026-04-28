"""Data models for LinkedIn Researcher Agent."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class PostType(str, Enum):
    """Type of LinkedIn post."""
    ORIGINAL = "original"
    SHARED = "shared"
    ARTICLE = "article"
    COMMENT = "comment"


class CrawlSource(str, Enum):
    """How the post was discovered."""
    HOME_FEED = "home_feed"
    SEARCH = "search"
    PROFILE = "profile"
    LINKED_POST = "linked_post"
    COMMENT_THREAD = "comment_thread"
    X_FEED = "x_feed"
    X_SEARCH = "x_search"
    X_PROFILE = "x_profile"
    FB_FEED = "fb_feed"
    FB_SEARCH = "fb_search"
    FB_GROUP = "fb_group"
    FB_PROFILE = "fb_profile"


class QueueItemType(str, Enum):
    """Type of item in the exploration queue."""
    POST_URL = "post_url"
    PROFILE_URL = "profile_url"
    SEARCH_KEYWORD = "search_keyword"


class QueueItemStatus(str, Enum):
    """Status of a queue item."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Author(BaseModel):
    """LinkedIn post author information."""
    name: str = ""
    headline: str = ""
    profile_url: str = ""
    linkedin_id: str = ""


class LinkedInPost(BaseModel):
    """A single LinkedIn post with all extracted data."""
    post_id: str = ""
    url: str = ""
    author: Author = Field(default_factory=Author)
    content: str = ""
    post_type: PostType = PostType.ORIGINAL
    published_date: str = ""
    reactions_count: int = 0
    comments_count: int = 0
    reposts_count: int = 0

    # Extracted links and mentions
    mentioned_profiles: list[str] = Field(default_factory=list)
    external_links: list[str] = Field(default_factory=list)
    linked_posts: list[str] = Field(default_factory=list)

    # Relevance analysis (filled by GPT-5.4 nano)
    relevance_score: int = 0
    relevance_topics: list[str] = Field(default_factory=list)
    summary: str = ""
    should_follow_links: bool = False
    follow_targets: list[str] = Field(default_factory=list)

    # Novelty analysis
    novelty_score: int = 0
    novelty_reason: str = ""
    is_novel: bool = False

    # Metadata
    platform: str = "linkedin"
    crawl_source: CrawlSource = CrawlSource.HOME_FEED
    crawled_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    is_relevant: bool = False


class QueueItem(BaseModel):
    """An item in the exploration queue."""
    url: str
    item_type: QueueItemType
    priority: int = 50  # 0-100, higher = more important
    status: QueueItemStatus = QueueItemStatus.PENDING
    source_post_id: str = ""  # Which post led to this URL
    added_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    reason: str = ""  # Why this was added to the queue


class AgentStatus(str, Enum):
    """Current status of the agent."""
    INITIALIZING = "initializing"
    WAITING_LOGIN = "waiting_login"
    RUNNING = "running"
    PAUSED = "paused"
    SYNTHESIZING = "synthesizing"
    ERROR = "error"
    STOPPED = "stopped"


class AgentStats(BaseModel):
    """Runtime statistics for the agent."""
    status: AgentStatus = AgentStatus.INITIALIZING
    started_at: str = ""
    first_started_at: str = ""
    total_posts_scanned: int = 0
    relevant_posts_found: int = 0
    total_sessions: int = 0
    posts_since_last_synthesis: int = 0
    last_synthesis_at: str = ""
    queue_size: int = 0
    current_url: str = ""
    current_action: str = ""
    errors: list[str] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=lambda: TokenUsage())


class TokenUsage(BaseModel):
    """Track token usage and estimated costs."""
    nano_input_tokens: int = 0
    nano_output_tokens: int = 0
    powerful_input_tokens: int = 0
    powerful_output_tokens: int = 0

    @property
    def nano_cost(self) -> float:
        """Estimated cost for GPT-5.4 nano usage."""
        return (self.nano_input_tokens * 0.20 + self.nano_output_tokens * 1.25) / 1_000_000

    @property
    def powerful_cost(self) -> float:
        """Estimated cost for GPT-5.4 usage."""
        return (self.powerful_input_tokens * 2.50 + self.powerful_output_tokens * 15.00) / 1_000_000

    @property
    def total_cost(self) -> float:
        return self.nano_cost + self.powerful_cost


class ActivityLog(BaseModel):
    """A single activity log entry."""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    action: str = ""
    detail: str = ""
    level: str = "info"  # info, warning, error
