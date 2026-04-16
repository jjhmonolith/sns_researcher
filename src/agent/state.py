"""Shared state module — holds the global crawler reference.

This module is a single source of truth for the crawler instance.
Both main.py and dashboard/app.py import from here, ensuring they
reference the same object in the same process.
"""

from __future__ import annotations

_crawler = None


def set_crawler(crawler) -> None:
    global _crawler
    _crawler = crawler


def get_crawler():
    return _crawler
