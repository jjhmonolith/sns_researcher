"""FastAPI web dashboard for monitoring the LinkedIn research agent."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.config import KNOWLEDGE_DIR, HEARTBEAT_FILE, STATS_FILE

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(title="LinkedIn AX Research Agent", docs_url=None, redoc_url=None)
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        return templates.TemplateResponse(request, "index.html")

    @app.get("/api/status")
    async def api_status():
        from src.agent.state import get_crawler

        crawler = get_crawler()
        if crawler is not None:
            return JSONResponse(crawler.get_status_dict())

        # Crawler not in this process — read from files
        return JSONResponse(_build_offline_status())

    @app.post("/api/pause")
    async def api_pause():
        from src.agent.state import get_crawler

        crawler = get_crawler()
        if crawler is None:
            return JSONResponse({"error": "Agent not running"}, status_code=400)
        crawler.request_pause()
        return JSONResponse({"ok": True, "paused": crawler._pause_requested})

    @app.post("/api/stop")
    async def api_stop():
        from src.agent.state import get_crawler

        crawler = get_crawler()
        if crawler is None:
            return JSONResponse({"error": "Agent not running"}, status_code=400)
        crawler.request_stop()
        return JSONResponse({"ok": True})

    @app.get("/api/posts")
    async def api_recent_posts():
        from src.knowledge.store import KnowledgeStore

        store = KnowledgeStore()
        posts = store.get_recent_posts(days=7, limit=100)
        return JSONResponse(posts)

    @app.get("/api/knowledge/{path:path}")
    async def api_knowledge_file(path: str):
        filepath = KNOWLEDGE_DIR / path
        if not filepath.exists() or not filepath.is_file():
            return JSONResponse({"error": "File not found"}, status_code=404)
        if not str(filepath.resolve()).startswith(str(KNOWLEDGE_DIR.resolve())):
            return JSONResponse({"error": "Access denied"}, status_code=403)
        content = filepath.read_text(encoding="utf-8")
        return JSONResponse({"path": path, "content": content})

    @app.get("/api/knowledge")
    async def api_knowledge_tree():
        tree = _build_tree(KNOWLEDGE_DIR)
        return JSONResponse(tree)

    @app.get("/api/atoms")
    async def api_atoms():
        from src.knowledge.store import KnowledgeStore

        store = KnowledgeStore()
        atoms = store.get_all_atoms()
        return JSONResponse(atoms)

    @app.get("/api/atoms/{atom_id}")
    async def api_atom_detail(atom_id: str):
        from src.knowledge.store import KnowledgeStore

        store = KnowledgeStore()
        data = store.get_atom_by_id(atom_id)
        if not data:
            return JSONResponse({"error": "Atom not found"}, status_code=404)
        return JSONResponse(data)

    return app


def _build_offline_status() -> dict:
    """Build a status dict from heartbeat + stats files when the crawler isn't in-process."""
    heartbeat = {}
    if HEARTBEAT_FILE.exists():
        try:
            heartbeat = json.loads(HEARTBEAT_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass

    stats = {}
    if STATS_FILE.exists():
        try:
            stats = json.loads(STATS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Determine effective status
    hb_timestamp = heartbeat.get("timestamp", "")
    heartbeat_age = None
    effective_status = "not_started"

    if hb_timestamp:
        try:
            hb_time = datetime.fromisoformat(hb_timestamp)
            heartbeat_age = int((datetime.now() - hb_time).total_seconds())
            # If heartbeat is recent, agent is running in another process
            if heartbeat_age < 60:
                effective_status = heartbeat.get("status", "running")
            elif heartbeat_age < 300:
                effective_status = "stale"
            else:
                effective_status = "dead"
        except Exception:
            pass

    cum_scanned = stats.get("total_posts_scanned", 0)
    cum_relevant = stats.get("relevant_posts_found", 0)
    cum_nano_in = stats.get("nano_input_tokens", 0)
    cum_nano_out = stats.get("nano_output_tokens", 0)
    cum_pow_in = stats.get("powerful_input_tokens", 0)
    cum_pow_out = stats.get("powerful_output_tokens", 0)
    nano_cost = (cum_nano_in * 0.20 + cum_nano_out * 1.25) / 1_000_000
    pow_cost = (cum_pow_in * 2.50 + cum_pow_out * 15.00) / 1_000_000

    return {
        "status": effective_status,
        "started_at": "",
        "uptime_seconds": 0,
        "current_cycle": heartbeat.get("cycle", 0),
        "pid": heartbeat.get("pid", 0),
        "total_sessions": stats.get("total_sessions", 0),
        "first_started_at": stats.get("first_started_at", ""),
        "total_posts_scanned": cum_scanned,
        "relevant_posts_found": cum_relevant,
        "relevance_rate": (
            f"{(cum_relevant / cum_scanned * 100):.1f}%" if cum_scanned > 0 else "N/A"
        ),
        "session_posts_scanned": 0,
        "session_relevant_found": 0,
        "posts_since_last_synthesis": 0,
        "last_synthesis_at": "Not yet",
        "atom_count": 0,
        "queue_size": 0,
        "queue_stats": {},
        "current_action": heartbeat.get("current_action", ""),
        "current_url": "",
        "heartbeat_age_seconds": heartbeat_age,
        "last_heartbeat": hb_timestamp,
        "token_usage": {
            "nano_input": cum_nano_in,
            "nano_output": cum_nano_out,
            "powerful_input": cum_pow_in,
            "powerful_output": cum_pow_out,
            "nano_cost": f"${nano_cost:.4f}",
            "powerful_cost": f"${pow_cost:.4f}",
            "total_cost": f"${nano_cost + pow_cost:.4f}",
        },
        "recent_errors": [],
        "posts_saved_today": 0,
        "activity_log": [],
    }


def _build_tree(path: Path, prefix: str = "") -> list[dict]:
    items = []
    if not path.exists():
        return items
    for entry in sorted(path.iterdir()):
        rel = str(entry.relative_to(KNOWLEDGE_DIR))
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            children = _build_tree(entry, rel)
            items.append(
                {
                    "name": entry.name,
                    "path": rel,
                    "type": "directory",
                    "children": children,
                }
            )
        elif entry.suffix in (".md", ".json"):
            items.append({"name": entry.name, "path": rel, "type": "file"})
    return items
