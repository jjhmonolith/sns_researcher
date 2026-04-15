"""FastAPI web dashboard for monitoring the LinkedIn research agent."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.config import KNOWLEDGE_DIR

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    """Create and configure the FastAPI dashboard application."""
    app = FastAPI(title="LinkedIn AX Research Agent", docs_url=None, redoc_url=None)

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    # Serve static files if directory exists
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        """Main dashboard page."""
        return templates.TemplateResponse(request, "index.html")

    @app.get("/api/status")
    async def api_status():
        """Get current agent status as JSON."""
        from src.main import get_crawler

        crawler = get_crawler()
        if crawler is None:
            return JSONResponse(
                {"status": "not_started", "message": "Agent has not started yet."},
                status_code=200,
            )
        return JSONResponse(crawler.get_status_dict())

    @app.post("/api/pause")
    async def api_pause():
        """Toggle pause/resume."""
        from src.main import get_crawler

        crawler = get_crawler()
        if crawler is None:
            return JSONResponse({"error": "Agent not running"}, status_code=400)
        crawler.request_pause()
        return JSONResponse({"ok": True, "paused": crawler._pause_requested})

    @app.post("/api/stop")
    async def api_stop():
        """Request graceful stop."""
        from src.main import get_crawler

        crawler = get_crawler()
        if crawler is None:
            return JSONResponse({"error": "Agent not running"}, status_code=400)
        crawler.request_stop()
        return JSONResponse({"ok": True})

    @app.get("/api/posts")
    async def api_recent_posts():
        """Get recent collected posts."""
        from src.knowledge.store import KnowledgeStore

        store = KnowledgeStore()
        posts = store.get_recent_posts(days=7, limit=100)
        return JSONResponse(posts)

    @app.get("/api/knowledge/{path:path}")
    async def api_knowledge_file(path: str):
        """Read a knowledge base file."""
        filepath = KNOWLEDGE_DIR / path
        if not filepath.exists() or not filepath.is_file():
            return JSONResponse({"error": "File not found"}, status_code=404)
        if not str(filepath.resolve()).startswith(str(KNOWLEDGE_DIR.resolve())):
            return JSONResponse({"error": "Access denied"}, status_code=403)
        content = filepath.read_text(encoding="utf-8")
        return JSONResponse({"path": path, "content": content})

    @app.get("/api/knowledge")
    async def api_knowledge_tree():
        """Get the knowledge base directory tree."""
        tree = _build_tree(KNOWLEDGE_DIR)
        return JSONResponse(tree)

    @app.get("/api/atoms")
    async def api_atoms():
        """Get all atom notes (lightweight metadata list)."""
        from src.knowledge.store import KnowledgeStore
        store = KnowledgeStore()
        atoms = store.get_all_atoms()
        return JSONResponse(atoms)

    @app.get("/api/atoms/{atom_id}")
    async def api_atom_detail(atom_id: str):
        """Get full content of a single atom note."""
        from src.knowledge.store import KnowledgeStore
        store = KnowledgeStore()
        data = store.get_atom_by_id(atom_id)
        if not data:
            return JSONResponse({"error": "Atom not found"}, status_code=404)
        return JSONResponse(data)

    return app


def _build_tree(path: Path, prefix: str = "") -> list[dict]:
    """Build a file tree structure for the knowledge base."""
    items = []
    if not path.exists():
        return items

    for entry in sorted(path.iterdir()):
        rel = str(entry.relative_to(KNOWLEDGE_DIR))
        if entry.name.startswith("."):
            continue

        if entry.is_dir():
            children = _build_tree(entry, rel)
            items.append({
                "name": entry.name,
                "path": rel,
                "type": "directory",
                "children": children,
            })
        elif entry.suffix == ".md":
            items.append({
                "name": entry.name,
                "path": rel,
                "type": "file",
            })
        elif entry.suffix == ".json":
            items.append({
                "name": entry.name,
                "path": rel,
                "type": "file",
            })

    return items
