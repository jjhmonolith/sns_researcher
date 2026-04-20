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
        """Get status of all crawlers."""
        from src.main import get_crawlers

        crawlers = get_crawlers()
        if not crawlers:
            return JSONResponse(
                {"status": "not_started", "message": "No crawlers running."},
                status_code=200,
            )

        # Primary status from LinkedIn crawler (backward compat)
        linkedin = crawlers.get("linkedin")
        if linkedin and hasattr(linkedin, "get_status_dict"):
            result = linkedin.get_status_dict()
        else:
            result = {"status": "not_started"}

        # Add X crawler status if available
        x = crawlers.get("x")
        if x and hasattr(x, "get_status_dict"):
            result["x_status"] = x.get_status_dict()

        result["platforms"] = list(crawlers.keys())
        return JSONResponse(result)

    @app.post("/api/pause")
    async def api_pause():
        """Toggle pause/resume on all crawlers."""
        from src.main import get_crawlers

        crawlers = get_crawlers()
        if not crawlers:
            return JSONResponse({"error": "No crawlers running"}, status_code=400)
        for crawler in crawlers.values():
            if hasattr(crawler, "request_pause"):
                crawler.request_pause()
        return JSONResponse({"ok": True})

    @app.post("/api/stop")
    async def api_stop():
        """Request graceful stop on all crawlers."""
        from src.main import get_crawlers

        crawlers = get_crawlers()
        if not crawlers:
            return JSONResponse({"error": "No crawlers running"}, status_code=400)
        for crawler in crawlers.values():
            if hasattr(crawler, "request_stop"):
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

    @app.get("/api/people")
    async def api_people():
        """Get key people profiles with followed author data."""
        import frontmatter

        # Load followed authors
        followed_file = KNOWLEDGE_DIR / "followed_authors.json"
        followed_map: dict[str, dict] = {}
        if followed_file.exists():
            try:
                data = json.loads(followed_file.read_text())
                for a in data.get("authors", []):
                    url = a.get("profile_url", "").split("?")[0].rstrip("/")
                    if url:
                        followed_map[url] = a
            except Exception:
                pass

        # Load people profiles
        people_dir = KNOWLEDGE_DIR / "insights" / "people"
        results = []
        if people_dir.exists():
            for md_file in sorted(people_dir.glob("*.md"), reverse=True):
                try:
                    fm = frontmatter.load(str(md_file))
                    meta = dict(fm.metadata)
                    profile_url = (meta.get("profile_url") or "").split("?")[0].rstrip("/")

                    # Merge with followed author data
                    fa = followed_map.get(profile_url, {})

                    results.append({
                        "slug": md_file.stem,
                        "name": meta.get("name", md_file.stem),
                        "headline": meta.get("headline", ""),
                        "profile_url": profile_url,
                        "is_followed": profile_url in followed_map,
                        "relevant_post_count": fa.get("relevant_post_count", 0),
                        "visit_count": fa.get("visit_count", 0),
                        "last_visited": fa.get("last_visited", ""),
                        "content_preview": fm.content[:300],
                        "path": str(md_file.relative_to(KNOWLEDGE_DIR)),
                    })
                except Exception:
                    continue

        # Filter out placeholder/descriptive names (GPT-generated slugs)
        noise_keywords = ["작성자", "발표자", "실무자", "author", "writer", "analyst", "speaker"]

        def is_real_name(name: str) -> bool:
            if not name:
                return False
            if any(kw in name.lower() for kw in noise_keywords):
                return False
            if len(name) > 25:
                return False
            return True

        results = [r for r in results if is_real_name(r["name"])]

        # Sort: followed first (by post count), then others
        results.sort(key=lambda x: (x["is_followed"], x["relevant_post_count"]), reverse=True)
        return JSONResponse(results)

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
