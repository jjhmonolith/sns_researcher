"""FastAPI web dashboard — multi-session research agent."""

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


def _get_orchestrator():
    import sys
    main_mod = sys.modules.get("src.main") or sys.modules.get("__main__")
    if main_mod and hasattr(main_mod, "_orchestrator"):
        return main_mod._orchestrator
    return None


def _get_session_config(session_id: str):
    orch = _get_orchestrator()
    if orch:
        return orch.manager.get_session(session_id)
    return None


def _get_session_store(session_id: str):
    config = _get_session_config(session_id)
    if not config:
        return None
    from src.knowledge.store import KnowledgeStore
    return KnowledgeStore(base_dir=config.knowledge_dir)


def create_app() -> FastAPI:
    app = FastAPI(title="Research Agent", docs_url=None, redoc_url=None)
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # ── Session List Page ──
    @app.get("/", response_class=HTMLResponse)
    async def sessions_page(request: Request):
        return templates.TemplateResponse(request, "sessions.html")

    # ── Individual Session Dashboard ──
    @app.get("/session/{session_id}", response_class=HTMLResponse)
    async def session_dashboard(request: Request, session_id: str):
        config = _get_session_config(session_id)
        if not config:
            return HTMLResponse("<h1>Session not found</h1>", status_code=404)
        return templates.TemplateResponse(request, "index.html", {
            "session_id": session_id,
            "session_name": config.name,
            "session_topic": config.topic_description[:80],
        })

    # ── Sessions CRUD API ──
    @app.get("/api/sessions")
    async def api_list_sessions():
        orch = _get_orchestrator()
        if not orch:
            return JSONResponse([])
        sessions = orch.manager.list_sessions()
        result = []
        for s in sessions:
            result.append({
                "id": s.id,
                "name": s.name,
                "topic_description": s.topic_description[:100],
                "keywords": s.keywords,
                "platforms": s.platforms,
                "status": s.status,
                "is_running": orch.is_session_running(s.id),
                "created_at": s.created_at,
            })
        return JSONResponse(result)

    @app.post("/api/sessions")
    async def api_create_session(request: Request):
        orch = _get_orchestrator()
        if not orch:
            return JSONResponse({"error": "Not ready"}, status_code=503)
        body = await request.json()
        config = orch.manager.create_session(
            name=body.get("name", ""),
            topic_description=body.get("topic_description", ""),
            keywords=body.get("keywords", []),
            platforms=body.get("platforms", ["linkedin"]),
        )
        return JSONResponse({"ok": True, "id": config.id})

    @app.delete("/api/sessions/{session_id}")
    async def api_delete_session(session_id: str):
        orch = _get_orchestrator()
        if not orch:
            return JSONResponse({"error": "Not ready"}, status_code=503)
        if orch.is_session_running(session_id):
            return JSONResponse({"error": "Stop session first"}, status_code=400)
        orch.manager.delete_session(session_id)
        return JSONResponse({"ok": True})

    @app.post("/api/sessions/{session_id}/start")
    async def api_start_session(session_id: str):
        orch = _get_orchestrator()
        if not orch:
            return JSONResponse({"error": "Not ready"}, status_code=503)
        ok = orch.start_session_threadsafe(session_id)
        return JSONResponse({"ok": ok})

    @app.post("/api/sessions/{session_id}/stop")
    async def api_stop_session(session_id: str):
        orch = _get_orchestrator()
        if not orch:
            return JSONResponse({"error": "Not ready"}, status_code=503)
        ok = orch.stop_session_threadsafe(session_id)
        return JSONResponse({"ok": ok})

    @app.post("/api/sessions/{session_id}/pause")
    async def api_pause_session(session_id: str):
        orch = _get_orchestrator()
        if not orch:
            return JSONResponse({"error": "Not ready"}, status_code=503)
        crawlers = orch.get_session_crawlers(session_id)
        for c in crawlers.values():
            if hasattr(c, "request_pause"):
                c.request_pause()
        return JSONResponse({"ok": True})

    # ── Per-Session Data API ──
    @app.get("/api/sessions/{session_id}/status")
    async def api_session_status(session_id: str):
        orch = _get_orchestrator()
        if not orch:
            return JSONResponse({"status": "not_started"})
        crawlers = orch.get_session_crawlers(session_id)
        if not crawlers:
            config = _get_session_config(session_id)
            return JSONResponse({"status": config.status if config else "not_found"})

        # Primary: linkedin status, fallback to first crawler
        primary = crawlers.get("linkedin") or next(iter(crawlers.values()))
        result = primary.get_status_dict() if hasattr(primary, "get_status_dict") else {"status": "unknown"}

        x = crawlers.get("x")
        if x and hasattr(x, "get_status_dict"):
            result["x_status"] = x.get_status_dict()

        fb = crawlers.get("facebook")
        if fb and hasattr(fb, "get_status_dict"):
            result["fb_status"] = fb.get_status_dict()

        result["platforms"] = list(crawlers.keys())
        return JSONResponse(result)

    @app.get("/api/sessions/{session_id}/posts")
    async def api_session_posts(session_id: str):
        store = _get_session_store(session_id)
        if not store:
            return JSONResponse([])
        return JSONResponse(store.get_recent_posts(days=7, limit=100))

    @app.get("/api/sessions/{session_id}/atoms")
    async def api_session_atoms(session_id: str):
        store = _get_session_store(session_id)
        if not store:
            return JSONResponse([])
        return JSONResponse(store.get_all_atoms())

    @app.get("/api/sessions/{session_id}/atoms/{atom_id}")
    async def api_session_atom_detail(session_id: str, atom_id: str):
        store = _get_session_store(session_id)
        if not store:
            return JSONResponse({"error": "Not found"}, status_code=404)
        data = store.get_atom_by_id(atom_id)
        if not data:
            return JSONResponse({"error": "Atom not found"}, status_code=404)
        return JSONResponse(data)

    @app.get("/api/sessions/{session_id}/people")
    async def api_session_people(session_id: str):
        config = _get_session_config(session_id)
        if not config:
            return JSONResponse([])
        import frontmatter
        people_dir = config.knowledge_dir / "insights" / "people"
        followed_file = config.followed_authors_path

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

        results = []
        if people_dir.exists():
            for md_file in sorted(people_dir.glob("*.md"), reverse=True):
                try:
                    fm = frontmatter.load(str(md_file))
                    meta = dict(fm.metadata)
                    profile_url = (meta.get("profile_url") or "").split("?")[0].rstrip("/")
                    fa = followed_map.get(profile_url, {})

                    name = meta.get("name", md_file.stem)
                    noise = ["작성자", "발표자", "실무자", "author", "writer", "analyst", "speaker"]
                    if any(kw in name.lower() for kw in noise) or len(name) > 25:
                        continue

                    results.append({
                        "slug": md_file.stem,
                        "name": name,
                        "headline": meta.get("headline", ""),
                        "profile_url": profile_url,
                        "is_followed": profile_url in followed_map,
                        "relevant_post_count": fa.get("relevant_post_count", 0),
                        "visit_count": fa.get("visit_count", 0),
                        "content_preview": fm.content[:300],
                        "path": str(md_file.relative_to(config.knowledge_dir)),
                    })
                except Exception:
                    continue

        results.sort(key=lambda x: (x["is_followed"], x["relevant_post_count"]), reverse=True)
        return JSONResponse(results)

    @app.get("/api/sessions/{session_id}/knowledge")
    async def api_session_knowledge_tree(session_id: str):
        config = _get_session_config(session_id)
        if not config:
            return JSONResponse([])
        return JSONResponse(_build_tree(config.knowledge_dir, config.knowledge_dir))

    @app.get("/api/sessions/{session_id}/knowledge/{path:path}")
    async def api_session_knowledge_file(session_id: str, path: str):
        config = _get_session_config(session_id)
        if not config:
            return JSONResponse({"error": "Not found"}, status_code=404)
        filepath = config.knowledge_dir / path
        if not filepath.exists() or not filepath.is_file():
            return JSONResponse({"error": "File not found"}, status_code=404)
        if not str(filepath.resolve()).startswith(str(config.knowledge_dir.resolve())):
            return JSONResponse({"error": "Access denied"}, status_code=403)
        content = filepath.read_text(encoding="utf-8")
        return JSONResponse({"path": path, "content": content})

    return app


def _build_tree(path: Path, base_dir: Path) -> list[dict]:
    items = []
    if not path.exists():
        return items
    for entry in sorted(path.iterdir()):
        if entry.name.startswith("."):
            continue
        rel = str(entry.relative_to(base_dir))
        if entry.is_dir():
            items.append({
                "name": entry.name, "path": rel, "type": "directory",
                "children": _build_tree(entry, base_dir),
            })
        elif entry.suffix in (".md", ".json"):
            items.append({"name": entry.name, "path": rel, "type": "file"})
    return items
