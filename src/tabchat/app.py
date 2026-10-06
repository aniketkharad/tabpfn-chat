"""FastAPI application factory and instance for tabchat."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from tabchat.routes.api import router as api_router

FRONTEND_DIR = Path(__file__).parent / "frontend"


def create_app() -> FastAPI:
    """Create and configure the tabchat FastAPI application."""
    app = FastAPI(
        title="tabchat",
        description="Local-first chat & prediction web app for TabPFN-3.5",
        version="0.1.0",
    )

    # Allow local development origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API routes
    app.include_router(api_router)

    # Serve static assets and single-page application
    if FRONTEND_DIR.exists():
        app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
        samples_dir = FRONTEND_DIR / "samples"
        if samples_dir.exists():
            app.mount("/samples", StaticFiles(directory=samples_dir), name="samples")

        @app.get("/")
        @app.get("/index.html")
        async def serve_index() -> FileResponse:
            return FileResponse(FRONTEND_DIR / "index.html")

    return app


app = create_app()
