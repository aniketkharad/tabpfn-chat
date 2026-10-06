"""Tests for FastAPI static asset serving and frontend root endpoints."""

from __future__ import annotations

from starlette.testclient import TestClient

from tabchat.app import app


def test_serve_root_index_html() -> None:
    """GET / should serve index.html with 200 OK and valid HTML content."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "tabchat" in response.text
    assert "workspace-grid" in response.text
    assert "TabPFN-3.5" in response.text


def test_serve_index_html_alias() -> None:
    """GET /index.html should serve index.html with 200 OK."""
    client = TestClient(app)
    response = client.get("/index.html")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "tabchat" in response.text


def test_serve_static_style_css() -> None:
    """GET /static/style.css should serve the CSS stylesheet with 200 OK."""
    client = TestClient(app)
    response = client.get("/static/style.css")
    assert response.status_code == 200
    assert "text/css" in response.headers.get("content-type", "")
    assert "--bg-app" in response.text
    assert ".plan-card" in response.text
    assert ".results-card" in response.text


def test_serve_static_app_js() -> None:
    """GET /static/app.js should serve the JavaScript script with 200 OK."""
    client = TestClient(app)
    response = client.get("/static/app.js")
    assert response.status_code == 200
    assert "javascript" in response.headers.get("content-type", "")
    assert "renderMicroMarkdown" in response.text
    assert "IS_MOCK_MODE" in response.text
    assert "buildSvgChart" in response.text


def test_api_routes_not_shadowed_by_static() -> None:
    """API routes under /api must remain intact and not shadowed by static handlers."""
    client = TestClient(app)
    response = client.post("/api/session")
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["status"] == "created"
