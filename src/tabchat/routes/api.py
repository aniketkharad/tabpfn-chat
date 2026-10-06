"""FastAPI API routes for tabchat session management, upload, chat, and execution."""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Cookie, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from tabchat.config import get_settings
from tabchat.engine.executor import execute_plan
from tabchat.engine.narrator import narrate_results
from tabchat.engine.planner import plan_analysis
from tabchat.llm.client import FakeLLMClient, GeminiLLMClient
from tabchat.storage import SessionStore
from tabchat.validation import CSVValidationError, validate_and_parse_csv

logger = logging.getLogger("tabchat.routes")

router = APIRouter(prefix="/api", tags=["api"])


class ChatRequest(BaseModel):
    """Payload for POST /api/chat."""

    message: str = Field(..., min_length=1, description="User conversational prompt")


def get_store(request: Request) -> SessionStore:
    """Retrieve SessionStore from app state or instantiate default."""
    return getattr(request.app.state, "store", None) or SessionStore()


def get_llm(request: Request) -> GeminiLLMClient | FakeLLMClient:
    """Retrieve LLM client from app state or instantiate default."""
    return getattr(request.app.state, "llm_client", None) or GeminiLLMClient()


def get_tabpfn_api_caller(request: Request) -> Any:
    """Retrieve custom TabPFN API caller if injected in app state (e.g. for testing)."""
    return getattr(request.app.state, "tabpfn_api_caller", None)


def resolve_session_id(
    request: Request,
    session_id: Optional[str] = Cookie(default=None),
    required: bool = True,
) -> str:
    """Extract and validate session_id from HttpOnly cookie."""
    store = get_store(request)
    if not session_id or not store.session_exists(session_id):
        if required:
            raise HTTPException(
                status_code=404,
                detail="Active session not found. Please initialize session via POST /api/session.",
            )
        return ""
    return session_id


@router.post("/session")
async def create_session(request: Request, response: Response) -> dict[str, str]:
    """Initialize a new session and set HttpOnly cookie."""
    store = get_store(request)
    session_id = store.create_session()
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        samesite="lax",
        path="/",
    )
    return {"session_id": session_id, "status": "created"}


@router.get("/session")
async def get_session(
    request: Request,
    session_id: Optional[str] = Cookie(default=None),
) -> dict[str, Any]:
    """Resume session state from cookie."""
    store = get_store(request)
    s_id = resolve_session_id(request, session_id=session_id, required=True)

    dataset_card = None
    try:
        _, card = store.get_dataset(s_id)
        dataset_card = card
    except (FileNotFoundError, Exception):
        dataset_card = None

    history = store.get_history(s_id)
    job_spec = store.get_job_spec(s_id)
    results = store.get_results(s_id)

    return {
        "session_id": s_id,
        "dataset_card": dataset_card,
        "history": history,
        "job_spec": job_spec,
        "results": results,
    }


@router.post("/upload")
async def upload_dataset(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    session_id: Optional[str] = Cookie(default=None),
) -> Any:
    """Multipart CSV upload, boundary validation, and dataset card creation."""
    store = get_store(request)

    # If cookie missing or invalid, initialize a new session
    if not session_id or not store.session_exists(session_id):
        session_id = store.create_session()
        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            samesite="lax",
            path="/",
        )

    file_bytes = await file.read()

    try:
        _, card = validate_and_parse_csv(file_bytes)
    except CSVValidationError as exc:
        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors, "errors": exc.errors},
        )
    except Exception as exc:
        return JSONResponse(
            status_code=422,
            content={"detail": [f"Malformed CSV: {exc}"], "errors": [f"Malformed CSV: {exc}"]},
        )

    # Persist raw bytes and generated dataset card
    store.save_dataset(session_id, file_bytes, card)
    return JSONResponse(status_code=200, content=card)


@router.post("/chat")
async def chat_turn(
    payload: ChatRequest,
    request: Request,
    session_id: Optional[str] = Cookie(default=None),
) -> dict[str, Any]:
    """Conversational intent clarification and JobSpec compilation turn."""
    store = get_store(request)
    s_id = resolve_session_id(request, session_id=session_id, required=True)
    llm = get_llm(request)
    settings = get_settings()

    # Enforce turn limit (<= 15)
    history = store.get_history(s_id)
    user_turns = sum(1 for m in history if m.get("role") == "user")
    if user_turns >= settings.MAX_SESSION_TURNS:
        raise HTTPException(
            status_code=400,
            detail=f"Session turn limit of {settings.MAX_SESSION_TURNS} reached. Please execute your plan or start a new session.",
        )

    turns_left = max(0, settings.MAX_SESSION_TURNS - (user_turns + 1))

    # Run Planner
    planner_output = await plan_analysis(
        session_id=s_id,
        user_message=payload.message,
        llm_client=llm,
        store=store,
    )

    if planner_output.get("is_clarification"):
        reply = planner_output.get("clarification_message") or "Could you clarify your goal?"
        plan_card = None
    else:
        proposed = planner_output.get("proposed_spec")
        plan_card = proposed
        target = proposed.get("target_column") if proposed else "target"
        rationale = proposed.get("rationale") if proposed else ""
        reply = f"I've configured an analytical plan to predict '{target}'. {rationale}".strip()

    return {
        "reply": reply,
        "plan_card": plan_card,
        "turns_left": turns_left,
    }


@router.post("/run")
async def run_analysis(
    request: Request,
    session_id: Optional[str] = Cookie(default=None),
) -> dict[str, Any]:
    """Execute active JobSpec deterministically against TabPFN and narrate results."""
    store = get_store(request)
    s_id = resolve_session_id(request, session_id=session_id, required=True)
    llm = get_llm(request)
    api_caller = get_tabpfn_api_caller(request)

    job_spec = store.get_job_spec(s_id)
    if job_spec is None:
        raise HTTPException(
            status_code=400,
            detail="No active JobSpec found for session. Please chat with the planner to configure a plan before running.",
        )

    # 1. Deterministic execution (ZERO LLM CALLS)
    try:
        results = await execute_plan(
            session_id=s_id,
            store=store,
            api_caller=api_caller,
        )
    except Exception as exc:
        logger.error("Executor execution failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Execution error: {exc}")

    # 2. Grounded narration
    try:
        narration = await narrate_results(
            session_id=s_id,
            llm_client=llm,
            store=store,
        )
    except Exception as exc:
        logger.error("Narrator synthesis failed: %s", exc, exc_info=True)
        narration = "Results generated successfully, but automated narration encountered an error."

    return {
        "results": results,
        "narration": narration,
    }


@router.get("/log")
async def get_log(
    request: Request,
    session_id: Optional[str] = Cookie(default=None),
) -> PlainTextResponse:
    """Return raw history.jsonl records as text/plain for debugging."""
    store = get_store(request)
    s_id = resolve_session_id(request, session_id=session_id, required=True)

    history_path = store.sessions_dir / s_id / "history.jsonl"
    if history_path.exists():
        content = history_path.read_text(encoding="utf-8")
    else:
        content = ""

    return PlainTextResponse(content=content, media_type="text/plain")


@router.delete("/session")
async def delete_session(
    request: Request,
    response: Response,
    session_id: Optional[str] = Cookie(default=None),
) -> dict[str, str]:
    """Wipe session directory from disk and delete cookie."""
    store = get_store(request)
    if session_id and store.session_exists(session_id):
        store.delete_session(session_id)
    response.delete_cookie(key="session_id", path="/")
    return {"status": "deleted"}
