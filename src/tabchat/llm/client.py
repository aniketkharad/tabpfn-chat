"""Rate-limited, concurrency-governed LLM client for tabchat."""

from __future__ import annotations

import asyncio
import collections
import logging
import time
from typing import Any, TypeVar

from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from tabchat.config import Settings, get_settings

logger = logging.getLogger("tabchat.llm")

T = TypeVar("T", bound=BaseModel)

# Module-level concurrency semaphore initialized from settings
_settings = get_settings()
_CONCURRENCY_SEMAPHORE = asyncio.Semaphore(_settings.GEMINI_CONCURRENCY)


class LLMError(Exception):
    """Base exception for LLM operations."""


class LLMRateLimitError(LLMError):
    """Raised when transient RPM rate limits are exceeded."""


class DailyQuotaExhaustedError(LLMError):
    """Raised when the daily quota (RPD) is exhausted, tripping a fatal state."""


class SlidingWindowRateLimiter:
    """Sliding-window rate governor tracking invocations over a 60-second window."""

    def __init__(self, rpm_limit: int, window_seconds: float = 60.0) -> None:
        self.rpm_limit = rpm_limit
        self.window_seconds = window_seconds
        self._timestamps: collections.deque[float] = collections.deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a request slot is available in the sliding window."""
        while True:
            async with self._lock:
                now = time.monotonic()
                # Evict expired timestamps
                while self._timestamps and now - self._timestamps[0] >= self.window_seconds:
                    self._timestamps.popleft()

                if len(self._timestamps) < self.rpm_limit:
                    self._timestamps.append(now)
                    return

                # Calculate wait duration until oldest timestamp rolls off
                oldest = self._timestamps[0]
                wait_time = self.window_seconds - (now - oldest) + 0.05

            if wait_time > 0:
                logger.warning(
                    "Rate limit reached (%d calls in 60s). Pausing for %.2fs...",
                    self.rpm_limit,
                    wait_time,
                )
                await asyncio.sleep(wait_time)


# Module-level rate limiter initialized from settings
_RATE_LIMITER = SlidingWindowRateLimiter(rpm_limit=_settings.GEMINI_RPM_LIMIT)


def _is_daily_exhaustion(err_str: str) -> bool:
    """Determine if a 429 error corresponds to daily quota exhaustion (RPD)."""
    err_lower = err_str.lower()
    return any(
        kw in err_lower
        for kw in (
            "generaterequestsperday",
            "daily",
            "free_tier_requests",
            "perday",
            "quota exceeded for metric",
        )
    ) or ("retry in" in err_lower and ("h" in err_lower or "m" in err_lower))


def _format_contents(messages: list[dict[str, Any]]) -> list[types.Content]:
    """Convert standard message dicts into Google GenAI types.Content list."""
    contents: list[types.Content] = []
    for msg in messages:
        role = msg.get("role", "user")
        if role == "assistant":
            role = "model"
        text = str(msg.get("content", ""))
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))
    return contents


class GeminiLLMClient:
    """Production Gemini client enforcing concurrency, rate limiting, and fallback."""

    def __init__(
        self,
        settings: Settings | None = None,
        semaphore: asyncio.Semaphore | None = None,
        rate_limiter: SlidingWindowRateLimiter | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.semaphore = semaphore or _CONCURRENCY_SEMAPHORE
        self.rate_limiter = rate_limiter or _RATE_LIMITER
        self.client = genai.Client(api_key=self.settings.GEMINI_API_KEY)

    async def _execute_with_governance(self, call_fn, model_override: str | None = None) -> Any:
        """Execute a call within the rate limiter and concurrency semaphore with fallback."""
        model = model_override or self.settings.LLM_PRIMARY_MODEL

        # Concurrency and Rate Limiter acquire
        await self.rate_limiter.acquire()
        async with self.semaphore:
            try:
                return await call_fn(model)
            except errors.APIError as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    if _is_daily_exhaustion(err_str):
                        logger.error("Daily quota exhausted on Gemini API: %s", err_str)
                        raise DailyQuotaExhaustedError(
                            f"Daily Gemini quota exhausted. Service temporarily unavailable: {e}"
                        ) from e

                    # Burst rate limit 429: attempt retry once with fallback model
                    logger.warning(
                        "Encountered 429 burst rate limit on %s. Retrying once with fallback model %s...",
                        model,
                        self.settings.LLM_FALLBACK_MODEL,
                    )
                    await asyncio.sleep(2.0)
                    try:
                        return await call_fn(self.settings.LLM_FALLBACK_MODEL)
                    except Exception as fallback_err:
                        if _is_daily_exhaustion(str(fallback_err)):
                            raise DailyQuotaExhaustedError(
                                f"Daily Gemini quota exhausted on fallback model: {fallback_err}"
                            ) from fallback_err
                        raise LLMRateLimitError(
                            f"Gemini API rate limit exceeded on primary and fallback: {fallback_err}"
                        ) from fallback_err

                raise LLMError(f"Gemini API error: {e}") from e
            except Exception as e:
                raise LLMError(f"Unexpected error calling Gemini API: {e}") from e

    async def generate_text(self, messages: list[dict[str, Any]], system_prompt: str = "") -> str:
        """Generate unstructured text response from Gemini."""
        contents = _format_contents(messages)

        async def _call(model: str) -> str:
            config = types.GenerateContentConfig()
            if system_prompt:
                config.system_instruction = system_prompt

            response = await self.client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            return response.text or ""

        return await self._execute_with_governance(_call)

    async def generate_structured(
        self,
        messages: list[dict[str, Any]],
        response_schema: type[T],
        system_prompt: str = "",
    ) -> T:
        """Generate structured response validated against a Pydantic BaseModel schema."""
        contents = _format_contents(messages)

        async def _call(model: str) -> T:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
            )
            if system_prompt:
                config.system_instruction = system_prompt

            response = await self.client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            text = response.text or "{}"
            return response_schema.model_validate_json(text)

        return await self._execute_with_governance(_call)


class FakeLLMClient:
    """Offline mock client implementing identical signatures without making network calls."""

    def __init__(
        self,
        mock_text: str = "This is a simulated response.",
        mock_structured: Any = None,
        latency_seconds: float = 0.0,
        semaphore: asyncio.Semaphore | None = None,
        rate_limiter: SlidingWindowRateLimiter | None = None,
    ) -> None:
        self.mock_text = mock_text
        self.mock_structured = mock_structured
        self.latency_seconds = latency_seconds
        self.semaphore = semaphore or _CONCURRENCY_SEMAPHORE
        self.rate_limiter = rate_limiter or _RATE_LIMITER
        self.call_count = 0
        self.call_history: list[dict[str, Any]] = []

    async def generate_text(self, messages: list[dict[str, Any]], system_prompt: str = "") -> str:
        """Simulate generate_text with concurrency and rate limiting governance."""
        await self.rate_limiter.acquire()
        async with self.semaphore:
            start_t = time.monotonic()
            if self.latency_seconds > 0:
                await asyncio.sleep(self.latency_seconds)
            self.call_count += 1
            self.call_history.append({
                "type": "text",
                "messages": messages,
                "start": start_t,
                "end": time.monotonic(),
            })
            text = self.mock_text(messages, system_prompt) if callable(self.mock_text) else self.mock_text
            return text

    async def generate_structured(
        self,
        messages: list[dict[str, Any]],
        response_schema: type[T],
        system_prompt: str = "",
    ) -> T:
        """Simulate generate_structured with concurrency and rate limiting governance."""
        await self.rate_limiter.acquire()
        async with self.semaphore:
            start_t = time.monotonic()
            if self.latency_seconds > 0:
                await asyncio.sleep(self.latency_seconds)
            self.call_count += 1
            self.call_history.append({
                "type": "structured",
                "schema": response_schema.__name__,
                "start": start_t,
                "end": time.monotonic(),
            })

            if self.mock_structured is not None:
                current_mock = self.mock_structured
                if callable(current_mock):
                    current_mock = current_mock(messages, response_schema)
                elif isinstance(current_mock, list) and current_mock:
                    current_mock = current_mock.pop(0)

                if isinstance(current_mock, response_schema):
                    return current_mock
                if isinstance(current_mock, dict):
                    return response_schema.model_validate(current_mock)
                if isinstance(current_mock, str):
                    return response_schema.model_validate_json(current_mock)

            # Synthesize valid default instance if possible
            try:
                return response_schema()
            except Exception:
                # Return empty json validate
                return response_schema.model_validate({})
