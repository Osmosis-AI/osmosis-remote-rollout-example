"""Core rollout execution logic for the Remote Rollout Server.

Protocol flow (async-init):
- Training -> RolloutServer: POST /init (returns 202 with tools).
- RolloutServer -> Training: POST {server_url}/v1/chat/completions (LLM generations).
- RolloutServer -> Training: POST {server_url}/v1/rollout/completed (final result).

Rollouts are executed in background tasks keyed by rollout_id (idempotency key).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from rollout_server.config import settings
from rollout_server.schemas import (
    CompletionsRequest,
    CompletionsResponse,
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    RolloutStatus,
)
from rollout_server.tools.calculator import execute_calculator_calls

logger = logging.getLogger(__name__)

MessageDict = Dict[str, Any]
ToolSchemaDict = Dict[str, Any]
ToolCallDict = Dict[str, Any]


@dataclass
class RolloutRecord:
    tools: List[ToolSchemaDict]
    # Task is released on completion to avoid retaining coroutine locals.
    task: Optional[asyncio.Task]
    created_at: float
    completed_at: Optional[float] = None
    last_accessed_at: float = 0.0


class AppState:
    """Shared application state for the FastAPI service."""

    def __init__(self) -> None:
        self.http_client: Optional[httpx.AsyncClient] = None
        self._http_client_lock = asyncio.Lock()
        self._rollout_semaphore: Optional[asyncio.Semaphore] = None

        # Idempotency registry keyed by rollout_id.
        self.rollout_records: Dict[str, RolloutRecord] = {}

    def prune_rollout_records(self) -> None:
        """Prune completed rollout records to bound memory usage.

        Notes:
        - We keep completed rollout_id records for a short TTL to preserve idempotency
          across transient client retries (e.g., /init response lost).
        - We drop references to finished asyncio.Tasks to avoid retaining large local
          objects (messages, responses) after completion.
        """
        if not self.rollout_records:
            return

        now = time.time()
        ttl_seconds = float(getattr(settings, "rollout_record_ttl_seconds", 0.0))
        max_records = int(getattr(settings, "max_rollout_records", 0))

        # 1) TTL pruning (completed only)
        if ttl_seconds > 0:
            expired_ids: List[str] = []
            for rollout_id, record in list(self.rollout_records.items()):
                if record.completed_at is None:
                    continue
                if (now - record.completed_at) > ttl_seconds:
                    expired_ids.append(rollout_id)
            for rollout_id in expired_ids:
                self.rollout_records.pop(rollout_id, None)

        # 2) Size-based pruning (evict oldest completed first)
        if max_records > 0 and len(self.rollout_records) > max_records:
            completed: List[tuple[str, float]] = [
                (rollout_id, float(record.completed_at))
                for rollout_id, record in self.rollout_records.items()
                if record.completed_at is not None
            ]
            completed.sort(key=lambda x: x[1])

            to_evict = len(self.rollout_records) - max_records
            evict_count = min(to_evict, len(completed))
            for rollout_id, _ts in completed[:evict_count]:
                self.rollout_records.pop(rollout_id, None)

            if len(self.rollout_records) > max_records:
                # Remaining overflow is from in-progress tasks (we do not evict them).
                logger.warning(
                    "rollout_records exceeded max_rollout_records=%s with in-progress tasks=%s. "
                    "Consider increasing MAX_ROLLOUT_RECORDS.",
                    max_records,
                    sum(
                        1
                        for r in self.rollout_records.values()
                        if r.task is not None and not r.task.done()
                    ),
                )

    @property
    def rollout_semaphore(self) -> asyncio.Semaphore:
        if self._rollout_semaphore is None:
            self._rollout_semaphore = asyncio.Semaphore(settings.max_concurrent_rollouts)
        return self._rollout_semaphore

    async def initialize(self) -> None:
        self.http_client = httpx.AsyncClient(
            timeout=settings.http_client_timeout,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )

    async def cleanup(self) -> None:
        if self.http_client is not None:
            await self.http_client.aclose()
            self.http_client = None

    async def get_http_client(self) -> httpx.AsyncClient:
        """Get or lazily create the shared HTTP client."""
        if self.http_client is not None:
            return self.http_client
        async with self._http_client_lock:
            if self.http_client is None:
                self.http_client = httpx.AsyncClient(
                    timeout=settings.http_client_timeout,
                    limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
                )
        # mypy: self.http_client is not None here
        return self.http_client


app_state = AppState()


def start_rollout(request: RolloutRequest, tools: List[ToolSchemaDict]) -> List[ToolSchemaDict]:
    """Start a background rollout task (idempotent by rollout_id).

    Returns the tool list to include in the InitResponse.
    """
    rollout_id = request.rollout_id

    # Opportunistic pruning on new init requests.
    app_state.prune_rollout_records()

    record = app_state.rollout_records.get(rollout_id)
    if record is not None:
        record.last_accessed_at = time.time()
        # Ensure we don't retain a finished task object indefinitely.
        if record.task is not None and record.task.done():
            record.completed_at = record.completed_at or time.time()
            record.task = None
        return record.tools

    tools_value = list(tools)
    created_at = time.time()
    task = asyncio.create_task(_run_rollout_task(request=request, tools=tools_value))

    def _on_done(t: asyncio.Task) -> None:
        try:
            t.result()
        except asyncio.CancelledError:
            logger.info(f"[{rollout_id}] Rollout task cancelled")
        except Exception:
            logger.exception(f"[{rollout_id}] Rollout task crashed")
        finally:
            # Mark record complete and drop task reference to avoid memory retention.
            record = app_state.rollout_records.get(rollout_id)
            if record is not None:
                record.completed_at = record.completed_at or time.time()
                record.task = None
            app_state.prune_rollout_records()

    task.add_done_callback(_on_done)
    app_state.rollout_records[rollout_id] = RolloutRecord(
        tools=tools_value,
        task=task,
        created_at=created_at,
        completed_at=None,
        last_accessed_at=created_at,
    )
    return tools_value


def _auth_headers(api_key: Optional[str]) -> Dict[str, str]:
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def _create_metrics(
    start_time: float,
    llm_latency_ms: float,
    tool_latency_ms: float,
    num_llm_calls: int,
    num_tool_calls: int,
    prompt_tokens: int,
    response_tokens: int,
    max_context_tokens: int,
) -> RolloutMetrics:
    return RolloutMetrics(
        total_latency_ms=(time.time() - start_time) * 1000.0,
        llm_latency_ms=llm_latency_ms,
        tool_latency_ms=tool_latency_ms,
        num_llm_calls=num_llm_calls,
        num_tool_calls=num_tool_calls,
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
        max_context_tokens=max_context_tokens,
    )


def _parse_tool_calls(assistant_message: MessageDict) -> List[ToolCallDict]:
    tool_calls = assistant_message.get("tool_calls") or []
    if isinstance(tool_calls, list):
        return tool_calls
    return []


def _normalize_stop(stop: Any) -> Optional[List[str]]:
    if stop is None:
        return None
    if isinstance(stop, str):
        return [stop]
    if isinstance(stop, list):
        return [str(s) for s in stop]
    return None


async def _post_rollout_completed(
    server_url: str,
    api_key: Optional[str],
    response: RolloutResponse,
) -> None:
    client = await app_state.get_http_client()
    resp = await client.post(
        f"{server_url}/v1/rollout/completed",
        json=response.model_dump(mode="json", exclude_none=True),
        headers=_auth_headers(api_key),
    )
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError:
        logger.error(
            "[%s] /v1/rollout/completed callback failed: status=%s body=%s",
            response.rollout_id,
            resp.status_code,
            resp.text,
        )
        raise


def _categorize_trainer_http_error(error: httpx.HTTPStatusError) -> Dict[str, Any]:
    status_code = error.response.status_code
    if status_code == 429:
        return {"error_category": "rate_limited", "http_status": status_code}
    if status_code >= 500:
        return {"error_category": "trainer_server_error", "http_status": status_code}
    if status_code >= 400:
        return {"error_category": "trainer_client_error", "http_status": status_code}
    return {"error_category": "trainer_error", "http_status": status_code}


async def _call_chat_completions(
    server_url: str,
    api_key: Optional[str],
    rollout_id: str,
    messages: List[MessageDict],
    completion_params: Dict[str, Any],
) -> CompletionsResponse:
    client = await app_state.get_http_client()

    request = CompletionsRequest(
        rollout_id=rollout_id,
        messages=messages,
        temperature=float(completion_params.get("temperature", 1.0)),
        top_p=float(completion_params.get("top_p", 1.0)),
        max_tokens=int(completion_params.get("max_tokens", 512)),
        stop=_normalize_stop(completion_params.get("stop")),
        logprobs=bool(completion_params.get("logprobs", True)),
    )
    payload = request.model_dump(mode="json", exclude_none=True)

    resp = await client.post(
        f"{server_url}/v1/chat/completions",
        json=payload,
        headers=_auth_headers(api_key),
    )
    resp.raise_for_status()
    return CompletionsResponse.model_validate(resp.json())


async def _run_rollout_task(request: RolloutRequest, tools: List[ToolSchemaDict]) -> None:
    """Run a rollout and post completion callback."""
    del tools  # Tools are provided to the trainer via InitResponse; not needed here.

    rollout_id = request.rollout_id
    start_time = time.time()

    llm_latency_ms = 0.0
    tool_latency_ms = 0.0
    num_llm_calls = 0
    num_tool_calls = 0
    prompt_tokens = 0
    response_tokens = 0
    max_context_tokens = 0
    total_response_tokens = 0

    status: RolloutStatus = RolloutStatus.COMPLETED
    finish_reason: Optional[str] = None
    error_message: Optional[str] = None
    extra_fields: Dict[str, Any] = {}
    final_messages: List[MessageDict] = []

    # Copy messages to ensure we only append within this task.
    messages: List[MessageDict] = list(request.messages)

    try:
        async with app_state.rollout_semaphore:
            async with asyncio.timeout(settings.rollout_timeout_seconds):
                for turn in range(request.max_turns):
                    logger.info(f"[{rollout_id}] Turn {turn + 1}/{request.max_turns}")

                    llm_start = time.time()
                    completion = await _call_chat_completions(
                        server_url=request.server_url,
                        api_key=request.api_key,
                        rollout_id=rollout_id,
                        messages=messages,
                        completion_params=request.completion_params,
                    )
                    llm_latency_ms += (time.time() - llm_start) * 1000.0
                    num_llm_calls += 1

                    prompt_len = len(completion.prompt_token_ids)
                    resp_len = len(completion.token_ids)
                    prompt_tokens += prompt_len
                    response_tokens += resp_len
                    max_context_tokens = max(max_context_tokens, prompt_len)
                    total_response_tokens += resp_len

                    assistant_message = completion.choices[0].message
                    messages.append(assistant_message)

                    tool_calls = _parse_tool_calls(assistant_message)
                    if not tool_calls:
                        finish_reason = completion.choices[0].finish_reason or "stop"
                        break

                    tool_start = time.time()
                    tool_results = await execute_calculator_calls(tool_calls)
                    tool_latency_ms += (time.time() - tool_start) * 1000.0
                    num_tool_calls += len(tool_calls)

                    messages.extend(tool_results)

                    if total_response_tokens >= request.max_tokens_total:
                        finish_reason = "max_tokens"
                        break
                else:
                    finish_reason = "max_turns"

                final_messages = messages

    except TimeoutError:
        status = RolloutStatus.ERROR
        error_message = f"Rollout timeout exceeded ({settings.rollout_timeout_seconds}s)"
        extra_fields = {"error_category": "timeout"}

    except httpx.HTTPStatusError as e:
        status = RolloutStatus.ERROR
        extra_fields = _categorize_trainer_http_error(e)
        error_message = "Trainer returned an error"
        logger.error(
            f"[{rollout_id}] Trainer HTTP error: {e.response.status_code} - {e.response.text}"
        )

    except httpx.RequestError as e:
        status = RolloutStatus.ERROR
        error_message = "Network error communicating with trainer"
        extra_fields = {"error_category": "network_error"}
        logger.error(f"[{rollout_id}] Trainer network error: {e}")

    except Exception:
        status = RolloutStatus.ERROR
        error_message = "Internal server error"
        extra_fields = {"error_category": "internal_error"}
        logger.exception(f"[{rollout_id}] Unexpected error during rollout")

    metrics = _create_metrics(
        start_time=start_time,
        llm_latency_ms=llm_latency_ms,
        tool_latency_ms=tool_latency_ms,
        num_llm_calls=num_llm_calls,
        num_tool_calls=num_tool_calls,
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
        max_context_tokens=max_context_tokens,
    )

    rollout_response = RolloutResponse(
        rollout_id=rollout_id,
        status=status,
        final_messages=final_messages if status == RolloutStatus.COMPLETED else [],
        finish_reason=finish_reason if status == RolloutStatus.COMPLETED else None,
        error_message=error_message if status == RolloutStatus.ERROR else None,
        metrics=metrics,
        extra_fields=extra_fields,
    )

    try:
        await _post_rollout_completed(
            server_url=request.server_url,
            api_key=request.api_key,
            response=rollout_response,
        )
        logger.info(f"[{rollout_id}] Posted rollout completion: status={status.value}")
    except Exception:
        logger.exception(f"[{rollout_id}] Failed to post rollout completion callback")
