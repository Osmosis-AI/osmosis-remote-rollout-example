"""Unit tests for schema validation.

These tests verify basic validation behavior for protocol payloads.
"""

import pytest
from pydantic import ValidationError

from rollout_server.schemas import (
    CompletionsRequest,
    InitResponse,
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    RolloutStatus,
)


class TestRolloutIdValidation:
    def _base_request(self, rollout_id: str) -> RolloutRequest:
        return RolloutRequest(
            rollout_id=rollout_id,
            server_url="http://trainer:8081",
            messages=[{"role": "user", "content": "Hello"}],
            completion_params={"temperature": 0.7, "max_tokens": 64, "logprobs": True},
        )

    def test_standard_uuid_format(self):
        req = self._base_request("550e8400-e29b-41d4-a716-446655440000")
        assert req.rollout_id == "550e8400-e29b-41d4-a716-446655440000"

    def test_job_context_format(self):
        job_id = "job123-step0-idx0-abc12345"
        req = self._base_request(job_id)
        assert req.rollout_id == job_id

    def test_hex_only_format(self):
        hex_id = "a1b2c3d4e5f67890a1b2c3d4e5f67890"
        req = self._base_request(hex_id)
        assert req.rollout_id == hex_id

    def test_empty_rollout_id_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            self._base_request("")
        assert "rollout_id must be non-empty" in str(exc_info.value)

    def test_whitespace_only_rollout_id_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            self._base_request("   ")
        assert "rollout_id must be non-empty" in str(exc_info.value)

    def test_too_long_rollout_id_rejected(self):
        long_id = "x" * 257
        with pytest.raises(ValidationError) as exc_info:
            self._base_request(long_id)
        assert "256 characters" in str(exc_info.value)


class TestServerUrlValidation:
    def test_http_url_accepted(self):
        req = RolloutRequest(
            rollout_id="test",
            server_url="http://trainer:8081",
            messages=[{"role": "user", "content": "Hello"}],
            completion_params={},
        )
        assert req.server_url == "http://trainer:8081"

    def test_https_url_accepted(self):
        req = RolloutRequest(
            rollout_id="test",
            server_url="https://trainer.example.com:8081",
            messages=[{"role": "user", "content": "Hello"}],
            completion_params={},
        )
        assert req.server_url == "https://trainer.example.com:8081"

    def test_trailing_slash_removed(self):
        req = RolloutRequest(
            rollout_id="test",
            server_url="http://trainer:8081/",
            messages=[{"role": "user", "content": "Hello"}],
            completion_params={},
        )
        assert req.server_url == "http://trainer:8081"

    def test_invalid_scheme_rejected(self):
        with pytest.raises(ValidationError):
            RolloutRequest(
                rollout_id="test",
                server_url="ftp://trainer:8081",
                messages=[{"role": "user", "content": "Hello"}],
                completion_params={},
            )


class TestInitResponse:
    def test_init_response_defaults(self):
        resp = InitResponse(rollout_id="r")
        assert resp.rollout_id == "r"
        assert resp.tools == []


class TestRolloutMetrics:
    def test_metrics_default_values(self):
        metrics = RolloutMetrics()
        assert metrics.total_latency_ms == 0.0
        assert metrics.llm_latency_ms == 0.0
        assert metrics.tool_latency_ms == 0.0
        assert metrics.num_llm_calls == 0
        assert metrics.num_tool_calls == 0
        assert metrics.prompt_tokens == 0
        assert metrics.response_tokens == 0
        assert metrics.max_context_tokens == 0


class TestRolloutResponse:
    def test_rollout_response_includes_metrics(self):
        metrics = RolloutMetrics(total_latency_ms=1000.0, llm_latency_ms=800.0, num_llm_calls=2)
        resp = RolloutResponse(
            rollout_id="test-id",
            status=RolloutStatus.COMPLETED,
            final_messages=[{"role": "assistant", "content": "Done"}],
            finish_reason="stop",
            metrics=metrics,
        )
        assert resp.metrics is not None
        assert resp.metrics.total_latency_ms == 1000.0
        assert resp.status == RolloutStatus.COMPLETED


class TestCompletionsRequest:
    def test_completions_request_minimal(self):
        req = CompletionsRequest(
            rollout_id="r",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert req.rollout_id == "r"
        assert req.model == "default"
