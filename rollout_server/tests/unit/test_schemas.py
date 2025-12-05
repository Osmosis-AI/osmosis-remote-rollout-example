"""Unit tests for schemas validation - specifically traingate compatibility.

These tests verify that the schemas accept traingate's format while still
providing proper validation.
"""

import pytest
from pydantic import ValidationError

from rollout_server.schemas import (
    RolloutRequest,
    RolloutResponse,
    RolloutStatus,
    RolloutMetrics,
    Message,
    SamplingParams,
    CompletionsRequest,
)


class TestRolloutIdValidation:
    """Test rollout_id validation for traingate compatibility."""

    def test_standard_uuid_format(self):
        """Test that standard UUID format is accepted."""
        request = RolloutRequest(
            rollout_id="550e8400-e29b-41d4-a716-446655440000",
            server_url="http://trainer:8081",
            messages=[Message(role="user", content="Hello")],
            sampling_params=SamplingParams(),
        )
        assert request.rollout_id == "550e8400-e29b-41d4-a716-446655440000"

    def test_traingate_format_with_job_context(self):
        """Test that traingate's job context format is accepted.

        traingate generates rollout_id like: "{job_id}-step{step}-idx{index}-{uuid[:8]}"
        """
        traingate_id = "job123-step0-idx0-abc12345"
        request = RolloutRequest(
            rollout_id=traingate_id,
            server_url="http://trainer:8081",
            messages=[Message(role="user", content="Hello")],
            sampling_params=SamplingParams(),
        )
        assert request.rollout_id == traingate_id

    def test_traingate_format_hex_only(self):
        """Test that traingate's hex-only format is accepted (no job context)."""
        # When no job_id is provided, traingate uses uuid4().hex
        hex_id = "a1b2c3d4e5f67890a1b2c3d4e5f67890"
        request = RolloutRequest(
            rollout_id=hex_id,
            server_url="http://trainer:8081",
            messages=[Message(role="user", content="Hello")],
            sampling_params=SamplingParams(),
        )
        assert request.rollout_id == hex_id

    def test_traingate_complex_job_id(self):
        """Test traingate format with complex job IDs."""
        complex_id = "experiment-v2-gpu4x-step100-idx5-deadbeef"
        request = RolloutRequest(
            rollout_id=complex_id,
            server_url="http://trainer:8081",
            messages=[Message(role="user", content="Hello")],
            sampling_params=SamplingParams(),
        )
        assert request.rollout_id == complex_id

    def test_empty_rollout_id_rejected(self):
        """Test that empty rollout_id is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            RolloutRequest(
                rollout_id="",
                server_url="http://trainer:8081",
                messages=[Message(role="user", content="Hello")],
                sampling_params=SamplingParams(),
            )
        assert "rollout_id must be non-empty" in str(exc_info.value)

    def test_whitespace_only_rollout_id_rejected(self):
        """Test that whitespace-only rollout_id is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            RolloutRequest(
                rollout_id="   ",
                server_url="http://trainer:8081",
                messages=[Message(role="user", content="Hello")],
                sampling_params=SamplingParams(),
            )
        assert "rollout_id must be non-empty" in str(exc_info.value)

    def test_too_long_rollout_id_rejected(self):
        """Test that overly long rollout_id is rejected (DoS prevention)."""
        long_id = "x" * 257
        with pytest.raises(ValidationError) as exc_info:
            RolloutRequest(
                rollout_id=long_id,
                server_url="http://trainer:8081",
                messages=[Message(role="user", content="Hello")],
                sampling_params=SamplingParams(),
            )
        assert "256 characters" in str(exc_info.value)

    def test_max_length_rollout_id_accepted(self):
        """Test that max-length rollout_id (256 chars) is accepted."""
        max_id = "x" * 256
        request = RolloutRequest(
            rollout_id=max_id,
            server_url="http://trainer:8081",
            messages=[Message(role="user", content="Hello")],
            sampling_params=SamplingParams(),
        )
        assert len(request.rollout_id) == 256


class TestRolloutMetrics:
    """Test RolloutMetrics structure matches traingate expectations."""

    def test_metrics_default_values(self):
        """Test that RolloutMetrics has correct default values."""
        metrics = RolloutMetrics()
        assert metrics.total_latency_ms == 0.0
        assert metrics.llm_latency_ms == 0.0
        assert metrics.tool_latency_ms == 0.0
        assert metrics.num_llm_calls == 0
        assert metrics.num_tool_calls == 0
        assert metrics.prompt_tokens == 0
        assert metrics.response_tokens == 0
        assert metrics.max_context_tokens == 0

    def test_metrics_with_values(self):
        """Test RolloutMetrics with actual values."""
        metrics = RolloutMetrics(
            total_latency_ms=1523.4,
            llm_latency_ms=1200.0,
            tool_latency_ms=323.4,
            num_llm_calls=3,
            num_tool_calls=2,
            prompt_tokens=100,
            response_tokens=50,
            max_context_tokens=150,
        )
        assert metrics.total_latency_ms == 1523.4
        assert metrics.num_llm_calls == 3

    def test_metrics_in_rollout_response(self):
        """Test that RolloutResponse correctly includes metrics."""
        metrics = RolloutMetrics(
            total_latency_ms=1000.0,
            llm_latency_ms=800.0,
            num_llm_calls=2,
        )
        response = RolloutResponse(
            rollout_id="test-id",
            status=RolloutStatus.COMPLETED,
            final_messages=[Message(role="assistant", content="Done")],
            finish_reason="stop",
            metrics=metrics,
        )
        assert response.metrics is not None
        assert response.metrics.total_latency_ms == 1000.0
        assert response.metrics.llm_latency_ms == 800.0


class TestServerUrlValidation:
    """Test server_url validation."""

    def test_http_url_accepted(self):
        """Test that HTTP URLs are accepted."""
        request = RolloutRequest(
            rollout_id="test",
            server_url="http://trainer:8081",
            messages=[Message(role="user", content="Hello")],
            sampling_params=SamplingParams(),
        )
        assert request.server_url == "http://trainer:8081"

    def test_https_url_accepted(self):
        """Test that HTTPS URLs are accepted."""
        request = RolloutRequest(
            rollout_id="test",
            server_url="https://trainer.example.com:8081",
            messages=[Message(role="user", content="Hello")],
            sampling_params=SamplingParams(),
        )
        assert request.server_url == "https://trainer.example.com:8081"

    def test_trailing_slash_removed(self):
        """Test that trailing slashes are removed for consistency."""
        request = RolloutRequest(
            rollout_id="test",
            server_url="http://trainer:8081/",
            messages=[Message(role="user", content="Hello")],
            sampling_params=SamplingParams(),
        )
        assert request.server_url == "http://trainer:8081"

    def test_non_http_url_rejected(self):
        """Test that non-HTTP URLs are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            RolloutRequest(
                rollout_id="test",
                server_url="ftp://trainer:8081",
                messages=[Message(role="user", content="Hello")],
                sampling_params=SamplingParams(),
            )
        assert "HTTP/HTTPS URL" in str(exc_info.value)


class TestMessagesValidation:
    """Test messages validation."""

    def test_empty_messages_rejected(self):
        """Test that empty messages array is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            RolloutRequest(
                rollout_id="test",
                server_url="http://trainer:8081",
                messages=[],
                sampling_params=SamplingParams(),
            )
        # Should fail min_length=1 validation
        assert "at least 1" in str(exc_info.value).lower() or "min" in str(exc_info.value).lower()


class TestCompletionsRequestResponseMask:
    """Test response_mask validation in CompletionsRequest."""

    def test_response_mask_none_accepted(self):
        """Test that None response_mask is accepted (first turn)."""
        request = CompletionsRequest(
            rollout_id="test",
            messages=[Message(role="user", content="Hello")],
            response_mask=None,
        )
        assert request.response_mask is None

    def test_response_mask_zeros_accepted(self):
        """Test that all-zeros response_mask is accepted (tool outputs)."""
        request = CompletionsRequest(
            rollout_id="test",
            messages=[Message(role="user", content="Hello")],
            response_mask=[0, 0, 0, 0, 0],
        )
        assert request.response_mask == [0, 0, 0, 0, 0]

    def test_response_mask_ones_accepted(self):
        """Test that all-ones response_mask is accepted (LLM tokens)."""
        request = CompletionsRequest(
            rollout_id="test",
            messages=[Message(role="user", content="Hello")],
            response_mask=[1, 1, 1],
        )
        assert request.response_mask == [1, 1, 1]

    def test_response_mask_mixed_accepted(self):
        """Test that mixed 0/1 response_mask is accepted."""
        request = CompletionsRequest(
            rollout_id="test",
            messages=[Message(role="user", content="Hello")],
            response_mask=[0, 0, 1, 1, 0, 1],
        )
        assert request.response_mask == [0, 0, 1, 1, 0, 1]

    def test_response_mask_invalid_values_rejected(self):
        """Test that response_mask with values other than 0/1 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            CompletionsRequest(
                rollout_id="test",
                messages=[Message(role="user", content="Hello")],
                response_mask=[0, 1, 2],
            )
        assert "0 or 1" in str(exc_info.value)

    def test_response_mask_negative_rejected(self):
        """Test that negative values in response_mask are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            CompletionsRequest(
                rollout_id="test",
                messages=[Message(role="user", content="Hello")],
                response_mask=[0, -1, 1],
            )
        assert "0 or 1" in str(exc_info.value)

