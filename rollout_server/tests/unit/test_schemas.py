"""Unit tests for schemas validation.

These tests verify that the schemas accept various ID formats while still
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
    ToolsResponse,
    ToolDefinition,
    ToolFunction,
)


class TestRolloutIdValidation:
    """Test rollout_id validation for various formats."""

    def test_standard_uuid_format(self):
        """Test that standard UUID format is accepted."""
        request = RolloutRequest(
            rollout_id="550e8400-e29b-41d4-a716-446655440000",
            server_url="http://trainer:8081",
            messages=[Message(role="user", content="Hello")],
            sampling_params=SamplingParams(),
        )
        assert request.rollout_id == "550e8400-e29b-41d4-a716-446655440000"

    def test_job_context_format(self):
        """Test that job context format is accepted.

        Example format: "{job_id}-step{step}-idx{index}-{uuid[:8]}"
        """
        job_id = "job123-step0-idx0-abc12345"
        request = RolloutRequest(
            rollout_id=job_id,
            server_url="http://trainer:8081",
            messages=[Message(role="user", content="Hello")],
            sampling_params=SamplingParams(),
        )
        assert request.rollout_id == job_id

    def test_hex_only_format(self):
        """Test that hex-only format is accepted (no job context)."""
        # When no job_id is provided, uuid4().hex can be used
        hex_id = "a1b2c3d4e5f67890a1b2c3d4e5f67890"
        request = RolloutRequest(
            rollout_id=hex_id,
            server_url="http://trainer:8081",
            messages=[Message(role="user", content="Hello")],
            sampling_params=SamplingParams(),
        )
        assert request.rollout_id == hex_id

    def test_complex_job_id(self):
        """Test format with complex job IDs."""
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
    """Test RolloutMetrics structure."""

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

    def test_localhost_url_accepted(self):
        """Test that localhost URLs are accepted (internal training infrastructure)."""
        request = RolloutRequest(
            rollout_id="test",
            server_url="http://localhost:8081",
            messages=[Message(role="user", content="Hello")],
            sampling_params=SamplingParams(),
        )
        assert request.server_url == "http://localhost:8081"

    def test_private_ip_accepted(self):
        """Test that private IP ranges are accepted (trainers run on private networks)."""
        private_ips = [
            ("http://10.0.0.1:8081", "http://10.0.0.1:8081"),
            ("http://192.168.1.1:8081", "http://192.168.1.1:8081"),
            ("http://172.16.0.1:8081", "http://172.16.0.1:8081"),
        ]
        for url, expected in private_ips:
            request = RolloutRequest(
                rollout_id="test",
                server_url=url,
                messages=[Message(role="user", content="Hello")],
                sampling_params=SamplingParams(),
            )
            assert request.server_url == expected


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


class TestMessageRoleValidation:
    """Test Message role validation."""

    def test_valid_roles_accepted(self):
        """Test that all valid roles are accepted."""
        valid_roles = ["system", "user", "assistant", "tool", "function"]
        for role in valid_roles:
            msg = Message(role=role, content="test")
            assert msg.role == role

    def test_invalid_role_rejected(self):
        """Test that invalid roles are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Message(role="invalid_role", content="test")
        assert "Invalid role" in str(exc_info.value)

    def test_empty_role_rejected(self):
        """Test that empty role is rejected."""
        with pytest.raises(ValidationError):
            Message(role="", content="test")

    def test_message_with_tool_calls(self):
        """Test Message with tool_calls structure."""
        msg = Message(
            role="assistant",
            content="Let me calculate that.",
            tool_calls=[
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "add",
                        "arguments": '{"a": 1, "b": 2}'
                    }
                }
            ]
        )
        assert msg.role == "assistant"
        assert len(msg.tool_calls) == 1

    def test_tool_message_with_tool_call_id(self):
        """Test tool message with tool_call_id."""
        msg = Message(
            role="tool",
            content="3",
            tool_call_id="call_123"
        )
        assert msg.role == "tool"
        assert msg.tool_call_id == "call_123"

    def test_message_content_can_be_none(self):
        """Test that content can be None (for some assistant messages)."""
        msg = Message(role="assistant", content=None)
        assert msg.content is None

    def test_message_content_can_be_list(self):
        """Test that content can be a list (multimodal content)."""
        msg = Message(
            role="user",
            content=[
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
            ]
        )
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2


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


class TestToolsResponse:
    """Test ToolsResponse and related schemas for GET /tools endpoint."""

    def test_empty_tools_response(self):
        """Test ToolsResponse with empty tools list."""
        response = ToolsResponse(tools=[])
        assert response.tools == []

    def test_tools_response_with_single_tool(self):
        """Test ToolsResponse with a single tool definition."""
        tool = ToolDefinition(
            type="function",
            function=ToolFunction(
                name="add",
                description="Add two numbers",
                parameters={
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"}
                    },
                    "required": ["a", "b"]
                }
            )
        )
        response = ToolsResponse(tools=[tool])
        assert len(response.tools) == 1
        assert response.tools[0].type == "function"
        assert response.tools[0].function.name == "add"

    def test_tools_response_with_multiple_tools(self):
        """Test ToolsResponse with multiple tool definitions."""
        tools = [
            ToolDefinition(
                type="function",
                function=ToolFunction(
                    name="add",
                    description="Add two numbers"
                )
            ),
            ToolDefinition(
                type="function",
                function=ToolFunction(
                    name="subtract",
                    description="Subtract two numbers"
                )
            ),
        ]
        response = ToolsResponse(tools=tools)
        assert len(response.tools) == 2
        assert response.tools[0].function.name == "add"
        assert response.tools[1].function.name == "subtract"

    def test_tool_function_minimal(self):
        """Test ToolFunction with only required field (name)."""
        func = ToolFunction(name="test_tool")
        assert func.name == "test_tool"
        assert func.description is None
        assert func.parameters is None

    def test_tool_function_with_all_fields(self):
        """Test ToolFunction with all fields populated."""
        func = ToolFunction(
            name="calculate",
            description="Perform a calculation",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        )
        assert func.name == "calculate"
        assert func.description == "Perform a calculation"
        assert func.parameters["type"] == "object"

    def test_tool_definition_default_type(self):
        """Test that ToolDefinition defaults to type='function'."""
        tool = ToolDefinition(
            function=ToolFunction(name="test")
        )
        assert tool.type == "function"

    def test_tools_response_from_dict(self):
        """Test ToolsResponse can be created from raw dict (API response parsing)."""
        raw_response = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "add",
                        "description": "Add two numbers",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "number"},
                                "b": {"type": "number"}
                            },
                            "required": ["a", "b"]
                        }
                    }
                }
            ]
        }
        response = ToolsResponse.model_validate(raw_response)
        assert len(response.tools) == 1
        assert response.tools[0].function.name == "add"

    def test_tools_response_to_dict(self):
        """Test ToolsResponse serialization to dict (API response generation)."""
        response = ToolsResponse(
            tools=[
                ToolDefinition(
                    type="function",
                    function=ToolFunction(
                        name="multiply",
                        description="Multiply two numbers",
                        parameters={
                            "type": "object",
                            "properties": {
                                "a": {"type": "number"},
                                "b": {"type": "number"}
                            },
                            "required": ["a", "b"]
                        }
                    )
                )
            ]
        )
        result = response.model_dump()
        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["type"] == "function"
        assert result["tools"][0]["function"]["name"] == "multiply"

