"""Unit tests for RolloutSession response_mask calculation.

CRITICAL: These tests verify the CORE functionality of the remote rollout protocol.
Incorrect response_mask corrupts training data and breaks PPO training.

Test coverage:
1. First turn (no tools) - mask should be None
2. Second turn (after tools) - mask should be [0] * num_tool_tokens
3. Multi-turn conversations - mask pattern verification
4. Negative token count detection (context truncation)
5. Token count alignment verification
6. Zero new tokens edge case
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from rollout_server.session import RolloutSession
from rollout_server.schemas import CompletionsResponse, CompletionsChoice, Message


# Note: tokenizer fixture is provided by conftest.py (session-scoped for performance)
# It uses Qwen/Qwen3-8B with trust_remote_code controlled by TEST_TRUST_REMOTE_CODE env var


@pytest.fixture
def mock_http_client():
    """Mock httpx.AsyncClient for testing."""
    client = AsyncMock()
    # Mock raise_for_status to do nothing
    client.post.return_value.raise_for_status = Mock()
    return client


def create_mock_response(
    message_content: str,
    tool_calls=None,
    token_ids=None,
    prompt_token_ids=None
):
    """Helper to create mock CompletionsResponse."""
    if token_ids is None:
        token_ids = [100, 101, 102]  # Default 3 tokens

    if prompt_token_ids is None:
        prompt_token_ids = [1, 2, 3, 4]

    message = {"role": "assistant", "content": message_content}
    if tool_calls:
        message["tool_calls"] = tool_calls

    response_data = {
        "id": "test-response",
        "object": "chat.completion",
        "created": 123456,
        "model": "mock-model",
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "stop"
        }],
        "token_ids": token_ids,
        "logprobs": [0.0] * len(token_ids),
        "prompt_token_ids": prompt_token_ids
    }

    mock_response = Mock()
    mock_response.json.return_value = response_data
    mock_response.raise_for_status = Mock()
    return mock_response


class TestResponseMaskFirstTurn:
    """Test response_mask calculation for first turn (no tools)."""

    @pytest.mark.asyncio
    async def test_first_turn_mask_is_none(self, tokenizer, mock_http_client):
        """First turn should have response_mask=None (no previous tool outputs)."""
        session = RolloutSession(
            rollout_id="test-first-turn",
            tokenizer=tokenizer,
            server_url="http://mock-trainer:9001",
            http_client=mock_http_client
        )

        # Initialize with user message
        session.messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"}
        ]

        # Mock LLM response
        mock_http_client.post.return_value = create_mock_response(
            "Hi there!",
            token_ids=[100, 101, 102]
        )

        # Call LLM
        sampling_params = {"temperature": 1.0, "top_p": 1.0, "max_tokens": 512}
        await session.call_llm(sampling_params)

        # Verify request
        call_args = mock_http_client.post.call_args
        request_json = call_args.kwargs['json']

        # CRITICAL: First turn should have response_mask=None
        assert request_json['response_mask'] is None, \
            "First turn should have response_mask=None"

        # Verify last_prompt_length updated
        assert session.last_prompt_length > 0, \
            "last_prompt_length should be updated after first turn"
        assert session.turn_count == 1, \
            "turn_count should be 1 after first call"

    @pytest.mark.asyncio
    async def test_first_turn_with_empty_messages(self, tokenizer, mock_http_client):
        """First turn with just user message should work."""
        session = RolloutSession(
            rollout_id="test-empty",
            tokenizer=tokenizer,
            server_url="http://mock-trainer:9001",
            http_client=mock_http_client
        )

        session.messages = [{"role": "user", "content": "Test"}]

        mock_http_client.post.return_value = create_mock_response("Response")

        await session.call_llm({"temperature": 1.0})

        call_args = mock_http_client.post.call_args
        request_json = call_args.kwargs['json']

        assert request_json['response_mask'] is None


class TestResponseMaskMultiTurn:
    """Test response_mask calculation for multi-turn conversations with tools."""

    @pytest.mark.asyncio
    async def test_second_turn_after_tool_execution(self, tokenizer, mock_http_client):
        """Second turn (after tool execution) should have response_mask=[0]*N."""
        session = RolloutSession(
            rollout_id="test-multi-turn",
            tokenizer=tokenizer,
            server_url="http://mock-trainer:9001",
            http_client=mock_http_client
        )

        # Turn 1: User asks for calculation
        session.messages = [
            {"role": "user", "content": "Calculate 5 + 3"}
        ]

        # Mock Turn 1 LLM response (with tool call)
        mock_http_client.post.return_value = create_mock_response(
            "I'll calculate that.",
            tool_calls=[{
                "id": "call_123",
                "type": "function",
                "function": {"name": "add", "arguments": '{"a":5,"b":3}'}
            }],
            token_ids=[100, 101, 102, 103]  # 4 tokens
        )

        # Turn 1: Call LLM
        await session.call_llm({"temperature": 1.0})
        turn1_last_prompt_length = session.last_prompt_length

        # Append assistant message
        assistant_msg = mock_http_client.post.return_value.json()['choices'][0]['message']
        session.append_assistant_message(assistant_msg)

        # Append tool result
        tool_result = {
            "role": "tool",
            "content": "8",
            "tool_call_id": "call_123"
        }
        session.append_tool_outputs([tool_result])

        # Calculate expected tool token count
        messages_before_turn2 = session.messages.copy()
        tool_tokens = tokenizer.apply_chat_template(
            messages_before_turn2,
            add_generation_prompt=True,
            tokenize=True
        )
        expected_tool_token_count = len(tool_tokens) - turn1_last_prompt_length

        # Mock Turn 2 LLM response (after tool)
        mock_http_client.post.return_value = create_mock_response(
            "The answer is 8.",
            token_ids=[200, 201, 202]
        )

        # Turn 2: Call LLM
        await session.call_llm({"temperature": 1.0})

        # Verify Turn 2 request had correct response_mask
        # Get the second call (index 1)
        assert mock_http_client.post.call_count == 2
        turn2_call_args = mock_http_client.post.call_args_list[1]
        turn2_request_json = turn2_call_args.kwargs['json']

        # CRITICAL: response_mask should be [0] * num_tool_tokens
        assert turn2_request_json['response_mask'] is not None, \
            "Second turn should have response_mask (tool outputs added)"

        actual_mask = turn2_request_json['response_mask']
        assert len(actual_mask) == expected_tool_token_count, \
            f"Mask length mismatch: expected {expected_tool_token_count}, got {len(actual_mask)}"

        assert all(m == 0 for m in actual_mask), \
            "All tool output tokens should have mask=0"

    @pytest.mark.asyncio
    async def test_multiple_tool_outputs(self, tokenizer, mock_http_client):
        """Multiple tool outputs should all be masked."""
        session = RolloutSession(
            rollout_id="test-multi-tools",
            tokenizer=tokenizer,
            server_url="http://mock-trainer:9001",
            http_client=mock_http_client
        )

        session.messages = [{"role": "user", "content": "Calculate"}]

        # Turn 1: LLM with multiple tool calls
        mock_http_client.post.return_value = create_mock_response(
            "Calculating...",
            tool_calls=[
                {"id": "call_1", "type": "function", "function": {"name": "add", "arguments": '{"a":1,"b":2}'}},
                {"id": "call_2", "type": "function", "function": {"name": "multiply", "arguments": '{"a":3,"b":4}'}}
            ]
        )

        await session.call_llm({"temperature": 1.0})
        turn1_last_prompt = session.last_prompt_length

        # Add assistant message and tool results
        session.append_assistant_message({
            "role": "assistant",
            "content": "Calculating...",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "add", "arguments": '{"a":1,"b":2}'}},
                {"id": "call_2", "type": "function", "function": {"name": "multiply", "arguments": '{"a":3,"b":4}'}}
            ]
        })
        session.append_tool_outputs([
            {"role": "tool", "content": "3", "tool_call_id": "call_1"},
            {"role": "tool", "content": "12", "tool_call_id": "call_2"}
        ])

        # Turn 2: Calculate expected mask length
        expected_tokens = tokenizer.apply_chat_template(
            session.messages, add_generation_prompt=True, tokenize=True
        )
        expected_mask_length = len(expected_tokens) - turn1_last_prompt

        mock_http_client.post.return_value = create_mock_response("Done")

        await session.call_llm({"temperature": 1.0})

        turn2_call = mock_http_client.post.call_args_list[1]
        mask = turn2_call.kwargs['json']['response_mask']

        assert mask is not None
        assert len(mask) == expected_mask_length
        assert all(m == 0 for m in mask)


class TestResponseMaskErrorCases:
    """Test error handling in response_mask calculation."""

    @pytest.mark.asyncio
    async def test_negative_token_count_raises_error(self, tokenizer, mock_http_client):
        """Negative token count should raise ValueError (context truncation detected)."""
        session = RolloutSession(
            rollout_id="test-negative-tokens",
            tokenizer=tokenizer,
            server_url="http://mock-trainer:9001",
            http_client=mock_http_client
        )

        # Initialize session state with longer message
        session.messages = [{"role": "user", "content": "This is a long message"}]
        session.last_prompt_length = 100  # Simulate previous state

        # Replace with shorter message (simulate context truncation)
        session.messages = [{"role": "user", "content": "Hi"}]

        # CRITICAL: Should raise ValueError for negative token count
        with pytest.raises(ValueError) as exc_info:
            await session.call_llm({"temperature": 1.0})

        # Verify error message is informative
        error_msg = str(exc_info.value)
        assert "Negative token count detected" in error_msg
        assert "test-negative-tokens" in error_msg
        assert "context truncation" in error_msg.lower()

    @pytest.mark.asyncio
    async def test_zero_new_tokens_returns_none_mask(self, tokenizer, mock_http_client):
        """Zero new tokens should result in response_mask=None."""
        session = RolloutSession(
            rollout_id="test-zero-tokens",
            tokenizer=tokenizer,
            server_url="http://mock-trainer:9001",
            http_client=mock_http_client
        )

        # Set up scenario where num_new_tokens = 0
        # This is unusual but theoretically possible
        session.messages = [{"role": "user", "content": "Test"}]

        # First call to establish last_prompt_length
        mock_http_client.post.return_value = create_mock_response("Response1")
        await session.call_llm({"temperature": 1.0})

        saved_last_prompt = session.last_prompt_length

        # Manually set last_prompt_length to match current (creates zero delta)
        current_tokens = tokenizer.apply_chat_template(
            session.messages, add_generation_prompt=True, tokenize=True
        )
        session.last_prompt_length = len(current_tokens)

        # Second call
        mock_http_client.post.return_value = create_mock_response("Response2")
        await session.call_llm({"temperature": 1.0})

        # Check second call had mask=None
        turn2_call = mock_http_client.post.call_args_list[1]
        mask = turn2_call.kwargs['json']['response_mask']

        assert mask is None, "Zero new tokens should result in mask=None"


class TestResponseMaskTokenAlignment:
    """Test token count alignment between RolloutServer and trainer."""

    @pytest.mark.asyncio
    async def test_last_prompt_length_tracking(self, tokenizer, mock_http_client):
        """Verify last_prompt_length is correctly updated."""
        session = RolloutSession(
            rollout_id="test-tracking",
            tokenizer=tokenizer,
            server_url="http://mock-trainer:9001",
            http_client=mock_http_client
        )

        session.messages = [{"role": "user", "content": "Test"}]

        # Calculate expected prompt length before call
        expected_prompt = tokenizer.apply_chat_template(
            session.messages, add_generation_prompt=True, tokenize=True
        )
        expected_initial_length = len(expected_prompt)

        # Mock response with 5 tokens
        response_tokens = [10, 11, 12, 13, 14]
        mock_http_client.post.return_value = create_mock_response(
            "Response",
            token_ids=response_tokens
        )

        # Initial state
        assert session.last_prompt_length == 0

        # Call LLM
        await session.call_llm({"temperature": 1.0})

        # Verify last_prompt_length = prompt + response
        expected_total = expected_initial_length + len(response_tokens)
        assert session.last_prompt_length == expected_total, \
            f"Expected {expected_total}, got {session.last_prompt_length}"

    @pytest.mark.asyncio
    async def test_turn_count_increments(self, tokenizer, mock_http_client):
        """Verify turn_count increments correctly."""
        session = RolloutSession(
            rollout_id="test-turn-count",
            tokenizer=tokenizer,
            server_url="http://mock-trainer:9001",
            http_client=mock_http_client
        )

        session.messages = [{"role": "user", "content": "Test"}]
        mock_http_client.post.return_value = create_mock_response("Response")

        assert session.turn_count == 0

        await session.call_llm({"temperature": 1.0})
        assert session.turn_count == 1

        session.append_tool_outputs([{"role": "tool", "content": "result"}])
        await session.call_llm({"temperature": 1.0})
        assert session.turn_count == 2


class TestResponseMaskEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_tool_output(self, tokenizer, mock_http_client):
        """Empty tool output content should still create valid mask."""
        session = RolloutSession(
            rollout_id="test-empty-tool",
            tokenizer=tokenizer,
            server_url="http://mock-trainer:9001",
            http_client=mock_http_client
        )

        session.messages = [{"role": "user", "content": "Test"}]

        # Turn 1
        mock_http_client.post.return_value = create_mock_response("Response1")
        await session.call_llm({"temperature": 1.0})
        turn1_last = session.last_prompt_length

        # Add empty tool output
        session.append_assistant_message({"role": "assistant", "content": "Response1"})
        session.append_tool_outputs([
            {"role": "tool", "content": "", "tool_call_id": "call_1"}  # Empty content
        ])

        # Turn 2
        expected_tokens = tokenizer.apply_chat_template(
            session.messages, add_generation_prompt=True, tokenize=True
        )
        expected_mask_len = len(expected_tokens) - turn1_last

        mock_http_client.post.return_value = create_mock_response("Response2")
        await session.call_llm({"temperature": 1.0})

        turn2_call = mock_http_client.post.call_args_list[1]
        mask = turn2_call.kwargs['json']['response_mask']

        # Should still have a mask (even if small)
        assert mask is not None
        assert len(mask) == expected_mask_len

    @pytest.mark.asyncio
    async def test_large_tool_output(self, tokenizer, mock_http_client):
        """Large tool output should create appropriately sized mask."""
        session = RolloutSession(
            rollout_id="test-large-tool",
            tokenizer=tokenizer,
            server_url="http://mock-trainer:9001",
            http_client=mock_http_client
        )

        session.messages = [{"role": "user", "content": "Test"}]

        # Turn 1
        mock_http_client.post.return_value = create_mock_response("Response1")
        await session.call_llm({"temperature": 1.0})
        turn1_last = session.last_prompt_length

        # Add large tool output
        large_content = "Result: " + "x" * 1000  # Large output
        session.append_assistant_message({"role": "assistant", "content": "Response1"})
        session.append_tool_outputs([
            {"role": "tool", "content": large_content, "tool_call_id": "call_1"}
        ])

        # Turn 2
        expected_tokens = tokenizer.apply_chat_template(
            session.messages, add_generation_prompt=True, tokenize=True
        )
        expected_mask_len = len(expected_tokens) - turn1_last

        mock_http_client.post.return_value = create_mock_response("Response2")
        await session.call_llm({"temperature": 1.0})

        turn2_call = mock_http_client.post.call_args_list[1]
        mask = turn2_call.kwargs['json']['response_mask']

        assert mask is not None
        assert len(mask) == expected_mask_len
        assert len(mask) > 0  # Should be non-empty
        assert all(m == 0 for m in mask)


class TestSessionLifecycle:
    """Test session initialization and cleanup."""

    @pytest.mark.asyncio
    async def test_session_close_owns_client(self, tokenizer):
        """Session should close HTTP client if it created it."""
        # Session creates its own client
        session = RolloutSession(
            rollout_id="test-close-own",
            tokenizer=tokenizer,
            server_url="http://mock:9001"
        )

        assert session._owns_client is True
        assert session.http_client is not None

        # Close should work
        await session.close()

    @pytest.mark.asyncio
    async def test_session_close_shared_client(self, tokenizer, mock_http_client):
        """Session should NOT close HTTP client if it was provided."""
        # Session uses provided client
        session = RolloutSession(
            rollout_id="test-close-shared",
            tokenizer=tokenizer,
            server_url="http://mock:9001",
            http_client=mock_http_client
        )

        assert session._owns_client is False
        assert session.http_client is mock_http_client

        # Close should not close the client
        await session.close()
        mock_http_client.aclose.assert_not_called()

    def test_session_initialization(self, tokenizer):
        """Test session initializes with correct defaults."""
        session = RolloutSession(
            rollout_id="test-init",
            tokenizer=tokenizer,
            server_url="http://mock:9001"
        )

        assert session.rollout_id == "test-init"
        assert session.tokenizer is tokenizer
        assert session.server_url == "http://mock:9001"
        assert session.messages == []
        assert session.last_prompt_length == 0
        assert session.turn_count == 0
        assert session.callback_api_key is None

    def test_session_with_api_key(self, tokenizer):
        """Test session with callback API key."""
        session = RolloutSession(
            rollout_id="test-api-key",
            tokenizer=tokenizer,
            server_url="http://mock:9001",
            callback_api_key="secret-key-123"
        )

        assert session.callback_api_key == "secret-key-123"


class TestAppendMethods:
    """Test message appending methods."""

    def test_append_tool_outputs(self, tokenizer):
        """Test append_tool_outputs adds messages correctly."""
        session = RolloutSession(
            rollout_id="test-append",
            tokenizer=tokenizer,
            server_url="http://mock:9001"
        )

        session.messages = [{"role": "user", "content": "Test"}]
        initial_count = len(session.messages)

        tool_results = [
            {"role": "tool", "content": "result1", "tool_call_id": "call_1"},
            {"role": "tool", "content": "result2", "tool_call_id": "call_2"}
        ]

        session.append_tool_outputs(tool_results)

        assert len(session.messages) == initial_count + 2
        assert session.messages[-2] == tool_results[0]
        assert session.messages[-1] == tool_results[1]

    def test_append_assistant_message(self, tokenizer):
        """Test append_assistant_message adds message correctly."""
        session = RolloutSession(
            rollout_id="test-append-assistant",
            tokenizer=tokenizer,
            server_url="http://mock:9001"
        )

        session.messages = []
        assistant_msg = {"role": "assistant", "content": "Response"}

        session.append_assistant_message(assistant_msg)

        assert len(session.messages) == 1
        assert session.messages[0] == assistant_msg
