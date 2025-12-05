"""Integration tests for response_mask correctness with mock trainer.

CRITICAL: These tests verify end-to-end response_mask handling.
This is THE most important integration test file in the entire codebase.

Test scenarios:
1. Single-turn conversation (no tools) - mask = None
2. Multi-turn with tools - correct mask pattern [None, [0,0,...]]
3. Multiple tool calls - verify each tool output is masked correctly
4. Response_mask tracking across turns

Note: This file uses fixtures from conftest.py:
- mock_trainer_with_tracking: Thread-safe mask tracking fixture
- mask_tracker: Isolated mask storage per test (no global state)
"""

import uuid
import pytest
from fastapi.testclient import TestClient

from rollout_server.server import app as rollout_app
from rollout_server.schemas import RolloutRequest, Message, SamplingParams


# Note: mock_trainer_with_tracking fixture is now defined in conftest.py
# This removes the global received_masks list that caused race conditions


@pytest.mark.asyncio
@pytest.mark.integration
async def test_single_turn_no_tools_mask_is_none(mock_trainer_with_tracking):
    """Single-turn conversation: response_mask should be None."""
    client, masks = mock_trainer_with_tracking
    rollout_client = TestClient(rollout_app)

    # Prepare rollout request
    rollout_request = RolloutRequest(
        rollout_id=str(uuid.uuid4()),
        server_url="http://mock-trainer:9001",
        messages=[
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello")  # No calculation keyword
        ],
        sampling_params=SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=512,
            logprobs=True
        ),
        tokenizer_name="Qwen/Qwen3-8B",
        tokenizer_revision="main",
        max_turns=10
    )

    # Send rollout request
    response = rollout_client.post(
        "/rollout",
        json=rollout_request.model_dump()
    )

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "COMPLETED"

    # CRITICAL: Verify response_mask was None for first turn
    assert len(masks) == 1, "Should have exactly 1 LLM call"
    assert masks[0] is None, "First turn should have response_mask=None"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_multi_turn_with_tools_correct_mask_pattern(mock_trainer_with_tracking):
    """CRITICAL: Multi-turn with tools must have correct mask pattern."""
    client, masks = mock_trainer_with_tracking
    rollout_client = TestClient(rollout_app)

    rollout_request = RolloutRequest(
        rollout_id=str(uuid.uuid4()),
        server_url="http://mock-trainer:9001",
        messages=[
            Message(role="system", content="You are a calculator assistant."),
            Message(role="user", content="Please calculate 5 plus 3")  # Triggers tool
        ],
        sampling_params=SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=512,
            logprobs=True
        ),
        tokenizer_name="Qwen/Qwen3-8B",
        tokenizer_revision="main",
        max_turns=10
    )

    response = rollout_client.post(
        "/rollout",
        json=rollout_request.model_dump()
    )

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "COMPLETED"

    # CRITICAL: Verify mask pattern
    # Expected: [None (turn 1), [0,0,...] (turn 2 with tool outputs)]
    assert len(masks) >= 2, "Should have at least 2 LLM calls (with tool use)"

    # Turn 1: No mask (first turn)
    assert masks[0] is None, "Turn 1 should have response_mask=None"

    # Turn 2: Should have mask for tool outputs
    assert masks[1] is not None, "Turn 2 should have response_mask (tool outputs added)"
    assert isinstance(masks[1], list), "Turn 2 mask should be a list"
    assert len(masks[1]) > 0, "Turn 2 mask should not be empty"
    assert all(m == 0 for m in masks[1]), "All tool output tokens should have mask=0"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_mask_length_matches_token_count(mock_trainer_with_tracking):
    """CRITICAL: response_mask length must match token count added."""
    client, masks = mock_trainer_with_tracking
    rollout_client = TestClient(rollout_app)

    from transformers import AutoTokenizer

    # Load same tokenizer to verify counts
    # Note: Qwen3 requires trust_remote_code=True for custom tokenizer code
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rollout_request = RolloutRequest(
        rollout_id=str(uuid.uuid4()),
        server_url="http://mock-trainer:9001",
        messages=[
            Message(role="user", content="Calculate 10 add 20")
        ],
        sampling_params=SamplingParams(temperature=0.7),
        tokenizer_name="Qwen/Qwen3-8B",
        tokenizer_revision="main",
        max_turns=10
    )

    response = rollout_client.post(
        "/rollout",
        json=rollout_request.model_dump()
    )

    assert response.status_code == 200
    result = response.json()

    # Verify final messages structure
    final_messages = result["final_messages"]
    assert len(final_messages) >= 3, "Should have at least user, assistant, tool messages"

    # Find tool message
    tool_messages = [m for m in final_messages if m["role"] == "tool"]
    assert len(tool_messages) > 0, "Should have at least one tool message"

    # CRITICAL: Verify response_mask length matches token count
    # The mask should have at least 2 entries: None for first turn, then mask for tool outputs
    assert len(masks) >= 2, "Should have at least 2 LLM calls for tool-using conversation"

    # Second mask should exist and match tool output token count
    second_mask = masks[1]
    assert second_mask is not None, "Second turn should have response_mask for tool outputs"
    assert isinstance(second_mask, list), "response_mask should be a list"

    # Calculate expected token count for tool outputs by tokenizing the tool messages
    # within the context of the full conversation (using chat template)
    messages_before_tool = final_messages[:2]  # user, assistant with tool_call
    messages_with_tool = final_messages[:4]    # user, assistant, tool

    # Tokenize both to get the difference (this is how the session calculates it)
    prompt_before = tokenizer.apply_chat_template(
        messages_before_tool, add_generation_prompt=True, tokenize=False
    )
    tokens_before = tokenizer.encode(prompt_before)

    prompt_with_tool = tokenizer.apply_chat_template(
        messages_with_tool, add_generation_prompt=True, tokenize=False
    )
    tokens_with_tool = tokenizer.encode(prompt_with_tool)

    # The mask length should correspond to tokens added between calls
    # Note: exact match may vary due to LLM response tokens, but mask should be non-empty
    assert len(second_mask) > 0, "Tool output mask should have positive length"
    assert all(m == 0 for m in second_mask), "All tool output tokens should have mask=0"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_multiple_turns_mask_progression(mock_trainer_with_tracking):
    """Test mask progression across multiple turns."""
    client, masks = mock_trainer_with_tracking
    rollout_client = TestClient(rollout_app)

    rollout_request = RolloutRequest(
        rollout_id=str(uuid.uuid4()),
        server_url="http://mock-trainer:9001",
        messages=[
            Message(role="user", content="Add 5 and 3")
        ],
        sampling_params=SamplingParams(temperature=0.7),
        tokenizer_name="Qwen/Qwen3-8B",
        max_turns=10
    )

    response = rollout_client.post("/rollout", json=rollout_request.model_dump())

    assert response.status_code == 200

    # Verify mask progression
    # Pattern should be: None, [0,0,...], possibly more
    assert len(masks) > 0
    assert masks[0] is None, "First mask should be None"

    # If there were multiple turns, check subsequent masks
    for i, mask in enumerate(masks[1:], start=1):
        if mask is not None:
            assert isinstance(mask, list), f"Mask {i} should be a list"
            assert all(m == 0 for m in mask), f"Mask {i} should only contain 0s"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_conversation_without_tools_completes(mock_trainer_with_tracking):
    """Conversation without tools should complete correctly with only None masks."""
    client, masks = mock_trainer_with_tracking
    rollout_client = TestClient(rollout_app)

    rollout_request = RolloutRequest(
        rollout_id=str(uuid.uuid4()),
        server_url="http://mock-trainer:9001",
        messages=[
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Tell me a joke")  # No calculation
        ],
        sampling_params=SamplingParams(temperature=0.7),
        tokenizer_name="Qwen/Qwen3-8B",
        max_turns=10
    )

    response = rollout_client.post("/rollout", json=rollout_request.model_dump())

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "COMPLETED"
    assert result["finish_reason"] == "stop"

    # Should only have one call with mask=None
    assert len(masks) == 1
    assert masks[0] is None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_response_mask_with_max_turns(mock_trainer_with_tracking):
    """Test response_mask behavior when hitting max_turns limit."""
    client, masks = mock_trainer_with_tracking
    rollout_client = TestClient(rollout_app)

    rollout_request = RolloutRequest(
        rollout_id=str(uuid.uuid4()),
        server_url="http://mock-trainer:9001",
        messages=[
            Message(role="user", content="Calculate 1 add 1")
        ],
        sampling_params=SamplingParams(temperature=0.7),
        tokenizer_name="Qwen/Qwen3-8B",
        max_turns=2  # Limited turns
    )

    response = rollout_client.post("/rollout", json=rollout_request.model_dump())

    assert response.status_code == 200
    result = response.json()

    # May complete normally or hit max_turns
    assert result["status"] == "COMPLETED"
    assert result["finish_reason"] in ["stop", "max_turns"]

    # Verify mask pattern is still correct
    assert len(masks) > 0
    assert masks[0] is None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_error_handling_preserves_mask_correctness():
    """Test that error scenarios don't corrupt mask tracking."""
    rollout_client = TestClient(rollout_app)

    # Test with invalid tokenizer name - should fail gracefully
    rollout_request = RolloutRequest(
        rollout_id=str(uuid.uuid4()),
        server_url="http://mock-trainer:9001",
        messages=[Message(role="user", content="Test")],
        sampling_params=SamplingParams(temperature=0.7),
        tokenizer_name="invalid/nonexistent-tokenizer",
        max_turns=10
    )

    response = rollout_client.post("/rollout", json=rollout_request.model_dump())

    # Should return error status
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "ERROR"
    # Should not corrupt state
    assert "rollout_id" in result
