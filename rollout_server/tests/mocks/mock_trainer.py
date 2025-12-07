"""Mock trainer server for testing RolloutServer independently.

This mock trainer implements the /v1/chat/completions endpoint that RolloutServer
expects to call back to. It provides canned responses for testing without
requiring a real GPU-backed training cluster.

Usage:
    cd rollout_server
    python -m tests.mocks.mock_trainer
"""

import time
import uuid
from typing import List

from fastapi import FastAPI
from transformers import AutoTokenizer
import uvicorn

from rollout_server.schemas import CompletionsRequest, CompletionsResponse, CompletionsChoice, Message


app = FastAPI(title="Mock Trainer Server")


# Initialize a tokenizer for generating mock token IDs
# Using Qwen3-8B to match production training environment
# In real usage, this would match the actual model being trained
# Note: Qwen3 requires trust_remote_code=True for custom tokenizer code
TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
if TOKENIZER.pad_token is None:
    TOKENIZER.pad_token = TOKENIZER.eos_token


def generate_mock_response(messages: List[Message], use_tools: bool = False) -> dict:
    """Generate a mock LLM response.

    Args:
        messages: Conversation history
        use_tools: If True, generate a response with tool calls

    Returns:
        Mock assistant message with optional tool calls
    """
    if use_tools:
        # Generate response with calculator tool call
        response_text = "I'll calculate that for you."
        assistant_message = {
            "role": "assistant",
            "content": response_text,
            "tool_calls": [
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": "add",
                        "arguments": '{"a": 5, "b": 3}'
                    }
                }
            ]
        }
    else:
        # Generate simple text response without tools
        response_text = "I'm a mock LLM. The calculation result should be in the previous message."
        assistant_message = {
            "role": "assistant",
            "content": response_text
        }

    return assistant_message


@app.post("/v1/chat/completions")
async def completions(request: CompletionsRequest) -> CompletionsResponse:
    """Mock completions endpoint.

    This simulates what the real trainer's /v1/chat/completions endpoint does:
    1. Receives messages and rollout_id from RolloutServer
    2. Generates LLM response (with or without tool calls)
    3. Returns token IDs and logprobs
    """
    print(f"[Mock Trainer] Received request for rollout_id={request.rollout_id}")
    print(f"[Mock Trainer] Messages count: {len(request.messages)}")
    print(f"[Mock Trainer] Response mask: {request.response_mask}")

    # Determine if we should use tools based on conversation state
    # First call: use tools if user asks for calculation
    # Second call (after tool execution): respond normally
    last_message = request.messages[-1]
    use_tools = False

    if last_message.role == "user":
        # First turn - check if user asks for calculation
        user_content = last_message.content
        if isinstance(user_content, str) and any(word in user_content.lower()
                                                  for word in ["calculate", "add", "sum", "plus"]):
            use_tools = True

    # Generate mock response
    assistant_message = generate_mock_response(request.messages, use_tools=use_tools)

    # Tokenize response for token IDs
    response_text = assistant_message["content"]
    response_token_ids = TOKENIZER.encode(response_text, add_special_tokens=False)

    # Generate mock logprobs (all 0.0 for simplicity)
    response_logprobs = [0.0] * len(response_token_ids)

    # Tokenize full prompt for verification
    prompt_text = TOKENIZER.apply_chat_template(
        [msg.model_dump() for msg in request.messages],
        tokenize=False,
        add_generation_prompt=True
    )
    prompt_token_ids = TOKENIZER.encode(prompt_text, add_special_tokens=True)

    print(f"[Mock Trainer] Generated response: {response_text[:100]}...")
    print(f"[Mock Trainer] Response tokens: {len(response_token_ids)}")
    print(f"[Mock Trainer] Tool calls: {assistant_message.get('tool_calls', [])}")

    # Build response
    return CompletionsResponse(
        id=f"mock-{uuid.uuid4().hex[:8]}",
        object="chat.completion",
        created=int(time.time()),
        model="mock-qwen3-8b",
        choices=[
            CompletionsChoice(
                index=0,
                message=Message(**assistant_message),
                finish_reason="stop"
            )
        ],
        token_ids=response_token_ids,
        logprobs=response_logprobs,
        prompt_token_ids=prompt_token_ids
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "mock-trainer"}


if __name__ == "__main__":
    import os

    port = int(os.getenv("MOCK_TRAINER_PORT", "9001"))

    print("=" * 60)
    print(f"Starting Mock Trainer Server on port {port}")
    print("This server simulates the trainer's /v1/chat/completions endpoint")
    print("for testing RolloutServer independently.")
    print("=" * 60)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
