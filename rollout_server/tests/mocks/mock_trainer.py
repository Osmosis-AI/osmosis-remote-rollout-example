"""Mock trainer server for testing RolloutServer independently.

This mock trainer implements the /v1/chat/completions endpoint that RolloutServer
expects to call back to. It provides canned responses for testing without
requiring a real GPU-backed training cluster.

Usage:
    cd rollout_server
    python -m tests.mocks.mock_trainer
"""

import re
import time
import uuid
from typing import List, Optional, Tuple

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


def parse_user_request(messages: List[Message]) -> Tuple[float, float, float]:
    """Parse the first user request to extract addends and multiplier."""
    user_msg = next((m for m in messages if m.role == "user"), None)
    if user_msg is None or not isinstance(user_msg.content, str):
        return 0.0, 0.0, 1.0

    text = user_msg.content.lower()
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    a = float(numbers[0]) if len(numbers) > 0 else 0.0
    b = float(numbers[1]) if len(numbers) > 1 else 0.0
    multiplier = float(numbers[2]) if len(numbers) > 2 else 2.0

    return a, b, multiplier


def find_prev_assistant(messages: List[Message]) -> Optional[Message]:
    """Find the last assistant message before the final message."""
    for msg in reversed(messages[:-1]):
        if msg.role == "assistant":
            return msg
    return None


def build_add_tool_call(a: float, b: float) -> dict:
    """Create an assistant message that calls the add tool."""
    return {
        "role": "assistant",
        "content": "I'll calculate the sum first.",
        "tool_calls": [
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": "add",
                    "arguments": f'{{"a": {a}, "b": {b}}}'
                }
            }
        ]
    }


def build_multiply_tool_call(sum_result: float, multiplier: float) -> dict:
    """Create an assistant message that calls the multiply tool."""
    return {
        "role": "assistant",
        "content": "Continuing the calculation.",
        "tool_calls": [
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": "multiply",
                    "arguments": f'{{"a": {sum_result}, "b": {multiplier}}}'
                }
            }
        ]
    }


def build_final_answer(a: float, b: float, multiplier: float, final_value: float) -> dict:
    """Create a plain assistant message with the final answer."""
    sum_value = a + b
    return {
        "role": "assistant",
        "content": f"{a} plus {b} equals {sum_value}. Multiplying {sum_value} by {multiplier} gives {final_value}."
    }


@app.post("/v1/chat/completions")
async def completions(request: CompletionsRequest) -> CompletionsResponse:
    """Mock completions endpoint.

    This simulates what the real trainer's /v1/chat/completions endpoint does:
    1. Receives messages and rollout_id from RolloutServer
    2. Generates LLM response (with or without tool calls)
    3. Returns token IDs and logprobs
    """
    print("\n" + "=" * 80)
    print(f"[Mock Trainer] RECEIVED REQUEST")
    print("=" * 80)
    print(f"  rollout_id: {request.rollout_id}")
    print(f"  messages: {len(request.messages)} messages")
    print(f"  response_mask: {request.response_mask}")
    print(f"  sampling_params:")
    print(f"    - temperature: {request.temperature}")
    print(f"    - top_p: {request.top_p}")
    print(f"    - max_tokens: {request.max_tokens}")
    print(f"    - logprobs: {request.logprobs}")
    print(f"\n  Messages detail:")
    for idx, msg in enumerate(request.messages):
        role = msg.role
        content = msg.content if msg.content else '<empty>'
        if len(str(content)) > 100:
            content = str(content)[:100] + "..."
        tool_calls = msg.tool_calls if msg.tool_calls else []
        print(f"    [{idx}] {role.upper()}: {content}")
        if tool_calls:
            for tc in tool_calls:
                print(f"        -> tool_call: {tc.function.name}({tc.function.arguments})")
        if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
            print(f"        -> tool_call_id: {msg.tool_call_id}")
    print("=" * 80)

    last_message = request.messages[-1]
    a, b, multiplier = parse_user_request(request.messages)

    assistant_message: dict

    if last_message.role == "user":
        # First turn - start with add tool call when a calculation is requested
        user_content = last_message.content
        wants_math = isinstance(user_content, str) and any(
            word in user_content.lower() for word in ["calculate", "add", "sum", "plus"]
        )
        if wants_math:
            assistant_message = build_add_tool_call(a, b)
        else:
            assistant_message = {
                "role": "assistant",
                "content": "I'm a mock LLM. I will respond without tools."
            }
    elif last_message.role == "tool":
        prev_assistant = find_prev_assistant(request.messages)
        if prev_assistant and prev_assistant.tool_calls:
            func_name = prev_assistant.tool_calls[0].function.name

            if func_name == "add":
                # After add tool result, call multiply
                try:
                    sum_result = float(last_message.content)
                except (TypeError, ValueError):
                    sum_result = a + b
                assistant_message = build_multiply_tool_call(sum_result, multiplier)
            elif func_name == "multiply":
                # After multiply tool result, return final text answer
                try:
                    final_value = float(last_message.content)
                except (TypeError, ValueError):
                    final_value = (a + b) * multiplier
                assistant_message = build_final_answer(a, b, multiplier, final_value)
            else:
                assistant_message = {
                    "role": "assistant",
                    "content": "I'm a mock LLM. The calculation result should be in the previous message."
                }
        else:
            assistant_message = {
                "role": "assistant",
                "content": "I'm a mock LLM. The calculation result should be in the previous message."
            }
    else:
        # Fallback
        assistant_message = {
            "role": "assistant",
            "content": "I'm a mock LLM. The calculation result should be in the previous message."
        }

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

    # Build response
    completion_response = CompletionsResponse(
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

    # Print detailed response
    print("\n" + "=" * 80)
    print(f"[Mock Trainer] SENDING RESPONSE")
    print("=" * 80)
    print(f"  id: {completion_response.id}")
    print(f"  model: {completion_response.model}")
    print(f"  token_ids: {len(completion_response.token_ids)} tokens")
    print(f"  prompt_token_ids: {len(completion_response.prompt_token_ids)} tokens")
    print(f"  message:")
    print(f"    - role: {completion_response.choices[0].message.role}")
    content = completion_response.choices[0].message.content
    if content and len(content) > 100:
        content = content[:100] + "..."
    print(f"    - content: {content}")
    tool_calls = completion_response.choices[0].message.tool_calls
    if tool_calls:
        print(f"    - tool_calls: {len(tool_calls)}")
        for tc in tool_calls:
            print(f"        -> {tc.function.name}({tc.function.arguments})")
    print("=" * 80 + "\n")

    return completion_response


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
