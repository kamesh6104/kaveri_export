
"""The one LLM client every part of the assistant uses (midstone 4).

Save as src/llm_client.py

Any OpenAI-compatible provider works:
OpenRouter, Ollama, Groq, Gemini.
The provider is configured through .env:
LLM_BASE_URL, LLM_API_KEY, LLM_MODEL.
"""

from __future__ import annotations

import json
import os
import random
import time
import uuid

from dataclasses import dataclass, field
from typing import Any, Callable

import openai
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


# Retry configuration
MAX_RETRIES = 5
BASE_WAIT_S = 1.0
MAX_WAIT_S = 30.0


# --------------------------------------------------
# 1. REPLY CLASS
# --------------------------------------------------

@dataclass
class Reply:
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    model: str = ""
    streamed: bool = False
    chunks: int = 0
    latency_ms: int = 0


# --------------------------------------------------
# 2. RUN LOG CLASS
# --------------------------------------------------

class RunLog:
    """Save chatbot events into a JSONL file."""

    def __init__(self, path: str | None = None):
        self.path = path

    def write(self, **event) -> None:

        if not self.path:
            return

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    event,
                    ensure_ascii=False,
                    default=str
                ) + "\n"
            )


# --------------------------------------------------
# 3. LLM CLIENT CLASS
# --------------------------------------------------

class LLMClient:

    def __init__(
        self,
        log: RunLog | None = None,
        max_retries: int = MAX_RETRIES
    ):

        self.base_url = os.getenv("LLM_BASE_URL")
        self.model = os.getenv("LLM_MODEL")

        self._client = openai.OpenAI(
            base_url=self.base_url,
            api_key=os.getenv("LLM_API_KEY"),
            max_retries=0,
            timeout=float(os.getenv("LLM_TIMEOUT", "300"))
        )

        # API request method
        self.create = self._client.chat.completions.create

        self.log = log or RunLog()
        self.max_retries = max_retries

    # --------------------------------------------------
    # 4. API CALL WITH RETRY
    # --------------------------------------------------

    def _call(self, **kwargs):
        """Retry API requests when HTTP 429 occurs."""

        for attempt in range(self.max_retries + 1):

            try:

                return self.create(
                    model=self.model,
                    **kwargs
                )

            except openai.RateLimitError as err:

                if attempt == self.max_retries:
                    raise

                # Check Retry-After header
                retry_after = None

                if err.response is not None:
                    retry_after = err.response.headers.get(
                        "retry-after"
                    )

                try:
                    if retry_after is not None:
                        wait = min(
                            MAX_WAIT_S,
                            max(0.0, float(retry_after))
                        )
                    else:
                        # Exponential backoff + random jitter
                        backoff = min(
                            MAX_WAIT_S,
                            BASE_WAIT_S * (2 ** attempt)
                        )

                        jitter = random.uniform(
                            0,
                            min(1.0, backoff * 0.1)
                        )

                        wait = min(
                            MAX_WAIT_S,
                            backoff + jitter
                        )

                except (TypeError, ValueError):
                    backoff = min(
                        MAX_WAIT_S,
                        BASE_WAIT_S * (2 ** attempt)
                    )
                    jitter = random.uniform(0, min(1.0, backoff * 0.1))
                    wait = min(MAX_WAIT_S, backoff + jitter)

                self.log.write(
                    event="retry",
                    status=429,
                    attempt=attempt + 1,
                    wait_s=round(wait, 2)
                )

                time.sleep(wait)

    # --------------------------------------------------
    # 5. CHAT FUNCTION
    # --------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        stream: bool = True,
        on_token: Callable[[str], None] | None = None
    ) -> Reply:

        started = time.perf_counter()

        extra = {"tools": tools} if tools else {}

        reply = Reply(
            streamed=stream,
            model=self.model
        )

        # ----------------------------------------------
        # NON-STREAMING RESPONSE
        # ----------------------------------------------

        if not stream:

            resp = self._call(
                messages=messages,
                **extra
            )

            msg = resp.choices[0].message

            reply.content = msg.content or ""

            reply.tool_calls = [
                _tool_call(
                    tc.id,
                    tc.function.name,
                    tc.function.arguments
                )
                for tc in (msg.tool_calls or [])
            ]

        # ----------------------------------------------
        # STREAMING RESPONSE
        # ----------------------------------------------

        else:

            pieces: dict[int, dict] = {}

            text = []

            for chunk in self._call(
                messages=messages,
                stream=True,
                **extra
            ):

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # --------------------------------------
                # RECEIVE TEXT CHUNKS
                # --------------------------------------

                if delta.content:

                    reply.chunks += 1

                    text.append(delta.content)

                    if on_token:
                        on_token(delta.content)

                # --------------------------------------
                # RECEIVE TOOL CALL CHUNKS
                # --------------------------------------

                for tc in delta.tool_calls or []:

                    slot = pieces.setdefault(
                        tc.index,
                        {
                            "id": "",
                            "name": "",
                            "arguments": ""
                        }
                    )

                    # Combine streamed tool-call fragments
                    if tc.id:
                        slot["id"] += tc.id

                    if tc.function.name:
                        slot["name"] += tc.function.name

                    if tc.function.arguments:
                        slot["arguments"] += tc.function.arguments

            # Combine all text chunks
            reply.content = "".join(text)

            # Convert tool calls into dictionaries
            reply.tool_calls = [
                _tool_call(
                    p["id"],
                    p["name"],
                    p["arguments"]
                )
                for _i, p in sorted(pieces.items())
            ]

        # ----------------------------------------------
        # CALCULATE RESPONSE TIME
        # ----------------------------------------------

        reply.latency_ms = int(
            (time.perf_counter() - started) * 1000
        )

        return reply


# --------------------------------------------------
# 6. TOOL CALL CONVERSION
# --------------------------------------------------

def _tool_call(
    call_id: str,
    name: str,
    arguments: str | dict
) -> dict:

    if isinstance(arguments, str):

        arguments = json.loads(
            arguments or "{}"
        )

    return {
        "id": call_id or f"call_{uuid.uuid4().hex[:8]}",
        "name": name,
        "arguments": arguments
    }


# --------------------------------------------------
# 7. CONVERSATION CLASS
# --------------------------------------------------

class Conversation:
    """Chat history plus tool calling. One instance per conversation."""

    def __init__(
        self,
        client: LLMClient,
        system: str | None = None,
        tools: list[dict] | None = None,
        functions: dict[str, Callable[..., Any]] | None = None,
        conversation_id: str = "c1"
    ):

        self.client = client

        self.history: list[dict] = (
            [{"role": "system", "content": system}]
            if system else []
        )

        self.tools = tools

        self.functions = functions or {}

        self.id = conversation_id

        self.turn = 0

    # --------------------------------------------------
    # 8. ASK FUNCTION
    # --------------------------------------------------

    def ask(
        self,
        text: str,
        on_token: Callable[[str], None] | None = None
    ) -> str:

        self.turn += 1

        log = self.client.log

        # Add user message to history
        self.history.append({
            "role": "user",
            "content": text
        })

        # Log user message
        log.write(
            conversation_id=self.id,
            turn=self.turn,
            role="user",
            content=text
        )

        # Allow up to 4 model response rounds
        for _round in range(4):

            # Send conversation to AI
            reply = self.client.chat(
                self.history,
                tools=self.tools,
                on_token=on_token
            )

            # Prepare assistant log
            event = dict(
                conversation_id=self.id,
                turn=self.turn,
                role="assistant",
                content=reply.content,
                streamed=reply.streamed,
                chunks=reply.chunks,
                model=reply.model,
                latency_ms=reply.latency_ms
            )

            # ------------------------------------------
            # NORMAL AI RESPONSE
            # ------------------------------------------

            if not reply.tool_calls:

                # Add AI response to history
                self.history.append({
                    "role": "assistant",
                    "content": reply.content
                })

                # Save AI response in log
                log.write(**event)

                # Return final answer
                return reply.content

            # ------------------------------------------
            # AI REQUESTED A TOOL
            # ------------------------------------------

            log.write(
                **event,
                tool_calls=reply.tool_calls
            )

            # Add assistant tool request to history
            self.history.append({
                "role": "assistant",
                "content": reply.content or None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(
                                tc["arguments"]
                            )
                        }
                    }
                    for tc in reply.tool_calls
                ]
            })

            # ------------------------------------------
            # EXECUTE PYTHON FUNCTIONS
            # ------------------------------------------

            for tc in reply.tool_calls:

                name = tc["name"]

                if name not in self.functions:
                    raise ValueError(
                        f"Unknown tool requested: {name}"
                    )

                # Execute the registered Python function
                result = self.functions[name](
                    **tc["arguments"]
                )

                # Convert result into string
                result_text = str(result)

                # Add tool result to conversation history
                self.history.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_text
                })

                # Log tool result
                log.write(
                    conversation_id=self.id,
                    turn=self.turn,
                    role="tool",
                    tool_call_id=tc["id"],
                    name=name,
                    content=result_text
                )

        # Too many tool calls without a final answer
        raise RuntimeError(
            "The model kept calling tools without answering."
        )


# --------------------------------------------------
# 9. TOOL SPECIFICATION
# --------------------------------------------------

def tool_spec(
    name: str,
    description: str,
    properties: dict,
    required: list[str]
) -> dict:

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }
