"""Handler for Anthropic Messages API requests.

Converts Anthropic requests to OpenAI ChatCompletion format, delegates to
OpenAIServingChat for processing, and converts responses back to Anthropic format.
"""

from __future__ import annotations

import asyncio
import json
import os
import logging
import random
import time
import uuid
from typing import TYPE_CHECKING, AsyncGenerator, Optional, Union

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from sglang.srt.entrypoints.anthropic.protocol import (
    AnthropicContentBlock,
    AnthropicCountTokensRequest,
    AnthropicCountTokensResponse,
    AnthropicDelta,
    AnthropicError,
    AnthropicErrorResponse,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicStreamEvent,
    AnthropicUsage,
)
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    StreamOptions,
    Tool,
    ToolChoice,
    ToolChoiceFuncName,
)
from sglang.srt.observability.req_time_stats import monotonic_time

if TYPE_CHECKING:
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

logger = logging.getLogger(__name__)

# Map OpenAI finish reasons to Anthropic stop reasons
STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}

# Minimum max_tokens for Anthropic /v1/messages requests.
# Qwen3.6 is a thinking model that emits <think>...</think> before content.
# If max_tokens is too small, thinking consumes all tokens and no content
# is produced. A minimum of 4096 provides headroom for typical thinking
# (~300-800 tokens) plus content (~100-500 tokens).
# Operator can override via SGLANG_ANTHROPIC_MIN_MAX_TOKENS env var.
ANTHROPIC_MIN_MAX_TOKENS = int(os.environ.get("SGLANG_ANTHROPIC_MIN_MAX_TOKENS", "4096"))

# Fallback text injected when a thinking model emits only a <think>...</think>
# block followed by EOS, with no text/tool_use content. Without this fallback
# the Anthropic response carries only a thinking block, and clients that
# follow Anthropic's strict content contract (e.g. Claude Code) silently
# drop the turn — surfacing as a stalled or empty continuation with no
# error. Qwen3.6 exhibits this intermittently when </think> is generated
# very early and <|im_end|> follows immediately.
ANTHROPIC_EMPTY_TEXT_FALLBACK = os.environ.get(
    "SGLANG_ANTHROPIC_EMPTY_TEXT_FALLBACK",
    "(model produced a thinking block with no further content)",
)

# Server-side retry knobs for thinking-only responses.
#
# When the model emits only a <think>...</think> block with no text/tool_use,
# we have two options before falling back to the placeholder text:
#
#   1. Re-issue the same request to the local backend with a fresh seed and a
#      small temperature bump. Qwen3.6's short-thinking-stop is intermittent
#      and largely seed-sensitive; a single retry usually produces a healthy
#      thinking + content turn.
#   2. Inject the static placeholder (patch #21, v10 behavior) so strict
#      clients don't drop the turn.
#
# Path (1) is the primary fix; path (2) remains as a safety net if the retry
# also produces a thinking-only response, errors out, or exceeds the budget.
# Both knobs let an operator turn the retry off (RETRY_ENABLED=0) or tighten
# the timeout without rebuilding the image.
ANTHROPIC_THINKING_RETRY_ENABLED = os.environ.get(
    "SGLANG_ANTHROPIC_THINKING_RETRY_ENABLED", "1"
).lower() not in ("0", "false", "no", "off")
ANTHROPIC_THINKING_RETRY_TEMP_BUMP = float(
    os.environ.get("SGLANG_ANTHROPIC_THINKING_RETRY_TEMP_BUMP", "0.1")
)
ANTHROPIC_THINKING_RETRY_TIMEOUT_S = float(
    os.environ.get("SGLANG_ANTHROPIC_THINKING_RETRY_TIMEOUT_S", "60")
)
# Hard upper bound on the perturbed temperature so a misconfigured base
# temperature can't push the retry into degenerate sampling territory.
ANTHROPIC_THINKING_RETRY_TEMP_CEILING = 1.5


def _wrap_sse_event(data: str, event_type: str) -> str:
    """Format an Anthropic SSE event with event type and data lines."""
    return f"event: {event_type}\ndata: {data}\n\n"


class AnthropicServing:
    """Handler for Anthropic Messages API requests.

    Acts as a translation layer between Anthropic's Messages API and SGLang's
    OpenAI-compatible chat completion infrastructure.
    """

    def __init__(self, openai_serving_chat: OpenAIServingChat):
        self.openai_serving_chat = openai_serving_chat

    async def handle_messages(
        self,
        request: AnthropicMessagesRequest,
        raw_request: Request,
    ) -> Union[JSONResponse, StreamingResponse]:
        """Main entry point for /v1/messages endpoint."""
        try:
            chat_request = self._convert_to_chat_completion_request(request)
        except Exception as e:
            logger.exception("Error converting Anthropic request: %s", e)
            return self._error_response(
                status_code=400,
                error_type="invalid_request_error",
                message=str(e),
            )

        if request.stream:
            return await self._handle_streaming(chat_request, request, raw_request)
        else:
            return await self._handle_non_streaming(chat_request, request, raw_request)

    def _convert_to_chat_completion_request(
        self, anthropic_request: AnthropicMessagesRequest
    ) -> ChatCompletionRequest:
        """Convert an Anthropic Messages request to an OpenAI ChatCompletion request."""
        openai_messages = []

        def _convert_anthropic_image_source_to_openai_part(
            source: Optional[dict],
        ) -> Optional[dict]:
            if not isinstance(source, dict):
                return None

            source_type = source.get("type")
            if source_type == "base64":
                media_type = source.get("media_type", "image/png")
                data = source.get("data", "")
                if not data:
                    return None
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{data}",
                    },
                }

            url = source.get("url")
            if url:
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": url,
                    },
                }

            return None

        def _convert_tool_result_content(
            content: Optional[str | list[dict]],
        ) -> tuple[str | list[dict], str]:
            if isinstance(content, list):
                tool_content_parts = []
                tool_text_parts = []

                for item in content:
                    if not isinstance(item, dict):
                        continue

                    item_type = item.get("type")
                    if item_type == "text":
                        text = item.get("text", "")
                        if text:
                            tool_text_parts.append(text)
                            tool_content_parts.append({"type": "text", "text": text})
                    elif item_type == "image":
                        image_part = _convert_anthropic_image_source_to_openai_part(
                            item.get("source")
                        )
                        if image_part is not None:
                            tool_content_parts.append(image_part)
                    elif item_type == "tool_reference":
                        # Anthropic uses `tool_name`; the SGLang chat template
                        # matches on `name`. Translate at the boundary.
                        ref_name = item.get("tool_name") or item.get("name")
                        if ref_name:
                            tool_content_parts.append(
                                {"type": "tool_reference", "name": ref_name}
                            )

                tool_text = "\n".join(tool_text_parts)
                if (
                    len(tool_content_parts) == 1
                    and tool_content_parts[0]["type"] == "text"
                ):
                    return tool_content_parts[0]["text"], tool_text
                if tool_content_parts:
                    return tool_content_parts, tool_text
                return "", tool_text

            tool_text = str(content) if content else ""
            return tool_text, tool_text

        # Add system message if provided
        if anthropic_request.system:
            if isinstance(anthropic_request.system, str):
                openai_messages.append(
                    {"role": "system", "content": anthropic_request.system}
                )
            else:
                system_parts = []
                for block in anthropic_request.system:
                    if block.type == "text" and block.text:
                        system_parts.append(block.text)
                system_text = "\n".join(system_parts)
                openai_messages.append({"role": "system", "content": system_text})

        # Convert messages
        for msg in anthropic_request.messages:
            if isinstance(msg.content, str):
                openai_messages.append({"role": msg.role, "content": msg.content})
                continue

            # Complex content with blocks
            openai_msg = {"role": msg.role}
            content_parts = []
            tool_calls = []
            reasoning_parts: list[str] = []

            for block in msg.content:
                if block.type == "text" and block.text:
                    content_parts.append({"type": "text", "text": block.text})

                elif block.type == "image" and block.source:
                    image_part = _convert_anthropic_image_source_to_openai_part(
                        block.source
                    )
                    if image_part is not None:
                        content_parts.append(image_part)

                elif block.type == "thinking" and block.thinking:
                    # Round-trip extended-thinking blocks back to the underlying
                    # model as `reasoning_content` on the assistant message.
                    # Without this branch, prior-turn reasoning is silently
                    # dropped on multi-turn agent loops, which breaks the
                    # round-trip for thinking models that interleave reasoning
                    # with tool calls (e.g. Qwen3-Coder thinking variants).
                    # The chat template still decides whether to render prior
                    # reasoning into the next prompt (e.g. Qwen3 strips by
                    # default); we just stop discarding the signal here.
                    reasoning_parts.append(block.thinking)

                elif block.type == "redacted_thinking":
                    # Encrypted thinking has no plaintext to forward; skip.
                    continue

                elif block.type == "tool_use":
                    tool_call = {
                        "id": block.id or f"call_{uuid.uuid4().hex}",
                        "type": "function",
                        "function": {
                            "name": block.name or "",
                            "arguments": json.dumps(block.input or {}),
                        },
                    }
                    tool_calls.append(tool_call)

                elif block.type == "tool_result":
                    tool_content, tool_text = _convert_tool_result_content(
                        block.content
                    )

                    # Use tool_use_id (per spec) with fallback to id
                    tool_call_id = block.tool_use_id or block.id or ""

                    # Tool results from user become separate tool messages
                    if msg.role == "user":
                        openai_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": tool_content,
                            }
                        )
                    else:
                        content_parts.append(
                            {
                                "type": "text",
                                "text": f"Tool result: {tool_text}",
                            }
                        )

            # Attach tool calls to assistant messages
            if tool_calls:
                openai_msg["tool_calls"] = tool_calls

            # Attach reasoning content (assistant only). Joining with a blank
            # line preserves block boundaries when an assistant turn carries
            # multiple thinking segments interleaved with tool calls.
            if reasoning_parts and msg.role == "assistant":
                openai_msg["reasoning_content"] = "\n\n".join(reasoning_parts)

            # Attach content
            if content_parts:
                if len(content_parts) == 1 and content_parts[0]["type"] == "text":
                    openai_msg["content"] = content_parts[0]["text"]
                else:
                    openai_msg["content"] = content_parts
            elif not tool_calls and not openai_msg.get("reasoning_content"):
                continue

            openai_messages.append(openai_msg)

        # Enforce minimum max_tokens for thinking models.
        # Without this, Qwen3.6 may consume the entire budget in <think>
        # and produce no content, causing client retries.
        max_tokens = anthropic_request.max_tokens
        if max_tokens is not None and max_tokens < ANTHROPIC_MIN_MAX_TOKENS:
            logger.info(
                "Bumping max_tokens from %d to %d (SGLANG_ANTHROPIC_MIN_MAX_TOKENS) "
                "to ensure room for thinking+content on Qwen3.6",
                max_tokens,
                ANTHROPIC_MIN_MAX_TOKENS,
            )
            max_tokens = ANTHROPIC_MIN_MAX_TOKENS

        # Build ChatCompletionRequest
        request_data = {
            "messages": openai_messages,
            "model": anthropic_request.model,
            "max_tokens": max_tokens,
            "stream": anthropic_request.stream or False,
        }

        if anthropic_request.temperature is not None:
            request_data["temperature"] = anthropic_request.temperature
        if anthropic_request.top_p is not None:
            request_data["top_p"] = anthropic_request.top_p
        if anthropic_request.top_k is not None:
            request_data["top_k"] = anthropic_request.top_k
        if anthropic_request.stop_sequences is not None:
            request_data["stop"] = anthropic_request.stop_sequences

        # Enable usage in stream so we can report it
        if anthropic_request.stream:
            request_data["stream_options"] = StreamOptions(include_usage=True)

        chat_request = ChatCompletionRequest(**request_data)

        # Convert tools. Deferred tools stay in the list with defer_loading=True;
        # the chat template hides them from the initial <tools> block and renders
        # them on demand when a tool_reference block names them.
        if anthropic_request.tools:
            chat_request.tools = [
                Tool(
                    type="function",
                    defer_loading=tool.defer_loading,
                    function={
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.input_schema,
                    },
                )
                for tool in anthropic_request.tools
            ]

        # Convert tool choice
        if anthropic_request.tool_choice is not None:
            tc_type = anthropic_request.tool_choice.type
            if tc_type == "none":
                chat_request.tool_choice = "none"
            elif chat_request.tools:
                if tc_type == "auto":
                    chat_request.tool_choice = "auto"
                elif tc_type == "any":
                    chat_request.tool_choice = "required"
                elif tc_type == "tool":
                    chat_request.tool_choice = ToolChoice(
                        type="function",
                        function=ToolChoiceFuncName(
                            name=anthropic_request.tool_choice.name
                        ),
                    )
        elif chat_request.tools:
            chat_request.tool_choice = "auto"

        return chat_request

    async def _handle_non_streaming(
        self,
        chat_request: ChatCompletionRequest,
        anthropic_request: AnthropicMessagesRequest,
        raw_request: Request,
    ) -> JSONResponse:
        """Handle non-streaming Anthropic request by delegating to OpenAI handler."""
        received_time = monotonic_time()
        received_time_perf = time.perf_counter()

        # Validate
        error_msg = self.openai_serving_chat._validate_request(chat_request)
        if error_msg:
            return self._error_response(
                status_code=400,
                error_type="invalid_request_error",
                message=error_msg,
            )

        try:
            # Convert to internal request
            validation_time = time.perf_counter() - received_time_perf
            adapted_request, processed_request = (
                self.openai_serving_chat._convert_to_internal_request(
                    chat_request, raw_request
                )
            )
            adapted_request.validation_time = validation_time
            adapted_request.received_time = received_time
            adapted_request.received_time_perf = received_time_perf

            # Get response from OpenAI handler
            response = await self.openai_serving_chat._handle_non_streaming_request(
                adapted_request, processed_request, raw_request
            )
        except Exception as e:
            logger.exception("Error processing Anthropic request: %s", e)
            return self._error_response(
                status_code=500,
                error_type="internal_error",
                message="Internal server error",
            )

        # Check for error responses from OpenAI handler
        if not isinstance(response, ChatCompletionResponse):
            # It's an error response (ORJSONResponse)
            return self._error_response(
                status_code=500,
                error_type="internal_error",
                message="Internal processing error",
            )

        # Convert to Anthropic response. If the result would be thinking-only,
        # try a perturbed retry before falling through to the placeholder.
        anthropic_response = self._convert_response(response, inject_fallback=False)

        has_thinking = any(b.type == "thinking" for b in anthropic_response.content)
        has_text_or_tool = any(
            b.type in ("text", "tool_use") for b in anthropic_response.content
        )

        if has_thinking and not has_text_or_tool and ANTHROPIC_THINKING_RETRY_ENABLED:
            retry_resp = await self._retry_thinking_only(chat_request, raw_request)
            if retry_resp is not None:
                retry_text, retry_tool_calls = self._extract_retry_payload(retry_resp)
                if retry_text or retry_tool_calls:
                    logger.info(
                        "Anthropic non-streaming thinking-only retry yielded "
                        "usable content (text=%s, tool_calls=%d); replacing "
                        "placeholder.",
                        "yes" if retry_text else "no",
                        len(retry_tool_calls),
                    )
                    # Append tool_use blocks for any tool_calls the retry
                    # produced, then a text block for any retry text. Keep
                    # the original thinking block already in content.
                    for tc in retry_tool_calls:
                        try:
                            tc_input = json.loads(tc.function.arguments)
                        except (json.JSONDecodeError, TypeError, AttributeError):
                            tc_input = {}
                        anthropic_response.content.append(
                            AnthropicContentBlock(
                                type="tool_use",
                                id=tc.id,
                                name=tc.function.name,
                                input=tc_input,
                            )
                        )
                    if retry_text:
                        anthropic_response.content.append(
                            AnthropicContentBlock(type="text", text=retry_text)
                        )
                    # Update stop_reason / usage to match what we actually
                    # produced.
                    if retry_tool_calls:
                        anthropic_response.stop_reason = "tool_use"
                    if retry_resp.usage is not None and anthropic_response.usage is not None:
                        anthropic_response.usage.input_tokens += (
                            retry_resp.usage.prompt_tokens or 0
                        )
                        anthropic_response.usage.output_tokens += (
                            retry_resp.usage.completion_tokens or 0
                        )
                    has_text_or_tool = True

        # If we still have thinking-only after the optional retry, inject the
        # static placeholder so strict clients don't drop the turn.
        if has_thinking and not has_text_or_tool:
            logger.warning(
                "Anthropic non-streaming response has only a thinking block "
                "and retry produced no usable content; injecting fallback "
                "text block to satisfy strict clients."
            )
            anthropic_response.content.append(
                AnthropicContentBlock(type="text", text=ANTHROPIC_EMPTY_TEXT_FALLBACK)
            )

        # Structural minimum: if we have no content at all, emit empty text.
        if not anthropic_response.content:
            anthropic_response.content.append(
                AnthropicContentBlock(type="text", text="")
            )

        return JSONResponse(content=anthropic_response.model_dump(exclude_none=True))

    async def _handle_streaming(
        self,
        chat_request: ChatCompletionRequest,
        anthropic_request: AnthropicMessagesRequest,
        raw_request: Request,
    ) -> Union[StreamingResponse, JSONResponse]:
        """Handle streaming Anthropic request."""
        received_time = monotonic_time()
        received_time_perf = time.perf_counter()

        # Validate
        error_msg = self.openai_serving_chat._validate_request(chat_request)
        if error_msg:
            return self._error_response(
                status_code=400,
                error_type="invalid_request_error",
                message=error_msg,
            )

        try:
            validation_time = time.perf_counter() - received_time_perf
            adapted_request, processed_request = (
                self.openai_serving_chat._convert_to_internal_request(
                    chat_request, raw_request
                )
            )
            adapted_request.validation_time = validation_time
            adapted_request.received_time = received_time
            adapted_request.received_time_perf = received_time_perf
        except Exception as e:
            logger.exception("Error converting streaming request: %s", e)
            return self._error_response(
                status_code=500,
                error_type="internal_error",
                message="Internal server error",
            )

        return StreamingResponse(
            self._generate_anthropic_stream(
                adapted_request,
                processed_request,
                anthropic_request,
                raw_request,
            ),
            media_type="text/event-stream",
            background=self.openai_serving_chat.tokenizer_manager.create_abort_task(
                adapted_request
            ),
        )

    async def _generate_anthropic_stream(
        self,
        adapted_request,
        processed_request: ChatCompletionRequest,
        anthropic_request: AnthropicMessagesRequest,
        raw_request: Request,
    ) -> AsyncGenerator[str, None]:
        """Convert OpenAI chat stream to Anthropic event stream."""
        openai_stream = self.openai_serving_chat._generate_chat_stream(
            adapted_request, processed_request, raw_request
        )

        # State tracking
        first_chunk = True
        content_block_index = 0
        content_block_open = False
        # Type of the currently-open content block ("text" or "tool_use").
        # Anthropic deltas are typed (text_delta vs input_json_delta) and
        # must match the block type at their index. We track this so we
        # can close + reopen when the underlying OpenAI stream switches
        # between text content and tool_calls within a single turn.
        content_block_type: Optional[str] = None
        # Track what kinds of content we have actually emitted. Strict
        # Anthropic clients (Claude Code) silently drop responses that
        # contain only a `thinking` block — they require at least one
        # text or tool_use block to register the turn. Qwen3.6 sometimes
        # emits <think>...</think><|im_end|> with nothing after the
        # closing think tag; we synthesize a fallback text block at
        # [DONE] when that happens.
        emitted_thinking = False
        emitted_text_or_tool = False
        finish_reason: Optional[str] = None
        usage_info: Optional[dict] = None
        message_id = f"msg_{uuid.uuid4().hex}"
        model = anthropic_request.model

        async for sse_line in openai_stream:
            if not sse_line.startswith("data: "):
                continue

            data_str = sse_line[6:].strip()

            if data_str == "[DONE]":
                # Close any open content block
                if content_block_open:
                    stop_event = AnthropicStreamEvent(
                        type="content_block_stop",
                        index=content_block_index,
                    )
                    yield _wrap_sse_event(
                        stop_event.model_dump_json(exclude_none=True),
                        "content_block_stop",
                    )
                    content_block_open = False
                    content_block_type = None

                # Fallback path: the model emitted only a thinking block
                # (no text, no tool_use). Strict Anthropic clients (Claude
                # Code) silently drop turns without text/tool_use, so we
                # must surface SOMETHING. Try a perturbed retry first
                # (patch #22), then fall through to the v10 placeholder.
                if emitted_thinking and not emitted_text_or_tool:
                    retry_text: Optional[str] = None
                    retry_tool_calls: list = []
                    if ANTHROPIC_THINKING_RETRY_ENABLED:
                        retry_resp = await self._retry_thinking_only(
                            processed_request, raw_request
                        )
                        if retry_resp is not None:
                            retry_text, retry_tool_calls = (
                                self._extract_retry_payload(retry_resp)
                            )
                            # Account retry usage into the totals reported
                            # to the client so observability isn't lying
                            # about the actual cost of this turn.
                            if retry_resp.usage is not None:
                                if usage_info is None:
                                    usage_info = {
                                        "input_tokens": 0,
                                        "output_tokens": 0,
                                    }
                                usage_info["input_tokens"] = (
                                    usage_info.get("input_tokens", 0)
                                    + (retry_resp.usage.prompt_tokens or 0)
                                )
                                usage_info["output_tokens"] = (
                                    usage_info.get("output_tokens", 0)
                                    + (retry_resp.usage.completion_tokens or 0)
                                )

                    if retry_tool_calls or retry_text:
                        logger.info(
                            "Anthropic thinking-only retry yielded usable "
                            "content (text=%s, tool_calls=%d); emitting "
                            "instead of placeholder.",
                            "yes" if retry_text else "no",
                            len(retry_tool_calls),
                        )
                        # Emit tool_use blocks first (matches the order the
                        # backend would have produced), then a text block
                        # for any retry text.
                        for tc in retry_tool_calls:
                            content_block_index += 1
                            try:
                                tc_input = json.loads(tc.function.arguments)
                            except (json.JSONDecodeError, TypeError, AttributeError):
                                tc_input = {}
                            start_event = AnthropicStreamEvent(
                                type="content_block_start",
                                index=content_block_index,
                                content_block=AnthropicContentBlock(
                                    type="tool_use",
                                    id=tc.id or f"toolu_{uuid.uuid4().hex}",
                                    name=tc.function.name,
                                    input={},
                                ),
                            )
                            yield _wrap_sse_event(
                                start_event.model_dump_json(exclude_none=True),
                                "content_block_start",
                            )
                            # Emit arguments as a single input_json_delta so
                            # the SDK reassembles them into the input field.
                            if tc.function.arguments:
                                delta_event = AnthropicStreamEvent(
                                    type="content_block_delta",
                                    index=content_block_index,
                                    delta=AnthropicDelta(
                                        type="input_json_delta",
                                        partial_json=tc.function.arguments,
                                    ),
                                )
                                yield _wrap_sse_event(
                                    delta_event.model_dump_json(exclude_none=True),
                                    "content_block_delta",
                                )
                            stop_event = AnthropicStreamEvent(
                                type="content_block_stop",
                                index=content_block_index,
                            )
                            yield _wrap_sse_event(
                                stop_event.model_dump_json(exclude_none=True),
                                "content_block_stop",
                            )
                            # If we emitted at least one tool_use, the
                            # turn's stop reason should reflect that.
                            finish_reason = "tool_calls"

                        if retry_text:
                            content_block_index += 1
                            start_event = AnthropicStreamEvent(
                                type="content_block_start",
                                index=content_block_index,
                                content_block=AnthropicContentBlock(
                                    type="text", text=""
                                ),
                            )
                            yield _wrap_sse_event(
                                start_event.model_dump_json(exclude_none=True),
                                "content_block_start",
                            )
                            delta_event = AnthropicStreamEvent(
                                type="content_block_delta",
                                index=content_block_index,
                                delta=AnthropicDelta(
                                    type="text_delta",
                                    text=retry_text,
                                ),
                            )
                            yield _wrap_sse_event(
                                delta_event.model_dump_json(exclude_none=True),
                                "content_block_delta",
                            )
                            stop_event = AnthropicStreamEvent(
                                type="content_block_stop",
                                index=content_block_index,
                            )
                            yield _wrap_sse_event(
                                stop_event.model_dump_json(exclude_none=True),
                                "content_block_stop",
                            )
                    else:
                        # Retry disabled, errored, timed out, or also produced
                        # thinking-only. Fall back to the v10 placeholder.
                        logger.warning(
                            "Anthropic stream finished with only a thinking block "
                            "and retry produced no usable content; injecting "
                            "fallback text block to satisfy strict clients."
                        )
                        content_block_index += 1
                        start_event = AnthropicStreamEvent(
                            type="content_block_start",
                            index=content_block_index,
                            content_block=AnthropicContentBlock(type="text", text=""),
                        )
                        yield _wrap_sse_event(
                            start_event.model_dump_json(exclude_none=True),
                            "content_block_start",
                        )
                        delta_event = AnthropicStreamEvent(
                            type="content_block_delta",
                            index=content_block_index,
                            delta=AnthropicDelta(
                                type="text_delta",
                                text=ANTHROPIC_EMPTY_TEXT_FALLBACK,
                            ),
                        )
                        yield _wrap_sse_event(
                            delta_event.model_dump_json(exclude_none=True),
                            "content_block_delta",
                        )
                        stop_event = AnthropicStreamEvent(
                            type="content_block_stop",
                            index=content_block_index,
                        )
                        yield _wrap_sse_event(
                            stop_event.model_dump_json(exclude_none=True),
                            "content_block_stop",
                        )

                # Emit message_delta with stop_reason and usage
                stop_reason = STOP_REASON_MAP.get(finish_reason or "stop", "end_turn")
                delta_event = AnthropicStreamEvent(
                    type="message_delta",
                    delta=AnthropicDelta(stop_reason=stop_reason),
                    usage=AnthropicUsage(
                        input_tokens=(
                            usage_info.get("input_tokens", 0) if usage_info else 0
                        ),
                        output_tokens=(
                            usage_info.get("output_tokens", 0) if usage_info else 0
                        ),
                    ),
                )
                yield _wrap_sse_event(
                    delta_event.model_dump_json(exclude_none=True),
                    "message_delta",
                )

                # Emit message_stop
                stop_msg = AnthropicStreamEvent(type="message_stop")
                yield _wrap_sse_event(
                    stop_msg.model_dump_json(exclude_none=True),
                    "message_stop",
                )
                continue

            # Parse the OpenAI chunk
            try:
                chunk = ChatCompletionStreamResponse.model_validate_json(data_str)
            except Exception:
                logger.debug("Failed to parse stream chunk: %s", data_str)
                error_event = AnthropicStreamEvent(
                    type="error",
                    error=AnthropicError(
                        type="api_error", message="Stream processing error"
                    ),
                )
                yield _wrap_sse_event(
                    error_event.model_dump_json(exclude_none=True), "error"
                )
                continue

            # First chunk: emit message_start
            if first_chunk:
                first_chunk = False

                start_event = AnthropicStreamEvent(
                    type="message_start",
                    message=AnthropicMessagesResponse(
                        id=message_id,
                        content=[],
                        model=model,
                        usage=AnthropicUsage(
                            input_tokens=(
                                chunk.usage.prompt_tokens if chunk.usage else 0
                            ),
                            output_tokens=0,
                        ),
                    ),
                )
                yield _wrap_sse_event(
                    start_event.model_dump_json(exclude_none=True),
                    "message_start",
                )
                # Skip if this was just the role chunk with empty content
                if chunk.choices and chunk.choices[0].delta.content == "":
                    continue

            # Usage-only chunk (empty choices with usage info)
            if not chunk.choices and chunk.usage:
                usage_info = {
                    "input_tokens": chunk.usage.prompt_tokens,
                    "output_tokens": chunk.usage.completion_tokens or 0,
                }
                continue

            if not chunk.choices:
                continue

            choice = chunk.choices[0]

            # Capture finish reason
            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason
                continue

            delta = choice.delta

            # Handle reasoning content deltas (extended thinking).
            #
            # Models served with --reasoning-parser (e.g. qwen3,
            # deepseek_r1) emit their <think>...</think> output as
            # `delta.reasoning_content` on the underlying OpenAI stream.
            # In Anthropic's protocol this maps to a `thinking` content
            # block carrying `thinking_delta` events. Without this branch
            # the entire reasoning trace is silently dropped on the
            # /v1/messages streaming endpoint, even though the OpenAI
            # endpoint surfaces it correctly.
            reasoning_chunk = getattr(delta, "reasoning_content", None)
            if reasoning_chunk:
                # If a non-thinking block is open (e.g. a previous text
                # block), close it before emitting the thinking block.
                if content_block_open and content_block_type != "thinking":
                    stop_event = AnthropicStreamEvent(
                        type="content_block_stop",
                        index=content_block_index,
                    )
                    yield _wrap_sse_event(
                        stop_event.model_dump_json(exclude_none=True),
                        "content_block_stop",
                    )
                    content_block_index += 1
                    content_block_open = False
                    content_block_type = None

                if not content_block_open:
                    start_event = AnthropicStreamEvent(
                        type="content_block_start",
                        index=content_block_index,
                        content_block=AnthropicContentBlock(
                            type="thinking", thinking=""
                        ),
                    )
                    yield _wrap_sse_event(
                        start_event.model_dump_json(exclude_none=True),
                        "content_block_start",
                    )
                    content_block_open = True
                    content_block_type = "thinking"

                delta_event = AnthropicStreamEvent(
                    type="content_block_delta",
                    index=content_block_index,
                    delta=AnthropicDelta(
                        type="thinking_delta",
                        thinking=reasoning_chunk,
                    ),
                )
                yield _wrap_sse_event(
                    delta_event.model_dump_json(exclude_none=True),
                    "content_block_delta",
                )
                emitted_thinking = True
                # If the chunk also carried text/tool_calls fall through;
                # otherwise skip the empty branches below.
                if not delta.tool_calls and not delta.content:
                    continue

            # Handle tool call deltas
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    tc_id = tc.id
                    tc_func = tc.function

                    # New tool call: close previous block, start new one
                    if tc_func and tc_func.name:
                        # Close previous content block if open
                        if content_block_open:
                            stop_event = AnthropicStreamEvent(
                                type="content_block_stop",
                                index=content_block_index,
                            )
                            yield _wrap_sse_event(
                                stop_event.model_dump_json(exclude_none=True),
                                "content_block_stop",
                            )
                            content_block_index += 1
                            content_block_type = None

                        # Start tool_use content block
                        start_event = AnthropicStreamEvent(
                            type="content_block_start",
                            index=content_block_index,
                            content_block=AnthropicContentBlock(
                                type="tool_use",
                                id=tc_id or f"toolu_{uuid.uuid4().hex}",
                                name=tc_func.name,
                                input={},
                            ),
                        )
                        yield _wrap_sse_event(
                            start_event.model_dump_json(exclude_none=True),
                            "content_block_start",
                        )
                        content_block_open = True
                        content_block_type = "tool_use"
                        emitted_text_or_tool = True

                        # Stream initial arguments if present
                        if tc_func.arguments:
                            delta_event = AnthropicStreamEvent(
                                type="content_block_delta",
                                index=content_block_index,
                                delta=AnthropicDelta(
                                    type="input_json_delta",
                                    partial_json=tc_func.arguments,
                                ),
                            )
                            yield _wrap_sse_event(
                                delta_event.model_dump_json(exclude_none=True),
                                "content_block_delta",
                            )

                    elif tc_func and tc_func.arguments:
                        # Continuing arguments for current tool call
                        delta_event = AnthropicStreamEvent(
                            type="content_block_delta",
                            index=content_block_index,
                            delta=AnthropicDelta(
                                type="input_json_delta",
                                partial_json=tc_func.arguments,
                            ),
                        )
                        yield _wrap_sse_event(
                            delta_event.model_dump_json(exclude_none=True),
                            "content_block_delta",
                        )
                continue

            # Handle text content deltas
            if delta.content is not None and delta.content != "":
                # If a non-text block is currently open (e.g. a tool_use
                # opened by an earlier delta in the same turn), close it
                # before emitting the text. Otherwise the text_delta
                # would be stamped with the tool_use's index and crash
                # any Anthropic-SDK client with "Content block is not a
                # text block".
                if content_block_open and content_block_type != "text":
                    stop_event = AnthropicStreamEvent(
                        type="content_block_stop",
                        index=content_block_index,
                    )
                    yield _wrap_sse_event(
                        stop_event.model_dump_json(exclude_none=True),
                        "content_block_stop",
                    )
                    content_block_index += 1
                    content_block_open = False
                    content_block_type = None

                # Start a text content block if needed
                if not content_block_open:
                    start_event = AnthropicStreamEvent(
                        type="content_block_start",
                        index=content_block_index,
                        content_block=AnthropicContentBlock(type="text", text=""),
                    )
                    yield _wrap_sse_event(
                        start_event.model_dump_json(exclude_none=True),
                        "content_block_start",
                    )
                    content_block_open = True
                    content_block_type = "text"

                # Emit text delta
                delta_event = AnthropicStreamEvent(
                    type="content_block_delta",
                    index=content_block_index,
                    delta=AnthropicDelta(
                        type="text_delta",
                        text=delta.content,
                    ),
                )
                yield _wrap_sse_event(
                    delta_event.model_dump_json(exclude_none=True),
                    "content_block_delta",
                )
                emitted_text_or_tool = True

    async def _retry_thinking_only(
        self,
        original_chat_request: ChatCompletionRequest,
        raw_request: Request,
    ) -> Optional[ChatCompletionResponse]:
        """Re-issue the request non-streaming with a fresh seed and bumped
        temperature.

        Called after the primary generation produced only a thinking block.
        Qwen3.6's short-thinking-stop is intermittent and seed-sensitive, so
        one perturbed retry typically yields a healthy thinking+content turn.

        Returns:
            The retry's ChatCompletionResponse on success, or None if the
            retry errored, timed out, or returned a non-response (e.g. an
            internal error from the OpenAI handler). Caller falls back to the
            placeholder text on None.
        """
        try:
            # Deep-copy so mutating sampling params on the retry can't leak
            # back into any in-flight state of the original request.
            retry_request = original_chat_request.model_copy(deep=True)

            # Force non-streaming so we get one consolidated response we can
            # convert with _convert_response().
            retry_request.stream = False
            retry_request.stream_options = None

            # New seed: bump if caller set one, otherwise pick fresh.
            if retry_request.seed is not None:
                retry_request.seed = (retry_request.seed + 1) & 0x7FFFFFFF
            else:
                retry_request.seed = random.randint(0, 2**31 - 1)

            # Bump temperature within a ceiling. If the original was unset,
            # the backend default (typically 1.0) applies; we still bump from
            # the explicit base of 1.0 in that case for consistency with how
            # the model was actually sampled the first time.
            base_temp = (
                retry_request.temperature
                if retry_request.temperature is not None
                else 1.0
            )
            retry_request.temperature = min(
                base_temp + ANTHROPIC_THINKING_RETRY_TEMP_BUMP,
                ANTHROPIC_THINKING_RETRY_TEMP_CEILING,
            )

            logger.warning(
                "Anthropic thinking-only detected; retrying with seed=%s temp=%.3f "
                "(timeout=%.0fs)",
                retry_request.seed,
                retry_request.temperature,
                ANTHROPIC_THINKING_RETRY_TIMEOUT_S,
            )

            async def _run_retry() -> Optional[ChatCompletionResponse]:
                # Re-validate after mutation — cheap and protects against
                # accidental drift if validation rules grow.
                err = self.openai_serving_chat._validate_request(retry_request)
                if err:
                    logger.warning(
                        "Anthropic thinking-only retry: validation failed: %s",
                        err,
                    )
                    return None

                adapted, processed = (
                    self.openai_serving_chat._convert_to_internal_request(
                        retry_request, raw_request
                    )
                )
                adapted.received_time = monotonic_time()
                adapted.received_time_perf = time.perf_counter()
                adapted.validation_time = 0.0

                resp = await self.openai_serving_chat._handle_non_streaming_request(
                    adapted, processed, raw_request
                )
                if not isinstance(resp, ChatCompletionResponse):
                    logger.warning(
                        "Anthropic thinking-only retry: non-response result "
                        "(likely an error from the OpenAI handler); "
                        "falling back to placeholder."
                    )
                    return None
                return resp

            return await asyncio.wait_for(
                _run_retry(), timeout=ANTHROPIC_THINKING_RETRY_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Anthropic thinking-only retry timed out after %.0fs; "
                "falling back to placeholder text.",
                ANTHROPIC_THINKING_RETRY_TIMEOUT_S,
            )
            return None
        except Exception as e:
            logger.exception(
                "Anthropic thinking-only retry raised; falling back to "
                "placeholder text. Error: %s",
                e,
            )
            return None

    @staticmethod
    def _extract_retry_payload(
        retry_response: ChatCompletionResponse,
    ) -> tuple[Optional[str], list]:
        """Pull text + tool_calls out of a retry response.

        Returns a (text, tool_calls) tuple where either may be empty/None.
        Reasoning content from the retry is intentionally discarded: the
        original thinking block has already been streamed to the client and
        we want to replace only the missing text/tool segment.
        """
        if not retry_response.choices:
            return None, []
        msg = retry_response.choices[0].message
        text = (msg.content or None) if isinstance(msg.content, str) else None
        if text is not None and text.strip() == "":
            text = None
        tool_calls = list(msg.tool_calls or [])
        return text, tool_calls

    def _convert_response(
        self,
        response: ChatCompletionResponse,
        inject_fallback: bool = True,
    ) -> AnthropicMessagesResponse:
        """Convert an OpenAI ChatCompletionResponse to an Anthropic Messages response.

        Args:
            response: The OpenAI-format response from the local handler.
            inject_fallback: When True (the default — used by any external
                caller of this method), append the static placeholder text
                block on thinking-only responses. _handle_non_streaming
                passes False so it can attempt a retry first and inject the
                placeholder only as a last resort.
        """
        if not response.choices:
            return AnthropicMessagesResponse(
                content=[AnthropicContentBlock(type="text", text="")],
                model=response.model,
                stop_reason="end_turn",
                usage=AnthropicUsage(input_tokens=0, output_tokens=0),
            )

        choice = response.choices[0]
        content: list[AnthropicContentBlock] = []

        # Add thinking content from reasoning_content (extended thinking).
        # Non-streaming responses carry reasoning on the assistant message;
        # without this branch the trace is silently dropped (the streaming
        # path already surfaces it as a thinking block).
        reasoning_text = getattr(choice.message, "reasoning_content", None)
        if reasoning_text:
            content.append(
                AnthropicContentBlock(type="thinking", thinking=reasoning_text)
            )

        # Add text content
        if choice.message.content:
            content.append(
                AnthropicContentBlock(type="text", text=choice.message.content)
            )

        # Add tool calls
        if choice.message.tool_calls:
            for tool_call in choice.message.tool_calls:
                try:
                    tool_input = json.loads(tool_call.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    tool_input = {}

                content.append(
                    AnthropicContentBlock(
                        type="tool_use",
                        id=tool_call.id,
                        name=tool_call.function.name,
                        input=tool_input,
                    )
                )

        # Fallback: if the response carries only a thinking block (no text,
        # no tool_use), strict Anthropic clients silently drop the turn.
        # The primary path is _handle_non_streaming's retry; this static
        # placeholder is the last-resort safety net. Callers that drive
        # their own retry pass inject_fallback=False and append later.
        has_thinking = any(b.type == "thinking" for b in content)
        has_text_or_tool = any(
            b.type in ("text", "tool_use") for b in content
        )
        if inject_fallback and has_thinking and not has_text_or_tool:
            logger.warning(
                "Anthropic non-streaming response has only a thinking block; "
                "injecting fallback text block to satisfy strict clients."
            )
            content.append(
                AnthropicContentBlock(
                    type="text", text=ANTHROPIC_EMPTY_TEXT_FALLBACK
                )
            )

        # If we have no content at all (no thinking, no text, no tool_use),
        # emit an empty text block so the response is structurally valid.
        # Callers controlling fallback themselves manage this case post-hoc.
        if inject_fallback and not content:
            content.append(AnthropicContentBlock(type="text", text=""))

        # Map stop reason
        stop_reason = STOP_REASON_MAP.get(choice.finish_reason or "stop", "end_turn")

        return AnthropicMessagesResponse(
            id=f"msg_{uuid.uuid4().hex}",
            content=content,
            model=response.model,
            stop_reason=stop_reason,
            usage=AnthropicUsage(
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
            ),
        )

    def _error_response(
        self,
        status_code: int,
        error_type: str,
        message: str,
    ) -> JSONResponse:
        """Create an Anthropic-format error response."""
        error_resp = AnthropicErrorResponse(
            error=AnthropicError(type=error_type, message=message)
        )
        return JSONResponse(
            status_code=status_code,
            content=error_resp.model_dump(),
        )

    async def handle_count_tokens(
        self,
        request: AnthropicCountTokensRequest,
        raw_request: Request,
    ) -> JSONResponse:
        """Handle /v1/messages/count_tokens endpoint.

        Converts the request to a ChatCompletionRequest, applies the chat
        template via the OpenAI handler to tokenize, and returns the count.
        """
        try:
            # Build a minimal AnthropicMessagesRequest so we can reuse conversion
            messages_request = AnthropicMessagesRequest(
                model=request.model,
                messages=request.messages,
                max_tokens=1,  # dummy, not used for counting
                system=request.system,
                tools=request.tools,
                tool_choice=request.tool_choice,
            )
            chat_request = self._convert_to_chat_completion_request(messages_request)
        except Exception as e:
            logger.exception("Error converting count_tokens request: %s", e)
            return self._error_response(
                status_code=400,
                error_type="invalid_request_error",
                message=str(e),
            )

        try:
            is_multimodal = (
                self.openai_serving_chat.tokenizer_manager.model_config.is_multimodal
            )
            processed = self.openai_serving_chat._process_messages(
                chat_request, is_multimodal
            )

            if isinstance(processed.prompt_ids, list):
                input_tokens = len(processed.prompt_ids)
            else:
                # prompt_ids is a string (multimodal case) — tokenize it
                tokenizer = self.openai_serving_chat.tokenizer_manager.tokenizer
                input_tokens = len(tokenizer.encode(processed.prompt_ids))

            return JSONResponse(
                content=AnthropicCountTokensResponse(
                    input_tokens=input_tokens
                ).model_dump()
            )
        except Exception as e:
            logger.exception("Error counting tokens: %s", e)
            return self._error_response(
                status_code=500,
                error_type="internal_error",
                message="Internal server error",
            )
