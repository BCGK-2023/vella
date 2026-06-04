"""``OpenRouterProvider`` — a real :class:`~vella.agent.ModelProvider` (``[openrouter]``).

An out-of-gate adapter over OpenRouter's OpenAI-compatible
``/chat/completions`` endpoint with ``stream=true``. It translates the
OpenAI-dialect Server-Sent-Events stream (``choices[].delta`` with fragmented
``content`` / ``reasoning`` text and fragmented ``tool_calls[].function.arguments``
JSON, a terminal ``finish_reason``, and a ``usage`` object) into the canonical
:data:`~vella.agent.provider.TurnEvent` lifecycle, then folds those events through
the SHARED in-gate accumulator (:mod:`vella.agent._assembler`) — it never rolls its
own delta accumulator. The result is therefore byte-identical (under
``model_dump(mode="json")``) to feeding the equivalent canonical events to the
deterministic path the :class:`~vella.agent.MockProvider` exercises in the gate.

``httpx`` is imported **lazily** (inside ``__init__``/the streaming method, never at
module top) so importing ``vella.agent`` — or even this module — never pulls
``httpx``: the cognition core stays five-dep and the import-boundary invariant holds
whether or not the ``[openrouter]`` extra is installed.

Dialect → canonical mapping:

* ``delta.content`` fragment → :class:`~vella.agent.provider.TextDelta` on the
  turn's text block.
* ``delta.reasoning`` fragment → :class:`~vella.agent.provider.ThinkingDelta` on the
  turn's thinking block (OpenRouter surfaces reasoning here for capable models).
* ``delta.tool_calls[i]`` → a ``tool_use`` block keyed by the OpenAI per-call
  ``index``: ``id``/``function.name`` open it (:class:`ToolUseBlockStub`), and each
  ``function.arguments`` fragment is an :class:`~vella.agent.provider.InputJsonDelta`
  (assembled + parsed exactly once at block stop by the shared accumulator).
* ``choices[].finish_reason`` → the canonical :data:`~vella.agent.StopReason`.
* ``usage`` (incl. ``prompt_tokens_details.cached_tokens`` where present) → the
  canonical :class:`~vella.agent.Usage`.

Block index assignment (the accumulator orders by integer index): text gets index
``0``, thinking index ``1``, and each tool call gets ``2 + openai_tool_index`` — a
stable, collision-free mapping so a turn's blocks read text, then thinking, then
tool calls in tool-call order.
"""

# ``httpx`` is an OUT-OF-GATE extra (``[openrouter]``) imported lazily and NOT
# installed in the core type-check env, so its members type as ``Unknown`` here.
# Relax the resulting strict-only "unknown type" reports for THIS adapter file only
# (the gated core stays fully strict); with the extra installed, httpx's own stubs
# make these precise. Mirrors the siblings' per-module tolerance of optional deps.
# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional

from vella.core import ToolDeclaration

from .._assembler import drain
from ..provider import (
    ContentBlockDelta,
    ContentBlockStart,
    ContentBlockStop,
    InputJsonDelta,
    MessageDelta,
    MessageStart,
    MessageStop,
    TextBlockStub,
    TextDelta,
    ThinkingBlockStub,
    ThinkingDelta,
    ToolUseBlockStub,
    TurnEvent,
    TurnRequest,
)
from ..turn import AssistantTurn, Message, StopReason, Usage

if TYPE_CHECKING:
    import httpx

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_TEXT_INDEX = 0
_THINKING_INDEX = 1
_TOOL_INDEX_BASE = 2

# OpenAI/OpenRouter finish_reason -> canonical StopReason. "tool_calls" is the
# dialect's name for the canonical "tool_use"; "length" maps to "max_tokens";
# "content_filter" is a refusal. Anything else (incl. a missing reason) defaults
# to "end_turn".
_FINISH_REASON: dict[str, StopReason] = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}


def _stop_reason(raw: Optional[str]) -> StopReason:
    """Map an OpenAI-dialect ``finish_reason`` to the canonical stop reason."""
    if raw is None:
        return "end_turn"
    return _FINISH_REASON.get(raw, "end_turn")


def _usage(payload: Optional[dict[str, Any]]) -> Usage:
    """Map an OpenAI-dialect ``usage`` object to the canonical :class:`Usage`.

    Reads ``prompt_tokens``/``completion_tokens`` and, where present, the
    prompt-cache breakdown (``prompt_tokens_details.cached_tokens`` is OpenRouter's
    cache-read count) and reasoning-token count
    (``completion_tokens_details.reasoning_tokens``).

    Args:
        payload: The dialect ``usage`` dict, or ``None`` when absent.

    Returns:
        The canonical :class:`Usage` (zeroed when ``payload`` is ``None``).
    """
    if not payload:
        return Usage()
    prompt_details = payload.get("prompt_tokens_details") or {}
    completion_details = payload.get("completion_tokens_details") or {}
    return Usage(
        input_tokens=int(payload.get("prompt_tokens", 0) or 0),
        output_tokens=int(payload.get("completion_tokens", 0) or 0),
        cache_read_tokens=int(prompt_details.get("cached_tokens", 0) or 0),
        cache_write_tokens=int(prompt_details.get("cache_creation_tokens", 0) or 0),
        reasoning_tokens=int(completion_details.get("reasoning_tokens", 0) or 0),
    )


def _message_to_wire(message: Message) -> dict[str, Any]:
    """Lower a canonical :class:`Message` to an OpenAI-dialect request message.

    Text/thinking blocks fold into the ``content`` string; ``tool_use`` blocks
    become ``tool_calls`` entries; a ``tool_result`` block becomes a ``role:"tool"``
    message keyed by ``tool_call_id``. Kept deliberately small — the request shape
    the OpenRouter chat-completions endpoint accepts for the canonical roles.

    Args:
        message: The canonical message to lower.

    Returns:
        The OpenAI-dialect message dict.
    """
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in message.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "thinking":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(
                {
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input, sort_keys=True),
                    },
                }
            )
        elif block.type == "tool_result":
            content = block.content
            return {
                "role": "tool",
                "tool_call_id": block.tool_use_id,
                "content": content if isinstance(content, str) else json.dumps(content),
            }
    wire: dict[str, Any] = {"role": message.role, "content": "".join(text_parts)}
    if tool_calls:
        wire["tool_calls"] = tool_calls
    return wire


def _tool_to_wire(tool: ToolDeclaration) -> dict[str, Any]:
    """Lower a :class:`~vella.core.ToolDeclaration` to an OpenAI-dialect tool spec."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


class OpenRouterProvider:
    """A real :class:`~vella.agent.ModelProvider` over OpenRouter (``[openrouter]``).

    Construct it from the run's ``provider`` node (:class:`~vella.agent.ProviderData`)
    plus an API key; :meth:`stream` performs one streaming chat-completions request
    and yields the canonical :data:`~vella.agent.provider.TurnEvent` lifecycle, and
    :meth:`turn` drains that stream through the SHARED accumulator
    (:func:`vella.agent._assembler.drain`) into one canonical
    :class:`~vella.agent.AssistantTurn`. It satisfies the structural
    :class:`~vella.agent.ModelProvider` Protocol by shape.

    ``httpx`` is imported lazily in :meth:`__init__`, so importing this module does
    not require the ``[openrouter]`` extra to be installed.
    """

    def __init__(
        self,
        *,
        model_id: str,
        api_key: str,
        endpoint: Optional[str] = None,
        cache_capable: bool = False,
        max_output_tokens: Optional[int] = None,
        transport: Optional["httpx.AsyncBaseTransport"] = None,
    ) -> None:
        """Build a provider from a ``provider`` node's fields plus an API key.

        Args:
            model_id: The OpenRouter model slug (``provider`` node ``model_id``).
            api_key: The OpenRouter API key (Bearer credential).
            endpoint: The base URL, or ``None`` for OpenRouter's default
                (``provider`` node ``endpoint``).
            cache_capable: Whether the provider declares prompt-cache support
                (``provider`` node ``cache_capable``); threaded into the request as
                ``usage`` accounting only — the assembler owns breakpoint placement.
            max_output_tokens: The per-turn output ceiling, or ``None`` for the
                model default (``provider`` node ``limits.max_output_tokens``).
            transport: An optional ``httpx`` transport to inject — the seam a
                network-free smoke test feeds ``httpx.MockTransport`` through.

        Raises:
            ModuleNotFoundError: If the ``[openrouter]`` extra (``httpx``) is not
                installed — raised here, lazily, never at import time.
        """
        import httpx  # lazy: importing vella.agent must never pull httpx

        self._model_id = model_id
        self._cache_capable = cache_capable
        self._max_output_tokens = max_output_tokens
        base_url = (endpoint or _DEFAULT_BASE_URL).rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            transport=transport,
        )

    def _request_body(self, request: TurnRequest) -> dict[str, Any]:
        """Build the OpenAI-dialect streaming request body from a canonical request."""
        body: dict[str, Any] = {
            "model": self._model_id,
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [_message_to_wire(m) for m in request.messages],
        }
        if request.tools:
            body["tools"] = [_tool_to_wire(t) for t in request.tools]
        choice = request.params.tool_choice
        if choice == "forced":
            body["tool_choice"] = "required"
        elif choice == "none":
            body["tool_choice"] = "none"
        if request.params.max_tokens is not None:
            body["max_tokens"] = request.params.max_tokens
        elif self._max_output_tokens is not None:
            body["max_tokens"] = self._max_output_tokens
        if request.params.temperature is not None:
            body["temperature"] = request.params.temperature
        if request.params.stop:
            body["stop"] = list(request.params.stop)
        return body

    async def stream(self, request: TurnRequest) -> AsyncIterator[TurnEvent]:
        """Yield ``request``'s turn as the canonical lifecycle event stream.

        Performs one streaming chat-completions request and normalizes the
        OpenAI-dialect SSE chunks into canonical :data:`~vella.agent.provider.TurnEvent`
        events: a single ``message_start``, a ``content_block_start`` the first time
        a text / thinking / tool-call block is seen, ``content_block_delta`` per
        fragment in arrival order, ``content_block_stop`` for every opened block once
        the stream ends, then ``message_delta`` (carrying ``stop_reason`` + ``usage``)
        and ``message_stop``. Tool-argument JSON is emitted as raw fragments — the
        shared accumulator parses it exactly once at block stop, never here.

        Args:
            request: The assembled messages + tool schemas + per-turn params.

        Yields:
            The turn's canonical :data:`~vella.agent.provider.TurnEvent` lifecycle.
        """
        # No lazy import needed here: the httpx client was built (and httpx imported)
        # in __init__; this method only drives it through the duck-typed seam.
        yield MessageStart(usage=Usage())

        opened: set[int] = set()
        text_started = False
        thinking_started = False
        tool_meta_sent: set[int] = set()
        finish_reason: Optional[str] = None
        usage_payload: Optional[dict[str, Any]] = None

        body = self._request_body(request)
        async with self._client.stream(
            "POST", "/chat/completions", json=body
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                chunk: dict[str, Any] = json.loads(data)
                if chunk.get("usage"):
                    usage_payload = chunk["usage"]
                for choice in chunk.get("choices", ()):
                    delta = choice.get("delta") or {}
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]

                    content = delta.get("content")
                    if content:
                        if not text_started:
                            text_started = True
                            opened.add(_TEXT_INDEX)
                            yield ContentBlockStart(
                                index=_TEXT_INDEX, block=TextBlockStub()
                            )
                        yield ContentBlockDelta(
                            index=_TEXT_INDEX, delta=TextDelta(text=content)
                        )

                    reasoning = delta.get("reasoning")
                    if reasoning:
                        if not thinking_started:
                            thinking_started = True
                            opened.add(_THINKING_INDEX)
                            yield ContentBlockStart(
                                index=_THINKING_INDEX, block=ThinkingBlockStub()
                            )
                        yield ContentBlockDelta(
                            index=_THINKING_INDEX,
                            delta=ThinkingDelta(text=reasoning),
                        )

                    for call in delta.get("tool_calls") or ():
                        tool_index = _TOOL_INDEX_BASE + int(call.get("index", 0))
                        fn = call.get("function") or {}
                        if tool_index not in tool_meta_sent:
                            tool_meta_sent.add(tool_index)
                            opened.add(tool_index)
                            yield ContentBlockStart(
                                index=tool_index,
                                block=ToolUseBlockStub(
                                    id=call.get("id") or "",
                                    name=fn.get("name") or "",
                                ),
                            )
                        args = fn.get("arguments")
                        if args:
                            yield ContentBlockDelta(
                                index=tool_index,
                                delta=InputJsonDelta(partial_json=args),
                            )

        for index in sorted(opened):
            yield ContentBlockStop(index=index)
        yield MessageDelta(
            stop_reason=_stop_reason(finish_reason), usage=_usage(usage_payload)
        )
        yield MessageStop()

    async def turn(self, request: TurnRequest) -> AssistantTurn:
        """Drain :meth:`stream` through the shared accumulator into one turn.

        Folds the canonical event stream through
        :func:`vella.agent._assembler.drain` — the SAME deterministic accumulator the
        in-gate path uses — so the result is byte-identical to assembling the
        equivalent events directly.

        Args:
            request: The assembled messages + tool schemas + per-turn params.

        Returns:
            The assembled canonical :class:`~vella.agent.AssistantTurn`.
        """
        return await drain(self.stream(request))

    async def aclose(self) -> None:
        """Close the underlying ``httpx`` client (release its connection pool)."""
        await self._client.aclose()
