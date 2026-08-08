"""LLM providers.

The agent loop is written once, here, against a tiny interface: given a
conversation and a set of tools, return the assistant's next turn. Ollama and
Anthropic each implement it.

Default is Ollama, because it is free, runs locally and needs no API key. The
Anthropic path is fully implemented and one environment variable away — see
`Config.provider` — but nothing in this project requires paid API access.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import httpx

from .models import RunStats

logger = logging.getLogger(__name__)


@dataclass
class ToolSpec:
    """A tool, described once and adapted per provider."""

    name: str
    description: str
    #: JSON Schema for the arguments.
    parameters: dict[str, Any]
    fn: Callable[..., str]

    def call(self, arguments: dict[str, Any]) -> str:
        try:
            return self.fn(**arguments)
        except TypeError as exc:
            # A small model passing the wrong argument names is common; tell it
            # what it did wrong rather than crashing the run.
            return f"Tool {self.name} rejected those arguments: {exc}"
        except Exception as exc:  # noqa: BLE001 - surfaced back to the model
            logger.exception("tool %s failed", self.name)
            return f"Tool {self.name} failed: {type(exc).__name__}: {exc}"


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    #: Provider-assigned id, echoed back with the result. Ollama omits it.
    call_id: str | None = None


@dataclass
class Turn:
    """One assistant turn: some text, and/or some tool calls."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    #: Provider-native message, appended verbatim to history when present.
    raw: Any = None


class LLMProvider(Protocol):
    name: str

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
    ) -> Turn: ...

    def format_tool_result(self, call: ToolCall, result: str) -> dict[str, Any]: ...

    def format_assistant(self, turn: Turn) -> dict[str, Any]: ...


# --------------------------------------------------------------- ollama ----


class OllamaProvider:
    """Local models over Ollama's chat API. Free, offline, no key."""

    name = "ollama"

    def __init__(
        self,
        model: str = "llama3.1:8b",
        host: str = "http://localhost:11434",
        timeout: float = 180.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def available(self) -> tuple[bool, str]:
        try:
            resp = self._client.get(f"{self.host}/api/tags", timeout=5.0)
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            return False, f"Ollama is not reachable at {self.host} ({exc})."

        names = {m.get("name", "") for m in resp.json().get("models", [])}
        if self.model in names:
            return True, ""
        # Ollama accepts `llama3.1` for `llama3.1:8b`, so match on the stem too.
        stem = self.model.split(":")[0]
        if any(n.split(":")[0] == stem for n in names):
            return True, ""
        return False, (
            f"Model {self.model!r} is not pulled. Available: "
            f"{', '.join(sorted(names)) or 'none'}. Run: ollama pull {self.model}"
        )

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
    ) -> Turn:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": "system", "content": system}, *messages],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ],
            # Low temperature: this is a lookup-and-reason task, not a creative
            # one, and small models wander badly at higher settings.
            "options": {"temperature": 0.1},
        }

        resp = self._client.post(f"{self.host}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        message = data.get("message", {}) or {}

        calls = []
        for raw_call in message.get("tool_calls", []) or []:
            fn = raw_call.get("function", {}) or {}
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append(ToolCall(name=fn.get("name", ""), arguments=args or {}))

        return Turn(
            text=(message.get("content") or "").strip(),
            tool_calls=calls,
            input_tokens=data.get("prompt_eval_count", 0) or 0,
            output_tokens=data.get("eval_count", 0) or 0,
            raw=message,
        )

    def format_assistant(self, turn: Turn) -> dict[str, Any]:
        return turn.raw or {"role": "assistant", "content": turn.text}

    def format_tool_result(self, call: ToolCall, result: str) -> dict[str, Any]:
        return {"role": "tool", "content": result, "name": call.name}


# ------------------------------------------------------------ anthropic ----


class AnthropicProvider:
    """Claude via the Messages API.

    Kept fully working but off by default — the project is designed to need no
    paid API access. Set STEWARD_PROVIDER=anthropic to use it.
    """

    name = "anthropic"

    def __init__(self, model: str = "claude-opus-5", effort: str = "high") -> None:
        from anthropic import Anthropic  # imported lazily so the dep is optional

        self.model = model
        self.effort = effort
        # A bare constructor resolves ANTHROPIC_API_KEY or an `ant auth login`
        # profile, so there is usually nothing to configure.
        self.client = Anthropic()

    def available(self) -> tuple[bool, str]:
        import os

        if os.environ.get("ANTHROPIC_API_KEY"):
            return True, ""
        return False, (
            "ANTHROPIC_API_KEY is not set. Either export it, run `ant auth login`, "
            "or use the free local provider with STEWARD_PROVIDER=ollama."
        )

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
    ) -> Turn:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=8192,
            system=system,
            output_config={"effort": self.effort},
            tools=[
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ],
            messages=messages,
        )

        # Safety classifiers can decline with a normal 200 and empty content, so
        # stop_reason is checked before anything indexes into content.
        if response.stop_reason == "refusal":
            return Turn(text="The model declined to answer this request.")

        text = "\n".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        calls = [
            ToolCall(name=block.name, arguments=dict(block.input or {}), call_id=block.id)
            for block in response.content
            if block.type == "tool_use"
        ]
        return Turn(
            text=text,
            tool_calls=calls,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            raw=response.content,
        )

    def format_assistant(self, turn: Turn) -> dict[str, Any]:
        return {"role": "assistant", "content": turn.raw}

    def format_tool_result(self, call: ToolCall, result: str) -> dict[str, Any]:
        return {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": call.call_id, "content": result}
            ],
        }


# ----------------------------------------------------------------- loop ----


def run_agent_loop(
    provider: LLMProvider,
    system: str,
    question: str,
    tools: list[ToolSpec],
    stats: RunStats,
    max_iterations: int = 20,
) -> str:
    """Drive the model until it stops calling tools.

    Written once for every provider. The loop is deliberately manual rather than
    using a provider's own runner helper, because that is the only way both
    backends can share identical behaviour and instrumentation.
    """
    by_name = {t.name: t for t in tools}
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    final_text = ""

    for _ in range(max_iterations):
        turn = provider.complete(system, messages, tools)
        stats.input_tokens += turn.input_tokens
        stats.output_tokens += turn.output_tokens

        if turn.text:
            final_text = turn.text

        if not turn.tool_calls:
            return final_text

        messages.append(provider.format_assistant(turn))

        for call in turn.tool_calls:
            spec = by_name.get(call.name)
            if spec is None:
                # Small models invent tool names; correct rather than crash.
                result = (
                    f"No tool named {call.name!r}. Available: {', '.join(sorted(by_name))}."
                )
            else:
                result = spec.call(call.arguments)
            messages.append(provider.format_tool_result(call, result))

    logger.warning("agent hit the %d-iteration cap", max_iterations)
    return final_text or "Stopped after reaching the tool-call limit without concluding."


def build_provider(config) -> LLMProvider:
    """Pick a provider from config. Free/local by default."""
    if config.provider == "anthropic":
        return AnthropicProvider(model=config.model, effort=config.effort)
    return OllamaProvider(model=config.ollama_model, host=config.ollama_host)
