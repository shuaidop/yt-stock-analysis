"""Claude Code (``claude -p``) as an LLM backend.

Lets the pipeline run on a Claude subscription instead of an API key. The
object returned by ``build_llm_client`` duck-types the two SDK methods the
analyzer uses (``client.messages.parse`` / ``client.beta.messages.parse``) and
returns a response with the same attributes the analyzer reads
(``stop_reason``, ``parsed_output``, ``usage``, ``model``, ``_request_id``).

Differences from the API backend
- Web search is delegated to Claude Code's built-in WebSearch/WebFetch tools.
- No ``pause_turn``: the CLI runs its own agent loop.
- ``effort`` and refusal fallbacks are not passed through.
- Cost reported by the CLI is the list-price equivalent, not what you are billed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any

import anthropic
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from ytstock.config import Settings
from ytstock.log import get_logger

log = get_logger(__name__)


class ClaudeCliError(RuntimeError):
    """The ``claude`` process failed or returned an error envelope."""


@dataclass
class _ServerToolUse:
    web_search_requests: int = 0


@dataclass
class _Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    server_tool_use: _ServerToolUse = field(default_factory=_ServerToolUse)


@dataclass
class CliResponse:
    stop_reason: str
    parsed_output: Any
    usage: _Usage
    model: str
    content: list = field(default_factory=list)
    stop_details: Any = None
    _request_id: str = ""
    reported_cost_usd: float = 0.0


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block["text"])
        elif getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts)


class ClaudeCliClient:
    def __init__(self, settings: Settings) -> None:
        path = shutil.which(settings.claude_cli_path)
        if path is None:
            raise ClaudeCliError(
                f"claude CLI not found at {settings.claude_cli_path!r}; install Claude Code or "
                "set ANTHROPIC_API_KEY to use the API backend."
            )
        self._path = path
        self._settings = settings
        self.messages = _Messages(self)
        self.beta = _Beta(self)

    @retry(
        retry=retry_if_exception_type(ClaudeCliError),
        stop=stop_after_attempt(2),
        wait=wait_fixed(3),
        reraise=True,
    )
    def parse(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        output_format: type[BaseModel],
        system: Any = None,
        tools: list[dict[str, Any]] | None = None,
        **_ignored: Any,
    ) -> CliResponse:
        s = self._settings
        user_turns = [m for m in messages if m.get("role") == "user"]
        prompt = _text_of(user_turns[-1]["content"]) if user_turns else ""
        system_text = _text_of(system) if system else ""
        schema = output_format.model_json_schema()
        wants_web = any(str(t.get("type", "")).startswith("web_search") for t in (tools or []))

        cmd = [
            self._path,
            "-p",
            "--no-session-persistence",
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(schema),
            "--model",
            s.claude_cli_model or model,
            "--max-turns",
            str(s.claude_cli_max_turns if wants_web else 3),
        ]
        if system_text:
            cmd += ["--system-prompt", system_text]
        if wants_web:
            cmd += ["--tools", "WebSearch,WebFetch", "--allowedTools", "WebSearch,WebFetch"]
        else:
            cmd += ["--tools", ""]

        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}  # allow nesting
        with tempfile.TemporaryDirectory(prefix="ytstock-cli-") as cwd:
            try:
                proc = subprocess.run(
                    cmd,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=s.claude_timeout_seconds,
                    cwd=cwd,
                    env=env,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise ClaudeCliError(
                    f"claude -p timed out after {s.claude_timeout_seconds}s"
                ) from exc

        try:
            envelope = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise ClaudeCliError(
                f"claude -p exit={proc.returncode}; non-JSON output: "
                f"{(proc.stdout or proc.stderr)[:400]}"
            ) from exc

        if envelope.get("is_error") or envelope.get("subtype") != "success":
            raise ClaudeCliError(
                f"claude -p {envelope.get('subtype')}: {str(envelope.get('result'))[:400]}"
            )

        structured = envelope.get("structured_output")
        if structured is None:
            raw = envelope.get("result") or ""
            try:
                structured = json.loads(raw)
            except json.JSONDecodeError:
                structured = None
        parsed = output_format.model_validate(structured) if structured is not None else None

        u = envelope.get("usage") or {}
        model_usage = envelope.get("modelUsage") or {}
        served = next(iter(model_usage), None) or (s.claude_cli_model or model)
        usage = _Usage(
            input_tokens=int(u.get("input_tokens") or 0),
            output_tokens=int(u.get("output_tokens") or 0),
            cache_read_input_tokens=int(u.get("cache_read_input_tokens") or 0),
            cache_creation_input_tokens=int(u.get("cache_creation_input_tokens") or 0),
            server_tool_use=_ServerToolUse(
                int((u.get("server_tool_use") or {}).get("web_search_requests") or 0)
            ),
        )
        log.debug(
            "claude_cli.done",
            model=served,
            turns=envelope.get("num_turns"),
            reported_cost_usd=envelope.get("total_cost_usd"),
        )
        return CliResponse(
            stop_reason="end_turn" if parsed is not None else "max_tokens",
            parsed_output=parsed,
            usage=usage,
            model=served,
            _request_id=str(envelope.get("session_id") or ""),
            reported_cost_usd=float(envelope.get("total_cost_usd") or 0.0),
        )


class _Messages:
    def __init__(self, client: ClaudeCliClient) -> None:
        self._client = client

    def parse(self, **request: Any) -> CliResponse:
        return self._client.parse(**request)


class _Beta:
    def __init__(self, client: ClaudeCliClient) -> None:
        self.messages = _Messages(client)


def resolve_llm_backend(settings: Settings) -> str:
    if settings.llm_backend != "auto":
        return settings.llm_backend
    if (
        settings.anthropic_api_key
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    ):
        return "api"
    if shutil.which(settings.claude_cli_path):
        return "claude-cli"
    return "api"  # SDK may still find an `ant auth login` profile


def build_llm_client(settings: Settings) -> Any:
    backend = resolve_llm_backend(settings)
    log.info("llm.backend", backend=backend, model=settings.claude_model)
    if backend == "claude-cli":
        return ClaudeCliClient(settings)
    kwargs: dict[str, Any] = {"timeout": settings.claude_timeout_seconds, "max_retries": 3}
    if settings.anthropic_api_key:
        kwargs["api_key"] = settings.anthropic_api_key
    return anthropic.Anthropic(**kwargs)
