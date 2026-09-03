from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from ytstock.llm_cli import ClaudeCliClient, ClaudeCliError, resolve_llm_backend
from ytstock.schemas import VideoAnalysis

from .conftest import ANALYSIS_DICT


@pytest.fixture
def cli_settings(settings, monkeypatch):
    monkeypatch.setattr("ytstock.llm_cli.shutil.which", lambda _: "/usr/local/bin/claude")
    return settings.model_copy(update={"llm_backend": "claude-cli", "anthropic_api_key": ""})


def _envelope(**overrides):
    base = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": json.dumps(ANALYSIS_DICT),
        "structured_output": ANALYSIS_DICT,
        "session_id": "sess-1",
        "num_turns": 2,
        "total_cost_usd": 0.12,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "cache_read_input_tokens": 30,
            "cache_creation_input_tokens": 40,
            "server_tool_use": {"web_search_requests": 2},
        },
        "modelUsage": {"claude-opus-5": {}},
    }
    base.update(overrides)
    return base


def test_cli_parse_builds_command_and_parses(cli_settings, monkeypatch):
    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        captured["stdin"] = kw["input"]
        captured["env"] = kw["env"]
        return SimpleNamespace(returncode=0, stdout=json.dumps(_envelope()), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    client = ClaudeCliClient(cli_settings)
    resp = client.beta.messages.parse(
        model="claude-opus-5",
        max_tokens=100,
        system=[{"type": "text", "text": "SYS"}],
        messages=[{"role": "user", "content": "PROMPT"}],
        output_format=VideoAnalysis,
        tools=[{"type": "web_search_20260209", "name": "web_search"}],
        betas=["x"],
        fallbacks="default",
    )
    cmd = captured["cmd"]
    assert cmd[:2] == ["/usr/local/bin/claude", "-p"]
    assert "--json-schema" in cmd and "--system-prompt" in cmd
    assert cmd[cmd.index("--system-prompt") + 1] == "SYS"
    assert cmd[cmd.index("--tools") + 1] == "WebSearch,WebFetch"
    assert captured["stdin"] == "PROMPT"
    assert "CLAUDECODE" not in captured["env"]
    assert isinstance(resp.parsed_output, VideoAnalysis)
    assert resp.stop_reason == "end_turn" and resp.model == "claude-opus-5"
    assert resp.usage.cache_read_input_tokens == 30
    assert resp.usage.server_tool_use.web_search_requests == 2
    assert resp._request_id == "sess-1"


def test_cli_no_tools_when_not_requested(cli_settings, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kw: (
            captured.setdefault("cmd", cmd)
            and SimpleNamespace(returncode=0, stdout=json.dumps(_envelope()), stderr="")
        ),
    )
    ClaudeCliClient(cli_settings).messages.parse(
        model="claude-opus-5",
        max_tokens=1,
        messages=[{"role": "user", "content": "p"}],
        output_format=VideoAnalysis,
    )
    cmd = captured["cmd"]
    assert cmd[cmd.index("--tools") + 1] == ""


def test_cli_error_envelope_raises(cli_settings, monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kw: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(_envelope(is_error=True, result="Not logged in")),
            stderr="",
        ),
    )
    client = ClaudeCliClient(cli_settings)
    client.parse.retry.wait = lambda *_: 0  # type: ignore[attr-defined]
    with pytest.raises(ClaudeCliError, match="Not logged in"):
        client.messages.parse(
            model="m",
            max_tokens=1,
            messages=[{"role": "user", "content": "p"}],
            output_format=VideoAnalysis,
        )


def test_cli_unparseable_output_marks_max_tokens(cli_settings, monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kw: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(_envelope(structured_output=None, result="not json")),
            stderr="",
        ),
    )
    resp = ClaudeCliClient(cli_settings).messages.parse(
        model="m",
        max_tokens=1,
        messages=[{"role": "user", "content": "p"}],
        output_format=VideoAnalysis,
    )
    assert resp.parsed_output is None and resp.stop_reason == "max_tokens"


def test_resolve_backend(settings, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert resolve_llm_backend(settings) == "api"  # key set in fixture
    s = settings.model_copy(update={"anthropic_api_key": ""})
    monkeypatch.setattr("ytstock.llm_cli.shutil.which", lambda _: "/bin/claude")
    assert resolve_llm_backend(s) == "claude-cli"
    monkeypatch.setattr("ytstock.llm_cli.shutil.which", lambda _: None)
    assert resolve_llm_backend(s) == "api"
    assert resolve_llm_backend(s.model_copy(update={"llm_backend": "claude-cli"})) == "claude-cli"
