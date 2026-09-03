from __future__ import annotations

from typer.testing import CliRunner

from ytstock.cli import app

runner = CliRunner()


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in (
        "run",
        "brief",
        "discover",
        "transcribe",
        "analyze",
        "factcheck",
        "report",
        "status",
    ):
        assert cmd in result.output


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and "ytstock" in result.output


def test_status_on_empty_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'x.db'}")
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    from ytstock.config import get_settings

    get_settings.cache_clear()
    result = runner.invoke(app, ["status", "--date", "2026-09-02"])
    assert result.exit_code == 0, result.output
    assert "videos: 0" in result.output
    get_settings.cache_clear()


def test_bad_date():
    result = runner.invoke(app, ["status", "--date", "yesterday"])
    assert result.exit_code != 0
