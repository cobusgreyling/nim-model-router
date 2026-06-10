import json
import re

from typer.testing import CliRunner

from nim_model_router.cli import app

runner = CliRunner()


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_models_command_handles_auto_alias():
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    output = _strip_ansi(result.output)
    assert "nim-router/auto" in output
    assert "classifier" in output and "decides" in output


def test_route_command_json_output():
    result = runner.invoke(app, ["route", "Hi", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["task"] == "fast"
    assert "confidence" in data


def test_classify_command():
    result = runner.invoke(app, ["classify", "refactor this python code"])
    assert result.exit_code == 0
    assert "coding" in result.output.lower()


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "nim-router" in result.output
