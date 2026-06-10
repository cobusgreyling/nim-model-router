from typer.testing import CliRunner

from nim_model_router.cli import app

runner = CliRunner()


def test_models_command_handles_auto_alias():
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "nim-router/auto" in result.output
    assert "classifier decides" in result.output


def test_route_command():
    result = runner.invoke(app, ["route", "Hi"])
    assert result.exit_code == 0
    assert '"task": "fast"' in result.output