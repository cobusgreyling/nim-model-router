from __future__ import annotations

from pathlib import Path

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from nim_model_router import __version__
from nim_model_router.classifier import classify_request
from nim_model_router.config import Settings, load_registry
from nim_model_router.router import ModelRouter

app = typer.Typer(
    name="nim-router",
    help="Route OpenAI-compatible requests to the best NVIDIA NIM model by task.",
    add_completion=False,
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"nim-router {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """NIM Model Router CLI."""
    pass  # pragma: no cover


@app.command("serve")
def serve_cmd(
    host: str | None = typer.Option(None, "--host", help="Bind host."),
    port: int | None = typer.Option(None, "--port", help="Bind port."),
    config: Path | None = typer.Option(None, "--config", help="Model registry YAML."),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload."),
) -> None:
    """Start the OpenAI-compatible proxy server."""
    settings = Settings()
    if config:
        settings = settings.model_copy(update={"router_config": config})
    if host:
        settings = settings.model_copy(update={"router_host": host})
    if port:
        settings = settings.model_copy(update={"router_port": port})

    if not settings.nvidia_api_key:
        console.print("[red]NVIDIA_API_KEY is not set.[/red] Copy .env.example to .env first.")
        raise typer.Exit(1)

    console.print(
        f"[green]Starting NIM router[/green] on http://{settings.router_host}:{settings.router_port}"
    )
    console.print(f"Upstream: {settings.nim_base_url}")
    console.print(f"Registry: {settings.router_config}")

    uvicorn.run(
        "nim_model_router.proxy:create_app",
        factory=True,
        host=settings.router_host,
        port=settings.router_port,
        reload=reload,
    )


@app.command("models")
def models_cmd(
    config: Path | None = typer.Option(None, "--config", help="Model registry YAML."),
) -> None:
    """Show task → model mappings and aliases."""
    registry = load_registry(config)
    router = ModelRouter(registry)

    table = Table(title="NIM Model Router Registry")
    table.add_column("Alias / Task")
    table.add_column("Resolved model")
    table.add_column("Description")

    for alias, task_name in sorted(registry.aliases.items()):
        task_cfg = registry.tasks[task_name]
        table.add_row(alias, task_cfg.model, task_cfg.description)

    console.print(table)
    console.print()
    console.print("[bold]Direct NIM models[/bold]")
    for item in router.list_router_models():
        if item["id"].startswith("nim-router/"):
            continue
        console.print(f"  {item['task']:14} → {item['resolved_model']}")


@app.command("route")
def route_cmd(
    prompt: str = typer.Argument(..., help="Prompt text to classify."),
    tools: bool = typer.Option(False, "--tools", help="Simulate a tool-use request."),
    task: str | None = typer.Option(None, "--task", help="Force task type."),
    config: Path | None = typer.Option(None, "--config", help="Model registry YAML."),
) -> None:
    """Dry-run routing for a prompt (no API call)."""
    registry = load_registry(config)
    router = ModelRouter(registry)

    payload: dict = {
        "model": "nim-router/auto",
        "messages": [{"role": "user", "content": prompt}],
    }
    if tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search documents",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    decision = router.route_chat(payload, task_header=task)
    console.print_json(
        data={
            "task": decision.task.value,
            "model": decision.model,
            "reason": decision.reason,
            "extra_body": decision.extra_body,
        }
    )


@app.command("classify")
def classify_cmd(
    prompt: str = typer.Argument(..., help="Prompt text to classify."),
    tools: bool = typer.Option(False, "--tools", help="Simulate tool definitions."),
    config: Path | None = typer.Option(None, "--config", help="Model registry YAML."),
) -> None:
    """Show classifier output only (no registry resolution)."""
    registry = load_registry(config)
    task, reason = classify_request(
        messages=[{"role": "user", "content": prompt}],
        tools=[{"type": "function"}] if tools else None,
        config=registry.classifier,
    )
    console.print_json(data={"task": task.value, "reason": reason})


@app.command("client-example")
def client_example_cmd() -> None:
    """Print a minimal OpenAI SDK example for this router."""
    example = '''
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed",  # router injects NVIDIA_API_KEY upstream
)

# Auto-route by content
response = client.chat.completions.create(
    model="nim-router/auto",
    messages=[{"role": "user", "content": "Build a Python agent with tool calling"}],
)
print(response.choices[0].message.content)

# Force a task via alias
response = client.chat.completions.create(
    model="nim-router/fast",
    messages=[{"role": "user", "content": "Say hello"}],
)

# Or via header (curl: -H "X-NIM-Task: reasoning")
'''
    console.print(example)


if __name__ == "__main__":
    app()