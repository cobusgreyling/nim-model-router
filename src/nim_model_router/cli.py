from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from nim_model_router import __version__
from nim_model_router.catalog import fetch_nim_catalog, suggest_models_for_task
from nim_model_router.classifier import classify_request
from nim_model_router.config import Settings, load_registry
from nim_model_router.proxy import create_app
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
    if settings.router_api_key:
        console.print("Client auth: ROUTER_API_KEY enabled")

    def app_factory():
        return create_app(settings)

    uvicorn.run(
        app_factory,
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
    table.add_column("Fallbacks")
    table.add_column("Description")

    for alias, task_name in sorted(registry.aliases.items()):
        if task_name == "auto":
            table.add_row(alias, "(classifier decides)", "", "Auto-route by request content")
            continue
        task_cfg = registry.tasks.get(task_name)
        if task_cfg is None:
            table.add_row(alias, "(unknown task)", "", task_name)
            continue
        table.add_row(
            alias,
            task_cfg.model,
            ", ".join(task_cfg.fallbacks) or "-",
            task_cfg.description,
        )

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
    json_output: bool = typer.Option(False, "--json", help="Emit plain JSON (no colors)."),
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
    data = {
        "task": decision.task.value,
        "model": decision.model,
        "reason": decision.reason,
        "confidence": decision.confidence,
        "extra_body": decision.extra_body,
        "fallback_models": decision.fallback_models,
    }
    if json_output:
        typer.echo(json.dumps(data, indent=2))
    else:
        console.print_json(data=data)


@app.command("classify")
def classify_cmd(
    prompt: str = typer.Argument(..., help="Prompt text to classify."),
    tools: bool = typer.Option(False, "--tools", help="Simulate tool definitions."),
    config: Path | None = typer.Option(None, "--config", help="Model registry YAML."),
) -> None:
    """Show classifier output only (no registry resolution)."""
    registry = load_registry(config)
    result = classify_request(
        messages=[{"role": "user", "content": prompt}],
        tools=[{"type": "function"}] if tools else None,
        config=registry.classifier,
        policies_registry=registry,
    )
    console.print_json(
        data={"task": result.task.value, "reason": result.reason, "confidence": result.confidence}
    )


@app.command("catalog-sync")
def catalog_sync_cmd(
    task: str = typer.Option("coding", "--task", help="Task to suggest models for."),
) -> None:
    """Fetch NIM catalog and suggest models for a task."""
    settings = Settings()
    if not settings.nvidia_api_key:
        console.print("[red]NVIDIA_API_KEY is not set.[/red]")
        raise typer.Exit(1)

    keywords_map = {
        "coding": ["code", "coder", "llama", "nemotron"],
        "reasoning": ["ultra", "reason", "nemotron"],
        "embedding": ["embed"],
        "rerank": ["rerank", "rank"],
        "fast": ["8b", "small", "mini"],
    }
    keywords = keywords_map.get(task, [task])

    async def _run() -> list[str]:
        catalog = await fetch_nim_catalog(
            api_key=settings.nvidia_api_key,
            base_url=settings.nim_base_url,
        )
        return suggest_models_for_task(catalog, keywords=keywords)

    suggestions = asyncio.run(_run())
    console.print_json(data={"task": task, "suggestions": suggestions})


@app.command("client-example")
def client_example_cmd() -> None:
    """Print a minimal OpenAI SDK example for this router."""
    example = """
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

# Rerank endpoint
import httpx
httpx.post(
    "http://127.0.0.1:8080/v1/rerank",
    json={
        "model": "nim-router/rerank",
        "query": "What is NIM?",
        "documents": ["NIM is ...", "Unrelated text"],
        "top_n": 2,
    },
)
"""
    console.print(example)


if __name__ == "__main__":
    app()
