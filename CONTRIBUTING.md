# Contributing to NIM Model Router

Thanks for your interest in contributing!

## Development setup

```bash
git clone https://github.com/cobusgreyling/nim-model-router.git
cd nim-model-router
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

## Running checks

```bash
ruff check src tests
ruff format src tests
pytest --cov=nim_model_router --cov-report=term-missing
```

## Project layout

- `src/nim_model_router/classifier.py` — request classification
- `src/nim_model_router/router.py` — task → model resolution
- `src/nim_model_router/proxy.py` — FastAPI app and endpoints
- `src/nim_model_router/client.py` — upstream NIM HTTP client
- `src/nim_model_router/models.yaml` — canonical task registry

## Adding a custom classifier plugin

Register an entry point in your package:

```toml
[project.entry-points."nim_model_router.classifiers"]
my_classifier = "my_package.classifier:classify"
```

Then set in `models.yaml`:

```yaml
classifier:
  plugin_classifier: my_classifier
```

## Pull request guidelines

1. Add tests for behavior changes
2. Keep changes focused — one feature or fix per PR
3. Update `CHANGELOG.md` under `Unreleased` or the next version
4. Ensure CI passes (lint, format, tests, coverage ≥ 75%)

## Reporting issues

Use GitHub Issues and include:

- Steps to reproduce
- Expected vs actual routing decision (`nim-router route "..." --json`)
- Relevant config snippets (redact API keys)