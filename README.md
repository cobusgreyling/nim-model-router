<p align="center">
  <img src="images/nim-router-header.jpg" alt="NIM Model Router" width="100%">
</p>

# NIM Model Router

**OpenAI-compatible proxy that routes requests to the best NVIDIA NIM model by task.**

NVIDIA's [NIM catalog](https://build.nvidia.com/models) has 100+ models. Picking the right one for each request is tedious. This router sits in front of the NIM API and automatically selects a model based on what you're asking for — fast chat, agentic tool use, deep reasoning, coding, embeddings, and more.

Drop it into any OpenAI SDK client by changing `base_url`. No other code changes required.

## Quick start

```bash
git clone https://github.com/cobusgreyling/nim-model-router.git
cd nim-model-router
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# Edit .env and set NVIDIA_API_KEY

nim-router serve
```

The proxy listens on `http://127.0.0.1:8080`.

## Usage

### Auto-routing (recommended)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="local",  # not used — router injects NVIDIA_API_KEY upstream
)

response = client.chat.completions.create(
    model="nim-router/auto",
    messages=[{"role": "user", "content": "Build a Python agent with tool calling"}],
)
print(response.choices[0].message.content)
```

### Explicit task aliases

| Model alias | Task | Default NIM model |
|-------------|------|-------------------|
| `nim-router/auto` | Classify automatically | — |
| `nim-router/fast` | Short Q&A, classification | `meta/llama-3.1-8b-instruct` |
| `nim-router/agentic` | Tool use, agents | `nvidia/nemotron-3-super-120b-a12b` |
| `nim-router/reasoning` | Deep analysis | `nvidia/nemotron-3-ultra-550b-a55b` |
| `nim-router/long-context` | Large documents | `nvidia/nemotron-3-super-120b-a12b` |
| `nim-router/coding` | Code generation | `nvidia/llama-3.3-nemotron-super-49b-v1.5` |

### Force a task via header

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-NIM-Task: reasoning" \
  -d '{
    "model": "nim-router/auto",
    "messages": [{"role": "user", "content": "Analyze the root cause step by step"}]
  }'
```

### Passthrough to a specific NIM model

If you pass a concrete NIM model ID (e.g. `meta/llama-3.1-70b-instruct`), the router forwards it unchanged.

### Embeddings

```python
response = client.embeddings.create(
    model="nim-router/auto",
    input="semantic search query",
)
```

Use `X-NIM-Task: rerank` or model alias routing for reranker models.

## How routing works

```
Client request
    │
    ├─ model alias (nim-router/*)  ──► task
    ├─ X-NIM-Task header           ──► task
    ├─ concrete NIM model ID       ──► passthrough
    └─ nim-router/auto             ──► classifier
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ▼                     ▼                     ▼
                 tools?              long prompt?          keywords?
               → agentic           → long_context      → coding / reasoning
                    │                     │                     │
                    └─────────────────────┴─────────────────────┘
                                          ▼
                                   NIM model + extra_body


www.cobusgreyling.com
```

Classifier signals:

- **Tools present** → `agentic`
- **Large prompt** (>12k estimated tokens) → `long_context`
- **Reasoning keywords** (prove, root cause, step by step…) → `reasoning`
- **Coding keywords** (def, import, refactor, ```…) → `coding`
- **Short prompt** (≤120 chars) → `fast`
- **Default** → `agentic`

## CLI

```bash
# Start proxy
nim-router serve --port 8080

# Dry-run routing (no API call)
nim-router route "refactor this Python function"
nim-router route "hello" 
nim-router route "plan a multi-step agent" --tools

# Show registry
nim-router models

# Print OpenAI SDK example
nim-router client-example
```

## Observability

Every proxied response includes routing metadata:

| Header | Example |
|--------|---------|
| `X-NIM-Routed-Task` | `agentic` |
| `X-NIM-Routed-Model` | `nvidia/nemotron-3-super-120b-a12b` |
| `X-NIM-Router-Reason` | `request includes tool definitions` |

```bash
# Live stats
curl http://127.0.0.1:8080/v1/router/stats

# Task registry
curl http://127.0.0.1:8080/v1/router/tasks

# Dry-run endpoint
curl -X POST http://127.0.0.1:8080/v1/router/dry-run \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"debug my rust code"}]}'
```

Set `ROUTER_LOG_PATH=data/router.log.jsonl` to persist request logs.

## Customizing models

Edit `config/models.yaml` (or set `ROUTER_CONFIG`):

```yaml
tasks:
  agentic:
    model: nvidia/nemotron-3-nano-30b-a3b   # swap to a smaller model
    extra_body:
      enable_thinking: true
      reasoning_budget: 2048
```

Restart the server after changes.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
```

## Security

- Store `NVIDIA_API_KEY` in `.env` — never commit it.
- The local proxy does not require client authentication by default. Bind to `127.0.0.1` or add your own auth layer before exposing it publicly.

## License

MIT
