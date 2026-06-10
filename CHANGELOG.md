# Changelog

All notable changes to this project are documented in this file.

## [0.3.0] - 2026-06-10

### Added
- Cost estimation from token usage (`X-NIM-Estimated-Cost-USD` header, stats, Prometheus)
- `X-NIM-Fallback-Used` response header for observability
- Optional CORS support via `ROUTER_CORS_ORIGINS`
- System-prompt coding detection in the classifier
- 429 rate-limit fallback to the next model in the chain

### Changed
- Low-confidence routing now downgrades all expensive tasks (not just agentic) to `general`
- Cached tiktoken encoder for faster repeated token estimation
- LLM classifier tolerates markdown-wrapped JSON responses
- Expanded coding and reasoning keyword lists in `models.yaml`

## [0.2.0] - 2026-06-10

### Added
- Fallback chains when upstream models return 5xx errors
- Request policies (short-prompt guards, token limits, uncertain → general)
- `general` task tier and confidence scores on routing decisions
- Tiktoken-based token estimation and multimodal content awareness
- `/v1/rerank` and `/v1/ranking` proxy endpoints
- Config hot-reload via `POST /v1/router/reload`
- Optional client auth via `ROUTER_API_KEY`
- Prometheus metrics at `/metrics`
- Shared upstream HTTP client with retries and exponential backoff
- Latency-aware routing using rolling per-model latency stats
- A/B test configuration support in `models.yaml`
- `nim-router catalog-sync` CLI command
- Docker and docker-compose deployment files
- Expanded test suite with `respx` mocks and coverage reporting
- CONTRIBUTING guide, issue templates, and PyPI publish workflow

### Fixed
- `nim-router serve --config` now correctly applies the selected registry
- Flaky CLI tests caused by Rich ANSI color codes
- Duplicate `config/models.yaml` removed; `src/nim_model_router/models.yaml` is canonical

### Changed
- Default ambiguous prompts route to `general` instead of always `agentic`
- Version is sourced from package metadata

## [0.1.0] - 2026-06-09

### Added
- Initial OpenAI-compatible NIM task router
- Rule-based classifier, CLI, proxy server, and basic observability