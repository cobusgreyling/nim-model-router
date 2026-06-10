"""NVIDIA NIM model router — OpenAI-compatible task-based proxy."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("nim-model-router")
except PackageNotFoundError:  # pragma: no cover - editable install without metadata
    __version__ = "0.2.0"
