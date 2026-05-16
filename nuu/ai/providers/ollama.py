"""
Ollama provider implementation.
"""

from __future__ import annotations

import os

from ..event_stream import AssistantMessageEventStream
from ..types import Context, ModelInfo, SimpleStreamOptions
from .openai_completions import (
    OpenAICompletionsOptions,
    stream_openai_completions,
    stream_simple_openai_completions,
)


def resolve_ollama_base_url(base_url: str) -> str:
    configured_base_url = os.environ.get("OLLAMA_BASE_URL")
    if not configured_base_url:
        return base_url
    normalized = configured_base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


def _with_resolved_base_url(model: ModelInfo) -> ModelInfo:
    resolved_base_url = resolve_ollama_base_url(model.base_url)
    if resolved_base_url == model.base_url:
        return model
    return model.model_copy(update={"base_url": resolved_base_url})


def stream_ollama(
    model: ModelInfo,
    context: Context,
    options: OpenAICompletionsOptions | None = None,
) -> AssistantMessageEventStream:
    return stream_openai_completions(_with_resolved_base_url(model), context, options)


def stream_simple_ollama(
    model: ModelInfo,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    return stream_simple_openai_completions(
        _with_resolved_base_url(model), context, options
    )
