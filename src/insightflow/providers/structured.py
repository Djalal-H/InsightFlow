"""Prompted JSON output from a chat provider, validated with Pydantic.

The workflow asks for JSON in the prompt instead of a provider `response_format` option, so the
callers work with every LiteLLM provider. Malformed output is not an error: callers receive `None`
and use a safe fallback. Provider failures still raise `ProviderError`.
"""

import json
from collections.abc import Sequence

from pydantic import BaseModel, ValidationError

from insightflow.providers.llm import ChatMessage, ChatProvider


def parse_json_object[ModelT: BaseModel](text: str, model: type[ModelT]) -> ModelT | None:
    """Validate the first JSON object in `text`, ignoring code fences and surrounding prose."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(value, dict):
            try:
                return model.model_validate(value)
            except ValidationError:
                return None
        start = text.find("{", start + 1)
    return None


async def complete_json[ModelT: BaseModel](
    chat_provider: ChatProvider,
    messages: Sequence[ChatMessage],
    model: type[ModelT],
) -> ModelT | None:
    """Request a completion and parse it as `model`; return `None` for unusable output."""
    return parse_json_object(await chat_provider.complete(messages), model)
