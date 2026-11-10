"""JSON-schema checks for tool arguments, shared by MCP and local tools."""

from collections.abc import Mapping
from typing import Any

import jsonschema
from jsonschema.exceptions import best_match

ERROR_DETAIL_MAX_CHARS = 300


def argument_problem(schema: dict[str, Any], arguments: Mapping[str, Any]) -> str | None:
    """Return a short reason when the arguments do not match the JSON schema, else None."""
    try:
        validator_class = jsonschema.validators.validator_for(schema)
        validator_class.check_schema(schema)
    except jsonschema.SchemaError:
        return "The tool's input schema is not valid."
    error = best_match(validator_class(schema).iter_errors(dict(arguments)))
    if error is None:
        return None
    location = ".".join(str(part) for part in error.absolute_path)
    reason = f"{location}: {error.message}" if location else error.message
    return clip(f"Invalid arguments: {reason}")


def clip(text: str) -> str:
    if len(text) <= ERROR_DETAIL_MAX_CHARS:
        return text
    return text[: ERROR_DETAIL_MAX_CHARS - 3] + "..."
