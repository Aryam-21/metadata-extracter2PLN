from __future__ import annotations

import json
from typing import Any

from google import genai
from google.genai import types

from .backends import BackendUnavailable
from .models import Usage
from .structured_backend import StructuredBackend


class GeminiBackend(StructuredBackend):
    provider = "gemini"

    def __init__(
        self, *, api_key: str | None, model: str, timeout_seconds: float = 45.0
    ):
        self.name = model
        self.ready = bool(api_key)
        self._model = model
        self._timeout_ms = int(timeout_seconds * 1_000)
        self._client = genai.Client(api_key=api_key) if api_key else None

    def _request(self, prompt: str, schema: type) -> tuple[Any, Usage]:
        if self._client is None:
            raise BackendUnavailable("GEMINI_API_KEY is not configured")
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=_compact_schema(schema.model_json_schema()),
                    temperature=0,
                    http_options=types.HttpOptions(timeout=self._timeout_ms),
                ),
            )
        except Exception as exc:
            raise BackendUnavailable(f"Gemini request failed: {exc}") from exc
        parsed = response.parsed
        if parsed is None and response.text:
            try:
                parsed = json.loads(response.text)
            except json.JSONDecodeError as exc:
                raise BackendUnavailable("Gemini returned invalid JSON") from exc
        if parsed is None:
            raise BackendUnavailable("Gemini returned no structured result")
        metadata = response.usage_metadata
        return parsed, Usage(
            input_tokens=int(getattr(metadata, "prompt_token_count", 0) or 0),
            output_tokens=int(getattr(metadata, "candidates_token_count", 0) or 0),
        )


def _compact_schema(value: Any) -> Any:
    """Keep Gemini's constrained decoder small; strict validation happens afterwards."""
    supported = {
        "$defs",
        "$ref",
        "type",
        "properties",
        "required",
        "items",
        "enum",
        "anyOf",
    }
    if isinstance(value, dict):
        compact = {}
        for key, item in value.items():
            if key not in supported:
                continue
            if key in {"properties", "$defs"} and isinstance(item, dict):
                compact[key] = {
                    name: _compact_schema(child) for name, child in item.items()
                }
            else:
                compact[key] = _compact_schema(item)
        if compact.get("type") == "object" and isinstance(
            compact.get("properties"), dict
        ):
            compact["propertyOrdering"] = list(compact["properties"])
        return compact
    if isinstance(value, list):
        return [_compact_schema(item) for item in value]
    return value
