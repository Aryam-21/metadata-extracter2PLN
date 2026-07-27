from __future__ import annotations

import json
from typing import Any

import boto3
from botocore.config import Config

from .backends import BackendUnavailable
from .models import Usage
from .structured_backend import StructuredBackend


class BedrockBackend(StructuredBackend):
    provider = "bedrock"

    def __init__(
        self,
        *,
        model: str,
        region: str,
        access_key: str | None = None,
        secret_key: str | None = None,
        timeout_seconds: float = 45.0,
        max_tokens: int = 8192,
    ):
        self.name = model
        self.ready = bool(model and region)
        self._model = model
        self._max_tokens = max_tokens
        credentials = (
            {"aws_access_key_id": access_key, "aws_secret_access_key": secret_key}
            if access_key and secret_key
            else {}
        )
        self._client = None
        self._client_kwargs = {
            "region_name": region,
            "config": Config(
                connect_timeout=timeout_seconds,
                read_timeout=timeout_seconds,
                retries={"max_attempts": 3, "mode": "standard"},
            ),
            **credentials,
        }

    def _request(self, prompt: str, schema: type) -> tuple[Any, Usage]:
        schema_document = _bedrock_schema(schema.model_json_schema())
        try:
            client = self._client or boto3.client("bedrock-runtime", **self._client_kwargs)
            response = client.converse(
                modelId=self._model,
                system=[{"text": "Return only data that conforms to the requested JSON schema."}],
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"temperature": 0, "maxTokens": self._max_tokens},
                outputConfig={
                    "textFormat": {
                        "type": "json_schema",
                        "structure": {
                            "jsonSchema": {
                                "name": schema.__name__.lower(),
                                "description": "Validated metadata extraction result",
                                "schema": json.dumps(
                                    schema_document,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ),
                            }
                        },
                    }
                },
            )
        except Exception as exc:
            raise BackendUnavailable(
                f"Bedrock request failed: {exc.__class__.__name__}"
            ) from exc

        content = response.get("output", {}).get("message", {}).get("content", [])
        text = next(
            (item["text"] for item in content if isinstance(item, dict) and item.get("text")),
            None,
        )
        if not text:
            raise BackendUnavailable("Bedrock returned no structured result")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise BackendUnavailable("Bedrock returned invalid JSON") from exc
        usage = response.get("usage") or {}
        return parsed, Usage(
            input_tokens=int(usage.get("inputTokens") or 0),
            output_tokens=int(usage.get("outputTokens") or 0),
        )


def _bedrock_schema(value: Any) -> Any:
    supported = {
        "$defs",
        "$ref",
        "type",
        "properties",
        "required",
        "items",
        "enum",
        "const",
        "anyOf",
        "allOf",
        "format",
        "description",
        "additionalProperties",
    }
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key not in supported:
                continue
            if key in {"properties", "$defs"} and isinstance(item, dict):
                result[key] = {name: _bedrock_schema(child) for name, child in item.items()}
            else:
                result[key] = _bedrock_schema(item)
        if result.get("type") == "object":
            result["additionalProperties"] = False
        return result
    if isinstance(value, list):
        return [_bedrock_schema(item) for item in value]
    return value
