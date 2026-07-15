from __future__ import annotations

import json
import time
from typing import Any

import httpx

from .backends import BackendUnavailable
from .models import Usage
from .structured_backend import StructuredBackend


class AsiBackend(StructuredBackend):
    provider = "asi"

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        base_url: str = "https://api.asi1.ai/v1",
        timeout_seconds: float = 45.0,
    ):
        self.name = model
        self.ready = bool(api_key)
        self._model = model
        self._client = (
            httpx.Client(
                base_url=base_url.rstrip("/"),
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=timeout_seconds,
            )
            if api_key
            else None
        )

    def _request(self, prompt: str, schema: type) -> tuple[Any, Usage]:
        if self._client is None:
            raise BackendUnavailable("ASI_ONE_API_KEY is not configured")
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only JSON matching the supplied schema.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__.lower(),
                    "strict": True,
                    "schema": schema.model_json_schema(),
                },
            },
        }
        response = None
        for attempt in range(3):
            try:
                response = self._client.post("/chat/completions", json=payload)
            except httpx.HTTPError as exc:
                if attempt == 2:
                    raise BackendUnavailable(f"ASI request failed: {exc}") from exc
                time.sleep(2**attempt)
                continue
            if response.status_code not in {408, 429, 500, 502, 503, 504}:
                break
            if attempt == 2:
                break
            delay = response.headers.get("retry-after")
            try:
                seconds = min(10.0, max(0.0, float(delay))) if delay else 2**attempt
            except ValueError:
                seconds = 2**attempt
            time.sleep(seconds)
        if response is None:
            raise BackendUnavailable("ASI returned no response")
        try:
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except (httpx.HTTPError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise BackendUnavailable(
                f"ASI returned an invalid response (HTTP {response.status_code})"
            ) from exc
        usage = body.get("usage") or {}
        return parsed, Usage(
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
        )
