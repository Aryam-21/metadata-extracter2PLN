from __future__ import annotations

from typing import Any, Sequence

import httpx
from typesafe_sdk import Choice, TypeSafeClient

from .backends import BackendUnavailable
from .models import (
    PropertySpec,
    SemanticBatchResult,
    SemanticRecordResult,
    SemanticValue,
    Usage,
)
from .structured_backend import StructuredBackend


class JEVBackend(StructuredBackend):
    """JEV backend for semantic classification."""

    name = "jev"
    provider = "jev"

    def __init__(
        self,
        *,
        model: str = "jev-latest",
        api_key: str | None = None,
        timeout_seconds: float = 45.0,
        transport: str = "typesafe",
        openrouter_api_key: str | None = None,
        openrouter_model: str = "typesafe/jev-1.13",
        openrouter_base_url: str = "https://openrouter.ai/api/alpha/decisions",
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.transport = transport
        self.openrouter_api_key = openrouter_api_key
        self.openrouter_model = openrouter_model
        self.openrouter_base_url = openrouter_base_url

        self._client: TypeSafeClient | None = None

    @property
    def ready(self) -> bool:
        if self.transport == "openrouter":
            return bool(self.openrouter_api_key)
        return bool(self.api_key)

    def discover_plan(
        self,
        *,
        source_name: str,
        records: Sequence[dict[str, Any]],
        required_properties: Sequence[str],
    ) -> tuple[Any, Usage]:
        raise BackendUnavailable(
            "JEV is configured for semantic classification only"
        )

    def extract_semantics(
        self,
        *,
        texts: Sequence[str],
        properties: Sequence[PropertySpec],
    ) -> tuple[SemanticBatchResult, Usage]:
        if not self.ready:
            if self.transport == "openrouter":
                raise BackendUnavailable(
                    "JEV OpenRouter transport is not configured; "
                    "set OPENROUTER_API_KEY"
                )

            raise BackendUnavailable(
                "JEV is not configured; set TYPESAFE_API_KEY"
            )

        if not properties:
            return SemanticBatchResult(records=[]), Usage()

        for prop in properties:
            if not prop.allowed_values:
                raise ValueError(
                    f"JEV classification requires allowed_values "
                    f"for property {prop.name!r}"
                )

        if self.transport == "openrouter":
            return self._extract_with_openrouter(
                texts=texts,
                properties=properties,
            )

        return self._extract_with_typesafe(
            texts=texts,
            properties=properties,
        )

    def _extract_with_typesafe(
        self,
        *,
        texts: Sequence[str],
        properties: Sequence[PropertySpec],
    ) -> tuple[SemanticBatchResult, Usage]:
        client = self._get_client()

        records: list[SemanticRecordResult] = []
        total_input_tokens = 0
        total_output_tokens = 0

        for record_index, text in enumerate(texts):
            questions = self._build_questions(properties)

            try:
                result = client.system_one(
                    state=text[:12000],
                    questions=questions,
                    model=self.model,
                )
            except Exception as exc:
                raise BackendUnavailable(
                    f"JEV classification failed: {exc}"
                ) from exc

            values = self._convert_answers(
                properties=properties,
                answers=result.answers,
            )

            records.append(
                SemanticRecordResult(
                    record_index=record_index,
                    values=values,
                )
            )

            if result.usage.input_tokens is not None:
                total_input_tokens += result.usage.input_tokens

            if result.usage.output_tokens is not None:
                total_output_tokens += result.usage.output_tokens

        return (
            SemanticBatchResult(records=records),
            Usage(
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
            ),
        )

    def _extract_with_openrouter(
        self,
        *,
        texts: Sequence[str],
        properties: Sequence[PropertySpec],
    ) -> tuple[SemanticBatchResult, Usage]:
        records: list[SemanticRecordResult] = []
        total_input_tokens = 0
        total_output_tokens = 0

        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=self.timeout_seconds) as client:
            for record_index, text in enumerate(texts):
                questions = self._build_openrouter_questions(properties)

                payload = {
                    "model": self.openrouter_model,
                    "state": text[:12000],
                    "questions": questions,
                }

                try:
                    response = client.post(
                        self.openrouter_base_url,
                        headers=headers,
                        json=payload,
                    )
                    response.raise_for_status()
                    result = response.json()
                except Exception as exc:
                    raise BackendUnavailable(
                        f"OpenRouter JEV classification failed: {exc}"
                    ) from exc

                answers = result.get("answers")
                if not isinstance(answers, dict):
                    raise BackendUnavailable(
                        "OpenRouter JEV response did not contain "
                        "a valid 'answers' object"
                    )

                values = self._convert_answers(
                    properties=properties,
                    answers=answers,
                )

                records.append(
                    SemanticRecordResult(
                        record_index=record_index,
                        values=values,
                    )
                )

                usage = result.get("usage", {})

                if isinstance(usage, dict):
                    input_tokens = usage.get("input_tokens", 0) or 0
                    output_tokens = usage.get("output_tokens", 0) or 0

                    total_input_tokens += int(input_tokens)
                    total_output_tokens += int(output_tokens)

        return (
            SemanticBatchResult(records=records),
            Usage(
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
            ),
        )

    @staticmethod
    def _build_questions(
        properties: Sequence[PropertySpec],
    ) -> dict[str, Choice]:
        return {
            prop.name: Choice(
                instructions=prop.description,
                criteria={
                    value: None
                    for value in prop.allowed_values
                },
            )
            for prop in properties
        }

    @staticmethod
    def _build_openrouter_questions(
        properties: Sequence[PropertySpec],
    ) -> dict[str, dict[str, Any]]:
        return {
            prop.name: {
                "type": "choice",
                "instructions": prop.description,
                "criteria": {
                    value: None
                    for value in prop.allowed_values
                },
            }
            for prop in properties
        }

    @staticmethod
    def _convert_answers(
        *,
        properties: Sequence[PropertySpec],
        answers: dict[str, Any],
    ) -> list[SemanticValue]:
        values: list[SemanticValue] = []

        for prop in properties:
            answer = answers.get(prop.name)

            if answer is None:
                raise BackendUnavailable(
                    f"JEV response is missing answer for "
                    f"property {prop.name!r}"
                )

            if isinstance(answer, dict):
                choice = answer.get("choice")
                confidence = answer.get("confidence", 0.0)
                probabilities = answer.get("probabilities", {})
            else:
                choice = answer.choice
                confidence = answer.confidence
                probabilities = answer.probabilities

            if not isinstance(choice, str):
                raise BackendUnavailable(
                    f"JEV returned an invalid choice for "
                    f"property {prop.name!r}"
                )

            strength = probabilities.get(choice, 0.0)

            values.append(
                SemanticValue(
                    property_name=prop.name,
                    value=choice,
                    strength=float(strength),
                    confidence=float(confidence),
                    evidence_quote=None,
                )
            )

        return values

    def _get_client(self) -> TypeSafeClient:
        if self._client is None:
            self._client = TypeSafeClient(
                api_key=self.api_key,
                timeout=self.timeout_seconds,
            )

        return self._client

    def _request(
        self,
        prompt: str,
        schema: type,
    ) -> tuple[Any, Usage]:
        raise BackendUnavailable(
            "JEV does not support generic structured generation"
        )