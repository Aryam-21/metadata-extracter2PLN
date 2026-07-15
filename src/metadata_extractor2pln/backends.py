from __future__ import annotations

from typing import Any, Protocol, Sequence

from .models import PlanDraft, PropertySpec, SemanticResult, Usage


class ModelBackend(Protocol):
    name: str
    ready: bool

    def discover_plan(
        self,
        *,
        source_name: str,
        records: Sequence[dict[str, Any]],
        required_properties: Sequence[str],
    ) -> tuple[PlanDraft, Usage]: ...

    def extract_semantics(
        self,
        *,
        record: dict[str, Any],
        text: str,
        properties: Sequence[PropertySpec],
    ) -> tuple[SemanticResult, Usage]: ...


class BackendUnavailable(RuntimeError):
    pass
