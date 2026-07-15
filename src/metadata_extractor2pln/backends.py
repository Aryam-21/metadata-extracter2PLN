from __future__ import annotations

from typing import Any, Protocol, Sequence

from .models import PlanDraft, PropertySpec, SemanticBatchResult, Usage


class ModelBackend(Protocol):
    name: str
    provider: str
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
        texts: Sequence[str],
        properties: Sequence[PropertySpec],
    ) -> tuple[SemanticBatchResult, Usage]: ...


class BackendUnavailable(RuntimeError):
    pass
