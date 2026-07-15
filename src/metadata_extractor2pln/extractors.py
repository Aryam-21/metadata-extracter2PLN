from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from .backends import BackendUnavailable, ModelBackend
from .compiler import CONTRACT_VERSION, compile_fact
from .models import (
    Evidence,
    ExtractedProperty,
    ExtractionPlan,
    ExtractResponse,
    PropertySpec,
    RecordResult,
    SemanticValue,
    Usage,
)
from .utils import coerce_float, get_path, parse_datetime, stable_hash, text_from_value


@dataclass(frozen=True)
class Corpus:
    numeric: dict[str, list[float]]


def extract_records(
    *,
    namespace: str,
    plan: ExtractionPlan,
    records: Sequence[dict[str, Any]],
    backend: ModelBackend | None,
) -> ExtractResponse:
    corpus = _build_corpus(plan, records)
    results: list[RecordResult] = []
    usage = Usage()
    for index, record in enumerate(records):
        source_id = _source_id(record, plan, index)
        properties: list[ExtractedProperty] = []
        errors: list[str] = []
        semantic_specs: list[PropertySpec] = []
        for spec in plan.properties:
            if spec.extractor == "semantic_text":
                semantic_specs.append(spec)
                continue
            try:
                extracted = _extract_deterministic(spec, record, corpus)
                if extracted is not None:
                    properties.append(extracted)
                elif spec.required:
                    errors.append(
                        f"required property {spec.name!r} could not be extracted"
                    )
            except ValueError as exc:
                errors.append(f"{spec.name}: {exc}")

        if semantic_specs:
            if backend is None or not backend.ready:
                errors.extend(
                    f"required property {spec.name!r} needs a configured model backend"
                    for spec in semantic_specs
                    if spec.required
                )
            else:
                text = _record_text(record, plan)
                try:
                    result, semantic_usage = backend.extract_semantics(
                        record=record,
                        text=text,
                        properties=semantic_specs,
                    )
                    usage = Usage(
                        input_tokens=usage.input_tokens + semantic_usage.input_tokens,
                        output_tokens=usage.output_tokens
                        + semantic_usage.output_tokens,
                    )
                    extracted, semantic_errors = _validate_semantics(
                        result.values, semantic_specs, text
                    )
                    properties.extend(extracted)
                    errors.extend(semantic_errors)
                except (BackendUnavailable, RuntimeError, ValueError) as exc:
                    errors.append(f"semantic extraction failed: {exc}")

        facts = [
            compile_fact(namespace=namespace, entity_id=source_id, extracted=item)
            for item in properties
            if next(
                spec for spec in plan.properties if spec.name == item.name
            ).include_in_pln
        ]
        results.append(
            RecordResult(
                source_id=source_id,
                entity_id=f"{namespace}_{source_id}",
                properties=properties,
                facts=facts,
                errors=errors,
            )
        )
    return ExtractResponse(
        contract_version=CONTRACT_VERSION,
        plan_fingerprint=plan.fingerprint,
        records=results,
        usage=usage,
    )


def _extract_deterministic(
    spec: PropertySpec, record: dict[str, Any], corpus: Corpus
) -> ExtractedProperty | None:
    values = [get_path(record, path) for path in spec.field_paths]
    if spec.extractor == "structured_field":
        raw = next((value for value in values if value not in (None, "")), None)
        if raw is None:
            return None
        value = text_from_value(raw).strip()
        value = _canonical_allowed(spec, value)
        return _property(spec, value, 1.0, 1.0, f"copied from {spec.field_paths[0]}")
    if spec.extractor == "numeric_bucket":
        raw = next(
            (
                coerce_float(value)
                for value in values
                if coerce_float(value) is not None
            ),
            None,
        )
        if raw is None:
            return None
        population = corpus.numeric.get(spec.name, [])
        percentile = _percentile(raw, population)
        value = (
            "low" if percentile < 1 / 3 else "medium" if percentile < 2 / 3 else "high"
        )
        value = _canonical_allowed(spec, value)
        return _property(
            spec,
            value,
            0.15 + 0.8 * percentile,
            0.9,
            f"relative rank of numeric value {raw:g}",
        )
    if spec.extractor == "date_bucket":
        parsed = next(
            (parse_datetime(value) for value in values if parse_datetime(value)), None
        )
        if parsed is None:
            return None
        value = _canonical_allowed(spec, parsed.strftime("%Y-%m"))
        return _property(
            spec, value, 1.0, 1.0, f"calendar month derived from {parsed.isoformat()}"
        )
    if spec.extractor == "calculated_metric":
        if spec.metric in {"length", "reading-time"}:
            text = " ".join(text_from_value(value) for value in values).strip()
            if not text:
                return None
            words = len(text.split())
            if spec.metric == "length":
                value = (
                    "short" if words < 300 else "medium" if words < 1_000 else "long"
                )
                detail = f"classified from {words} words"
            else:
                minutes = max(1, math.ceil(words / 220))
                value = (
                    "quick" if minutes <= 2 else "medium" if minutes <= 5 else "long"
                )
                detail = f"estimated {minutes} minute reading time from {words} words"
            return _property(spec, _canonical_allowed(spec, value), 1.0, 0.98, detail)
        if spec.metric == "engagement":
            numbers = [coerce_float(value) for value in values]
            score = sum(value for value in numbers if value is not None)
            if not any(value is not None for value in numbers):
                return None
            population = corpus.numeric.get(spec.name, [])
            percentile = _percentile(score, population)
            value = (
                "low"
                if percentile < 1 / 3
                else "medium"
                if percentile < 2 / 3
                else "high"
            )
            detail = f"relative rank of combined engagement score {score:g}"
            return _property(
                spec,
                _canonical_allowed(spec, value),
                0.15 + 0.8 * percentile,
                0.9,
                detail,
            )
    return None


def _validate_semantics(
    values: Sequence[SemanticValue], specs: Sequence[PropertySpec], text: str
) -> tuple[list[ExtractedProperty], list[str]]:
    by_name = {spec.name: spec for spec in specs}
    seen: set[str] = set()
    extracted: list[ExtractedProperty] = []
    errors: list[str] = []
    for item in values:
        spec = by_name.get(item.property_name)
        if spec is None:
            errors.append(f"model returned unrequested property {item.property_name!r}")
            continue
        if item.property_name in seen:
            errors.append(f"model returned duplicate property {item.property_name!r}")
            continue
        seen.add(item.property_name)
        try:
            value = _canonical_allowed(spec, item.value)
        except ValueError as exc:
            errors.append(f"{spec.name}: {exc}")
            continue
        quote = item.evidence_quote
        if quote and quote not in text:
            errors.append(
                f"{spec.name}: discarded an evidence quote not found in source text"
            )
            quote = None
        extracted.append(
            ExtractedProperty(
                name=spec.name,
                value=value,
                strength=item.strength,
                confidence=item.confidence,
                evidence=Evidence(
                    method="semantic_text",
                    detail="model classification constrained by the extraction plan",
                    quote=quote,
                ),
            )
        )
    for spec in specs:
        if spec.required and spec.name not in seen:
            errors.append(
                f"required property {spec.name!r} was not returned by the model"
            )
    return extracted, errors


def _build_corpus(plan: ExtractionPlan, records: Sequence[dict[str, Any]]) -> Corpus:
    numeric: dict[str, list[float]] = {}
    for spec in plan.properties:
        if spec.extractor == "numeric_bucket":
            numeric[spec.name] = [
                number
                for record in records
                for path in spec.field_paths
                if (number := coerce_float(get_path(record, path))) is not None
            ]
        elif spec.extractor == "calculated_metric" and spec.metric == "engagement":
            scores = []
            for record in records:
                values = [
                    coerce_float(get_path(record, path)) for path in spec.field_paths
                ]
                if any(value is not None for value in values):
                    scores.append(sum(value for value in values if value is not None))
            numeric[spec.name] = scores
    return Corpus(numeric=numeric)


def _property(
    spec: PropertySpec, value: str, strength: float, confidence: float, detail: str
) -> ExtractedProperty:
    return ExtractedProperty(
        name=spec.name,
        value=value,
        strength=max(0.0, min(1.0, strength)),
        confidence=max(0.0, min(1.0, confidence)),
        evidence=Evidence(method=spec.extractor, detail=detail),
    )


def _canonical_allowed(spec: PropertySpec, value: str) -> str:
    value = str(value).strip()
    if not value:
        raise ValueError("extracted value is blank")
    if not spec.allowed_values:
        return value
    allowed = {item.casefold(): item for item in spec.allowed_values}
    try:
        return allowed[value.casefold()]
    except KeyError as exc:
        raise ValueError(f"value {value!r} is not in allowed_values") from exc


def _percentile(value: float, population: Sequence[float]) -> float:
    if not population:
        return 0.5
    below = sum(item < value for item in population)
    equal = sum(item == value for item in population)
    return (below + 0.5 * equal) / len(population)


def _record_text(record: dict[str, Any], plan: ExtractionPlan) -> str:
    return "\n".join(
        text_from_value(get_path(record, path)).strip()
        for path in plan.text_fields
        if text_from_value(get_path(record, path)).strip()
    )[:100_000]


def _source_id(record: dict[str, Any], plan: ExtractionPlan, index: int) -> str:
    for path in plan.id_fields:
        value = get_path(record, path)
        if value not in (None, ""):
            return str(value)[:256]
    return f"record-{index}-{stable_hash(record)[:12]}"
