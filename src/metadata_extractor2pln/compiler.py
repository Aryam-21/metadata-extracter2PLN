from __future__ import annotations

import hashlib
import json

from .models import CompiledFact, ExtractedProperty
from .utils import sanitize_symbol


CONTRACT_VERSION = "pettachainer.horn.v1"


def compile_fact(
    *,
    namespace: str,
    entity_id: str,
    extracted: ExtractedProperty,
) -> CompiledFact:
    predicate = extracted.name
    entity = sanitize_symbol(f"{namespace}_{entity_id}")
    value = json.dumps(extracted.value, ensure_ascii=False)
    body = f"({predicate} {entity} {value})"
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
    proof_id = sanitize_symbol(f"{namespace}_{digest}")
    source = (
        f"(: {proof_id} {body} "
        f"(STV {_number(extracted.strength)} {_number(extracted.confidence)}))"
    )
    return CompiledFact(
        source=source,
        proof_id=proof_id,
        idempotency_key=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        property_name=predicate,
    )


def _number(value: float) -> str:
    return format(value, ".15g")
