from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from typing import Any, Iterable, Mapping


def normalize_name(value: str, fallback: str = "property") -> str:
    normalized = str(value or "").strip().lower().replace("_", "-").replace(" ", "-")
    normalized = re.sub(r"[^a-z0-9-]+", "-", normalized)
    normalized = re.sub(r"-+", "-", normalized).strip("-")
    if not normalized or not normalized[0].isalpha():
        normalized = f"p-{normalized}" if normalized else fallback
    return normalized[:64]


def sanitize_symbol(value: Any, fallback: str = "unknown") -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "").strip()).strip("_")
    if not normalized:
        normalized = fallback
    if not (normalized[0].isalpha() or normalized[0] == "_"):
        normalized = f"n_{normalized}"
    return normalized[:128]


def get_path(data: Any, path: str) -> Any:
    current = data
    for raw_part in path.split("."):
        if current is None:
            return None
        wants_list = raw_part.endswith("[]")
        part = raw_part[:-2] if wants_list else raw_part
        if isinstance(current, Mapping):
            current = current.get(part)
        elif isinstance(current, list):
            values = [item.get(part) for item in current if isinstance(item, Mapping)]
            current = values
        else:
            return None
        if wants_list and not isinstance(current, list):
            current = [current]
    return first_scalar(current)


def first_scalar(value: Any) -> Any:
    if isinstance(value, list):
        for item in value:
            scalar = first_scalar(item)
            if scalar not in (None, ""):
                return scalar
        return None
    if isinstance(value, Mapping):
        for key in ("slug", "name", "title", "value", "content", "id"):
            if value.get(key) not in (None, ""):
                return value[key]
        return None
    return value


def text_from_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(filter(None, (text_from_value(item).strip() for item in value)))
    if isinstance(value, Mapping):
        if value.get("content") or value.get("text"):
            return str(value.get("content") or value.get("text"))
        return " ".join(
            filter(None, (text_from_value(item).strip() for item in value.values()))
        )
    return str(value)


def coerce_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def parse_datetime(value: Any) -> dt.datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(text)
        return parsed.replace(tzinfo=None)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return dt.datetime.strptime(text[:19], fmt)
        except ValueError:
            continue
    return None


def schema_paths(records: Iterable[Mapping[str, Any]], max_depth: int = 2) -> list[str]:
    paths: set[str] = set()

    def walk(value: Any, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                child = f"{prefix}.{key}" if prefix else str(key)
                paths.add(child)
                walk(item, child, depth + 1)
        elif isinstance(value, list) and value:
            child = f"{prefix}[]"
            paths.add(child)
            walk(value[0], child, depth + 1)

    for record in records:
        walk(record, "", 0)
    return sorted(paths)


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
