from __future__ import annotations

from contextvars import ContextVar

_request_deadline: ContextVar[float | None] = ContextVar(
    "request_deadline",
    default=None,
)


def set_request_deadline(deadline: float | None):
    return _request_deadline.set(deadline)


def reset_request_deadline(token) -> None:
    _request_deadline.reset(token)


def get_request_deadline() -> float | None:
    return _request_deadline.get()