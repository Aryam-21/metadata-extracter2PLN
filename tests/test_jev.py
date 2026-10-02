from types import SimpleNamespace

import httpx
import pytest
import time

from metadata_extractor2pln.request_context import (
    reset_request_deadline,
    set_request_deadline,
)

from metadata_extractor2pln.backends import BackendUnavailable
from metadata_extractor2pln.jev_backend import JEVBackend
from metadata_extractor2pln.models import PropertySpec


class FakeJEVClient:
    def __init__(self):
        self.calls = []

    def system_one(self, *, state, questions, model):
        self.calls.append(
            {
                "state": state,
                "questions": questions,
                "model": model,
            }
        )

        answers = {
            "sentiment": SimpleNamespace(
                choice="positive",
                confidence=0.90,
                probabilities={
                    "positive": 0.80,
                    "negative": 0.20,
                },
            )
        }

        return SimpleNamespace(
            answers=answers,
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=5,
            ),
        )


def make_sentiment_property() -> PropertySpec:
    return PropertySpec(
        name="sentiment",
        description="Classify the sentiment of the text.",
        extractor="semantic_text",
        field_paths=[],
        allowed_values=["positive", "negative"],
    )


def test_jev_extract_semantics():
    backend = JEVBackend(
        model="jev-test",
        api_key="test-key",
    )

    fake_client = FakeJEVClient()
    backend._client = fake_client

    result, usage = backend.extract_semantics(
        texts=["This is an excellent article."],
        properties=[make_sentiment_property()],
    )

    value = result.records[0].values[0]

    assert value.property_name == "sentiment"
    assert value.value == "positive"
    assert value.strength == 0.80
    assert value.confidence == 0.90
    assert value.evidence_quote is None

    assert usage.input_tokens == 10
    assert usage.output_tokens == 5

    assert len(fake_client.calls) == 1

    call = fake_client.calls[0]

    assert call["state"] == "This is an excellent article."
    assert call["model"] == "jev-test"

    question = call["questions"]["sentiment"]

    assert question.criteria == {
        "positive": None,
        "negative": None,
    }


def test_jev_requires_api_key():
    backend = JEVBackend()

    assert backend.ready is False

    with pytest.raises(BackendUnavailable, match="TYPESAFE_API_KEY"):
        backend.extract_semantics(
            texts=["Some text"],
            properties=[make_sentiment_property()],
        )


def test_jev_does_not_discover_plans():
    backend = JEVBackend(api_key="test-key")

    with pytest.raises(
        BackendUnavailable,
        match="classification only",
    ):
        backend.discover_plan(
            source_name="test-source",
            records=[{"text": "example"}],
            required_properties=["sentiment"],
        )


def test_jev_requires_allowed_values():
    backend = JEVBackend(api_key="test-key")

    property_without_choices = PropertySpec(
        name="sentiment",
        description="Classify the sentiment.",
        extractor="semantic_text",
        field_paths=[],
        allowed_values=[],
    )

    with pytest.raises(
        ValueError,
        match="allowed_values",
    ):
        backend.extract_semantics(
            texts=["Some text"],
            properties=[property_without_choices],
        )

def test_jev_openrouter_transport():
    backend = JEVBackend(
        transport="openrouter",
        openrouter_api_key="test-key",
        openrouter_model="typesafe/jev-1.13",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/alpha/decisions"
        assert request.headers["Authorization"] == "Bearer test-key"

        body = request.read()
        assert b"typesafe/jev-1.13" in body

        return httpx.Response(
            200,
            json={
                "model": "typesafe/jev-1.13-20260917",
                "answers": {
                    "sentiment": {
                        "type": "choice",
                        "choice": "positive",
                        "probabilities": {
                            "positive": 0.95,
                            "negative": 0.05,
                        },
                        "confidence": 0.98,
                    }
                },
                "usage": {
                    "input_tokens": 302,
                    "output_tokens": 32,
                },
            },
        )

    transport = httpx.MockTransport(handler)

    original_client = httpx.Client

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self._client = original_client(
                transport=transport,
                **{
                    key: value
                    for key, value in kwargs.items()
                    if key != "transport"
                },
            )

        def __enter__(self):
            return self._client.__enter__()

        def __exit__(self, *args):
            return self._client.__exit__(*args)

        def post(self, *args, **kwargs):
            return self._client.post(*args, **kwargs)

    original_httpx_client = httpx.Client
    httpx.Client = FakeClient

    try:
        result, usage = backend.extract_semantics(
            texts=["This article is excellent."],
            properties=[make_sentiment_property()],
        )
    finally:
        httpx.Client = original_httpx_client

    value = result.records[0].values[0]

    assert value.property_name == "sentiment"
    assert value.value == "positive"
    assert value.strength == 0.95
    assert value.confidence == 0.98
    assert usage.input_tokens == 302
    assert usage.output_tokens == 32
def test_jev_stops_before_request_deadline():
    calls = []

    class FakeClient:
        def system_one(self, *, state, questions, model):
            calls.append(state)
            return {
                "answers": {
                    "sentiment": {
                        "choice": "positive",
                        "confidence": 0.9,
                        "probabilities": {
                            "positive": 0.8,
                            "negative": 0.2,
                        },
                    }
                },
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                },
            }

    backend = JEVBackend(
        model="jev-latest",
        api_key="test-key",
        timeout_seconds=45.0,
        transport="typesafe",
    )

    backend._client = FakeClient()

    # Only 10 seconds remain, but one JEV call may take up to 45 seconds.
    deadline = time.monotonic() + 10.0

    token = set_request_deadline(deadline)
    try:
        result, usage = backend.extract_semantics(
            texts=["first record", "second record"],
            properties=[make_sentiment_property()],
        )
    finally:
        reset_request_deadline(token)

    assert calls == []
    assert result.records == []
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
def test_jev_stops_after_successful_record_when_deadline_is_too_close(
    monkeypatch,
):
    fake_client = FakeJEVClient()

    backend = JEVBackend(
        model="jev-test",
        api_key="test-key",
        timeout_seconds=45.0,
        transport="typesafe",
    )

    backend._client = fake_client

    # Start with 60 seconds remaining.
    current_time = [0.0]
    deadline = 60.0

    def fake_monotonic():
        return current_time[0]

    monkeypatch.setattr(
        "metadata_extractor2pln.jev_backend.time.monotonic",
        fake_monotonic,
    )

    original_system_one = fake_client.system_one

    def system_one_with_time(*, state, questions, model):
        result = original_system_one(
            state=state,
            questions=questions,
            model=model,
        )

        # Simulate the first provider call taking 50 seconds.
        current_time[0] = 50.0

        return result

    fake_client.system_one = system_one_with_time

    monkeypatch.setattr(
        "metadata_extractor2pln.jev_backend.get_request_deadline",
        lambda: deadline,
    )

    result, usage = backend.extract_semantics(
        texts=["first record", "second record"],
        properties=[make_sentiment_property()],
    )

    # Only the first record should have reached JEV.
    assert len(fake_client.calls) == 1
    assert fake_client.calls[0]["state"] == "first record"

    # The successful first result is preserved.
    assert len(result.records) == 1
    assert result.records[0].record_index == 0

    # Usage from the successful call is preserved.
    assert usage.input_tokens == 10
    assert usage.output_tokens == 5
def test_jev_preserves_successful_records_when_later_record_fails():
    class FailingOnSecondClient:
        def __init__(self):
            self.calls = []

        def system_one(self, *, state, questions, model):
            self.calls.append(state)

            if len(self.calls) == 2:
                raise RuntimeError("simulated JEV failure")

            return SimpleNamespace(
                answers={
                    "sentiment": SimpleNamespace(
                        choice="positive",
                        confidence=0.90,
                        probabilities={
                            "positive": 0.80,
                            "negative": 0.20,
                        },
                    )
                },
                usage=SimpleNamespace(
                    input_tokens=10,
                    output_tokens=5,
                ),
            )

    backend = JEVBackend(
        model="jev-test",
        api_key="test-key",
        transport="typesafe",
    )

    fake_client = FailingOnSecondClient()
    backend._client = fake_client

    result, usage = backend.extract_semantics(
        texts=["first record", "second record"],
        properties=[make_sentiment_property()],
    )

    assert fake_client.calls == ["first record", "second record"]

    assert len(result.records) == 1
    assert result.records[0].record_index == 0

    value = result.records[0].values[0]
    assert value.property_name == "sentiment"
    assert value.value == "positive"
    assert value.strength == 0.80
    assert value.confidence == 0.90

    assert usage.input_tokens == 10
    assert usage.output_tokens == 5
def test_jev_openrouter_preserves_successful_records_when_later_record_fails():
    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class FailingOnSecondClient:
        def __init__(self):
            self.calls = []

        def post(self, *args, **kwargs):
            self.calls.append(kwargs)

            if len(self.calls) == 2:
                raise RuntimeError("simulated OpenRouter failure")

            return FakeResponse(
                {
                    "answers": {
                        "sentiment": {
                            "choice": "positive",
                            "confidence": 0.90,
                            "probabilities": {
                                "positive": 0.80,
                                "negative": 0.20,
                            },
                        }
                    },
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                    },
                }
            )

    backend = JEVBackend(
        model="jev-test",
        api_key="test-key",
        transport="openrouter",
        openrouter_api_key="test-openrouter-key",
        openrouter_model="typesafe/jev-1.13",
    )

    fake_client = FailingOnSecondClient()

    class FakeClientContext:
        def __enter__(self):
            return fake_client

        def __exit__(self, *args):
            return None

    original_client = httpx.Client
    httpx.Client = lambda *args, **kwargs: FakeClientContext()

    try:
        result, usage = backend.extract_semantics(
            texts=["first record", "second record"],
            properties=[make_sentiment_property()],
        )
    finally:
        httpx.Client = original_client

    assert len(fake_client.calls) == 2
    assert len(result.records) == 1
    assert result.records[0].record_index == 0

    value = result.records[0].values[0]
    assert value.property_name == "sentiment"
    assert value.value == "positive"
    assert value.strength == 0.80
    assert value.confidence == 0.90

    assert usage.input_tokens == 10
    assert usage.output_tokens == 5