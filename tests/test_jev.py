from types import SimpleNamespace

import httpx
import pytest

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