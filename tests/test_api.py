from fastapi.testclient import TestClient

from metadata_extractor2pln.api import create_app
from metadata_extractor2pln.config import Settings


def _client():
    settings = Settings(
        environment="test",
        api_keys=("tester:secret",),
        allowed_hosts=("testserver",),
        rate_limit_per_minute=20,
    )
    return TestClient(create_app(settings))


def test_health_is_public_and_api_is_authenticated():
    with _client() as client:
        assert client.get("/health").status_code == 200
        assert client.post("/v1/run", json={}).status_code == 401


def test_offline_run_returns_deterministic_facts():
    payload = {
        "namespace": "demo",
        "source_name": "articles",
        "use_model": False,
        "required_properties": ["engagement"],
        "records": [
            {"id": "one", "content": "A short article.", "likes": 4, "comments": 1},
            {
                "id": "two",
                "content": "Another short article.",
                "likes": 20,
                "comments": 8,
            },
        ],
    }
    with _client() as client:
        response = client.post(
            "/v1/run",
            headers={"Authorization": "Bearer secret"},
            json=payload,
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["plan"]["fingerprint"]
    assert len(body["extraction"]["records"]) == 2
    engagement_fact = next(
        fact
        for fact in body["extraction"]["records"][0]["facts"]
        if fact["property_name"] == "engagement"
    )
    assert engagement_fact["atom"].startswith("(engagement ")


def test_body_size_limit_applies_before_validation():
    settings = Settings(
        environment="test",
        api_keys=("tester:secret",),
        allowed_hosts=("testserver",),
        max_request_bytes=20,
    )
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/run",
            headers={"Authorization": "Bearer secret"},
            content=b"x" * 21,
        )
    assert response.status_code == 413


def test_body_size_limit_cannot_be_bypassed_with_chunked_input():
    settings = Settings(
        environment="test",
        api_keys=("tester:secret",),
        allowed_hosts=("testserver",),
        max_request_bytes=10,
    )
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/run",
            headers={"Authorization": "Bearer secret"},
            content=iter([b"123456", b"789012"]),
        )
    assert response.status_code == 413
