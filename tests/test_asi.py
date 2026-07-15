import json

import httpx

from metadata_extractor2pln.asi import AsiBackend


def test_asi_uses_structured_chat_completion():
    captured = {}

    def respond(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "entity_type": "article",
                                    "id_fields": ["id"],
                                    "text_fields": ["content"],
                                    "properties": [],
                                }
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 8, "completion_tokens": 3},
            },
        )

    backend = AsiBackend(api_key="secret", model="asi1-mini")
    backend._client = httpx.Client(
        transport=httpx.MockTransport(respond), base_url="https://api.asi1.ai/v1"
    )

    plan, usage = backend.discover_plan(
        source_name="mindplex",
        records=[{"id": "one", "content": "text"}],
        required_properties=["engagement"],
    )

    assert plan.entity_type == "article"
    assert usage.input_tokens == 8
    assert captured["model"] == "asi1-mini"
    assert captured["response_format"]["type"] == "json_schema"
    assert captured["response_format"]["json_schema"]["strict"] is True
