import json

from metadata_extractor2pln.bedrock import BedrockBackend


class FakeBedrockClient:
    def __init__(self):
        self.request = None

    def converse(self, **request):
        self.request = request
        return {
            "output": {
                "message": {
                    "content": [
                        {
                            "text": json.dumps(
                                {
                                    "entity_type": "article",
                                    "id_fields": ["id"],
                                    "text_fields": ["content"],
                                    "properties": [],
                                }
                            )
                        }
                    ]
                }
            },
            "usage": {"inputTokens": 11, "outputTokens": 5},
        }


def test_bedrock_uses_converse_structured_output():
    client = FakeBedrockClient()
    backend = BedrockBackend(model="test-model", region="us-east-1")
    backend._client = client

    plan, usage = backend.discover_plan(
        source_name="mindplex",
        records=[{"id": "one", "content": "text"}],
        required_properties=["engagement"],
    )

    assert plan.entity_type == "article"
    assert usage.input_tokens == 11
    output = client.request["outputConfig"]["textFormat"]
    assert output["type"] == "json_schema"
    schema = json.loads(output["structure"]["jsonSchema"]["schema"])
    assert schema["additionalProperties"] is False
    assert "maxItems" not in json.dumps(schema)
