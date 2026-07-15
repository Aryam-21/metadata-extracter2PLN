from types import SimpleNamespace

from metadata_extractor2pln.gemini import GeminiBackend


class FakeModels:
    def __init__(self):
        self.config = None

    def generate_content(self, *, model, contents, config):
        self.config = config
        return SimpleNamespace(
            parsed={"entity_type": "item", "id_fields": ["id"], "properties": []},
            text=None,
            usage_metadata=SimpleNamespace(
                prompt_token_count=7,
                candidates_token_count=3,
            ),
        )


def test_uses_json_schema_path_instead_of_incompatible_openapi_translation():
    models = FakeModels()
    backend = GeminiBackend(api_key=None, model="gemini-test")
    backend._client = SimpleNamespace(models=models)

    result, usage = backend.discover_plan(
        source_name="test",
        records=[{"id": "one", "content": "text"}],
        required_properties=["engagement"],
    )

    assert result.id_fields == ["id"]
    assert usage.input_tokens == 7
    assert models.config.response_schema is None
    assert "additionalProperties" not in models.config.response_json_schema
    assert "maxItems" not in str(models.config.response_json_schema)
    assert models.config.response_json_schema["propertyOrdering"] == [
        "entity_type",
        "id_fields",
        "text_fields",
        "properties",
    ]


def test_model_plan_is_normalized_before_strict_validation():
    models = FakeModels()
    backend = GeminiBackend(api_key=None, model="gemini-test")
    backend._client = SimpleNamespace(models=models)
    models.generate_content = lambda **kwargs: SimpleNamespace(
        parsed={
            "entity_type": "Mindplex_Content",
            "id_fields": ["id"],
            "properties": [
                {
                    "name": "Audience_Expertise",
                    "description": "Reader expertise",
                    "extractor": "semantic_text",
                    "field_paths": ["content"],
                }
            ],
        },
        text=None,
        usage_metadata=SimpleNamespace(),
    )

    result, _ = backend.discover_plan(
        source_name="test",
        records=[{"id": "one", "content": "text"}],
        required_properties=["engagement"],
    )

    assert result.entity_type == "mindplex-content"
    assert result.properties[0].name == "audience-expertise"
    assert result.properties[0].extractor == "semantic_text"
