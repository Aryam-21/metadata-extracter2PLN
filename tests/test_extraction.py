from metadata_extractor2pln.extractors import extract_records
from metadata_extractor2pln.models import (
    PlanDraft,
    PropertySpec,
    SemanticResult,
    SemanticValue,
    Usage,
)
from metadata_extractor2pln.planner import sanitize_plan


class FakeBackend:
    name = "fake-model"
    ready = True

    def extract_semantics(self, *, record, text, properties):
        return (
            SemanticResult(
                values=[
                    SemanticValue(
                        property_name="audience-expertise",
                        value="Expert",
                        strength=0.8,
                        confidence=0.9,
                        evidence_quote="experienced engineers",
                    )
                ]
            ),
            Usage(input_tokens=10, output_tokens=4),
        )


RECORDS = [
    {"id": "a", "content": "For experienced engineers.", "likes": 1, "comments": 0},
    {"id": "b", "content": "For experienced engineers.", "likes": 100, "comments": 20},
]


def _plan():
    draft = PlanDraft(
        id_fields=["id"],
        text_fields=["content"],
        properties=[
            PropertySpec(
                name="audience-expertise",
                description="Intended reader expertise",
                extractor="semantic_text",
                field_paths=["content"],
                allowed_values=["Beginner", "Expert"],
                required=True,
            )
        ],
    )
    return sanitize_plan(
        source_name="articles",
        draft=draft,
        records=RECORDS,
        required_properties=["engagement", "audience-expertise"],
        planner="heuristic",
    )


def test_extracts_semantics_engagement_and_compiles_petta_facts():
    response = extract_records(
        namespace="news",
        plan=_plan(),
        records=RECORDS,
        backend=FakeBackend(),
    )

    first, second = response.records
    first_props = {item.name: item for item in first.properties}
    second_props = {item.name: item for item in second.properties}
    assert first_props["engagement"].value == "low"
    assert second_props["engagement"].value == "high"
    assert first_props["audience-expertise"].value == "Expert"
    assert first_props["audience-expertise"].evidence.quote == "experienced engineers"
    assert all(fact.source.startswith("(: news_") for fact in first.facts)
    assert all("(STV " in fact.source for fact in first.facts)
    assert response.usage.input_tokens == 20


def test_missing_model_is_an_explicit_record_error():
    response = extract_records(
        namespace="news", plan=_plan(), records=RECORDS[:1], backend=None
    )
    assert any(
        "needs a configured model backend" in error
        for error in response.records[0].errors
    )
