from metadata_extractor2pln.backends import BackendUnavailable
from metadata_extractor2pln.models import PlanDraft, PropertySpec
from metadata_extractor2pln.planner import (
    discover_plan,
    sanitize_plan,
    validate_and_pin_plan,
)


RECORDS = [
    {
        "article_id": "a-1",
        "body": "Short technical article.",
        "likes": 10,
        "comments": 2,
        "published_at": "2026-07-15T10:00:00Z",
    },
    {
        "article_id": "a-2",
        "body": "A second article with more detail.",
        "likes": 30,
        "comments": 8,
        "published_at": "2026-07-14T10:00:00Z",
    },
]


def test_heuristic_plan_discovers_paths_and_enforces_required_properties():
    plan, model, usage = discover_plan(
        source_name="articles",
        records=RECORDS,
        required_properties=["engagement", "audience_expertise"],
        use_model=False,
    )

    specs = {item.name: item for item in plan.properties}
    assert model is None
    assert usage.input_tokens == 0
    assert plan.fingerprint
    assert "body" in plan.text_fields
    assert specs["engagement"].extractor == "calculated_metric"
    assert specs["engagement"].required is True
    assert {"likes", "comments"}.issubset(specs["engagement"].field_paths)
    assert specs["audience-expertise"].extractor == "semantic_text"
    assert {
        "length-bucket",
        "reading-time",
        "topic",
        "tone",
        "content-type",
        "primary-goal",
        "sentiment",
        "complexity",
        "actionability",
    }.issubset(specs)


def test_sanitizer_replaces_unsafe_model_engagement_definition():
    draft = PlanDraft(
        id_fields=["missing", "article_id"],
        text_fields=["body", "invented"],
        properties=[
            PropertySpec(
                name="engagement",
                description="Model tries to copy arbitrary content",
                extractor="structured_field",
                field_paths=["body"],
            ),
            PropertySpec(
                name="title",
                description="Identity metadata must not become a semantic property",
                extractor="structured_field",
                field_paths=["body"],
            ),
        ],
    )

    plan = sanitize_plan(
        source_name="articles",
        draft=draft,
        records=RECORDS,
        required_properties=["engagement"],
        planner="bedrock",
    )

    engagement = next(item for item in plan.properties if item.name == "engagement")
    assert engagement.extractor == "calculated_metric"
    assert engagement.metric == "engagement"
    assert "title" not in {item.name for item in plan.properties}
    assert plan.id_fields == ["article_id"]
    assert plan.text_fields == ["body"]


def test_sanitizer_drops_unbounded_optional_semantic_properties():
    draft = PlanDraft(
        id_fields=["article_id"],
        text_fields=["body"],
        properties=[
            PropertySpec(
                name="summary",
                description="Free-form article summary",
                extractor="semantic_text",
                field_paths=["body"],
            )
        ],
    )

    plan = sanitize_plan(
        source_name="articles",
        draft=draft,
        records=RECORDS,
        required_properties=["engagement"],
        planner="bedrock",
    )

    assert "summary" not in {item.name for item in plan.properties}
    assert "topic" in {item.name for item in plan.properties}


def test_modified_plan_fingerprint_is_rejected():
    plan, _, _ = discover_plan(
        source_name="articles",
        records=RECORDS,
        required_properties=["engagement"],
        use_model=False,
    )
    modified = plan.model_copy(update={"entity_type": "changed"})

    try:
        validate_and_pin_plan(modified)
    except ValueError as exc:
        assert "fingerprint" in str(exc)
    else:
        raise AssertionError("modified plan should have been rejected")


def test_model_planning_failure_falls_back_to_deterministic_plan():
    class FailingBackend:
        name = "unavailable-model"
        ready = True

        def discover_plan(self, **kwargs):
            raise BackendUnavailable("rate limited")

    plan, model, usage = discover_plan(
        source_name="articles",
        records=RECORDS,
        required_properties=["engagement"],
        backend=FailingBackend(),
        use_model=True,
    )

    assert plan.planner == "heuristic"
    assert model is None
    assert usage.input_tokens == 0
