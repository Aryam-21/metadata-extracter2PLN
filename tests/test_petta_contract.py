import sys
from pathlib import Path

import pytest

from metadata_extractor2pln.compiler import compile_fact
from metadata_extractor2pln.models import Evidence, ExtractedProperty


def test_compiled_fact_passes_sibling_pettachainer_policy():
    sibling = Path(__file__).parents[2] / "PeTTaChainer"
    if not sibling.exists():
        pytest.skip("sibling PeTTaChainer checkout is unavailable")
    sys.path.insert(0, str(sibling))
    from pettachainer.server.policy import validate_statement_source

    fact = compile_fact(
        namespace="news",
        entity_id="article-42",
        extracted=ExtractedProperty(
            name="engagement",
            value="high",
            strength=0.8,
            confidence=0.9,
            evidence=Evidence(method="calculated_metric", detail="test fixture"),
        ),
    )

    parsed = validate_statement_source(fact.source, 32_768)
    assert parsed.kind == "fact"
