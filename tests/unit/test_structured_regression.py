"""Regression set for prompted-JSON parsing: real-world model output shapes per workflow role.

`evaluation/structured_cases.jsonl` holds raw outputs and the expected parse (`null` means the
caller's fallback runs: the agent searches once then answers, the judge reports `unchecked`,
and the retrieval steps leave the candidates unchanged). Add a case when a hosted model returns
a new shape.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from insightflow.agents.models import AgentAction, JudgeVerdict
from insightflow.providers.structured import parse_json_object
from insightflow.rag.corrective import RelevanceGrade
from insightflow.rag.query_expansion import ExpandedQueries
from insightflow.rag.refinement import SentenceSelection

DATASET = Path(__file__).parents[2] / "evaluation" / "structured_cases.jsonl"
MODELS: dict[str, type[BaseModel]] = {
    "agent": AgentAction,
    "judge": JudgeVerdict,
    "expand": ExpandedQueries,
    "grade": RelevanceGrade,
    "compress": SentenceSelection,
}


def load_cases() -> list[dict[str, Any]]:
    lines = DATASET.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


CASES = load_cases()


def test_the_dataset_covers_every_role_with_unique_ids() -> None:
    assert len({case["id"] for case in CASES}) == len(CASES)
    assert {case["role"] for case in CASES} == set(MODELS)
    for role in MODELS:
        role_cases = [case for case in CASES if case["role"] == role]
        assert any(case["expected"] is None for case in role_cases), role
        assert any(case["expected"] is not None for case in role_cases), role


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_model_output_parses_as_expected(case: dict[str, Any]) -> None:
    parsed = parse_json_object(case["output"], MODELS[case["role"]])
    actual = parsed.model_dump() if parsed is not None else None
    assert actual == case["expected"]
