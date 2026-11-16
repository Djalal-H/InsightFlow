"""The prompt-injection regression set: every case gives exactly its expected signals.

Attack cases must be found, benign controls must stay clean (false positives are failures
too), and answer cases must be changed exactly as expected.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from insightflow.guardrails.output import AnswerGuard
from insightflow.guardrails.screen import screen

DATASET = Path(__file__).parents[2] / "evaluation" / "injection_cases.jsonl"
CASES: list[dict[str, Any]] = [
    json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()
]
CANARY = "ifc-0123456789ab"


def test_case_ids_are_unique_and_both_kinds_are_present() -> None:
    assert len({case["id"] for case in CASES}) == len(CASES) >= 30
    screened = [case for case in CASES if case["target"] != "answer"]
    assert any(case["signals"] for case in screened)
    assert any(not case["signals"] for case in screened)


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_injection_case(case: dict[str, Any]) -> None:
    if case["target"] == "answer":
        text = case["text"].replace("{canary}", CANARY)
        result = AnswerGuard().check(text, allowed_urls=case["allowed_urls"], canary=CANARY)
        assert result.signals == case["signals"]
        assert result.text == case["answer"]
        return
    untrusted = case["target"] in ("web", "document")
    assert screen(case["text"], untrusted=untrusted) == sorted(case["signals"])
