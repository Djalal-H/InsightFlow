"""Tests for prompted JSON parsing used by the workflow's LLM nodes."""

import pytest

from insightflow.agents.models import JudgeVerdict, RouteDecision
from insightflow.agents.nodes.verify import JUDGE_SYSTEM_PROMPT
from insightflow.agents.structured import complete_json, parse_json_object
from insightflow.core.exceptions import ProviderTimeoutError
from tests.fakes import ScriptedChatProvider


@pytest.mark.parametrize(
    "text",
    [
        '{"route": "direct", "standalone_query": " hi "}',
        '```json\n{"route": "direct", "standalone_query": "hi"}\n```',
        'Here is the decision: {"route": "direct", "standalone_query": "hi"} Done.',
        '{not json} {"route": "direct", "standalone_query": "hi"}',
        '{"route": "direct", "standalone_query": "hi", "reason": "greeting"}',
    ],
)
def test_parse_json_object_finds_the_first_valid_object(text: str) -> None:
    assert parse_json_object(text, RouteDecision) == RouteDecision(
        route="direct", standalone_query="hi"
    )


@pytest.mark.parametrize(
    "text",
    [
        "",
        "no json here",
        '["route", "direct"]',
        '{"route": "elsewhere", "standalone_query": "hi"}',
        '{"route": "direct"}',
        '{"route": "direct", "standalone_query": "unterminated',
    ],
)
def test_parse_json_object_returns_none_for_unusable_output(text: str) -> None:
    assert parse_json_object(text, RouteDecision) is None


async def test_complete_json_parses_the_model_reply() -> None:
    chat = ScriptedChatProvider(judge='{"grounded": false, "unsupported_claims": ["x"]}')

    verdict = await complete_json(
        chat, [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}], JudgeVerdict
    )

    assert verdict == JudgeVerdict(grounded=False, unsupported_claims=["x"])


async def test_complete_json_propagates_provider_errors() -> None:
    chat = ScriptedChatProvider(judge=ProviderTimeoutError())

    with pytest.raises(ProviderTimeoutError):
        await complete_json(
            chat, [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}], JudgeVerdict
        )
