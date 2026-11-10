"""Tests for prompted JSON parsing used by the workflow's LLM nodes."""

import pytest

from insightflow.agents.models import AgentAction, JudgeVerdict
from insightflow.agents.nodes.verify import JUDGE_SYSTEM_PROMPT
from insightflow.agents.structured import complete_json, parse_json_object
from insightflow.core.exceptions import ProviderTimeoutError
from tests.fakes import ScriptedChatProvider


@pytest.mark.parametrize(
    "text",
    [
        '{"thought": " x ", "action": "Answer"}',
        '```json\n{"thought": "x", "action": "answer"}\n```',
        'Here is the step: {"thought": "x", "action": "answer"} Done.',
        '{not json} {"thought": "x", "action": "answer"}',
        '{"thought": "x", "action": "answer", "confidence": "high"}',
    ],
)
def test_parse_json_object_finds_the_first_valid_object(text: str) -> None:
    assert parse_json_object(text, AgentAction) == AgentAction(thought="x", action="answer")


@pytest.mark.parametrize(
    "text",
    [
        "",
        "no json here",
        '["action", "answer"]',
        '{"thought": "x", "action": "  "}',
        '{"thought": "x"}',
        '{"thought": "x", "action": "unterminated',
    ],
)
def test_parse_json_object_returns_none_for_unusable_output(text: str) -> None:
    assert parse_json_object(text, AgentAction) is None


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
