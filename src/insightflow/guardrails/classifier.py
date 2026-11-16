"""An optional hosted content-safety classifier (Llama Guard style), behind a protocol.

Llama Guard models (for example `nvidia_nim/meta/llama-guard-4-12b`) classify a user message,
or a user message and a reply, as `safe` or `unsafe` with hazard categories (`S1`, `S2`, ...).
They detect harmful content, not prompt injection; the other guardrails cover injection.

The classifier is an extra layer, so it fails open: an unreadable reply or a provider failure
is audited and the text passes. Each check is one billable (or rate-limited free-tier) call.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Final, Protocol

from insightflow.core.exceptions import ProviderError
from insightflow.providers.llm import ChatProvider
from insightflow.security.audit import audit

logger = logging.getLogger(__name__)

UNSAFE_ANSWER: Final = "I cannot help with that request."
UNSAFE_SIGNAL: Final = "answer:unsafe"
_CATEGORY = re.compile(r"\bS\d{1,2}\b")


@dataclass(frozen=True)
class SafetyVerdict:
    safe: bool
    categories: list[str] = field(default_factory=list)


class SafetyClassifier(Protocol):
    async def check_input(self, text: str) -> SafetyVerdict: ...

    async def check_output(self, question: str, answer: str) -> SafetyVerdict: ...


class LlamaGuardClassifier:
    """`SafetyClassifier` over any chat model that answers in the Llama Guard format."""

    def __init__(self, chat_provider: ChatProvider) -> None:
        self._chat = chat_provider

    async def check_input(self, text: str) -> SafetyVerdict:
        return await self._classify([{"role": "user", "content": text}], "input")

    async def check_output(self, question: str, answer: str) -> SafetyVerdict:
        return await self._classify(
            [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
            "output",
        )

    async def _classify(self, messages: list[dict[str, str]], target: str) -> SafetyVerdict:
        try:
            reply = await self._chat.complete(messages)
        except ProviderError as exc:
            audit("safety_check_failed", target=target, error=type(exc).__name__)
            return SafetyVerdict(safe=True)
        verdict = parse_verdict(reply)
        if verdict is None:
            audit("safety_check_unreadable", target=target)
            return SafetyVerdict(safe=True)
        if not verdict.safe:
            audit("safety_check_unsafe", target=target, categories=verdict.categories)
        return verdict


def parse_verdict(reply: str) -> SafetyVerdict | None:
    """Read `safe`, or `unsafe` followed by categories; `None` for anything else."""
    lines = [line.strip() for line in reply.strip().splitlines() if line.strip()]
    if not lines:
        return None
    first = lines[0].lower()
    if first == "safe":
        return SafetyVerdict(safe=True)
    if first.startswith("unsafe"):
        return SafetyVerdict(safe=False, categories=sorted(set(_CATEGORY.findall(reply))))
    return None
