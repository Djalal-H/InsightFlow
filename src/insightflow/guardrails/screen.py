"""A heuristic screen for prompt-injection patterns (OWASP LLM01 input screening).

The screen is a signal, not a boundary: attackers can phrase instructions in ways that no
pattern list covers. Its results feed the audit log, the response metadata, and the policy
for untrusted web pages. It reads a normalized copy (NFKC, case-folded, invisible characters
removed, length-limited) and never changes the text that the model receives.

Typo-tolerant matching: a word with the same first and last letters and the same letters in
between as a keyword ("ignroe", "instrcutions") counts as the keyword, because models read
such words correctly.
"""

import base64
import binascii
import re
import unicodedata
from collections.abc import Iterable
from typing import Final

from insightflow.guardrails.text import strip_invisible

SCREEN_MAX_CHARS: Final = 20_000

INSTRUCTION_OVERRIDE: Final = "instruction_override"
ROLE_OVERRIDE: Final = "role_override"
PROMPT_DISCLOSURE: Final = "prompt_disclosure"
ROLE_MARKER: Final = "role_marker"
HIDDEN_TEXT: Final = "hidden_text"
ENCODED_PAYLOAD: Final = "encoded_payload"
EXFILTRATION_MARKUP: Final = "exfiltration_markup"

KEYWORDS: Final = (
    "ignore",
    "disregard",
    "forget",
    "override",
    "bypass",
    "previous",
    "instructions",
    "instruction",
    "prompt",
    "system",
    "reveal",
    "developer",
    "jailbreak",
)

_OVERRIDE = re.compile(
    r"\b(?:ignore|disregard|forget|override|bypass|skip)\b"
    r"(?:\s+\w+){0,3}?\s+"
    r"(?:instructions?|prompts?|rules|directions|guidelines|guardrails|constraints|context)\b"
)
_NEW_INSTRUCTIONS = re.compile(r"\b(?:new|updated|real|actual)\s+(?:system\s+)?instructions?\s*:")
_ROLE = re.compile(
    r"\byou\s+are\s+now\b"
    r"|\bdeveloper\s+mode\b"
    r"|\bdo\s+anything\s+now\b"
    r"|\bjailbr(?:oken|eak)\b"
    r"|\bact\s+as\s+(?:an?\s+)?(?:unrestricted|unfiltered|uncensored|evil)\b"
    r"|\bpretend\s+(?:that\s+)?you\s+(?:have\s+no|are\s+not\s+bound)"
)
_DISCLOSURE = re.compile(
    r"\b(?:reveal|show|print|repeat|output|display|leak|tell\s+me|write\s+out)\b"
    r"(?:\s+\w+){0,2}?\s+"
    r"(?:your|the)\s+(?:\w+\s+){0,2}?(?:system\s+prompt|instructions|initial\s+prompt|prompt)\b"
)
_MARKER = re.compile(
    r"(?im)^\s*(?:#{1,3}\s*)?(?:system|assistant|developer)\s*:"
    r"|</?\s*(?:system|assistant|instructions?)\s*>"
    r"|<\|im_(?:start|end)\|>"
    r"|\[/?inst\]"
    r"|<<\s*/?sys\s*>>"
)
_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\(\s*https?://[^)\s]+\?[^)\s]*\)")
_BASE64_CANDIDATE = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")
_WORD = re.compile(r"[a-z]+")


def _fingerprint(word: str) -> tuple[str, str, str, int]:
    return word[0], word[-1], "".join(sorted(word[1:-1])), len(word)


_KEYWORD_BY_FINGERPRINT: Final = {_fingerprint(word): word for word in KEYWORDS}


def normalize(text: str) -> str:
    """The copy that the screen reads: limited, NFKC, case-folded, typos mapped to keywords."""
    limited = text[:SCREEN_MAX_CHARS]
    folded = unicodedata.normalize("NFKC", strip_invisible(limited)[0]).casefold()
    return _WORD.sub(_canonical_word, folded)


def _canonical_word(match: re.Match[str]) -> str:
    word = match.group(0)
    if len(word) < 4:
        return word
    return _KEYWORD_BY_FINGERPRINT.get(_fingerprint(word), word)


def screen(text: str, *, untrusted: bool = False) -> list[str]:
    """Return the names of the suspicious patterns in `text`, sorted; empty when none.

    `untrusted` (web pages, passages, tool output) also checks for links that would send data
    out when a client renders them.
    """
    signals: set[str] = set()
    if strip_invisible(text[:SCREEN_MAX_CHARS])[1]:
        signals.add(HIDDEN_TEXT)
    signals.update(_pattern_signals(normalize(text)))
    if untrusted and _MARKDOWN_IMAGE.search(text[:SCREEN_MAX_CHARS]):
        signals.add(EXFILTRATION_MARKUP)
    for decoded in _decoded_payloads(text[:SCREEN_MAX_CHARS]):
        if any(_pattern_signals(normalize(decoded))):
            signals.add(ENCODED_PAYLOAD)
            break
    return sorted(signals)


def _pattern_signals(normalized: str) -> Iterable[str]:
    if _OVERRIDE.search(normalized) or _NEW_INSTRUCTIONS.search(normalized):
        yield INSTRUCTION_OVERRIDE
    if _ROLE.search(normalized):
        yield ROLE_OVERRIDE
    if _DISCLOSURE.search(normalized):
        yield PROMPT_DISCLOSURE
    if _MARKER.search(normalized):
        yield ROLE_MARKER


def _decoded_payloads(text: str) -> Iterable[str]:
    """Base64 runs that decode to readable text, for inspection only."""
    for match in _BASE64_CANDIDATE.finditer(text):
        candidate = match.group(0)
        try:
            raw = base64.b64decode(candidate + "=" * (-len(candidate) % 4), validate=True)
            decoded = raw.decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        if decoded.isprintable() or "\n" in decoded:
            yield decoded
