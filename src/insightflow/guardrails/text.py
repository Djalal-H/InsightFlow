"""Remove characters that a model reads but a person does not see."""

import re
from typing import Final

INVISIBLE_PATTERN: Final = re.compile(
    "["
    "​"  # zero-width space
    "‎‏"  # left-to-right and right-to-left marks
    "‪-‮"  # bidirectional embeddings and overrides ("Trojan Source")
    "⁠-⁤"  # word joiner and invisible operators
    "⁦-⁩"  # bidirectional isolates
    "﻿"  # zero-width no-break space
    "\U000e0000-\U000e007f"  # tag characters: invisible ASCII ("ASCII smuggling")
    "\U000e0100-\U000e01ef"  # variation selectors supplement (data hidden after a character)
    "]"
)
"""Zero-width joiner and non-joiner (U+200C, U+200D) stay: scripts and emoji need them."""


def strip_invisible(text: str) -> tuple[str, int]:
    """Return the text without invisible characters, and how many were removed."""
    cleaned, count = INVISIBLE_PATTERN.subn("", text)
    return cleaned, count


def clean(text: str) -> str:
    """The text without invisible characters."""
    return INVISIBLE_PATTERN.sub("", text)
