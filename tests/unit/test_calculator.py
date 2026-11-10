"""Unit tests for the safe arithmetic evaluator behind the calculator tool."""

import math
import re

import pytest

from insightflow.tools.calculator import (
    MAX_EXPRESSION_CHARS,
    CalculationError,
    evaluate,
    format_number,
)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1 + 2 * 3", 7),
        ("(1 + 2) * 3", 9),
        ("7 / 2", 3.5),
        ("7 // 2", 3),
        ("7 % 4", 3),
        ("-3 + +5", 2),
        ("2 ** 10", 1024),
        ("2 ** -1", 0.5),
        ("0.15 * 80", 12.0),
        ("abs(-4)", 4),
        ("round(2.567, 2)", 2.57),
        ("round(2.4)", 2),
        ("min(3, 1, 2)", 1),
        ("max(3, 1, 2)", 3),
        ("sqrt(16)", 4.0),
        ("floor(2.7) + ceil(2.1)", 5),
        ("log10(1000)", 3.0),
        ("log(8, 2)", 3.0),
        ("pi * 2", math.pi * 2),
        ("  12  ", 12),
    ],
)
def test_evaluate_supported_arithmetic(expression: str, expected: float) -> None:
    """Allow-listed operators, functions, and constants give Python's arithmetic result."""
    assert evaluate(expression) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("expression", "reason"),
    [
        ("", "empty"),
        ("1 +", "not valid arithmetic"),
        ("1 / 0", "division by zero"),
        ("5 % 0", "division by zero"),
        ("2 ^ 3", "use ** for powers"),
        ("x + 1", "unknown name 'x'"),
        ("__import__('os')", "unknown function"),
        ("(1).real", "unsupported syntax"),
        ("[1, 2]", "unsupported syntax"),
        ("1, 200", "unsupported syntax"),
        ("'a' * 3", "only numbers"),
        ("True + 1", "only numbers"),
        ("1j * 2", "only numbers"),
        ("round(1.5, ndigits=1)", "keyword arguments"),
        ("round(1.5, 0.5)", "whole number of digits"),
        ("sqrt(1, 2)", "wrong number of arguments"),
        ("min(1)", "wrong number of arguments"),
        ("sqrt(-1)", "not defined"),
        ("(-8) ** (1 / 3)", "not defined"),
        ("0 ** -1", "not defined"),
        ("9 ** 9 ** 9", "too large"),
        ("10 ** 101", "too large"),
        ("exp(1000)", "too large"),
        ("1e308 * 10", "too large"),
    ],
)
def test_evaluate_rejects_unsafe_or_invalid_input(expression: str, reason: str) -> None:
    """Unsupported syntax and undefined or oversized results raise a readable error."""
    with pytest.raises(CalculationError, match=re.escape(reason)):
        evaluate(expression)


def test_evaluate_rejects_long_expressions() -> None:
    """The length limit applies before parsing."""
    with pytest.raises(CalculationError, match="longer than"):
        evaluate("1+" * MAX_EXPRESSION_CHARS + "1")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (12, "12"),
        (12.0, "12"),
        (0.1 + 0.2, "0.3"),
        (2 / 3, "0.666666666667"),
        (1e20, "1e+20"),
        (10**30, str(10**30)),
    ],
)
def test_format_number(value: float, expected: str) -> None:
    """Whole numbers have no `.0` and floats are shown with 12 significant digits."""
    assert format_number(value) == expected
