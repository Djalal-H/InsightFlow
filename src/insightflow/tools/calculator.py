"""Safe arithmetic evaluation for the calculator tool.

The expression is parsed with `ast` and only an allow-listed set of nodes is evaluated. `eval`
is never used. Size limits stop expressions such as `9**9**9` before they use memory or time.
"""

import ast
import math
import operator
from collections.abc import Callable

Number = int | float

MAX_EXPRESSION_CHARS = 200
MAX_ABS_VALUE = 10**100
MAX_INT_EXPONENT = 1000


class CalculationError(ValueError):
    """The expression is not supported or its value cannot be calculated."""


def _round(value: Number, ndigits: Number = 0) -> Number:
    if not isinstance(ndigits, int):
        raise CalculationError("round() needs a whole number of digits")
    return round(value) if ndigits == 0 else round(value, ndigits)


def _log(value: Number, base: Number = math.e) -> float:
    return math.log(value, base)


FUNCTIONS: dict[str, tuple[Callable[..., Number], int, int]] = {
    # name: (function, minimum arguments, maximum arguments)
    "abs": (abs, 1, 1),
    "round": (_round, 1, 2),
    "min": (min, 2, 10),
    "max": (max, 2, 10),
    "sqrt": (math.sqrt, 1, 1),
    "floor": (math.floor, 1, 1),
    "ceil": (math.ceil, 1, 1),
    "log": (_log, 1, 2),
    "log10": (math.log10, 1, 1),
    "exp": (math.exp, 1, 1),
}

CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e}

BINARY_OPERATORS: dict[type[ast.operator], Callable[[Number, Number], Number]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}

UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Number], Number]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def evaluate(expression: str) -> Number:
    """Return the value of an arithmetic expression, or raise `CalculationError`."""
    text = expression.strip()
    if not text:
        raise CalculationError("the expression is empty")
    if len(text) > MAX_EXPRESSION_CHARS:
        raise CalculationError(f"the expression is longer than {MAX_EXPRESSION_CHARS} characters")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise CalculationError("the expression is not valid arithmetic") from exc
    try:
        return _evaluate(tree.body)
    except CalculationError:
        raise
    except ZeroDivisionError as exc:
        raise CalculationError("division by zero") from exc
    except OverflowError as exc:
        raise CalculationError("the result is too large") from exc
    except (ValueError, TypeError) as exc:
        raise CalculationError("the operation is not defined for these values") from exc


def format_number(value: Number) -> str:
    """Render a result for people: whole numbers without `.0`, others with 12 significant digits."""
    if isinstance(value, int):
        return str(value)
    if value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    return format(value, ".12g")


def _evaluate(node: ast.expr) -> Number:
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise CalculationError("only numbers are allowed")
        return _checked(value)
    if isinstance(node, ast.Name):
        if node.id not in CONSTANTS:
            raise CalculationError(f"unknown name {node.id!r}")
        return CONSTANTS[node.id]
    if isinstance(node, ast.UnaryOp):
        unary = UNARY_OPERATORS.get(type(node.op))
        if unary is None:
            raise CalculationError("unsupported operator")
        return _checked(unary(_evaluate(node.operand)))
    if isinstance(node, ast.BinOp):
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow):
            return _checked(_power(left, right))
        binary = BINARY_OPERATORS.get(type(node.op))
        if binary is None:
            hint = "; use ** for powers" if isinstance(node.op, ast.BitXor) else ""
            raise CalculationError(f"unsupported operator{hint}")
        return _checked(binary(left, right))
    if isinstance(node, ast.Call):
        return _checked(_call(node))
    raise CalculationError("unsupported syntax")


def _call(node: ast.Call) -> Number:
    if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
        raise CalculationError("unknown function")
    if node.keywords:
        raise CalculationError("keyword arguments are not supported")
    function, minimum, maximum = FUNCTIONS[node.func.id]
    if not minimum <= len(node.args) <= maximum:
        raise CalculationError(f"{node.func.id}() got the wrong number of arguments")
    arguments = [_evaluate(argument) for argument in node.args]
    result: Number = function(*arguments)
    return result


def _power(base: Number, exponent: Number) -> Number:
    if isinstance(base, int) and isinstance(exponent, int) and exponent >= 0:
        if exponent > MAX_INT_EXPONENT:
            raise CalculationError("the result is too large")
        if abs(base) > 1 and exponent * math.log10(abs(base)) > math.log10(MAX_ABS_VALUE):
            raise CalculationError("the result is too large")
        result: int = base**exponent
        return result
    # math.pow raises for a negative base with a fractional exponent instead of returning a
    # complex number, as `**` does.
    return math.pow(base, exponent)


def _checked(value: Number) -> Number:
    if isinstance(value, float) and not math.isfinite(value):
        raise CalculationError("the result is too large")
    if abs(value) > MAX_ABS_VALUE:
        raise CalculationError("the result is too large")
    return value
