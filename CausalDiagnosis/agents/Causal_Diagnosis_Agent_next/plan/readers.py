"""The typed readers of every value that a text from outside brings into the agent (CausalDiagnosis/docs/general-agent-plan.md,
section The check, No text raises). Each tests the type of its value before it uses it and returns (value, None) or (None,
reason), the reason in the words of the prompt sentence of its rule. name is the argument or the expectation key alone, as
"cycles" or "expect.top_rank"; whoever renders the reason puts the head "step k (t): " ahead of it."""

import math
import sys

MAX_COUNT = sys.maxsize


def copy_json(value):
    """Copy JSON containers with a stack, preserving their contents at any depth."""
    if not isinstance(value, (dict, list)):
        return value
    root = {} if isinstance(value, dict) else []
    memo, pending = {id(value): root}, [(value, root)]
    while pending:
        source, target = pending.pop()
        entries = source.items() if isinstance(source, dict) else enumerate(source)
        for key, child in entries:
            if isinstance(child, (dict, list)):
                if id(child) not in memo:
                    memo[id(child)] = {} if isinstance(child, dict) else []
                    pending.append((child, memo[id(child)]))
                child = memo[id(child)]
            if isinstance(target, dict):
                target[key] = child
            else:
                target.append(child)
    return root


def read_name(value, names, name):
    """A string that is one of the names given; a list or an object is refused before any lookup."""
    if not isinstance(value, str) or value not in names:
        return None, f"{name}: must name one of: " + ", ".join(names)
    return value, None


def read_count(value, minimum, name):
    """An integer that is no boolean, a float of integral value, or a string of the digits 0 to 9 alone, at least minimum; no
    sign and no digit of another script."""
    if isinstance(value, bool):
        return None, f"{name}: must be an integer"
    if isinstance(value, str):
        if not value or not value.isascii() or not value.isdecimal():
            return None, f"{name}: must contain only the digits 0 to 9"
        digits = value.lstrip("0") or "0"
        largest = str(MAX_COUNT)
        if len(digits) > len(largest) or len(digits) == len(largest) and digits > largest:
            return None, f"{name}: exceeds the interpreter's index range"
        value = int(digits)
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None, f"{name}: must be a finite integer"
        value = int(value)
    if not isinstance(value, int):
        return None, f"{name}: must be an integer"
    if value > MAX_COUNT or value < -MAX_COUNT - 1:
        return None, f"{name}: exceeds the interpreter's index range"
    if value < minimum:
        return None, f"{name}: must be at least {minimum}"
    return value, None


def read_number(value, low, high, name):
    """An integer or a float that is finite and inside the range given, high None for no upper bound; a boolean, a NaN and an
    infinity are refused."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, f"{name}: must be a number"
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        return None, f"{name}: must be a finite number"
    if low is not None and value < low:
        return None, f"{name}: must be at least {low}"
    if high is not None and value > high:
        return None, f"{name}: must be at most {high}"
    return value, None


def read_text(value, name):
    """A string that holds more than white space."""
    if not isinstance(value, str) or not value.strip():
        return None, f"{name}: must be nonempty text"
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return None, f"{name}: must contain complete UTF-8 characters"
    if any(ord(char) < 32 and char not in "\r\n" for char in value):
        return None, f"{name}: must contain no raw control characters other than CR and LF"
    return value.strip(), None


def read_list(value, name):
    """A list."""
    if not isinstance(value, list):
        return None, f"{name}: must be a list"
    return copy_json(value), None


def read_object(value, name):
    """A JSON object."""
    if not isinstance(value, dict):
        return None, f"{name}: must be an object"
    return copy_json(value), None
