"""The typed readers of every value that a text from outside brings into the agent (CausalDiagnosis/docs/general-agent-plan.md,
section The check, No text raises). Each tests the type of its value before it uses it and returns (value, None) or (None,
reason), the reason in the words of the prompt sentence of its rule. name is the argument or the expectation key alone, as
"cycles" or "expect.top_rank"; whoever renders the reason puts the head "step k (t): " ahead of it. Stubs of the freeze; the reader
of a reply and the check build them."""


def read_name(value, names, name):
    """A string that is one of the names given; a list or an object is refused before any lookup."""
    raise NotImplementedError


def read_count(value, minimum, name):
    """An integer that is no boolean, a float of integral value, or a string of the digits 0 to 9 alone, at least minimum; no
    sign and no digit of another script."""
    raise NotImplementedError


def read_number(value, low, high, name):
    """An integer or a float that is finite and inside the range given, high None for no upper bound; a boolean, a NaN and an
    infinity are refused."""
    raise NotImplementedError


def read_text(value, name):
    """A string that holds more than white space."""
    raise NotImplementedError


def read_list(value, name):
    """A list."""
    raise NotImplementedError


def read_object(value, name):
    """A JSON object."""
    raise NotImplementedError
