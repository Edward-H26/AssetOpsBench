"""The comparison of a whole summary with a hand-computed one for the tests of the performance package: every leaf of the nested
summary is compared, floats within the tolerance of pytest.approx, None and the other values exactly, so that no figure of a summary
goes untested."""
import pytest


def leaves(value, path=""):
    """The (path, value) pairs of the leaves of a nested summary, a dict walked by its keys."""
    if isinstance(value, dict):
        return [leaf for key in value for leaf in leaves(value[key], f"{path}{key}.")]
    return [(path[:-1], value)]


def assert_same(found, expected):
    """Every leaf of found equals the leaf of expected at the same path, and the two hold the same paths."""
    found, expected = dict(leaves(found)), dict(leaves(expected))
    assert sorted(found) == sorted(expected)
    differ = [path for path in expected if not (found[path] == pytest.approx(expected[path]) if isinstance(expected[path], float)
                                                 else found[path] == expected[path] and type(found[path]) is type(expected[path]))]
    assert differ == [], {path: (found[path], expected[path]) for path in differ}
