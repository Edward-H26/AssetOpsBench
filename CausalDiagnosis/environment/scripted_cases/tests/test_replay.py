"""Invented scripts distinguish public safety rules from hidden requirements."""

import copy

import pytest

from ..replay import ScriptedEpisode, condition


def episode(**limits):
    empty = {"type": "object", "properties": {}, "additionalProperties": False}
    tools = [{"name": "read", "schema": empty, "cost": 2},
             {"name": "off", "schema": empty, "cost": 1, "requires": ["power == 'ON'"], "updates": {"power": "OFF"}},
             {"name": "repair", "schema": empty, "cost": 3},
             {"name": "live_test", "schema": empty, "cost": 2, "requires": ["power == 'ON'"]},
             {"name": "submit", "schema": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}, "ends": True}]
    actions = [{"id": "reading", "tool": "read", "args": {}, "observation": {"text": "released reading"}},
               {"id": "replacement", "tool": "repair", "args": {}, "requires": ["power == 'OFF'"],
                "observation": "replacement made", "updates": {"fixed": True}},
               {"id": "test", "tool": "live_test", "args": {}, "observation": "test made"}]
    return ScriptedEpisode({"power": "ON"}, tools, actions, **limits)


def test_exact_matching_charges_answered_calls_and_refusals_reveal_no_other_script():
    env = episode()
    refused = env.call("read", {"unexpected": "replacement"})
    assert "refused" in refused and env.calls == 1 and env.spent == 0
    assert "replacement made" not in str(refused) and "reading" not in str(refused)
    answered = env.call("read", {})
    assert answered["result"] == {"text": "released reading"} and env.spent == 2
    env.call("read", {})
    assert env.spent == 4 and env.calls == 3


def test_hidden_requirement_refuses_and_a_state_change_lifts_it_without_ending():
    env = episode()
    refused = env.call("repair", {})
    assert "refused" in refused and not env.over and env.spent == 0
    assert "power == 'OFF'" in refused["refused"]
    env.call("off", {})
    assert env.call("repair", {})["result"] == "replacement made"
    assert env.state["fixed"] and env.spent == 4


def test_public_requirement_ends_but_repeating_a_state_write_only_refuses():
    env = episode()
    env.call("off", {})
    assert "already" in env.call("off", {})["refused"] and not env.over
    assert env.calls == 2 and env.spent == 1
    assert "refused" in env.call("live_test", {})
    assert env.over and env.end_reason == "public_precondition" and env.spent == 1


def test_submission_needs_a_read_but_does_not_compare_or_return_private_answers():
    env = episode()
    assert "refused" in env.call("submit", {"answer": "anything"}) and not env.over
    env.call("read", {})
    assert env.call("submit", {"answer": "anything"})["result"] == {"accepted": True}
    assert env.over and env.submission == {"answer": "anything"}
    assert "is_correct" not in str(env.call("read", {}))


def test_only_explicit_limits_end_a_run_and_unaffordable_calls_make_no_state_change():
    env = episode(budget=1)
    assert "refused" in env.call("read", {}) and env.end_reason == "budget" and env.spent == 0
    env = episode(max_calls=1)
    assert env.call("read", {})["result"] and env.end_reason == "call_limit"
    env = episode()
    for _ in range(8):
        env.call("read", {})
    assert not env.over


def test_conditions_are_data_and_input_containers_are_copied():
    assert condition("power == 'ON' and locked != True", {"power": "ON", "locked": False})
    assert not condition("unknown == 'ON'", {})
    with pytest.raises(ValueError):
        condition("__import__('os').getcwd()", {})
    env = episode()
    before = copy.deepcopy(env.state)
    result = env.call("read", {})
    result["status"]["state"]["power"] = "OTHER"
    assert env.state == before
