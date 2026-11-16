"""A refused action whose handler has no row in the contract (the no_call step today) takes the class none in the log, the class of its
own log entry, and never asks the contract for a class it does not hold."""


def test_a_refused_action_whose_handler_has_no_contract_row_takes_class_none(make_env, monkeypatch):
    env = make_env()
    env.reset(0)
    monkeypatch.setitem(env._handlers, "not_a_tool", lambda action, think: env._refuse(action, "refused for the test", think))
    env.step({"type": "not_a_tool"})
    assert env.log[-1]["event"] == "refused" and env.log[-1]["class"] == "none"
