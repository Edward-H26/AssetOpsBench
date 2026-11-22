"""The refusal of a second work order on a component whose order is open names the close of that order, the cycle that its
expected_close_t states in the answer of the refusal, so that an agent learns when the open order closes instead of reading it as a
failed one. It holds on every clocked preset and on a row without a clock whose repair card keeps an order open until the next
call that is not an order. A first order, an order on another component and an order after the close are answered as before,
and on a row whose orders close inside their call a second order is no refusal."""
import json

import pytest

from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers import compose
from CausalDiagnosis.utils.paths import preset_path

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.tests.conftest import advance, faulty_components, order, submit_truth

PUMP_AND_VALVE = (100, 80, 2, 130)  # two faulty components of the asset file, the valve repaired before the pump by priority


def preset_env(world, name):
    cycles, blocks = world
    return HydraulicRigEnv(cycles, blocks, compose(json.loads(preset_path(name).read_text(encoding="utf-8"))))


def open_order_reason(component, close):
    return f"a work order on the {component} is open; it closes at cycle {close}"


def first_order(env, seed=0):
    """The diagnosis of the truth and one order on the first faulty component, or on the cooler at a start without a fault."""
    env.reset(seed)
    submit_truth(env)
    component = (faulty_components(env.initial) or ["cooler"])[0]
    obs, *_ = order(env, component)
    return component, obs["result"]["work_order"]


@pytest.mark.parametrize("name", ["repair_rules", "deterioration_thinking_off", "deterioration_token_clock", "static_zero_durations"])
def test_a_second_order_on_a_component_whose_order_is_open_names_its_close(world, name):
    env = preset_env(world, name)
    component, work_order = first_order(env)
    assert work_order["status"] != "closed"
    obs, cost, done, info = order(env, component)
    reason = open_order_reason(component, work_order["expected_close_t"])
    assert info == {"refused": reason} and obs["refused"] == reason and cost == 0 and not done
    assert env.log[-1]["event"] == "refused" and env.log[-1]["reason"] == reason
    listed = next(o for o in obs["open_work_orders"] if o["component"] == component)
    assert listed["expected_close_t"] == work_order["expected_close_t"]


def test_the_close_named_is_the_schedule_of_the_open_order_at_the_refusal(make_env):
    """The pump ordered first and the valve right after it form one visit, and the valve's smaller priority number moves the
    pump's repair after the valve's, so the refusal names the moved close, the one the answer lists."""
    env = make_env(start={"combination": list(PUMP_AND_VALVE), "entry_offset": 0})
    env.reset(0)
    submit_truth(env)
    first, *_ = order(env, "pump")
    order(env, "valve")
    obs, cost, done, info = order(env, "pump")
    listed = next(o for o in obs["open_work_orders"] if o["component"] == "pump")
    assert listed["expected_close_t"] > first["result"]["work_order"]["expected_close_t"]
    assert info == {"refused": open_order_reason("pump", listed["expected_close_t"])} and cost == 0


@pytest.mark.parametrize("name", ["repair_rules", "deterioration_thinking_off"])
def test_a_first_order_an_order_on_another_component_and_an_order_after_the_close_are_answered_as_before(world, name):
    env = preset_env(world, name)
    component, work_order = first_order(env)
    assert work_order["id"] == "wo-1" and work_order["component"] == component
    other = next(c for c in COMPONENTS if c != component)
    obs, _cost, _done, info = order(env, other)
    assert not info.get("refused") and obs["result"]["work_order"]["component"] == other
    opened = obs["open_work_orders"]
    _obs, _charged, done, _info = advance(env, max(o["expected_close_t"] for o in opened) - env.time)
    assert len(opened) == 2 and not done and not env.public_work_orders()
    obs, _cost, _done, info = order(env, component)
    assert not info.get("refused") and obs["result"]["work_order"]["id"] == "wo-3"


def test_on_a_row_whose_orders_close_inside_their_call_a_second_order_is_no_refusal(world):
    env = preset_env(world, "base")
    component, work_order = first_order(env)
    assert work_order["status"] == "closed"
    obs, _cost, _done, info = order(env, component)
    assert not info.get("refused") and obs["result"]["work_order"]["id"] == "wo-2"
