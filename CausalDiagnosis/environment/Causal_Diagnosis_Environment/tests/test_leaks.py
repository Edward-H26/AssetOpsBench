"""The leak fixes of CausalDiagnosis/environment/Causal_Diagnosis_Environment (env.py, episode_clock.py, config.py, oracle.py and policies.py): a change
summary lists alarms only for the channels of the sensor groups the agent paid to read, and the closing date of a past
work order no longer tells a healthy component from a degraded one. No test states a value computed from the rig data."""
import random

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_4_repair_rules import draw_fix_history
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS, HEALTHY, SENSORS, TIER_SENSORS
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import mean_field

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.tests.conftest import advance, read, summary

VALVE_AND_COOLER = (3, 90, 0, 130)


def alarm_channels(env):
    return {a["channel"] for a in summary(env, 0)[0]["result"]["alarms"]}


def test_a_change_summary_lists_alarms_only_for_the_channels_of_groups_a_charged_read_named(make_env):
    env = make_env(start={"combination": list(VALVE_AND_COOLER), "entry_offset": 0})
    env.reset(0)
    advance(env, 3)
    assert alarm_channels(env) == set()
    env.step({"type": "read_sensors", "sensor_group": "tier9", "cycles": 1})  # refused, so nothing is paid
    assert alarm_channels(env) == set()
    read(env, "tier1", 1)
    tier1 = alarm_channels(env)
    assert tier1 and tier1 <= {mean_field(s) for s in TIER_SENSORS[1]}
    read(env, "tier3", 1)
    every = alarm_channels(env)
    # the valve leaves channels that only tier3 holds outside their normal range, which the tier1 summary had to keep back
    assert tier1 < every and every - {mean_field(s) for s in TIER_SENSORS[1]} and every <= {mean_field(s) for s in SENSORS}


def test_the_closing_date_of_a_past_order_tells_nothing_about_whether_its_component_is_healthy():
    config = EpisodeConfig()
    ages = {"healthy": set(), "degraded": set()}
    for seed in range(3000):
        for entry in draw_fix_history(random.Random(seed), config, VALVE_AND_COOLER):
            level = VALVE_AND_COOLER[COMPONENTS.index(entry["component"])]
            ages["healthy" if level == HEALTHY[entry["component"]] else "degraded"].add(-entry["closed_t"])
    assert ages["healthy"] == ages["degraded"] and len(ages["healthy"]) > 1
