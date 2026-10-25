"""The price table is checked when the environment is built (src/envs/hydraulic_rig/env.py, _checked_prices): a table that names a
measure tier without sensors, leaves a component without a part price, has no downtime charge, has no measure prices or
holds a price that is not a number, not finite or negative, or prices a measurement at zero raises ValueError at construction, so no measure or replace action can count a step
or charge the budget and then fail on the table. Valid tables, the default and a custom one, build and charge every
tier and part they price."""
import copy
import re
from pathlib import Path

import pytest

from envs.hydraulic_rig.data import COMPONENTS, build_blocks, default_facts_path, load_cycles
from envs.hydraulic_rig.env import DEFAULT_PRICES, TinyHydraulicEnv

if not Path(default_facts_path()).exists():
    pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH", allow_module_level=True)

CYCLES = load_cycles(default_facts_path())
BLOCKS = build_blocks(CYCLES)
# fewer tiers, other prices, a float price and a part the world does not model
CUSTOM_PRICES = {"measure": {1: 2, 3: 7}, "parts": {"cooler": 5, "valve": 6, "pump": 7, "accumulator": 8, "filter": 1}, "downtime": 4.5}
BAD_TABLES = {  # one kind of fault in the default table each, with the problem the error must name
    "tier without sensors": (lambda p: p["measure"].update({4: 5}), "measure tiers [4] have no sensors in TIER_SENSORS"),
    "component without a part price": (lambda p: p["parts"].pop("pump"), "components ['pump'] have no part price"),
    "no downtime charge": (lambda p: p.pop("downtime"), "no downtime charge"),
    "no measure prices": (lambda p: p.pop("measure"), "no measure prices"),
    # a string and a bool: True is an int in Python, so the check has to exclude bools explicitly
    "price that is not a number": (lambda p: p.update(parts={**p["parts"], "pump": "30"}, downtime=True),
                                   "prices ['part pump', 'downtime'] are not numbers"),
    "price that is not finite": (lambda p: p.update(downtime=float("inf")), "prices ['downtime'] are not finite"),
    "negative price": (lambda p: p["parts"].update({"valve": -1}), "prices ['part valve'] are negative"),
    "free measurement": (lambda p: p["measure"].update({1: 0}), "measure ['tier 1'] are not positive"),
}


@pytest.mark.parametrize("name", list(BAD_TABLES))
def test_a_bad_price_table_raises_when_the_environment_is_built(name):
    change, problem = BAD_TABLES[name]
    prices = copy.deepcopy(DEFAULT_PRICES)
    change(prices)
    with pytest.raises(ValueError, match=re.escape(problem)):
        TinyHydraulicEnv(CYCLES, BLOCKS, prices=prices)


@pytest.mark.parametrize("prices", [None, DEFAULT_PRICES, CUSTOM_PRICES], ids=["omitted", "default", "custom"])
def test_a_valid_price_table_builds_and_charges_every_tier_and_part_it_prices(prices):
    table = prices or DEFAULT_PRICES
    for comp in COMPONENTS:
        env = TinyHydraulicEnv(CYCLES, BLOCKS, prices=prices)
        assert env.prices == table
        env.reset(0)
        for tier, price in table["measure"].items():
            assert env.step({"type": "measure", "tier": tier})[1:3] == (price, False)
        env.step({"type": "submit", "insufficient_evidence": True, "next_action": "replace"})
        assert env.step({"type": "replace", "component": comp})[1] == table["parts"][comp] + table["downtime"]
