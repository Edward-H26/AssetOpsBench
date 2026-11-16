"""Big layer 1, priced reads and writes. A read costs the price of its sensor group per
newly paid cycle, a work order its part and the repair charge, the prices paid fall from a budget that ends the episode when
an action cannot pay, and the misdiagnosis price is what a wrong diagnosis wastes, which the environment adds to the hidden
ledger at the write of a wrong diagnosis, outside the budget. Its mode budget sets the price to the budget of the asset. Off,
every price is zero and no budget stops the episode, so the cost of an episode counts its calls."""
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.small_switch import Switch, asset_value

NAME = "priced"
FIELDS = ("prices.read", "prices.parts", "prices.repair_charge", "budget", "prices.misdiagnosis")


def _price(name, key, off, modes=None):
    """The switch of one key of the price table: off, the value off(asset value), on, the asset's value, and its modes."""
    return Switch(name, (f"prices.{key}",), off=lambda profile: {f"prices.{key}": off(profile.config_fields()["prices"][key])},
                  on=lambda profile: {f"prices.{key}": profile.config_fields()["prices"][key]}, modes=modes)


SWITCHES = {s.name: s for s in (
    _price("read_prices", "read", lambda table: {group: 0 for group in table}),
    _price("part_prices", "parts", lambda table: {component: 0 for component in table}),
    _price("repair_charge", "repair_charge", lambda value: 0),
    Switch("budget", ("budget",), off=lambda profile: {"budget": None}, on=asset_value("budget")),
    _price("misdiagnosis", "misdiagnosis", lambda value: 0,
           modes={"budget": lambda profile: {"prices.misdiagnosis": profile.config_fields()["budget"]}}),
)}


def cost_unit(prices):
    """The unit of every cost on the episode's row of the environment table: money from the priced
    row on, where a read, a part, the repair charge or the visit has a price, and calls on the free base row, where every
    tool call costs one call, so that the oracle policy's cost, the denominator of the normalized cost, stays above zero.
    The diagnostics server's decision_rule.py imports this function and applies it to the released prices."""
    priced = [*prices["read"].values(), *prices["parts"].values(), prices.get("repair_charge", 0), prices.get("visit", 0)]
    return "money" if any(p > 0 for p in priced) else "calls"


def free(config):
    """Whether nothing has a price and no budget stops the episode, as on the base row, which the metering layer serves
    without a receipt line while no clock runs either."""
    return cost_unit(config.prices) == "calls" and config.budget is None


def read_price(config, sensor_group, cycles):
    return cycles * config.prices["read"][sensor_group]


def order_price(config, component, mode="standard"):
    """The price of a work order without the visit charge: the part plus the repair charge, times the surcharge for an
    emergency order."""
    price = config.prices["parts"][component] + config.prices["repair_charge"]
    return price * config.emergency["surcharge"] if mode == "emergency" else price


def tool_price(config, tool):
    """The price of a call as the tool manifest publishes it: per cycle read by sensor group, per work order by mode and
    component with the visit charge of the order that opens a visit, and zero for every other tool."""
    if tool == "read_sensors":
        return dict(config.prices["read"])
    if tool == "create_work_order":
        price = {"standard": {c: order_price(config, c) for c in config.components}, "visit": config.prices["visit"]}
        if config.emergency_on():
            price["emergency"] = {c: order_price(config, c, "emergency") for c in config.components}
        return price
    return 0


def affordable(config, prices_paid, price):
    """Whether the budget pays a price on top of the prices paid, which an episode without a budget always does. The budget
    limits the prices paid alone, so the downtime and the penalties of the hidden ledger never stop an action."""
    return config.budget is None or prices_paid + price <= config.budget


def cheapest_group(read_prices, sensor_groups, channels):
    """The sensor group a metered read of these channels is charged to: the cheapest group that covers them, then the
    smaller, then by name. A configuration holds a group of every channel (EpisodeConfig._check_sensor_groups), so some
    group covers any channels of its asset."""
    covering = [g for g, members in sensor_groups.items() if channels <= set(members)]
    return min(covering, key=lambda g: (read_prices[g], len(sensor_groups[g]), g))
