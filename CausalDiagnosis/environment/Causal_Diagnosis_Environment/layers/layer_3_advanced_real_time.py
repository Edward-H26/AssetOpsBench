"""Big layer 3, the advanced real-time clock: every tool call is a durative action of its declared
length, a read of cycles not yet recorded lasts the cycles it plays, a work order waits its lead time and stops the rig for
its repair downtime, and every elapsed cycle charges the downtime price of the hidden condition to the hidden ledger, the
stopped rate while a repair stops the rig. Off, every declared duration and every downtime rate is zero."""
from CausalDiagnosis.datasets.dataset_loader import rule_cards as rules

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_2_real_time import any_duration
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.small_switch import Switch, asset_value

NAME = "advanced_real_time"
FIELDS = ("durations.read_per_cycle", "durations.lead_time", "durations.repair_downtime", "downtime_rate")


def _duration(key, off):
    return Switch(key, (f"durations.{key}",), off=lambda profile: {f"durations.{key}": off(profile.config_fields()["durations"][key])},
                  on=lambda profile: {f"durations.{key}": profile.config_fields()["durations"][key]})


def _downtime_off(profile):
    card = {"kind": "downtime", "parameters": rules.kind_of("downtime").off(profile)}
    return {"downtime_rate": rules.apply_card({}, card)["downtime_rate"]}


SWITCHES = {s.name: s for s in (
    _duration("read_per_cycle", lambda value: 0),
    _duration("lead_time", lambda table: {component: 0 for component in table}),
    _duration("repair_downtime", lambda table: {component: 0 for component in table}),
    Switch("downtime_rate", ("downtime_rate",), off=_downtime_off, on=asset_value("downtime_rate")),
)}


def durations_on(config):
    """Whether a read, a lead time or a repair downtime takes declared time."""
    return any_duration(config.durations)


def lead_time(config, component, mode="standard"):
    return config.emergency["lead"] if mode == "emergency" else config.durations["lead_time"][component]


def tool_duration(config, tool):
    """The declared cycles of a call per unit, per cycle read for read_sensors and per cycle waited for
    wait_for_notification, and zero for every other tool, whose answer takes no time."""
    if tool == "read_sensors":
        return config.durations["read_per_cycle"]
    if tool == "wait_for_notification":
        return 1
    return 0


def level_rate(config, component, level):
    """The downtime rate per cycle of a component at a level; the table keys are the levels as JSON spells them."""
    return config.downtime_rate[component][str(level)]


def downtime_on(config):
    """A table of zero rates, the stopped rate included, switches the downtime rule off, the off value of its rule card."""
    rates = [r for c in config.components for r in config.downtime_rate[c].values()] + [config.downtime_rate[rules.STOPPED]]
    return any(r > 0 for r in rates)


def accrue(clock, t, stopped):
    """Downtime of the cycle: the stopped rate while a repair stops the rig, else the rate of every degraded component."""
    if stopped:
        charge = clock.config.downtime_rate[rules.STOPPED]
    else:
        charge = sum(level_rate(clock.config, c, level) for c, level in clock.levels.items())
    if charge:
        clock.downtime.append({"t": t, "charge": charge})
        clock.downtime_total += charge
    return charge
