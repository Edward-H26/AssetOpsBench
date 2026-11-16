"""Big layer 2, the real-time clock, which follows the System app of ARE. In the
generation_time mode the agent's thinking between two calls runs the clock, the wall seconds adding up from call to call into
one cycle per think_seconds_per_cycle with the rest carried, and every answer carries the notifications of the public events.
The switch time_tools lists the two time tools get_current_time and wait_for_notification on the task server of the
AssetOpsBench rows, while the rig's action server lists them on every row as its wait. Off, the clock
mode is declared and the time tools are not listed, so time passes only in a wait of the rig's own tools or in a declared
duration of the advanced real-time clock. The switch think_seconds_per_cycle also takes the mapping {"scale": x}, x times
the rate of the asset file, so the thinking price presets follow the value the asset file declares."""
import math

from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.small_switch import Switch, asset_value

NAME = "real_time"
FIELDS = ("clock_mode", "seconds_per_token", "think_seconds_per_cycle", "time_tools")
CLOCK_MODES = ("declared", "generation_time", "instant", "generated_tokens")
# the mode whose thinking time is the tokens the agent's model generated since the previous call times seconds_per_token, so
# that the charge of a call is the same whatever the load of the machine that serves the model
TOKEN_MODE = "generated_tokens"
THINKING_MODES = ("generation_time", TOKEN_MODE)  # the modes that charge the agent's thinking time before an action


def _scaled_think_seconds(mapping, profile):
    """The thinking rate of a mapping {"scale": x}: x times the asset file's think_seconds_per_cycle, x a positive finite
    number. A whole result is kept as an integer, so the briefing states 20 seconds and not 20.0."""
    scale = mapping.get("scale")
    if set(mapping) != {"scale"} or isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) \
            or scale <= 0:
        raise ValueError(f"the small switch think_seconds_per_cycle takes on, off, a number or the mapping {{'scale': x}} with x a "
                         f"positive number, not {mapping!r}")
    seconds = scale * profile.config_fields()["think_seconds_per_cycle"]
    return {"think_seconds_per_cycle": int(seconds) if float(seconds).is_integer() else seconds}


SWITCHES = {s.name: s for s in (
    # seconds_per_token follows the mode: the configuration takes the asset's rate in TOKEN_MODE and None in every other mode
    Switch("clock_mode", ("clock_mode", "seconds_per_token"), off=lambda profile: {"clock_mode": "declared"},
           on=lambda profile: {"clock_mode": "generation_time"},
           modes={mode: (lambda profile, mode=mode: {"clock_mode": mode}) for mode in CLOCK_MODES}),
    Switch("think_seconds_per_cycle", ("think_seconds_per_cycle",), off=asset_value("think_seconds_per_cycle"), on=asset_value("think_seconds_per_cycle"),
           parameters=_scaled_think_seconds),
    Switch("time_tools", ("time_tools",), off=lambda profile: {"time_tools": False}, on=lambda profile: {"time_tools": True}),
)}


def has_clock(clock_mode, durations):
    """Whether time passes while the agent does not wait: not on the rows without a clock, the declared
    mode with every declared duration zero, where time passes only in a wait. The clock runs where the clock mode of this
    layer is on or a duration of the advanced real-time clock is. The diagnostics server's belief_filter.py imports this function
    and applies it to the released rules, so that its replay settles a closure where the environment did."""
    return clock_mode != "declared" or any_duration(durations)


def any_duration(durations):
    """Whether a read per cycle, a lead time or a repair downtime of the durations takes declared time."""
    return any(d > 0 for d in (durations["read_per_cycle"], *durations["lead_time"].values(), *durations["repair_downtime"].values()))


def thinking(config, think_cycles):
    """The thinking cycles a step charges before its action: all of them in the two thinking modes, none otherwise."""
    return think_cycles if config.clock_mode in THINKING_MODES else 0


def action_cycles(config, cycles, per_cycle):
    """The cycles an action moves the clock: its declared duration, cycles times per_cycle, or one cycle for every agent
    step in the instant mode, a refused step included."""
    if config.clock_mode == "instant":
        return 1
    return cycles * per_cycle


def wait_timeout(config, timeout):
    """The timeout of a wait in cycles, one cycle in the instant mode, where a wait takes its cycle like every other call."""
    return 1 if config.clock_mode == "instant" else timeout


def token_rate(config):
    """The seconds_per_token of a configuration: the given value, else the one of the clock of the asset file, in the
    generated_tokens mode, and None in every other mode, which charges no token, so that the configuration of another mode
    keeps its bytes. An asset whose file states no rate leaves None, which the configuration refuses in that mode."""
    if config.clock_mode != TOKEN_MODE:
        return None
    if config.seconds_per_token is not None:
        return config.seconds_per_token
    return get_profile(config.asset_id, engine=False).config_fields()["seconds_per_token"]
