"""The rules of the clock mode generation_time beside the chess clock (servers/thinking_clock.py), which measures the
seconds charged: their conversion into cycles of thinking, the one rule that the token clock (token_clock.py) applies to its
own seconds as well, whether a mode charges wall seconds at all, and the monotonic seconds between two calls that the state
file keeps beside them (monotonic_gap), which no replay reads."""
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_2_real_time import THINKING_MODES


def think_cycles(config, carried, seconds):
    """The thinking time of a call in cycles, charged in the two thinking modes only: the call's seconds, its wall seconds in
    the generation_time mode and its generated tokens times seconds_per_token in the generated_tokens mode (token_seconds),
    join the seconds carried from the calls before it, every full think_seconds_per_cycle of them is one cycle and the rest is
    carried to the next call, so an episode is charged the seconds of all its calls over think_seconds_per_cycle rounded down
    once, not a whole cycle per call. Returns (cycles, the seconds charged, the seconds carried after the call), and the other
    modes charge nothing and keep the seconds carried."""
    if config.clock_mode not in THINKING_MODES:
        return 0, 0.0, carried
    cycles, carried = divmod(carried + seconds, config.think_seconds_per_cycle)
    return int(cycles), seconds, carried


def charges_wall_seconds(config):
    """Whether the clock charges the wall seconds between two calls, which the load of the machine moves: in the generation_time
    mode alone. A reader that needs a charge a replay repeats, such as the digests that replay a scripted episode, asks this
    and not thinking(), since the generated_tokens mode charges thinking time a replay repeats."""
    return config.clock_mode == "generation_time"


def monotonic_gap(config, last, now):
    """The monotonic seconds between two calls in the mode that charges the wall seconds between them (charges_wall_seconds),
    zero in the other modes, none below zero, and None without both readings."""
    if not charges_wall_seconds(config):
        return 0.0
    if last is None or now is None:
        return None
    return max(0.0, now - last)
