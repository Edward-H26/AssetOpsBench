"""The oracle of an episode, written from the hidden state in the form ARE gives it (CausalDiagnosis/docs/environment.md,
section Oracle): the diagnosis of h(t0), one work order per component that is faulty at some time of the episode, each
with the cycle by which its repair has to close, and a finish. The verifier compares the agent's writes with it.
"""
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_5_deterioration_and_failure import failure_on, worst_level


def oracle(env):
    """{"diagnosis": {component: level of h(t0)}, "work_orders": [{"component", "window_end"}], "finish": True}.

    The work order set holds every component that is faulty at t0 or dropped by the schedule or the hazard before the
    episode ends (before the horizon while the episode runs). A drop at cycle s happens when the clock passes s, so an
    episode that ends at time s never saw it. window_end is the
    cycle at which the failure event would fire, that is the cycle at which the component reaches `patience` cycles at its
    worst level, or the horizon when it never would, which holds for every component while the failure rule is off
    (patience 0); an order that closes at or before window_end and restores the component averts the failure, since the
    close is processed before the failure check of that cycle, while an order that is open but not closed by then does
    not. window_end is the deadline of the failure card on the first run of the component at its worst level. A later
    run, which opens when the component comes back to its worst level, has a window end of its own, and the timing check
    of the verifier reads the window end of the run that is open at each close (verifier._run_windows)."""
    config, clock = env.config, env.clock
    end = env.time if env.done else config.horizon
    levels = dict(zip(config.components, env.initial))
    faulty = {c for c in config.components if levels[c] != config.healthy[c]}
    drops = sorted((d for d in clock.schedule + clock.hazard_drops if d["t"] < end), key=lambda d: d["t"])
    faulty |= {d["component"] for d in drops}
    work_orders = []
    for c in config.components:
        if c not in faulty:
            continue
        worst = worst_level(config, c)
        at_worst = [d["t"] for d in drops if d["component"] == c and d["to_level"] == worst]
        since = 0 if levels[c] == worst else (at_worst[0] if at_worst else None)
        failure_t = None if since is None or not failure_on(config) else since + config.failure["patience"] - 1
        window_end = failure_t if failure_t is not None and failure_t < config.horizon else config.horizon
        work_orders.append({"component": c, "window_end": window_end})
    return {"diagnosis": dict(levels), "work_orders": work_orders, "finish": True}
