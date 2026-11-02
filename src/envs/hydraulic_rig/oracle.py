"""The oracle of an episode, written from the hidden state in the form ARE gives it (docs/hydraulic_rig/environment.md,
section Oracle): the diagnosis of h(t0), one work order per component that is faulty at some time of the episode, each
with the cycle by which its repair has to close, and a finish. The verifier compares the agent's writes with it.
"""
from .clock import worst_level


def oracle(env):
    """{"diagnosis": {component: level of h(t0)}, "work_orders": [{"component", "window_end"}], "finish": True}.

    The work order set holds every component that is faulty at t0 or dropped by the schedule before the episode ends
    (before the horizon while the episode runs), except a component the external event restores before any agent order on
    it. A drop at cycle s happens when the clock passes s, so an episode that ends at time s never saw it. window_end is the
    cycle at which the failure event would fire, that is the cycle at which the component reaches `patience` cycles at its
    worst level, or the horizon when it never would; an order closed at or before window_end averts the failure, since
    the close is processed before the failure check of that cycle."""
    config, clock = env.config, env.clock
    end = env.time if env.done else config.horizon
    levels = dict(zip(config.components, env.initial))
    faulty = {c for c in config.components if levels[c] != config.healthy[c]}
    drops = [d for d in clock.schedule if d["t"] < end]
    faulty |= {d["component"] for d in drops}
    external = clock.external
    if external is not None and external["t"] < end:
        agent_orders = [e for e in env.log if e["event"] == "work_order_created" and e["component"] == external["component"]]
        if not any(e["t_start"] < external["t"] for e in agent_orders):
            faulty.discard(external["component"])
    work_orders = []
    for c in config.components:
        if c not in faulty:
            continue
        worst = worst_level(config, c)
        at_worst = [d["t"] for d in drops if d["component"] == c and d["to_level"] == worst]
        since = 0 if levels[c] == worst else (at_worst[0] if at_worst else None)
        failure_t = None if since is None else since + config.failure["patience"] - 1
        window_end = failure_t if failure_t is not None and failure_t < config.horizon else config.horizon
        work_orders.append({"component": c, "window_end": window_end})
    return {"diagnosis": dict(levels), "work_orders": work_orders, "finish": True}
