from . import episode_log


def fault_occurrences(log, hidden, asset):
    """Every fault occurrence of the episode as (component, began, ended), positions in the log: began is None for a fault present
    at the start and otherwise the entry that took the component off its healthy level, a deterioration transition, the expiry of
    an emergency repair or a new fault after a restoring close; ended is the entry that brought it back to its healthy level, None
    for a fault that lasts to the end. The one source of the onset of a fault for the time to work order, the time to repair and
    the time to repair in declared cycles, for any asset, in the order of the components and of their onsets."""
    current = dict(hidden)
    since = {c: None for c in asset.components if current[c] != asset.healthy[c]}
    occurrences = []
    for i, e in enumerate(log):
        after = episode_log._next_combination(current, e, asset)
        for c in asset.components:
            was_faulty, is_faulty = current[c] != asset.healthy[c], after[c] != asset.healthy[c]
            if is_faulty and not was_faulty:
                since[c] = i
            elif was_faulty and not is_faulty:
                occurrences.append((c, since.pop(c), i))
        current = after
    occurrences += [(c, began, None) for c, began in since.items()]
    order = {c: k for k, c in enumerate(asset.components)}
    return sorted(occurrences, key=lambda o: (order[o[0]], -1 if o[1] is None else o[1]))


def fault_onset(log, began):
    """The cycle at which a fault occurrence began, the cycle of the entry at `began`, and cycle 0 for a fault present at the start,
    whatever the agent read: the hidden record dates every fault, so an unread fault and a fault read late have the same onset."""
    return 0 if began is None else episode_log._time(log[began])


def restored_by_agent(log, ended, component):
    """Whether the entry at `ended`, which brought the component back to its healthy level, is a close of an agent work order on
    it, a restoring close."""
    e = log[ended]
    return e["event"] == "work_order_closed" and e.get("source") == "agent" and e["component"] == component


def _largest(delays, component, value):
    delays[component] = value if delays[component] is None else max(delays[component], value)


def _detection_delays(log, hidden, asset):
    """Per component the time to work order of the paper, the largest over its fault occurrences: for an occurrence, the cycles
    from its onset (fault_onset) to the first work order the agent created on the component at or after the onset and before the
    entry that ended the occurrence, so an order placed before the fault or after its end does not count. None for a component
    that never left its healthy level or got no such order in any occurrence."""
    orders = [(i, e) for i, e in enumerate(log) if e["event"] == "work_order_created"]
    delays = dict.fromkeys(asset.components)
    for component, began, ended in fault_occurrences(log, hidden, asset):
        onset = fault_onset(log, began)
        first = next((e["t_start"] for i, e in orders
                      if e["component"] == component and e["t_start"] >= onset and (ended is None or i < ended)), None)
        if first is not None:
            _largest(delays, component, first - onset)
    return delays


def _repair_delays(log, hidden, asset):
    """Per component the time to repair of the paper, the largest over its fault occurrences that a restoring close ended: the
    cycles from the onset (fault_onset) to that close. A fault present at the start counts from cycle 0 and a fault that began
    during the episode from the entry that began it, read by the agent or not, so a delay never waits for the agent's own reading.
    None for a component that never left its healthy level or whose occurrences no restoring close ended."""
    delays = dict.fromkeys(asset.components)
    for component, began, ended in fault_occurrences(log, hidden, asset):
        if ended is not None and restored_by_agent(log, ended, component):
            _largest(delays, component, episode_log._time(log[ended]) - fault_onset(log, began))
    return delays


def _unrestored(log, hidden, asset):
    """The components with a fault occurrence that no restoring close ended, in the order of the components: a fault that lasts to
    the end, and a fault that an emergency repair or the end of a hold took back to its level without an agent close. The reward
    takes the delay of such an episode at the horizon."""
    left = {c for c, began, ended in fault_occurrences(log, hidden, asset) if ended is None or not restored_by_agent(log, ended, c)}
    return [c for c in asset.components if c in left]
