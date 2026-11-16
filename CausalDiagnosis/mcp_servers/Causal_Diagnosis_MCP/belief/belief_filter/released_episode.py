import math
from collections import namedtuple
import numpy as np
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import has_clock  # the environment's rule of a row without a clock, kept in one place
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.served_recordings import group_recordings


# recorded: a record names the reading by its cycle, as on a row that reads a record (serving.same_recording)
Reading = namedtuple("Reading", "n t group values standing recorded", defaults=(False, False))
Closure = namedtuple("Closure", "t component source id mode after", defaults=(None,))
Observation = namedtuple("Observation", "fields index x numbers t standing", defaults=(False,))
# stops: the (first cycle, cycle after the last) of every run of cycles that a repair stopped
Episode = namedtuple("Episode", "time readings closures history pending clock stops", defaults=((), True, ()))


# ----- the released evidence -----
def parse_released(released):
    """The released evidence as the filter reads it: the readings, each standing when a read of zero duration served it
    while the clock stood at its cycle, before the kernels of that cycle, and recorded when its read event
    names the tool that served it, which the engine of a row that reads a record writes; the closures with the mode of their
    work order and the number of the last reading released before each (after); the fix history when the agent has asked
    for it; the open work orders with the cycle at which each will close; whether time passes on the row while the agent
    does not wait; and the runs of cycles that a repair stopped, from the cycle of the event
    work_order_executed of an order to the cycle of its closure, or to the time of the status while it is in repair."""
    events = released["events"]
    modes = {e["id"]: e.get("mode") or "standard" for e in events if e.get("event") == "work_order_created" and e.get("id")}
    closures, read_end, last, of_record, executed, closed = [], {}, 0, set(), {}, {}
    for e in events:
        if e.get("event") == "read":
            for o in e.get("served", ()):
                read_end[o["n"]] = e["t_end"]
                last = max(last, o["n"])
                if "tool" in e:
                    of_record.add(o["n"])
        elif e.get("event") == "work_order_executed":
            executed[e["id"]] = int(e["t_end"])
        elif e.get("event") == "work_order_closed":
            closures.append(Closure(int(e["t_end"]), e["component"], e.get("source", "agent"), e["id"], modes.get(e["id"], "standard"), last))
            closed[e["id"]] = int(e["t_end"])
    histories = [e.get("history") for e in events if e.get("event") == "fix_history"]
    readings = [Reading(o["n"], o["t"], o["sensor_group"], dict(o["values"]), o["t"] == read_end.get(o["n"]), o["n"] in of_record)
                for o in released["observations"]]
    status, rules = released["status"], released["rules"]
    pending = [Closure(int(o["expected_close_t"]), o["component"], o.get("source", "agent"), o["id"], o.get("mode") or "standard")
               for o in status.get("open_work_orders", []) if o.get("expected_close_t") is not None]
    stops = tuple(sorted((first, closed.get(order, int(status["time"]))) for order, first in executed.items()))
    return Episode(int(status["time"]), readings, closures, histories[-1] if histories else None, pending,
                   has_clock(rules["clock_mode"], rules["durations"]), stops)


def merge_observations(readings, space):
    """One observation per recording among the readings (serving.group_recordings), in the order of their first reading,
    with the summaries the effect model knows, which keeps the cycle and the standing mark of its first reading. A
    recording with no summary that the model knows is left out."""
    out = []
    for item in group_recordings(readings):
        fields = tuple(f for f in space.fields if f in item.values)
        if fields:
            out.append(Observation(fields, space.field_indices(fields), np.array([item.values[f] for f in fields]), tuple(item.numbers),
                                   item.t, item.standing))
    return out


def release_order(items):
    """The readings and closures the rig served while the clock stood at one cycle, in the order it released them: a
    closure after the readings numbered up to its after and before the next one."""
    return sorted(items, key=lambda item: (item.numbers[0], 0) if isinstance(item, Observation) else
                  (math.inf if item.after is None else item.after, 1))
