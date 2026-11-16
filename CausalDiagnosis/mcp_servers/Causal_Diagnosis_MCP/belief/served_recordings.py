"""Which readings show one recording, and what the row of an episode can still serve to a reading
(CausalDiagnosis/docs/causal-diagnosis-mcp.md, sections The filter and E7 to E12).

A recording enters the belief once. Two readings show one recording when a record names both by one cycle, as on a row that
reads a record, where a cycle index names the one recording the record holds for it (Reading.recorded), or when they agree
on every summary they share and share at least one, as when a window wrapped and a recording was served again. The filter
and the count below group the readings by this one rule (group_recordings).

A test is a reading that the row can still serve. A sensor group is open until it has been shown one window of recordings
in full since the last released closure, after which the rule acts. On a row that reads a record an open group can be
served every recorded cycle since that closure whose recording lacks a summary of the group or was never read, and the
record holds the window of cycles before the episode and every cycle the clock played that no repair stopped
(recorded_cycles). On a row that plays a recording per read a group can be shown the window less the recordings it has
been shown in full (left_to_serve). The count reads the released evidence and the one public length of the window, never
a status field that names an extent of the record, a recording index or a hidden event.

Two limits stay. The count ends at one window, so on a row with a clock later cycles can still show recordings that no
reading has shown. A level drop or the end of an emergency hold starts a new window that no public event names, so the
count does not start again there.
"""
from collections import namedtuple
from dataclasses import dataclass

# the recordings a reading can still show a sensor group, and the summaries of the group that those recordings would add
Left = namedtuple("Left", "cycles summaries")


@dataclass
class Recording:
    """One recording among the readings of an episode: every summary shown of it, the (cycle, number) of each reading that
    showed it, the cycle and the standing mark of its first reading, and whether a record names it by its cycle."""
    values: dict
    served: list
    t: int
    standing: bool
    recorded: bool

    @property
    def numbers(self):
        """The numbers of its readings in the order of their (cycle, number)."""
        return [n for _t, n in sorted(self.served)]

    @property
    def cycles(self):
        """The cycles at which it was served."""
        return {t for t, _n in self.served}


def same_recording(item, reading):
    """Whether a reading shows the recording item: a record names both by one cycle, or the two agree on every summary they
    share and share at least one, which is a recording served again after its window wrapped. Two readings of one cycle
    on a row that plays a recording per read are two recordings unless their summaries say otherwise."""
    if item.recorded and reading.recorded and reading.t in item.cycles:
        return True
    shared = [f for f in reading.values if f in item.values]
    return bool(shared) and all(item.values[f] == reading.values[f] for f in shared)


def group_recordings(readings):
    """The distinct recordings among the readings, in the order of the (cycle, number) of their first reading, with every
    released summary, known to the effect model or not. A reading joins every recording it is the same as (same_recording),
    and those become one, so that a read of a wrapped cycle through another group leaves no recording in two items."""
    found = []
    for reading in sorted(readings, key=lambda r: (r.t, r.n)):
        same = [item for item in found if same_recording(item, reading)]
        if not same:
            found.append(Recording(dict(reading.values), [(reading.t, reading.n)], reading.t, reading.standing, reading.recorded))
            continue
        item, joined = same[0], same[1:]
        for other in joined:
            item.values.update(other.values)
            item.served += other.served
        found = [kept for kept in found if not any(kept is other for other in joined)]
        item.values.update(reading.values)
        item.served.append((reading.t, reading.n))
    return found


def recordings(readings):
    """The distinct recordings among the readings, each as {summary: value} (group_recordings)."""
    return [item.values for item in group_recordings(readings)]


def left_of(view, shown, window):
    """{sensor group: Left} for the recordings shown so far, each given as the summaries shown of it: for every sensor
    group of the view, the window less the recordings that hold every summary of the group, never below zero, and the
    summaries of the group that the recordings not yet shown in full still lack. Before any reading every group has the
    whole window left."""
    left = {}
    for group in view.sensor_groups:
        fields = set(view.group_fields(group))
        missing = [len(fields - set(summaries)) for summaries in shown]
        cycles = max(0, window - sum(1 for lacking in missing if lacking == 0))
        unseen = max(0, window - len(shown))
        left[group] = Left(cycles, sum(missing) + unseen * len(fields) if cycles else 0)
    return left


def left_to_serve(view, readings, window):
    """left_of for the recordings among the readings the server has been shown (recordings)."""
    return left_of(view, recordings(readings), window)


def shows(reading, closure):
    """Whether a reading shows the state after a released closure: recorded at or after the closure's cycle and released
    after it, which leaves out a reading that a read of zero duration served while the clock stood at that cycle before the
    closure."""
    return reading.t >= closure.t and reading.n > closure.after


def recorded_cycles(episode, window):
    """The cycles that the record of a row that reads one holds: the window of cycles -window to -1 before the episode,
    then every cycle from 0 to the time less one that no repair stopped (Episode.stops). Time passes in the waits of a row
    without a clock and in every call of a row with one, and the record holds each cycle that the asset played."""
    return [*range(-window, 0), *(t for t in range(episode.time) if not any(first <= t < after for first, after in episode.stops))]


def since_closure(episode, window):
    """The readings that show the state since the last released closure, every reading when none was released, and the
    first cycle that counts, the cycle of that closure or -window."""
    if not episode.closures:
        return list(episode.readings), -window
    last = episode.closures[-1]
    return [r for r in episode.readings if shows(r, last)], last.t


def left_in_episode(view, episode, window):
    """{sensor group: Left}: what the row can still serve each group. A group is open while the recordings shown since the
    last closure that hold every summary of the group number less than the window. On a row that reads a record the cycles
    of an open group are the recorded cycles since that closure whose recording lacks a summary of the group or was never
    read, and its summaries are those that these cycles would add. On a row that plays a recording per read the count is
    left_to_serve of the readings since the closure. A closed group has zero cycles and zero summaries."""
    readings, first = since_closure(episode, window)
    if not view.reads_record:
        return left_to_serve(view, readings, window)
    shown = group_recordings(readings)
    of_cycle = {t: item for item in shown for t in item.cycles}
    cycles = [t for t in recorded_cycles(episode, window) if t >= first]
    left = {}
    for group in view.sensor_groups:
        fields = set(view.group_fields(group))
        if sum(1 for item in shown if fields <= set(item.values)) >= window:
            left[group] = Left(0, 0)
            continue
        lacking = [len(fields - set(of_cycle[t].values)) if t in of_cycle else len(fields) for t in cycles]
        left[group] = Left(sum(1 for n in lacking if n), sum(lacking))
    return left
