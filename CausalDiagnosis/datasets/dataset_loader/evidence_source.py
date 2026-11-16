"""The evidence an asset serves (CausalDiagnosis/docs/adding-a-dataset.md): the protocol through which the clock and the environment
reach the records of an asset, and RecordingSource, the source of the recorded condition levels kind.

A recording source serves the first BLOCK_CAP stable recordings of every combination, so that block length reveals nothing
about the combination, cut by one entry offset o that the episode draws at its start, the same way for every combination,
into two segments of SEGMENT recordings: the segment at o, which the engine of the AssetOpsBench rows records
before cycle 0, and the segment at o + SEGMENT, which every live window plays and then wraps among. The clock opens a window
on every change of the combination, on the live segment of a combination met for the first time in the episode, and for a
combination it returns to the window it left, resumed where the rig left it. A read returns the served statistics of the profile for every channel of the group at six
significant digits under the public field names of the profile, and the normal ranges and the alarms come from the recordings
the rig played, never from a label. The names with band in the code (BAND_QUANTILES, band_verdict and the others) name
the normal range of a channel, the term of every text an agent reads.
"""
from typing import Optional, Protocol, runtime_checkable

from CausalDiagnosis.datasets.splits import development_split
from CausalDiagnosis.datasets.dataset_loader import recordings as records

# the public normal range of a channel mean, from its smallest to its largest value over the stable cycles of the healthy
# combination, so that no healthy recording stands outside it
BAND_QUANTILES = (0.0, 1.0)
BLOCK_CAP = 10  # recordings served per combination, so that block length reveals nothing about the combination
SEGMENT = BLOCK_CAP // 2  # recordings of a segment, the prehistory or the live window of an episode
WORLDS_KEPT = 2  # the worlds whose tables a store keeps, as many as world.load_world keeps per process
_BANDS = {}  # the normal ranges of a recording source, per profile, recordings and blocks
_MOVED = {}  # the combinations that the development rule moved out of the test set, per profile and blocks


def segment_positions(index):
    """The block positions of a segment: SEGMENT positions from its index on, modulo the recordings a combination serves."""
    return [(index + k) % BLOCK_CAP for k in range(SEGMENT)]


def served_values(cycle, channels, profile):
    """What a reading returns for one recorded cycle: the served statistics of the profile for every channel, under the
    public field names of the profile and at six significant digits."""
    summaries = records.group_features(cycle, channels, profile, tuple(f"_{s}" for s in profile.statistics["served"]))
    return {records.public_name(k, profile): float(f"{v:.6g}") for k, v in sorted(summaries.items())}


def world_table(store, objects, compute):
    """compute(), kept in store for the very objects of the world it was computed from, for the WORLDS_KEPT worlds asked for
    last. An entry holds those objects, so no other object can take their identity while the entry stands."""
    key = tuple(id(o) for o in objects)
    kept = store.pop(key, None)
    if kept is None:
        kept = (objects, compute())
    store[key] = kept  # the entry asked for last stands last
    while len(store) > WORLDS_KEPT:
        del store[next(iter(store))]
    return kept[1]


def band_field(channel, profile):
    """The public field of the plain statistic of a channel, the served statistic listed ahead of the others, which carries
    the normal range and the alarm."""
    return f"{profile.field_name(channel)}_{profile.plain_statistic()}"


mean_field = band_field  # the former name, which the tests of the rig read


def loop(profile):
    """The loop of the readings of the recorded levels kind: an unchanged state replays its SEGMENT live recordings, and a
    start shows SEGMENT recordings from before its first cycle."""
    return SEGMENT, SEGMENT


@runtime_checkable
class EvidenceSource(Protocol):
    """What the clock and the environment read from the evidence of an asset. normal_ranges is the public normal range of
    the plain statistic of every channel, and window is the window open now, whose start and entry offset the clock
    reports. A source may also have fix_history(state), the recorded work orders of a start in the form of
    draw_fix_history (CausalDiagnosis/environment/Causal_Diagnosis_Environment/layers/layer_4_repair_rules.py), which the environment then serves while the fix
    history card is on. A source may also have draw_start(state, rng, offset), the window start and entry offset that
    open would draw, which lets the environment learn the start state of a seed without a reset, and outside(record), the
    normal range fields of a record outside their normal range, which lets it check the alarms of the cycles played since its
    last check alone."""

    normal_ranges: dict
    window: Optional[dict]

    def start_candidates(self, config) -> list:
        """The hidden starts that reset may draw under the fault rule and config.allowed, sorted."""

    def starts(self, combination) -> list:
        """Every (window start, entry offset) that open can draw for the combination."""

    def forget(self) -> None:
        """Drop the windows of the last episode, before the clock opens the start window of the next."""

    def open(self, state: tuple, t: int, rng, offset: Optional[int] = None) -> None:
        """Serve the state from cycle t on, drawing from rng what the serving needs, or entering at the given offset."""

    def prehistory(self, state: tuple) -> list:
        """The records of the state from before the episode, oldest first, which no live window plays."""

    def play(self) -> Optional[tuple]:
        """The record that the running asset produces in its next cycle, with whether it was played before, or None when
        the source holds no recording for the cycle, which then serves no reading."""

    def served_values(self, record, channels) -> dict:
        """The public answer of a read of the channels on one record."""

    def alarms(self, played: list, since: int, channels) -> list:
        """The channels outside their public normal range in a cycle played since `since`, each with its first such cycle."""


class RecordingSource:
    """The recorded condition levels kind: the stable blocks of the recordings, served through the segments of one entry offset per episode."""

    def __init__(self, profile, cycles, blocks):
        """cycles maps a recording index to its Cycle, and blocks maps a combination to the indices of its stable cycles,
        at least BLOCK_CAP of them, since an entry offset and the fit it names must exist for every combination alike."""
        short = sorted(combination for combination, block in blocks.items() if len(block) < BLOCK_CAP)
        if short:
            raise ValueError(f"every combination needs at least {BLOCK_CAP} stable recordings, and {len(short)} hold fewer, first {short[0]}")
        self.profile = profile
        self.cycles = cycles
        self.blocks = blocks
        self.window = None
        self.entry = None  # the entry offset of the episode, drawn with its start window, which every later window shares
        self.served = {}  # combination -> the window that served it in this episode, which play() advances in place
        self._values = {}  # (recording index, channels) -> the served values, formatted once per source
        self._outside = {}  # recording index -> the normal range fields outside their normal range, in channel order
        self.normal_ranges = world_table(_BANDS, (profile, cycles, blocks), self._normal_ranges)

    def start_candidates(self, config):
        """The one definition of the test set of an asset: under config.allowed the listed combinations within the fault
        rule, else the recorded combinations within the fault rule less those that the development rule moved to
        development (CausalDiagnosis/datasets/splits/development_split.py)."""
        if config.allowed is not None:
            allowed = {tuple(k) for k in config.allowed}
            return sorted(k for k in self.blocks if config.within_fault_rule(k) and k in allowed)
        moved = world_table(_MOVED, (self.profile, self.blocks), lambda: development_split.moved_to_development(self.profile, self.blocks))
        return sorted(k for k in self.blocks if config.within_fault_rule(k) and k not in moved)

    def starts(self, combination):
        """Every (window start, entry offset) that open can draw for the combination: BLOCK_CAP entries, whatever the length
        of its block, since every window starts at the first recording, and none for a combination without a recording."""
        return [(0, e) for e in range(BLOCK_CAP)] if combination in self.blocks else []

    def forget(self):
        """Drop the windows of the last episode, before the clock opens the start window of the next."""
        self.served, self.entry = {}, None

    def draw_start(self, state, rng, offset=None):
        """The (window start, entry offset) of the first window of an episode in the state, with the draws from rng that open
        makes and nothing else changed: a seeded draw, or the given offset of an anchored start, which the environment
        has checked against starts(). Every window starts at the first recording."""
        if offset is not None:
            return 0, offset
        # the first draw stays so that the later draws of the episode's generator hold
        rng.randrange(len(self.blocks[state]) - BLOCK_CAP + 1)
        return 0, rng.randrange(BLOCK_CAP)

    def open(self, state, t, rng, offset=None):
        """Serve the live segment of the state. The first window of an episode names its entry offset, a seeded draw or the
        given offset of an anchored start, and every later window shares it and draws nothing, so the segments of an episode
        tell nothing about its combinations. The clock plays cycles[(offset + played) % SEGMENT], so the segment is laid out
        from the entry offset on, which keeps the offset, the identity of the start and the index of the fits. A combination
        served before in the episode resumes its window where the rig left it, since a fresh window would put recordings
        back on the timeline that the window already showed."""
        if state in self.served:
            self.window = self.served[state]
            return
        block = self.blocks[state]
        if self.entry is None:
            _start, self.entry = self.draw_start(state, rng, offset)
        live = [block[p] for p in segment_positions(self.entry + SEGMENT)]
        self.window = {"t": t, "cycles": [live[(i - self.entry) % SEGMENT] for i in range(SEGMENT)], "start": 0, "offset": self.entry, "played": 0}
        self.served[state] = self.window

    def prehistory(self, state):
        """The segment of the state at the entry offset in play order, the recordings before cycle 0 of a row that records
        them, which no live window of the episode plays."""
        return [self.blocks[state][p] for p in segment_positions(self.entry)]

    def play(self):
        """Play the next recorded cycle of the window: (recording index, repeated), where repeated says that the window has
        wrapped and this cycle was played before in the current window."""
        w = self.window
        played = w["played"]
        w["played"] += 1
        return w["cycles"][(w["offset"] + played) % len(w["cycles"])], played >= len(w["cycles"])

    def served_values(self, index, channels):
        """What a read of the channels returns for one recording, formatted once per source and handed out as a new dict, so
        that no log entry shares its values with the store."""
        key = (index, tuple(channels))
        if key not in self._values:
            self._values[key] = served_values(self.cycles[index], channels, self.profile)
        return dict(self._values[key])

    def outside(self, index):
        """The normal range fields of one recording outside their normal range, in the channel order of the profile, computed once
        per source from the served values of every channel."""
        if index not in self._outside:
            channels, bands = tuple(self.profile.channels), self.normal_ranges
            values = self.served_values(index, channels)
            fields = (band_field(s, self.profile) for s in channels)
            self._outside[index] = tuple(f for f in fields if not bands[f]["low"] <= values[f] <= bands[f]["high"])
        return self._outside[index]

    def _normal_ranges(self):
        """{public field of the plain statistic of a channel: {"low", "high"}}: the normal range quantiles over the stable cycles of
        the healthy combination, its smallest and its largest value, computed on the served values so that the normal range and
        the alarm check see the same numbers and no healthy recording raises an alarm."""
        channels = tuple(self.profile.channels)
        served = [self.served_values(i, channels) for i in self.blocks[self.profile.healthy_state()]]
        bands = {}
        for s in channels:
            field = band_field(s, self.profile)
            values = [v[field] for v in served]
            bands[field] = {"low": records.percentile(values, BAND_QUANTILES[0]), "high": records.percentile(values, BAND_QUANTILES[1])}
        return bands

    def alarms(self, played, since, channels):
        """The plain statistics of the channels outside their normal range in any cycle the rig ran since `since`, with the
        first such cycle, from the recordings it played. played lists (cycle, recording index or None while the rig was
        stopped)."""
        wanted = {band_field(s, self.profile) for s in channels}
        first = {}
        for t, index in played:
            if t < since or index is None or not wanted:
                continue
            for field in self.outside(index):
                if field in wanted and field not in first:
                    first[field] = t
        return [{"channel": field, "first_t": t} for field, t in sorted(first.items(), key=lambda kv: (kv[1], kv[0]))]
