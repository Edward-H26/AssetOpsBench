"""The registry of evidence kinds (CausalDiagnosis/docs/adding-a-dataset.md).

The evidence kind of an asset says how the engine plays its episodes: whether it plays them at all, whether the records
show the asset after a change of its condition, the keys of the records table beside the data keys, the check of a raw
asset file that only the kind can make, the loader of the data file, the source that serves the records, the truth model of
the verifier and the loop of the readings. The engine builds none of them by name. The built in kinds are recorded_levels,
the recorded condition levels with their segments, and scripted_cases, whose episodes the engine does not play. The
package of an asset registers a further kind with register_kind, and a field may name its object by a "module:attribute"
reference, resolved on first use, so that the package imports no module outside the asset layer.
"""
from dataclasses import dataclass

from CausalDiagnosis.datasets.dataset_loader import evidence_source as evidence, dataset_packages as ports, recordings as records, ground_truth as truth


@dataclass(frozen=True)
class EvidenceKind:
    """An evidence kind, the value of evidence_kind in the asset file of its assets."""

    name: str
    plays_episodes: bool = True  # the engine plays its episodes: the asset file holds every table of the engine
    replays_repairs: bool = True  # the records show the asset after a change of its condition
    record_keys: tuple = ()  # the keys of records beside the data keys
    check: object = None  # check(data) -> the problems of a raw asset file that only the kind can find
    loader: object = None  # loader(path, profile) -> (records, blocks)
    source: object = None  # source(profile, records by index, blocks) -> an EvidenceSource
    truth: object = None  # truth(profile) -> a TruthModel, LevelsTruth when None
    loop: object = None  # loop(profile) -> (live, before), None for a kind whose source does not loop

    def part(self, name):
        """The object of a field, its "module:attribute" reference resolved on first use."""
        value = getattr(self, name)
        return ports.resolve(value, f"the evidence kind {self.name}: {name}") if isinstance(value, str) else value


BUILTIN_KINDS = {kind.name: kind for kind in (
    EvidenceKind("recorded_levels", record_keys=("label_keys", "stable_flag_stable"), check=records.record_problems,
                 loader=records.load_recordings, source=evidence.RecordingSource, truth=truth.LevelsTruth, loop=evidence.loop),
    EvidenceKind("scripted_cases", plays_episodes=False, replays_repairs=False),
)}
_REGISTERED = {}


def register_kind(kind):
    """Register a further evidence kind under its name and return it. The name of a built in kind or of an earlier
    registered one is refused, and so is a kind that plays episodes without a loader or a source."""
    if not isinstance(kind, EvidenceKind):
        raise TypeError(f"an evidence kind is a kinds.EvidenceKind, not {kind!r}")
    if kind.name in BUILTIN_KINDS or kind.name in _REGISTERED:
        raise ValueError(f"the evidence kind {kind.name!r} is already declared")
    missing = [field for field in ("loader", "source") if getattr(kind, field) is None]
    if kind.plays_episodes and missing:
        raise ValueError(f"the evidence kind {kind.name!r} plays episodes, so it needs a {' and a '.join(missing)}")
    _REGISTERED[kind.name] = kind
    return kind


def unregister_kind(name):
    """Remove a registered kind, as a test or the fixture of a made-up asset does. A built in kind cannot be removed."""
    if name in BUILTIN_KINDS:
        raise ValueError(f"the built in evidence kind {name!r} cannot be removed")
    _REGISTERED.pop(name, None)


def kind_of(name):
    """The kind of the name, built in or registered, or None."""
    if not isinstance(name, str):
        return None
    return BUILTIN_KINDS.get(name) or _REGISTERED.get(name)


def names():
    """The names of every kind, the built in ones first."""
    return tuple(BUILTIN_KINDS) + tuple(_REGISTERED)
