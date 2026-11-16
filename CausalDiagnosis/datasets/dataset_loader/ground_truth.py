"""The truth of an asset (CausalDiagnosis/docs/adding-a-dataset.md): what the verifier reads from an asset to score an episode, and
LevelsTruth, the truth of the recorded condition levels kind.

A recorded levels asset holds one level per component and has recovered when every component is back at its
healthy level. Its profile names the components each channel indicates, the failure modes a served answer releases, and
the words that the contradiction heuristic of the verifier reads as the whole asset. The oracle of an episode stays with
oracle.py, which reads the configuration and the clock of the engine, and the cost normalizer stays the run of the
oracle policy in the drivers.
"""
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class TruthModel(Protocol):
    """What the verifier reads from an asset to score an episode."""

    components: tuple
    healthy: dict
    whole_asset_words: tuple

    def levels(self, combination) -> Optional[dict]:
        """A combination as {component: level}, from the list in component order that a log entry holds or from a
        dict."""

    def diagnosis_correct(self, submitted: Optional[dict], truth: dict) -> dict:
        """Per component, whether the submitted level equals the true one, None for every component without a
        submission."""

    def recovered(self, final: dict) -> bool:
        """Whether the final combination is the healthy one."""

    def faulty(self, levels: dict) -> set:
        """The components off their healthy level."""

    def indicated(self, field: str) -> tuple:
        """The components that a channel, or the channel of a public field, indicates."""

    def fmea_ids(self, component: str) -> set:
        """The identifiers of the failure modes of a component, or of every component when it is empty, that a served
        answer releases."""


def misdiagnosed(model, submitted, truth):
    """Whether a submitted diagnosis misses the truth in the level of any component, false without a submission: the
    predicate behind diagnosis_all_correct of the verifier and behind the misdiagnosis charge of every engine and driver,
    so that a score and a charge never disagree on one diagnosis."""
    return submitted is not None and not all(model.diagnosis_correct(submitted, truth).values())


class LevelsTruth:
    """The recorded condition levels kind, read from the profile of the asset."""

    def __init__(self, profile):
        self.profile = profile
        self.components = profile.components
        self.healthy = profile.healthy
        self.whole_asset_words = tuple(profile.text["whole_asset_patterns"])

    def levels(self, combination):
        if combination is None:
            return None
        if isinstance(combination, dict):
            return {c: combination.get(c) for c in self.components}
        return dict(zip(self.components, combination))

    def diagnosis_correct(self, submitted, truth):
        return {c: (submitted.get(c) == truth[c]) if submitted else None for c in self.components}

    def recovered(self, final):
        return final == self.healthy

    def faulty(self, levels):
        return {c for c in self.components if levels.get(c) != self.healthy[c]}

    def indicated(self, field):
        return tuple(self.profile.knowledge["indicates"].get(self.profile.channel_of(field), ()))

    def fmea_ids(self, component):
        # the tool serves the modes that describe a fault, so a healthy or an informational entry of the profile is never released
        source = self.profile.knowledge["fmea_source"]
        return {f"fmea/{source}/{e['code']}" for e in self.profile.fault_modes() if not component or e["component"] == component}
