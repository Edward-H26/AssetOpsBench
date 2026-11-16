from typing import Protocol, runtime_checkable
import numpy as np


# ----- the protocol of a diagnostics model -----
@runtime_checkable
class DiagnosticsModel(Protocol):
    """What rank_root_causes, the decision rule, the troubleshooting plans and verify_outcome read from an asset. A model
    without numeric evidence sets calibrated to False and
    returns None where a quantity needs the effect model."""
    calibrated: bool

    def hypotheses(self, view):
        """Every hypothesis of the asset, every combination of the profile for the recorded levels kind."""

    def prior(self, view, released):
        """The belief at cycle 0: the profile weights times the history term once a history is released."""

    def replay(self, view, released):
        """The belief now: the prior, then the released readings and closures in the order the environment served them."""

    def predict(self, belief, cycle, events):
        """One cycle of the kernels of the active rule cards, with the public events of that cycle."""

    def update(self, belief, observation):
        """The likelihood of one reading."""

    def tests(self, view):
        """The candidate tests by name with price and duration."""

    def expected_elimination(self, belief, test, draws, seed):
        """The expected number of hypotheses a test rules out, or None without an outcome model."""

    def information_gain(self, belief, test, draws, seed):
        """The expected drop of the entropy of the belief, in nats, that one reading of a test brings, never negative, which
        ranks the reads (E21), or None without an outcome model."""

    def after_repair(self, belief, components, view):
        """The belief after the repairs of one visit, under the dependency rule."""

    def repair_plan(self, hypothesis, view):
        """The repair steps that restore the hypothesis, priced and timed from the public rules, visits included."""

    def outcome_check(self, prediction, observation):
        """The verdict of a stored prediction against a later observation, or None without an outcome model."""


PROTOCOL_MEMBERS = tuple(sorted({*DiagnosticsModel.__annotations__,
                                 *(name for name, member in vars(DiagnosticsModel).items() if callable(member) and not name.startswith("_"))}))


def diagnostics_backend(model):
    """The model when it has every member of DiagnosticsModel, else a TypeError that names the members it lacks, so that a
    backend is refused where the server binds it rather than at the first ranking that reads a missing member."""
    missing = [name for name in PROTOCOL_MEMBERS if not hasattr(model, name)]
    if missing:
        raise TypeError(f"{type(model).__name__} is not a DiagnosticsModel: it lacks {', '.join(missing)}")
    return model


class EffectDiagnostics:
    """The DiagnosticsModel of the recorded levels kind. Its hypotheses are the combinations of the view, its belief is the
    filter of belief_filter.py over the augmented states weighed by the effect model E1, its repair plan is the one visit of
    schedule_visit, and its outcome check is the conformal verdict of E16 and E17. Every member reads the view, the
    released evidence and the episode's fitted models, never a hidden table, a seed or the fault rule of the starts."""
    calibrated = True

    def __init__(self, models, block=True, resampled=True):
        """models are the fits of one episode (adapter.Models.without); block False is the literal likelihood of the ranking
        model joint; resampled carries the refitted mean tables of E6 within the work bound of the filter."""
        self.models, self.block, self.resampled = models, block, resampled

    def _filter(self, view, released, support=None, replay=True):
        from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_filter import Episode, Filter, parse_released

        episode = released if isinstance(released, Episode) else parse_released(released)
        return Filter(view, self.models.effect, episode, resampled=self.resampled, block=self.block, support=support, replay=replay)

    def hypotheses(self, view):
        return list(view.combinations)

    def prior(self, view, released, support=None):
        """The belief at cycle 0: uniform over the hypotheses, or over the support, times the likelihood of the released fix
        history under the history rule and the draw of the schedule rule."""
        return self._filter(view, released, support, replay=False)

    def replay(self, view, released):
        """The belief now: the prior, then the released readings and closures in the order the rig served them, with the
        predict step of every cycle the clock ran (Filter._replay)."""
        return self._filter(view, released)

    def predict(self, belief, cycle, events):
        """One cycle of the kernels of the rule cards in the clock's order, with the closures among `events` stamped with
        the cycle, on a copy of the belief."""
        return belief.copy().predict(cycle, events)

    def update(self, belief, observation):
        """The likelihood of one recorded cycle, on a copy of the belief."""
        return belief.copy().observe(observation)

    def tests(self, view):
        """The reads a test can choose, one cycle of a sensor group each, with its price and its duration."""
        prices, per_cycle = view.rules["prices"]["read"], view.rules["durations"]["read_per_cycle"]
        return {g: {"price": float(prices[g]), "duration": int(per_cycle)} for g in sorted(view.sensor_groups)}

    def _one_cycle(self, belief, test, draws, seed):
        """The masses of the belief one cycle of a sensor group meets and those masses after each seeded draw of its reading."""
        ahead = belief.next_reading().posterior()
        fields = self.models.effect.space.groups[test]
        return ahead.probabilities, ahead.update_with(fields, ahead.sample_next(fields, draws, np.random.default_rng(seed)))

    def expected_elimination(self, belief, test, draws, seed):
        """The expected elimination of one cycle of a sensor group, reported beside the information gain of E21: the effective
        number of hypotheses of the belief the reading meets minus its mean after the draws of the reading."""
        from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.read_choice import dropped
        from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_at_one_cycle import effective_number

        before, after = self._one_cycle(belief, test, draws, seed)
        return dropped(float(effective_number(before)), effective_number(after))

    def information_gain(self, belief, test, draws, seed):
        """E21 of one cycle of a sensor group: the entropy of the belief the reading meets minus its mean after
        the draws of the reading, in nats."""
        from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.read_choice import gained
        from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_at_one_cycle import entropy

        before, after = self._one_cycle(belief, test, draws, seed)
        return gained(float(entropy(before)), entropy(after))

    def after_repair(self, belief, components, view):
        """The belief that the first reading after one visit on the components, created now, meets: the visit's closures on
        the cycles of schedule_visit, each taking only when its dependencies are healthy, and the rule cards meanwhile."""
        from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.troubleshooting_plans import visit_schedule

        visit = visit_schedule(view, components)
        cycles = visit[-1].close_offset + 1 if visit else 1
        return belief.ahead(cycles, [(s.component, "standard", belief.time + s.close_offset) for s in visit])

    def repair_plan(self, hypothesis, view):
        """The repairs that restore the hypothesis in one visit, in the technician's order with their close offsets."""
        from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.troubleshooting_plans import visit_schedule

        return visit_schedule(view, view.faulty(hypothesis))

    def outcome_check(self, prediction, observation):
        """E16 and E17 of a stored prediction against the values of one reading: per summary read the
        standardized deviation, then one combined score, the largest of them, its conformal p value against the calibration
        cycles scored the same way over the same summaries, and the verdict at the prediction's error level. None for the
        three when the reading holds no predicted summary."""
        from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.conformal_intervals import verdict

        index, sigma, calibration = self.models.effect.space.field_index, self.models.sigma, self.models.calibration
        checks = []
        for field, (mean, low, high, _quantile) in prediction["predicted"].items():
            if field not in observation:
                continue
            checks.append((field, mean, observation[field], low, high, abs(observation[field] - mean) / float(sigma[index[field]])))
        if not checks:
            return checks, None, None, None
        combined, p_value = calibration.combined_p_value([index[c[0]] for c in checks], [c[5] for c in checks])
        return checks, combined, p_value, verdict(p_value, prediction["alpha"])

__all__ = ["DiagnosticsModel", "EffectDiagnostics", "PROTOCOL_MEMBERS", "diagnostics_backend"]
