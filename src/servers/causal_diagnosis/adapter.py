"""The four tools of the diagnostics server as plain methods over the released evidence of one episode and the fitted
models of models.json (docs/hydraulic_rig/causal-diagnosis-server.md), testable without the mcp package.

No belief is stored between calls: every call replays the released observations and closed work orders into a fresh
posterior, so that a server process started per tool call, as the Plan-Execute runner does, answers exactly as a long
lived one would. The only file the tools write is predictions.json beside the released evidence, which verify_outcome
reads back. Every threshold in a result traces to a price, a duration, a downtime rate or the declared error level, and
the assumptions of the result say how.
"""
import json
from collections import namedtuple
from functools import lru_cache
from pathlib import Path

import numpy as np

from envs.hydraulic_rig import knowledge as kb
from envs.hydraulic_rig.data import COMPONENTS, HEALTHY, LEVELS
from servers.json_files import write_json_atomically

from .conformal import DEFAULT_ALPHA, Calibration, interval, verdict
from .decision import MONTE_CARLO_SAMPLES, MONTE_CARLO_SEED, costs, decide, expected_downtime_rate, unseparated
from .discriminant_urn import urn_probabilities
from .effect_model import FIELD_INDEX, GROUP_FIELDS, HEALTHY_COMBO, SUMMARY_FIELDS, EffectModel, channel_field, combo_key, statistic_of
from .models import (Candidate, ChannelCheck, ChannelShare, ClosedWorkOrder, ComponentAttribution, EpisodeEvidence, ErrorResult, EvidenceItem,
                     Fault, InterventionEffect, MonteCarlo, OutcomeVerification, PredictedSummary, RootCauses, SeparatingTest, Stability,
                     TimeWindow, Uncertainty)
from .posterior import DEFAULT_RANKING_MODEL, RANKING_MODELS, ClosedOrder, Posterior, Reading, apply_orders

MODELS_PATH = Path(__file__).resolve().with_name("models.json")
MODELS_FORMAT = 2
RELEASED_FORMAT = 2
PREDICTIONS_NAME = "predictions.json"
UNIT = "cycle"
MONITOR = "monitor"
REPLACE = "replace:"
ABSTENTION_RULES = ("decision", "fixed_cut")  # E10, or the v1 posterior cut kept for ablation A8
DEFAULT_ABSTENTION_CUT = 0.95  # the v1 cut, used only under abstention_rule fixed_cut and named in the result
Episode = namedtuple("Episode", "released rules status events readings closed_orders")

A_POOL = ("every model is fitted once on the fixed pool, the stable cycles of every combination outside the evaluation split of the "
          "default fault rule, so no model has seen a combination an episode can start from and no model depends on the episode")
A_E1 = ("E1: the mean of each of the 34 summaries (mean and standard deviation of the 17 channels) under a combination is an intercept plus "
        "one main effect per non healthy level of each component plus the pairwise interactions that lowered the cross fitted log "
        "likelihood over the pool, with the pooled within combination covariance shrunk toward its diagonal by the Ledoit and Wolf rule")
A_E2 = ("E2: the posterior over the start combinations the public fault rule allows is proportional to a uniform prior times the product "
        "over readings of the Gaussian density of the read summaries under the combination current at the reading; the current combination "
        "is the start combination with every component replaced by a work order closed at or before the reading's cycle set to its healthy "
        "level; a scheduled drop of a component is not modeled")
A_BLOCK = ("joint_block: the cycles of one recorded block share a latent offset from the model mean with the covariance of the cross "
           "fitted block misfits, so repeated readings of one block add less than independent draws would; the default ranking model")
A_URN = ("discriminant_urn: the v1 ranking model, one linear discriminant per sensor group and component read on the same 34 summaries, "
         "repeated readings combined by a Polya urn around the confusion row of the cross fit within the pool; the lookahead and the "
         "predictions keep the joint model's Gaussian predictive around this posterior")
A_DEDUPE = ("a cycle served twice between two work order closures, because the recording window wrapped or two groups read the same "
            "cycle, enters once with the union of its summaries")
A_E3 = "E3: the anomaly score of a summary is its absolute deviation from the median of the healthy recordings over their median absolute deviation; it enters no decision"
A_E4 = ("E4: a summary's predicted deviation under the top candidate splits into the main effects of its faulty components and halves of "
        "their interactions, and a component's explained anomaly is the sum over summaries of the anomaly score times its share")
A_E6 = ("E6: the effect model is refit on resamples of the pool drawn by recorded block, and the interval of the top marginal is its "
        "range over the resampled fits; no band constant is used")
A_E14 = ("E14: the predicted mean of a summary after the action is the posterior weighted mean of the model means, and its interval is the "
         "split conformal interval mean plus or minus q times the within standard deviation, with q the smallest held out calibration "
         "score such that at least (1 - alpha)(n + 1) units of block weight lie at or below it, n the number of pool blocks; the interval "
         "is calibrated for a known combination, so a spread posterior adds spread the interval does not carry")
A_E16 = ("E16 and E17: the p value of a channel is (1 + weight of calibration scores at or above its score) / (n + 1); the outcome is "
         "inconsistent when the smallest p value is at or below alpha / K with K channels read (Bonferroni, valid under any dependence), "
         "and the Holm step down is reported beside it; the recovery check compares the predicted recovery with the posterior of the "
         "healthy combination from the post repair readings alone under a uniform prior over the combinations the repair can reach")
A_TIME = "time counts rig cycles from the start of the episode; refused calls release nothing and no cycle index of the recording is released"
A_SIX_DIGITS = "values are the released ones, at six significant digits"
A_BANDS = "the healthy band of a channel mean is the public band of the asset profile, served by the action server, not a fitted threshold"


class NoEpisode(Exception):
    """A tool needs released evidence that this server does not have."""


class Models:
    """models.json parsed once per process: the effect model, its conformal calibration and the v1 ranking model."""

    def __init__(self, raw):
        if raw.get("format") != MODELS_FORMAT:
            raise ValueError(f"models.json has format {raw.get('format')}, this server reads format {MODELS_FORMAT}; refit with "
                             "`python -m servers.causal_diagnosis.fit`")
        self.raw = raw
        self.effect = EffectModel.from_json(raw["effect"])
        calibration = raw["effect"]["calibration"]
        self.calibration = Calibration(calibration["scores"], calibration["weights"], calibration["n_blocks"])
        self.sigma = np.sqrt(np.diag(self.effect.sigma_within))
        self.urn = raw["discriminant_urn"]
        self.pool = raw["pool"]


@lru_cache(maxsize=1)
def load_models(path=MODELS_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(f"{Path(path).name} is generated from the rig data and not part of the repository: run "
                                "`python -m servers.causal_diagnosis.fit` with HYD_FACTS_PATH set")
    return Models(json.loads(Path(path).read_text(encoding="utf-8")))


def parse_episode(released):
    """The released evidence as the posterior reads it: readings and closed work orders in time order."""
    readings = [Reading(o["n"], o["t"], o["sensor_group"], dict(o["values"])) for o in released["observations"]]
    closed = [ClosedOrder(e["t_end"], e["component"], e.get("source", "agent"), e["id"])
              for e in released["events"] if e.get("event") == "work_order_closed"]
    return Episode(released, released["rules"], released["status"], released["events"], readings, closed)


def combo_text(combo):
    return "(" + ", ".join(f"{c} {v}" for c, v in zip(COMPONENTS, combo)) + ")"


def rule_text(rules, n_candidates):
    rule = rules["fault_rule"]
    return (f"the candidates are the {n_candidates} combinations the public rule allows: between {rule['min_faulty']} and "
            f"{rule['max_faulty']} components off their healthy level at the start")


def observation_ids(readings):
    return [f"obs/{r.n}/{field}" for r in readings for field in sorted(r.values)]


def order_ids(closed_orders):
    return [f"wo/{o.id}" for o in closed_orders]


def band_of(rules, field):
    """The public healthy band of a channel mean, keyed by the summary field or by the channel field."""
    bands = rules.get("healthy_bands") or {}
    band = bands.get(field)
    if band is None and statistic_of(field) == "mean":
        band = bands.get(channel_field(field))
    return band


class CausalTools:
    """The four tools. With released_path None the server serves no episode and every tool answers with an error."""

    def __init__(self, models, released_path=None, ranking_model=DEFAULT_RANKING_MODEL, abstention_rule="decision", abstention_cut=DEFAULT_ABSTENTION_CUT,
                 alpha=DEFAULT_ALPHA, samples=MONTE_CARLO_SAMPLES, seed=MONTE_CARLO_SEED):
        if ranking_model not in RANKING_MODELS:
            raise ValueError(f"ranking_model must be one of {RANKING_MODELS}")
        if abstention_rule not in ABSTENTION_RULES:
            raise ValueError(f"abstention_rule must be one of {ABSTENTION_RULES}")
        if not 0 < float(abstention_cut) <= 1:
            raise ValueError("abstention_cut must be a probability in (0, 1]")
        self.models = models
        self.released_path = None if released_path is None else Path(released_path)
        self.ranking_model = ranking_model
        self.abstention_rule = abstention_rule
        self.abstention_cut = float(abstention_cut)
        self.alpha = alpha
        self.samples = samples
        self.seed = seed

    # ----- the episode -----
    def _released(self):
        if self.released_path is None:
            raise NoEpisode("this server serves no episode; start one through the hydraulic_rig server")
        if not self.released_path.exists():
            raise NoEpisode("no released evidence yet; the episode has not been started")
        released = json.loads(self.released_path.read_text(encoding="utf-8"))
        if released.get("format") != RELEASED_FORMAT:
            raise NoEpisode(f"the released evidence has format {released.get('format')}; this server reads format {RELEASED_FORMAT}")
        return parse_episode(released)

    def _posterior(self, episode, resampled=True, readings=None, closed_orders=None, time=None, starts=None):
        readings = episode.readings if readings is None else readings
        closed_orders = episode.closed_orders if closed_orders is None else closed_orders
        time = episode.status["time"] if time is None else time
        post = Posterior(self.models.effect, episode.rules, readings, closed_orders, time,
                         block_variance=self.ranking_model == "joint_block", resampled=resampled, starts=starts)
        if self.ranking_model == "discriminant_urn" and starts is None:
            post.override(urn_probabilities(self.models.urn, episode.rules, readings, closed_orders))
        return post

    def _model_assumptions(self):
        notes = [A_POOL, A_E1, A_E2, A_DEDUPE]
        if self.ranking_model == "joint_block":
            notes.append(A_BLOCK)
        if self.ranking_model == "discriminant_urn":
            notes.append(A_URN)
        return notes

    def _predictions_path(self):
        return self.released_path.with_name(PREDICTIONS_NAME)

    def _predictions(self):
        path = self._predictions_path()
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

    @staticmethod
    def _window(episode, readings=None):
        readings = episode.readings if readings is None else readings
        start = min((r.t for r in readings), default=episode.status["time"])
        return TimeWindow(start=start, end=episode.status["time"], unit=UNIT)

    # ----- get_episode_evidence -----
    def get_episode_evidence(self, sensor_group):
        try:
            episode = self._released()
        except NoEpisode as e:
            return ErrorResult(error=str(e))
        group = sensor_group.strip()
        groups = episode.rules["sensor_groups"]
        if group == "all":
            channels = None
        elif group in groups:
            channels = set(groups[group])
        else:
            sensor = kb.sensor_of(group)
            if sensor is None:
                return ErrorResult(error=f"unknown sensor_group {group}: use all, {', '.join(sorted(groups))}, a channel such as PS1 or a field "
                                         "such as PS1_Pressure_bar_100Hz")
            channels = {kb.HYD1_FIELDS[sensor]}
        items = []
        for reading in episode.readings:
            anomaly = self.models.effect.anomaly_scores(reading.values)
            for field, value in sorted(reading.values.items()):
                if channels is not None and channel_field(field) not in channels:
                    continue
                sensor = kb.sensor_of(field)
                unit, rate = kb.unit_and_rate(sensor)
                band = band_of(episode.rules, field)
                items.append(EvidenceItem(evidence_id=f"obs/{reading.n}/{field}", field=field, channel=sensor, statistic=statistic_of(field),
                                          value=value, unit=unit, rate=rate, n=reading.n, t=reading.t, sensor_group=reading.group,
                                          band_low=None if band is None else band["low"], band_high=None if band is None else band["high"],
                                          inside_band=None if band is None else bool(band["low"] <= value <= band["high"]),
                                          anomaly_score=anomaly.get(field)))
        closed = [ClosedWorkOrder(id=o.id, component=o.component, source=o.source, t=o.t) for o in episode.closed_orders]
        outside = sum(1 for i in items if i.inside_band is False)
        scored = [i.anomaly_score for i in items if i.anomaly_score is not None]
        return EpisodeEvidence(
            sensor_group=group, n_observations=len(episode.readings), items=items, work_orders_closed=closed,
            evidence_ids=[i.evidence_id for i in items] + order_ids(episode.closed_orders), time_window=self._window(episode),
            assumptions=[A_BANDS, A_E3, A_TIME, A_SIX_DIGITS],
            uncertainty=Uncertainty(kind="largest_anomaly_score", value=max(scored) if scored else 0.0),
            message=(f"{len(items)} items from {len(episode.readings)} observation(s); {outside} channel means outside their public healthy band; "
                     f"{len(closed)} work order(s) closed"))

    # ----- rank_root_causes -----
    def rank_root_causes(self, top_k):
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            return ErrorResult(error="top_k must be a positive integer")
        try:
            episode = self._released()
        except NoEpisode as e:
            return ErrorResult(error=str(e))
        post = self._posterior(episode)
        cost = costs(episode.rules, episode.status["time"])
        decision = decide(post, cost, self.samples, self.seed)
        order = post.order()
        marginals, current = post.component_marginals(), post.current_marginals()
        levels = post.level_probability()
        insufficient, best, separating, abstention_text = self._abstention(post, cost, decision, order, levels)
        attribution, _anomaly, _combo = post.attribution(order[0])
        stability = post.stability()
        ranked = [Candidate(rank=i + 1, combination=dict(zip(COMPONENTS, post.candidates[k])), current_combination=dict(zip(COMPONENTS, post.current[k])),
                            faults=[Fault(component=c, level=v, code=kb.level_code(c, v)) for c, v in zip(COMPONENTS, post.candidates[k]) if v != HEALTHY[c]],
                            probability=float(post.probabilities[k]), evidence_ids=self._bearing_ids(post.candidates[k], episode))
                  for i, k in enumerate(order[:top_k])]
        rate = expected_downtime_rate(post.current, post.probabilities, cost)
        test = SeparatingTest(sensor_group=best, cycles=1, price=cost.test_price[best], duration=cost.test_duration[best], cost=decision.test_cost[best],
                              expected_loss=decision.expected_loss_test[best]) if insufficient else None
        top = post.candidates[order[0]]
        if self.abstention_rule == "fixed_cut":
            verdict_text = (f"insufficient evidence under the fixed cut {self.abstention_cut:g}: read one cycle of {best}" if insufficient else
                            f"evidence sufficient under the fixed cut {self.abstention_cut:g}")
        else:
            verdict_text = (f"insufficient evidence: one cycle of {best} lowers the expected loss from {decision.expected_loss_act:.3g} to "
                            f"{decision.expected_loss_test[best]:.3g}" if insufficient else
                            f"evidence sufficient: no single reading lowers the expected loss of acting now ({decision.expected_loss_act:.3g})")
        unsep_text = f"; unseparated: {', '.join(separating)}" if separating else ""
        cost_text = (f"E7 to E10: an unnecessary replacement costs the part plus the repair charge plus the stopped downtime rate over the repair "
                     f"downtime ({', '.join(f'{c} {v:g}' for c, v in cost.unnecessary.items())}); a missed fault costs its downtime rate over the "
                     f"remaining horizon of {cost.remaining} cycles; a test costs one cycle of its read price plus one cycle of the expected "
                     f"downtime rate {rate:.3g}; the flag, the separating test and the unseparated components follow from these costs and the "
                     f"posterior through a one step lookahead with {decision.samples} Monte Carlo draws per test (seed {decision.seed}); no fixed "
                     f"confidence cut is used")
        return RootCauses(
            ranking_model=self.ranking_model, n_candidates=len(post.candidates), candidates=ranked, component_fault_probability=marginals,
            current_fault_probability=current, level_probability={c: {str(lv): p for lv, p in levels[c].items()} for c in COMPONENTS},
            attribution=[ComponentAttribution(component=c, fault_probability=marginals[c], current_fault_probability=current[c],
                                              explained_anomaly=attribution[c]["explained_anomaly"],
                                              channels=[ChannelShare(**item) for item in attribution[c]["channels"]]) for c in COMPONENTS],
            stability=None if stability is None else Stability(**stability), insufficient_evidence=insufficient,
            separating_test=test, expected_saving=decision.expected_saving, expected_loss_act=decision.expected_loss_act,
            expected_loss_test=decision.expected_loss_test, unseparated=sorted(separating, key=COMPONENTS.index),
            separating_channels={c: separating[c] for c in sorted(separating, key=COMPONENTS.index)},
            monte_carlo=MonteCarlo(samples=decision.samples, seed=decision.seed, standard_error=decision.standard_error),
            costs={"unnecessary_replacement": cost.unnecessary, "remaining_horizon": cost.remaining, "test_cost": decision.test_cost,
                   "expected_downtime_rate": rate},
            evidence_ids=observation_ids(episode.readings) + order_ids(episode.closed_orders), time_window=self._window(episode),
            assumptions=[rule_text(episode.rules, len(post.candidates))] + self._model_assumptions() + [A_E3, A_E4, A_E6, cost_text, abstention_text, A_TIME],
            uncertainty=Uncertainty(kind="top_marginal_resampling_interval", value=stability["top_marginal"] if stability else float(max(marginals.values())),
                                    detail=None if stability is None else {"low": stability["interval_low"], "high": stability["interval_high"],
                                                                           "top_share": stability["top_share"]}),
            message=(f"{len(post.candidates)} candidate combination(s); top {combo_text(top)} at {post.probabilities[order[0]]:.3f}; fault probabilities "
                     + ", ".join(f"{c} {marginals[c]:.3f}" for c in COMPONENTS) + f"; {verdict_text}{unsep_text}"))

    def _abstention(self, post, cost, decision, order, levels):
        """The abstention flag, the separating test and the unseparated components under the configured rule: E10 and E12
        by default, or for ablation A8 the v1 rule, a fixed cut on the top posterior with the cheapest sensor group as the
        test and the components whose most probable level is below the cut as unseparated."""
        if self.abstention_rule == "decision":
            separating = unseparated(post, cost, decision)
            text = ("abstention_rule decision: the flag, the separating test and the unseparated components are E10 and E12, decided by "
                    "the expected losses above")
            return decision.insufficient_evidence, decision.separating_test, separating, text
        cheapest = min(cost.test_price, key=lambda g: (cost.test_price[g], g))
        insufficient = bool(post.probabilities[order[0]] < self.abstention_cut)
        separating = {c: list(GROUP_FIELDS[cheapest]) for c in COMPONENTS if max(levels[c].values()) < self.abstention_cut}
        text = (f"abstention_rule fixed_cut (ablation A8) with cut {self.abstention_cut:g}: insufficient_evidence is true when the top "
                f"posterior is below the cut, the separating test is the cheapest sensor group ({cheapest}) and a component is unseparated "
                f"when its most probable level is below the cut, as the v1 server did; the expected losses of E8 to E10 are reported beside "
                f"it and decide nothing here")
        return insufficient, cheapest, separating, text

    @staticmethod
    def _bearing_ids(combo, episode):
        """The released summaries whose channel indicates a faulty component of the combination, and the closed orders."""
        faulty = {c for c, v in zip(COMPONENTS, combo) if v != HEALTHY[c]}
        ids = [f"obs/{r.n}/{field}" for r in episode.readings for field in sorted(r.values)
               if faulty & set(kb.INDICATES.get(kb.sensor_of(field), ()))]
        return ids + order_ids(episode.closed_orders)

    # ----- estimate_intervention_effect -----
    def estimate_intervention_effect(self, action):
        try:
            episode = self._released()
        except NoEpisode as e:
            return ErrorResult(error=str(e))
        action = action.strip()
        if action == MONITOR:
            component = None
        elif action.startswith(REPLACE) and action[len(REPLACE):] in COMPONENTS:
            component = action[len(REPLACE):]
        else:
            return ErrorResult(error=f"action must be {MONITOR} or {REPLACE}<component> with the component one of {', '.join(COMPONENTS)}")
        rules, status = episode.rules, episode.status
        post = self._posterior(episode)
        cost = costs(rules, status["time"])
        rate = expected_downtime_rate(post.current, post.probabilities, cost)
        if component is None:
            after, addresses, recovery = post.current_posterior(), 0.0, post.current_posterior().get(HEALTHY_COMBO, 0.0)
            price, duration, downtime = 0.0, rules["durations"]["advance_per_cycle"], rate * rules["durations"]["advance_per_cycle"]
            resampled = post.resampled_recovery(None)
        else:
            after, addresses, recovery = post.post_repair(component)
            price = rules["prices"]["parts"][component] + rules["prices"]["repair_charge"]
            duration = rules["durations"]["lead_time"][component] + rules["durations"]["repair_downtime"][component]
            downtime = rules["downtime_rate"]["stopped"] * rules["durations"]["repair_downtime"][component]
            resampled = post.resampled_recovery(component)
        low = high = band = None
        if resampled is not None:
            low, high = float(resampled.min()), float(resampled.max())
            band = (high - low) / 2
        predicted = self._predict(after)
        risk_before = 1 - post.current_posterior().get(HEALTHY_COMBO, 0.0)
        risk_after = 1 - recovery
        if band is not None and risk_after < risk_before - band:
            direction = "lower"
        elif band is not None and risk_after > risk_before + band:
            direction = "higher"
        else:
            direction = "no change" if band is not None else "undetermined"
        predictions = self._predictions()
        prediction_id = f"pred/{len(predictions) + 1}"
        predictions.append({"id": prediction_id, "action": action, "component": component, "events": len(episode.events),
                            "observations": len(episode.readings), "time": status["time"], "addresses": addresses, "recovery": recovery,
                            "recovery_interval": None if band is None else [low, high], "alpha": self.alpha, "ranking_model": self.ranking_model,
                            "after": {combo_key(h): p for h, p in after.items()},
                            "predicted": {p.field: [p.mean, p.low, p.high, p.quantile] for p in predicted}})
        write_json_atomically(self._predictions_path(), predictions, indent=1)
        notes = self._model_assumptions() + [
            "E13: the posterior after the action moves every candidate's current combination through the replacement; the recovery probability "
            "is the moved probability of the healthy combination; a replaced component is at its healthy level by the replacement rule",
            A_E14, A_E6,
            "E15: risk is one minus the posterior of the healthy combination before and after the action, and the direction uses a dead band "
            "equal to the half width of the resampling interval of the recovery probability",
            f"the price is the part plus the repair charge of the public price list, the duration the lead time plus the repair downtime, and the "
            f"downtime during the action the stopped rate over the repair downtime; for {MONITOR} the duration is one cycle at the expected downtime "
            f"rate {rate:.3g}, since no observation is free",
            f"alpha = {self.alpha:g} is the declared error level of the intervals"]
        if component is not None and any(o.get("component") == component for o in status.get("open_work_orders", [])):
            notes.append("a work order on this component is open; the environment refuses a second one")
        if component is not None and not any(e.get("event") in ("submit_diagnosis", "declare_insufficient_evidence") for e in episode.events):
            notes.append("the environment accepts a work order only after a diagnosis or a declaration of insufficient evidence")
        if status["episode_over"]:
            notes.append("the episode is over, so the action can no longer be taken")
        return InterventionEffect(
            prediction_id=prediction_id, action=action, component=component, addresses_fault=addresses, recovery=recovery,
            recovery_interval_low=low, recovery_interval_high=high, price=price, duration=duration, downtime_during_action=downtime,
            predicted=predicted, alpha=self.alpha, risk_before=risk_before, risk_after=risk_after, direction=direction, dead_band=band,
            evidence_ids=observation_ids(episode.readings) + order_ids(episode.closed_orders) + [prediction_id],
            time_window=TimeWindow(start=status["time"], end=status["time"] + duration, unit=UNIT), assumptions=notes,
            uncertainty=Uncertainty(kind="recovery_probability", value=recovery, detail=None if band is None else {"low": low, "high": high}),
            message=(f"{action}: addresses a fault with probability {addresses:.3f}, recovery {recovery:.3f}"
                     + (f" (resampling interval {low:.3f} to {high:.3f})" if band is not None else "")
                     + f", price {price:g}, duration {duration} cycle(s); risk {risk_before:.3f} to {risk_after:.3f}, {direction}; "
                     f"{len(predicted)} summaries predicted with {1 - self.alpha:g} conformal intervals as {prediction_id}"))

    def _predict(self, after):
        """E14 for every summary: the posterior weighted mean and the conformal interval at alpha."""
        combos = list(after)
        weights = np.array([after[h] for h in combos])
        means = weights @ self.models.effect.mu_rows(combos)
        out = []
        for k, field in enumerate(SUMMARY_FIELDS):
            q, _capped = self.models.calibration.quantile(k, self.alpha)
            low, high = interval(float(means[k]), float(self.models.sigma[k]), q)
            out.append(PredictedSummary(field=field, mean=float(means[k]), low=low, high=high, quantile=q))
        return out

    # ----- verify_outcome -----
    def verify_outcome(self, prediction_id):
        try:
            episode = self._released()
        except NoEpisode as e:
            return ErrorResult(error=str(e))
        prediction = next((p for p in self._predictions() if p["id"] == prediction_id.strip()), None)
        if prediction is None:
            return ErrorResult(error=f"unknown prediction_id {prediction_id}; estimate_intervention_effect returns one such as pred/1")
        events, component, first = episode.events, prediction["component"], prediction["events"]
        later = [r for r in episode.readings if r.n > prediction["observations"]]
        if component is None:
            since, status, recovered_text = prediction["time"], "", "not applicable: monitor repairs nothing"
            closed_between = []
        else:
            closing = next((e for e in events[first:] if e.get("event") == "work_order_closed" and e.get("component") == component), None)
            if closing is None:
                refusals = [e for e in events[first:] if e.get("event") == "refused" and isinstance(e.get("action"), dict)
                            and e["action"].get("type") == "create_work_order" and e["action"].get("component") == component]
                if refusals:
                    status = f"action refused: {refusals[-1].get('reason')}"
                elif any(o.get("component") == component for o in episode.status.get("open_work_orders", [])):
                    status = "not yet observable: the work order is open and has not closed"
                else:
                    status = "not yet observable: no work order on the component has closed since the prediction"
                return self._verification(prediction, status, None, [], None, "not observable: the repair has not happened")
            since = closing["t_end"]
            later = [r for r in later if r.t >= since]
            closed_between = [e for e in events[first:] if e.get("event") == "work_order_closed" and e.get("component") != component and e["t_end"] <= since]
            recovered_text = ""
        if not later:
            return self._verification(prediction, "not yet observable: no reading since the action", None, [], None,
                                      recovered_text or "not observable: no reading since the repair closed")
        observed = min(later, key=lambda r: (r.t, r.n))
        others = [e for e in events[first:] if e.get("event") == "work_order_closed" and e.get("component") != component and e["t_end"] <= observed.t]
        if others or closed_between:
            names = ", ".join(f"replace:{e['component']}" for e in others + closed_between)
            return self._verification(prediction, f"not comparable: {names} closed between the prediction and the reading", None, [], None,
                                      recovered_text or "not observable: another repair changed the combination")
        checks, p_values = [], {}
        for field, (mean, low, high, _q) in prediction["predicted"].items():
            if field not in observed.values:
                continue
            k = FIELD_INDEX[field]
            score = abs(observed.values[field] - mean) / float(self.models.sigma[k])
            p_values[field] = self.models.calibration.p_value(k, score)
            checks.append((field, mean, observed.values[field], low, high, score))
        family = verdict(p_values, prediction["alpha"])
        channels = [ChannelCheck(field=f, predicted=m, observed=o, low=lo, high=hi, score=s, p_value=p_values[f], bonferroni=family["bonferroni"][f], holm=family["holm"][f])
                    for f, m, o, lo, hi, s in checks]
        recovery_posterior = None
        if component is not None:
            support = [tuple(int(v) for v in key.split(",")) for key in prediction["after"]]
            post = self._posterior(episode, resampled=False, readings=later, closed_orders=[], time=episode.status["time"], starts=support)
            recovery_posterior = post.current_posterior().get(HEALTHY_COMBO, 0.0)
            recovered_text = f"predicted recovery {prediction['recovery']:.3f}, posterior of the healthy combination from the post repair readings {recovery_posterior:.3f}"
        return self._verification(prediction, "verified", observed, channels, family, recovered_text, recovery_posterior, since)

    def _verification(self, prediction, status, observed, channels, family, recovery_status, recovery_posterior=None, since=None):
        interval_ = prediction.get("recovery_interval")
        n = len(channels)
        consistent = None if family is None else family["consistent"]
        bonferroni = sum(1 for c in channels if c.bonferroni)
        holm = sum(1 for c in channels if c.holm)
        ids = [prediction["id"]] + ([f"obs/{observed.n}/{c.field}" for c in channels] if observed else [])
        window = TimeWindow(start=since if since is not None else prediction["time"], end=observed.t if observed else prediction["time"], unit=UNIT)
        if consistent is None:
            verdict_text = ""
        else:
            verdict_text = (f"; {'consistent' if consistent else 'inconsistent'} at alpha {prediction['alpha']:g}: {bonferroni} of {n} channels rejected by "
                            f"Bonferroni, {holm} by Holm")
        return OutcomeVerification(
            prediction_id=prediction["id"], action=prediction["action"], status=status, observation=observed.n if observed else None,
            observed_t=observed.t if observed else None, channels=channels, n_channels=n, alpha=prediction["alpha"], consistent=consistent,
            bonferroni_rejections=bonferroni, holm_rejections=holm, recovery_predicted=prediction["recovery"],
            recovery_interval_low=None if interval_ is None else interval_[0], recovery_interval_high=None if interval_ is None else interval_[1],
            recovery_posterior=recovery_posterior, recovery_status=recovery_status, evidence_ids=ids, time_window=window,
            assumptions=self._model_assumptions() + [A_E14, A_E16, f"alpha = {prediction['alpha']:g} is the declared error level of the verdict", A_TIME],
            uncertainty=Uncertainty(kind="smallest_p_value", value=min(p_values_of(channels), default=1.0)),
            message=f"{prediction['id']} ({prediction['action']}): {status}{verdict_text}; recovery {recovery_status}")


def p_values_of(channels):
    return [c.p_value for c in channels]
