"""The seven tools of the causal-diagnosis server as plain methods over the released answers of one episode and the fitted
models of models.json (docs/hydraulic_rig/causal-diagnosis-server.md), testable without the mcp package.

No belief is stored between calls: every call replays the released events, which are few, so that a server process
started per tool call, as the Plan-Execute runner does, answers exactly as a long-lived one would. The only file the
tools write is predictions.json beside the released answers, which verify_outcome reads back.
"""
import json
from collections import namedtuple
from math import comb
from functools import lru_cache
from pathlib import Path

from envs.hydraulic_rig.classifiers import COMBOS, TIERS, lda_predict, signature
from envs.hydraulic_rig.data import COMPONENTS, HEALTHY, LEVELS, SENSORS
from envs.hydraulic_rig.env import HEALTHY_COMBO, n_faulty
from servers.hydraulic_rig.names import ASSET_ID, HYD1_FIELDS, tier_fields
from servers.json_files import write_json_atomically

from . import knowledge as kb
from .effect import BAND, combo_key, design, mixture
from .models import (AssetContext, Candidate, ChannelContext, ComponentContext, EpisodeEvidence, ErrorResult, EvidenceItem, Fault,
                     FmeaPath, FmeaPaths, IndicatingChannel, InterventionEffect, OutcomeVerification, PredictedSummary, Residual,
                     RootCauses, SymptomReport, TimeWindow, Uncertainty)

MODELS_PATH = Path(__file__).resolve().with_name("models.json")
RELEASED_FORMAT = 1
PREDICTIONS_NAME = "predictions.json"
CONFIDENCE = 0.95  # below this top posterior the ranking reports insufficient evidence, as the greedy reference policy does
BAND_COVERAGE = BAND[1] - BAND[0]
MISS_RATE = 1 - BAND_COVERAGE  # the share of a calibrated prediction's summaries expected outside their bands
FALSE_ALARM = 0.05  # the largest probability with which a calibrated prediction may be called inconsistent
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four"}
TIER_GROUPS = tuple(f"tier{t}" for t in TIERS)

Measurement = namedtuple("Measurement", "number tier result repeated")

A_CLASSIFIER = ("a measurement is read by one linear discriminant per component, fitted on the pool and applied to the released values; "
                "the confusion rows come from a cross-fit within the pool, counted on its one- and three-fault combinations")
A_URN = ("repeated readings of one served block combine by a Polya urn whose counts restart under the hypotheses in which a replaced "
         "component was faulty; a cycle read again at the same tier adds nothing")
A_REPLACED = "a replaced component is at its healthy level by the replacement rule, whatever the classifiers read afterwards"
A_BAND = ("a healthy band is the 1st to 99th percentile over the pool cycles in which every component the channel indicates is healthy; "
          "a channel that indicates no component takes all pool cycles")
A_TIME = "time counts accepted measurements; refused calls are not counted, and no cycle number or timestamp is released"
A_SIX_DIGITS = "values are the released ones, at six significant digits"
A_ASSOC = "an association between a channel and a component is diagnostic evidence, not a causal edge"
A_GROUPS = "the measurement groups are nested, tier2 adds channels to tier1 and tier3 to tier2; a channel's group is the cheapest tier that serves it"
A_CORR = ("the correlations are those of Helwig et al. 2015, Table II, on interval features of the fixed working cycle, "
          "not on the served cycle statistics")
A_ACCURACY = ("held-out accuracy per group is the share of the pool's one- and three-fault cycles whose predicted level equals the true level "
              "in the cross-fit within the pool")
A_CODES = ("the failure-mode codes, severities and recommended actions are the 16 distinct entries of the IndustryAssetEQA hydraulic facts, "
           "extracted without their cycles; the level that carries a code is the component's condition level")
A_TEXT = "the task text names no component, level or physical effect; it is composed from the public rule when the scenario carries none"
A_EFFECT = ("predicted summaries come from an additive model of the component levels fitted once per combination on the pool without that "
            "combination's own recordings; a combination outside the pool takes the fit on the whole pool; the band is the 1st to 99th "
            "percentile of the posterior-weighted mixture, from residual quantiles stored on a grid")
A_NEW_BLOCK = ("where the replaced component was faulty the environment serves recordings of the new combination after the repair, "
               "which restart the urn")
A_PRICE = "the price is the part plus the downtime charge of the public price list"
A_RECOVERY = ("recovery ends the episode at once, so an episode that continues after a repair did not recover; an episode that ends with "
              "the repair reports only that it is over, which a recovery, a used-up repair limit and the step cap share")
A_CONSISTENT = ("consistent means that no more observed summaries lie outside their bands than n_outside_tolerated, the smallest count "
                "that a calibrated prediction exceeds with probability at most 0.05 under a binomial model of independent summaries "
                "that each miss their band with probability 0.02, and, where recovery is observable, that the outcome matches the "
                "predicted recovery probability rounded to the nearer of 0 and 1; p_value_outside is the probability under that model "
                "of at least as many summaries outside as observed")


def binomial_tail(n, k, p=MISS_RATE):
    """P(X >= k) for X ~ Binomial(n, p)."""
    return sum(comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(max(k, 0), n + 1))


def misses_tolerated(n, p=MISS_RATE, level=FALSE_ALARM):
    """The smallest number of band misses to tolerate so that a calibrated prediction of n summaries is called
    inconsistent with probability at most level; requiring every summary inside its band would reject a calibrated
    prediction of 17 summaries with probability 1 - 0.98 ** 17, about 0.29."""
    return next(k for k in range(n + 1) if binomial_tail(n, k + 1, p) <= level)


def outcome_consistent(n_outside, n, recovered, recovery_predicted):
    return n_outside <= misses_tolerated(n) and (recovered is None or recovered == (recovery_predicted >= 0.5))


class NoEpisode(Exception):
    """A tool needs released answers that this server does not have."""


@lru_cache(maxsize=1)
def load_models(path=MODELS_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(f"{Path(path).name} is generated from the rig data and not part of the repository: run "
                                "`python -m servers.causal_diagnosis.fit` with HYD_FACTS_PATH set")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def level_index(combo):
    return tuple(LEVELS[c].index(v) for c, v in zip(COMPONENTS, combo))


def classify(models, tier, result):
    """The level index per component that the fixed-pool classifiers of the tier read from a released measurement."""
    tier_model = models["tiers"][str(tier)]
    x = [result[name] for name in tier_model["public_names"]]
    return tuple(lda_predict((m["mean"], m["scale"], m["coef"], m["bias"]), x) for m in tier_model["lda"])


def task_text(low, high):
    """The task text of a scenario class, worded as docs/hydraulic_rig/environment.md words S1 and S2, from the public rule."""
    opening = f"Hydraulic unit {ASSET_ID} was flagged by the performance monitor."
    if low == high == 1:
        rule = "Exactly one component is off its healthy level. Find it"
    elif low == high:
        rule = f"Exactly {NUMBER_WORDS[low]} components are off their healthy levels. Find them"
    else:
        rule = f"Between {NUMBER_WORDS[low]} and {NUMBER_WORDS[high]} components are off their healthy levels. Find them"
    return f"{opening} {rule} and restore the unit within the budget."


def pool_text(models):
    pool = models["pool"]
    return (f"every model is fitted once on the fixed pool, the stable cycles of the {len(pool['combinations'])} combinations outside "
            f"the evaluation class S2 ({pool['cycles']} cycles), so no model depends on the episode")


def rule_text(rules, n_candidates):
    return (f"the candidates are the {n_candidates} combinations the public rule allows: between {rules['min_faulty']} and "
            f"{rules['max_faulty']} components off their healthy level")


def static_window():
    return TimeWindow(start=0, end=0, unit="static")


def measurement_window(measurements):
    return TimeWindow(start=1 if measurements else 0, end=len(measurements), unit="measurement")


def observation_ids(measurements):
    return [f"obs/{m.number}/{field}" for m in measurements for field in sorted(m.result)]


def combo_text(combo):
    return "(" + ", ".join(f"{c} {v}" for c, v in zip(COMPONENTS, combo)) + ")"


class Belief:
    """Posterior over the start combinations the public rule allows, replayed from the released answers.

    Hypotheses are kept per start combination, each with its current combination (after repairs), its urn counts per
    tier and the readings it has absorbed, so that under a range of fault counts two start combinations brought to one
    combination by a repair keep their different blocks."""

    def __init__(self, candidates, models):
        self.models = models
        self.hypotheses = {h: self._fresh(h, 1.0 / len(candidates)) for h in candidates}

    @staticmethod
    def _fresh(current, weight):
        return {"current": current, "weight": weight,
                "counts": {t: [[0] * len(LEVELS[c]) for c in COMPONENTS] for t in TIERS},
                "seen": {t: set() for t in TIERS}}

    def measure(self, tier, result):
        """Bayes update by the urn of the greedy reference policy: the next reading of a block follows its confusion row
        pulled towards the block's earlier readings at that tier; a cycle read again at the same tier adds nothing."""
        read = classify(self.models, tier, result)
        key = signature(result)
        tier_model = self.models["tiers"][str(tier)]
        rows, alpha = tier_model["rows"], tier_model["alpha"]
        for s in self.hypotheses.values():
            if key in s["seen"][tier]:
                continue
            idx = level_index(s["current"])
            like = 1.0
            for j in range(len(COMPONENTS)):
                n = s["counts"][tier][j]
                like *= (alpha[j] * rows[j][idx[j]][read[j]] + n[read[j]]) / (alpha[j] + sum(n))
            s["weight"] *= like
            s["seen"][tier].add(key)
            for j in range(len(COMPONENTS)):
                s["counts"][tier][j][read[j]] += 1
        total = sum(s["weight"] for s in self.hypotheses.values())
        for s in self.hypotheses.values():
            s["weight"] /= total
        return read

    def replace(self, component, continued):
        """Move every hypothesis through the replacement; where the component was faulty the environment serves a new
        block, so the urn starts again, and where it was healthy the block and its readings stay. The environment ends
        an episode at once when every component is healthy, a public rule of the briefing, so when the episode continued
        after the replacement, the hypotheses that it brought to the healthy combination drop out."""
        j = COMPONENTS.index(component)
        for s in self.hypotheses.values():
            if s["current"][j] != HEALTHY[component]:
                after = s["current"][:j] + (HEALTHY[component],) + s["current"][j + 1:]
                s.update(self._fresh(after, s["weight"]))
        if continued:
            self.hypotheses = {h: s for h, s in self.hypotheses.items() if s["current"] != HEALTHY_COMBO}
            total = sum(s["weight"] for s in self.hypotheses.values())
            for s in self.hypotheses.values():
                s["weight"] /= total

    def posterior(self):
        """Posterior over the current combinations, summed over the start hypotheses that lead to each."""
        out = {}
        for s in self.hypotheses.values():
            out[s["current"]] = out.get(s["current"], 0.0) + s["weight"]
        return out

    def level_probability(self):
        post = self.posterior()
        return {c: {lv: sum(w for h, w in post.items() if h[j] == lv) for lv in LEVELS[c]} for j, c in enumerate(COMPONENTS)}

    def fault_probability(self):
        levels = self.level_probability()
        return {c: sum(p for lv, p in levels[c].items() if lv != HEALTHY[c]) for c in COMPONENTS}


class CausalTools:
    """The seven tools. With released_path None the server serves no episode and only the two static tools answer."""

    def __init__(self, models, released_path=None):
        self.models = models
        self.released_path = None if released_path is None else Path(released_path)

    # ----- the episode -----
    def _released(self):
        if self.released_path is None:
            raise NoEpisode("this server serves no episode; only get_asset_context and query_fmea_paths are available")
        if not self.released_path.exists():
            raise NoEpisode("no released answers yet; the episode has not been started")
        released = json.loads(self.released_path.read_text(encoding="utf-8"))
        if released.get("format") != RELEASED_FORMAT:
            raise NoEpisode("unknown format of the released answers")
        return released

    def _replay(self, released):
        """The candidates of the public rule, the belief after every released event and the accepted measurements."""
        rules = released["rules"]
        candidates = [h for h in COMBOS if rules["min_faulty"] <= n_faulty(h) <= rules["max_faulty"]]
        belief = Belief(candidates, self.models)
        measurements = []
        events = released["events"]
        for i, event in enumerate(events):
            if event["event"] == "measure":
                measurements.append(Measurement(len(measurements) + 1, event["tier"], event["result"], event["repeated"]))
                belief.measure(event["tier"], event["result"])
            elif event["event"] == "replace":
                belief.replace(event["component"], continued=i + 1 < len(events) or not released["status"]["episode_over"])
        return candidates, belief, measurements

    def _predictions_path(self):
        return self.released_path.with_name(PREDICTIONS_NAME)

    def _predictions(self):
        path = self._predictions_path()
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

    # ----- static knowledge -----
    def get_asset_context(self, asset_id):
        profile = kb.asset_profile(asset_id)
        if profile is None:
            return ErrorResult(error=f"unknown asset_id {asset_id}; this server knows {ASSET_ID}")
        components = [ComponentContext(name=c, levels=list(LEVELS[c]), healthy=HEALTHY[c], n_fault_codes=len(kb.fault_codes_of(c)))
                      for c in COMPONENTS]
        channels = []
        for s in SENSORS:
            unit, rate = kb.unit_and_rate(s)
            channels.append(ChannelContext(channel=s, field=HYD1_FIELDS[s], unit=unit, rate=rate, group=kb.group_of(s),
                                           statistics=list(kb.STATISTICS), indicates=list(kb.INDICATES.get(s, ()))))
        evidence = [f"assoc/{kb.ASSOCIATION_SOURCE}/{s}_{c}" for s in SENSORS for c in kb.INDICATES.get(s, ())]
        return AssetContext(
            asset_id=profile["asset_id"], asset_type=profile["asset_type"], description=profile["description"], site=profile["site"],
            components=components, channels=channels, n_failure_codes=len(kb.FAILURE_CODES), n_fault_codes=len(kb.FAULT_CODES),
            evidence_ids=evidence, time_window=static_window(), assumptions=[A_ASSOC, A_GROUPS, A_CODES],
            uncertainty=Uncertainty(kind="none", value=0.0),
            message=(f"{profile['asset_type']} {asset_id}: {len(components)} components, {len(channels)} channels in three nested groups, "
                     f"{len(kb.FAILURE_CODES)} failure-mode codes of which {len(kb.FAULT_CODES)} are faults"))

    def query_fmea_paths(self, asset_id, node):
        if asset_id != ASSET_ID:
            return ErrorResult(error=f"unknown asset_id {asset_id}; this server knows {ASSET_ID}")
        node = node.strip()
        sensor = kb.sensor_of(node)
        if node in COMPONENTS:
            kind, entries = "component", kb.fault_codes_of(node)
        elif node in kb.CODES:
            kind, entries = "failure_mode", [kb.CODES[node]]
        elif sensor is not None:
            kind, entries = "channel", [e for c in kb.INDICATES.get(sensor, ()) for e in kb.fault_codes_of(c)]
        else:
            return ErrorResult(error=(f"unknown node {node}: give a component (cooler, valve, pump, accumulator), a failure-mode code "
                                      "such as valve_switching_critical, or a channel such as PS1"))
        paths = [self._path(e) for e in entries]
        accuracies = {f"{p.component}/{group}": a for p in paths for group, a in p.held_out_accuracy.items()}
        lowest = min((p.held_out_accuracy["tier3"] for p in paths if p.held_out_accuracy), default=0.0)
        if kind == "channel" and not paths:
            message = f"channel {sensor} indicates no component in the association table"
        else:
            message = f"{len(paths)} path(s) for {kind} {node}: " + "; ".join(f"{p.component} {p.code} ({p.severity}, {p.recommended_action})" for p in paths)
        return FmeaPaths(asset_id=asset_id, node=node, node_kind=kind, paths=paths, evidence_ids=[i for p in paths for i in p.evidence_ids],
                         time_window=static_window(), assumptions=[A_ASSOC, A_CORR, A_ACCURACY, A_CODES],
                         uncertainty=Uncertainty(kind="held_out_accuracy_tier3", value=lowest, detail=accuracies or None), message=message)

    def _path(self, entry):
        comp = entry["component"]
        channels = [IndicatingChannel(channel=s, field=HYD1_FIELDS[s], correlation=kb.HELWIG_CORRELATIONS.get(comp, {}).get(s), group=kb.group_of(s))
                    for s in SENSORS if comp in kb.INDICATES.get(s, ())]
        accuracy = ({f"tier{t}": self.models["tiers"][str(t)]["accuracy"][COMPONENTS.index(comp)] for t in TIERS}
                    if comp in COMPONENTS else {})
        ids = [f"fmea/{kb.FMEA_SOURCE}/{entry['code']}"] + [f"assoc/{kb.ASSOCIATION_SOURCE}/{c.channel}_{comp}" for c in channels]
        return FmeaPath(component=comp, level=entry["level"], code=entry["code"], severity=entry["severity"], recommended_action=entry["action"],
                        channels=channels, held_out_accuracy=accuracy, evidence_ids=ids)

    # ----- the episode's tools -----
    def get_symptom_report(self):
        try:
            released = self._released()
        except NoEpisode as e:
            return ErrorResult(error=str(e))
        rules = released["rules"]
        n_candidates = sum(1 for h in COMBOS if rules["min_faulty"] <= n_faulty(h) <= rules["max_faulty"])
        text = released.get("task_text") or task_text(rules["min_faulty"], rules["max_faulty"])
        return SymptomReport(
            asset_id=released["asset_id"], task_text=text, report_time=0, fault_count_min=rules["min_faulty"], fault_count_max=rules["max_faulty"],
            n_candidates=n_candidates, budget=rules["budget"], step_cap=rules["max_steps"], repairs=rules["max_repairs"], prices=rules["prices"],
            status=released["status"], evidence_ids=[], time_window=TimeWindow(start=0, end=0, unit="measurement"),
            assumptions=[A_TEXT, rule_text(rules, n_candidates), A_TIME], uncertainty=Uncertainty(kind="candidates", value=float(n_candidates)),
            message=f"{text} Reported at measurement 0 with {n_candidates} candidate combinations under the rule.")

    def get_episode_evidence(self, sensor_group):
        try:
            released = self._released()
        except NoEpisode as e:
            return ErrorResult(error=str(e))
        group = sensor_group.strip()
        if group == "all":
            fields = None
        elif group in TIER_GROUPS:
            fields = set(tier_fields(int(group[len("tier"):])))
        else:
            sensor = kb.sensor_of(group)
            if sensor is None:
                return ErrorResult(error=(f"unknown sensor_group {group}: use all, tier1, tier2, tier3, a channel such as PS1 or a field "
                                          "such as PS1_Pressure_bar_100Hz"))
            fields = {HYD1_FIELDS[sensor]}
        _candidates, _belief, measurements = self._replay(released)
        items = []
        for m in measurements:
            for field, value in sorted(m.result.items()):
                if fields is not None and field.rsplit("_", 1)[0] not in fields:
                    continue
                sensor = kb.sensor_of(field)
                unit, rate = kb.unit_and_rate(sensor)
                band = self.models["bands"][field]
                items.append(EvidenceItem(evidence_id=f"obs/{m.number}/{field}", field=field, channel=sensor, statistic=kb.statistic_of(field),
                                          value=value, unit=unit, rate=rate, release_time=m.number, repeated=m.repeated, band_low=band["low"],
                                          band_high=band["high"], inside_band=band["low"] <= value <= band["high"], band_cycles=band["cycles"],
                                          band_healthy=list(band["healthy"])))
        outside = sum(1 for i in items if not i.inside_band)
        return EpisodeEvidence(
            sensor_group=group, n_measurements=len(measurements), items=items, evidence_ids=[i.evidence_id for i in items],
            time_window=measurement_window(measurements), assumptions=[A_BAND, A_TIME, A_SIX_DIGITS],
            uncertainty=Uncertainty(kind="band_coverage", value=BAND_COVERAGE),
            message=f"{len(items)} items from {len(measurements)} measurement(s); {outside} outside their healthy band")

    def rank_root_causes(self, top_k):
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            return ErrorResult(error="top_k must be a positive integer")
        try:
            released = self._released()
        except NoEpisode as e:
            return ErrorResult(error=str(e))
        candidates, belief, measurements = self._replay(released)
        post = belief.posterior()
        order = sorted(post, key=lambda h: (-post[h], h))
        levels = belief.level_probability()
        fault = belief.fault_probability()
        unseparated = [c for c in COMPONENTS if max(levels[c].values()) < CONFIDENCE]
        separating = {c: [HYD1_FIELDS[s] for s in SENSORS if c in kb.INDICATES.get(s, ())] for c in unseparated}
        top = post[order[0]]
        ranked = [Candidate(rank=i + 1, combination=dict(zip(COMPONENTS, h)),
                            faults=[Fault(component=c, level=v, code=kb.level_code(c, v)) for c, v in zip(COMPONENTS, h) if v != HEALTHY[c]],
                            probability=post[h], evidence_ids=self._bearing_ids(h, measurements))
                  for i, h in enumerate(order[:top_k])]
        insufficient = top < CONFIDENCE
        message = f"{len(post)} candidate combination(s); top {combo_text(order[0])} at {top:.3f}; " + (
            "insufficient evidence, unseparated: " + ", ".join(unseparated) if insufficient else "evidence sufficient")
        return RootCauses(
            n_candidates=len(post), candidates=ranked, component_fault_probability=fault,
            level_probability={c: {str(lv): p for lv, p in levels[c].items()} for c in COMPONENTS}, insufficient_evidence=insufficient,
            unseparated=unseparated, separating_channels=separating, evidence_ids=observation_ids(measurements),
            time_window=measurement_window(measurements),
            assumptions=[rule_text(released["rules"], len(candidates)), pool_text(self.models), A_CLASSIFIER, A_URN, A_REPLACED, A_TIME],
            uncertainty=Uncertainty(kind="posterior", value=top), message=message)

    @staticmethod
    def _bearing_ids(combo, measurements):
        """The released fields whose channel indicates a faulty component of the combination."""
        faulty = {c for c, v in zip(COMPONENTS, combo) if v != HEALTHY[c]}
        return [f"obs/{m.number}/{field}" for m in measurements for field in sorted(m.result)
                if faulty & set(kb.INDICATES.get(kb.sensor_of(field), ()))]

    def estimate_intervention_effect(self, action):
        try:
            released = self._released()
        except NoEpisode as e:
            return ErrorResult(error=str(e))
        action = action.strip()
        if action == "monitor":
            component = None
        elif action.startswith("replace:") and action[len("replace:"):] in COMPONENTS:
            component = action[len("replace:"):]
        else:
            return ErrorResult(error="action must be monitor or replace:<component> with the component one of cooler, valve, pump, accumulator")
        rules, status = released["rules"], released["status"]
        _candidates, belief, measurements = self._replay(released)
        post = belief.posterior()
        if component is None:
            addresses, after, price = 0.0, dict(post), 0.0
        else:
            j = COMPONENTS.index(component)
            addresses = sum(w for h, w in post.items() if h[j] != HEALTHY[component])
            after = {}
            for h, w in post.items():
                moved = h[:j] + (HEALTHY[component],) + h[j + 1:]
                after[moved] = after.get(moved, 0.0) + w
            price = rules["prices"]["parts"][component] + rules["prices"]["downtime"]
        recovery = after.get(HEALTHY_COMBO, 0.0)
        predicted = [self._predict(field, after) for field in sorted(self.models["effect_pool"])]
        predictions = self._predictions()
        prediction_id = f"pred/{len(predictions) + 1}"
        predictions.append({"id": prediction_id, "action": action, "component": component, "events": len(released["events"]),
                            "measurements": len(measurements), "addresses": addresses, "recovery": recovery,
                            "predicted": {p.field: [p.mean, p.band_low, p.band_high] for p in predicted}})
        write_json_atomically(self._predictions_path(), predictions, indent=1)
        notes = [A_EFFECT, A_REPLACED, A_NEW_BLOCK, A_PRICE, A_RECOVERY]
        if component is not None and status["phase"] == "investigate":
            notes.append("the environment accepts a repair only after the diagnosis has been submitted")
        if component is not None and status["repairs_left"] == 0:
            notes.append("no repair is left in this episode")
        if status["episode_over"]:
            notes.append("the episode is over, so the action can no longer be taken")
        return InterventionEffect(
            prediction_id=prediction_id, action=action, component=component, addresses_fault=addresses, recovery=recovery, price=price,
            predicted=predicted, evidence_ids=observation_ids(measurements) + [prediction_id],
            time_window=TimeWindow(start=len(measurements), end=len(measurements) + 1, unit="measurement"), assumptions=notes,
            uncertainty=Uncertainty(kind="recovery_probability", value=recovery),
            message=(f"{action}: addresses a fault with probability {addresses:.3f}, recovery {recovery:.3f}, price {price:g}; "
                     f"{len(predicted)} summaries of measurement {len(measurements) + 1} predicted as {prediction_id}"))

    def _predict(self, field, weights):
        parts = []
        for h, w in weights.items():
            entry = self.models["effect"].get(combo_key(h))
            if entry is not None:
                values = entry[field]
                parts.append((w, values[0], values[1:]))
            else:
                pool = self.models["effect_pool"][field]
                parts.append((w, sum(b * v for b, v in zip(pool["beta"], design(h))), pool["q"]))
        mean, low, high = mixture(parts, self.models["effect_grid"])
        return PredictedSummary(field=field, mean=mean, band_low=low, band_high=high)

    def verify_outcome(self, prediction_id):
        try:
            released = self._released()
        except NoEpisode as e:
            return ErrorResult(error=str(e))
        prediction = next((p for p in self._predictions() if p["id"] == prediction_id.strip()), None)
        if prediction is None:
            return ErrorResult(error=f"unknown prediction_id {prediction_id}; estimate_intervention_effect returns one such as pred/1")
        events = released["events"]
        first = prediction["events"]  # the index of the first event released after the prediction
        component = prediction["component"]
        if component is None:
            start, recovered, recovery_status = first, None, "not applicable: monitor repairs nothing"
            measured_before = sum(1 for e in events[:first] if e["event"] == "measure")
        else:
            action_index = next((i for i in range(first, len(events)) if events[i]["event"] == "replace" and events[i]["component"] == component), None)
            if action_index is None:
                refusals = [e for e in events[first:] if e["event"] == "refused" and isinstance(e.get("action"), dict)
                            and e["action"].get("type") == "replace" and e["action"].get("component") == component]
                status = f"action refused: {refusals[-1]['reason']}" if refusals else "not yet observable: the action has not been taken"
                return self._verification(prediction, status, None, [], None, "not observable: the action has not been taken", None)
            start = action_index + 1
            if start < len(events) or not released["status"]["episode_over"]:
                recovered, recovery_status = False, "not recovered: the episode continued after the repair"
            else:
                recovered, recovery_status = None, ("not observable: the episode ended with the repair, which a recovery, a used-up repair "
                                                    "limit and the step cap share")
            measured_before = sum(1 for e in events[:action_index] if e["event"] == "measure")
        observed_at = next((i for i in range(start, len(events)) if events[i]["event"] == "measure"), None)
        if observed_at is None:
            return self._verification(prediction, "not yet observable: no measurement since the action", None, [], recovered, recovery_status, None)
        # another replacement changes the combination whose recordings the measurement reads, so the prediction no longer applies
        others = [f"replace:{e['component']}" for e in events[first:observed_at] if e["event"] == "replace" and e["component"] != component]
        if others:
            return self._verification(prediction, f"not comparable: {', '.join(others)} came between the prediction and the measurement",
                                      None, [], recovered, recovery_status, None)
        observed = events[observed_at]
        number = measured_before + 1
        residuals = [Residual(field=field, predicted=mean, observed=observed["result"][field], residual=observed["result"][field] - mean,
                              inside_band=low <= observed["result"][field] <= high)
                     for field, (mean, low, high) in prediction["predicted"].items() if field in observed["result"]]
        consistent = outcome_consistent(sum(1 for r in residuals if not r.inside_band), len(residuals), recovered, prediction["recovery"])
        return self._verification(prediction, "verified", number, residuals, recovered, recovery_status, consistent)

    def _verification(self, prediction, status, number, residuals, recovered, recovery_status, consistent):
        inside = sum(1 for r in residuals if r.inside_band)
        tolerated = misses_tolerated(len(residuals)) if residuals else None
        p_value = binomial_tail(len(residuals), len(residuals) - inside) if residuals else None
        ids = [prediction["id"]] + ([f"obs/{number}/{r.field}" for r in residuals] if number else [])
        return OutcomeVerification(
            prediction_id=prediction["id"], action=prediction["action"], status=status, observation=number, residuals=residuals, n_inside=inside,
            n_outside=len(residuals) - inside, n_outside_tolerated=tolerated, p_value_outside=p_value,
            recovery_predicted=prediction["recovery"], recovered=recovered, recovery_status=recovery_status,
            consistent=consistent, evidence_ids=ids, time_window=TimeWindow(start=number or 0, end=number or 0, unit="measurement"),
            assumptions=[A_RECOVERY, A_EFFECT, A_CONSISTENT], uncertainty=Uncertainty(kind="band_coverage", value=BAND_COVERAGE),
            message=f"{prediction['id']} ({prediction['action']}): {status}; {inside} of {len(residuals)} observed summaries inside their band"
                    f"{f', at most {tolerated} outside tolerated' if residuals else ''}; recovery {recovery_status}")
