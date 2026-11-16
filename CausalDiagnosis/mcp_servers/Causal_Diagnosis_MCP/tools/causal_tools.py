import json
import time
from pathlib import Path
import numpy as np
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import STOPPED
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.verifier import COMMIT_EVENTS, RELEASED_FORMAT  # the released writes after which no advice is given
from CausalDiagnosis.mcp_servers.utils import transcript
from CausalDiagnosis.utils.atomic_writes_and_locks import locked, write_json_atomically
from CausalDiagnosis.utils.public_rules.repair_completion import verifies
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import LIVE_WINDOW, channel_field, profile_view, statistic_of
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.diagnostics_model import EffectDiagnostics, diagnostics_backend
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.troubleshooting_plans import (BRANCH_DETAILS, DEFAULT_BRANCH_DETAIL, SEPARATION, branch_previews, credible_set, moved_masses, plan_completion,
                       plan_terms)
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.conformal_intervals import DEFAULT_ALPHA, fits_for_level, most_probable, predict_summaries
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.decision_rule import MONTE_CARLO_SAMPLES, MONTE_CARLO_SEED, cost_unit, costs, decide, declare_above, expected_downtime_rate, unseparated
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.discriminant_urn_ranking import urn_probabilities
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.effect_model import combo_key, parse_combo
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.read_choice import (ACT_CALLS, BATCH_INNER_SAMPLES, BATCH_OUTCOMES, EC2_DRAWS, IG_EPSILON, ec2_cuts, eliminations, one_read, per_cost,
                          read_batch, read_units, reads_in_turn, record_read, rule_group)
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_filter import parse_released, shows
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tool_result_schemas import (Advice, BatchRead, Candidate, ChannelCheck, E10Rule, ChannelShare, ClosedWorkOrder, ComponentAttribution, Elimination, EpisodeEvidence,
                     ErrorResult, EvidenceItem, Fault, InterventionEffect, MonteCarlo, MoreCandidates, SetMass, OutcomeVerification, RootCauses, SeparatingTest, Stability,
                     TimeWindow, Uncertainty)
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_at_one_cycle import DEFAULT_RANKING_MODEL, RANKING_MODELS
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.served_recordings import left_in_episode
from . import tool_texts, episode_reading, rank_root_causes, intervention_effect


ABSTENTION_RULES = ("decision", "fixed_cut")  # E10, or the v1 posterior cut kept for the ablation of the stop rule
# the candidates a ranking returns when the caller names no number; a caller may ask for more, with no maximum
DEFAULT_TOP_K = 5
DEFAULT_ABSTENTION_CUT = 0.95  # the v1 cut, used only under abstention_rule fixed_cut and named in the result


def verifying_read_price(view, components, events):
    """The price of one cycle of the cheapest sensor group whose reading verifies a repair of the components
    (public_rules.repair_completion.verifies) under the indicating channels the released events show the agent was told
    (ProfileView.told_indicating_channels), the read the agent's plan check keeps beside the repair before it buys a read, so
    that the reserve of a batch is the agent's; nothing for a plan without a component and nothing on a row whose rules price
    nothing, where the read is a call that plan_completion counts already."""
    prices = view.rules["prices"]
    if not components or cost_unit(prices) == "calls":
        return 0.0
    indicating = view.told_indicating_channels(events)
    verifying = [g for g, channels in view.sensor_groups.items() if verifies(channels, list(components), indicating, view.sensor_groups)]
    # the agent's Board.verification_group falls back on the cheapest group where none verifies
    return float(min((prices["read"][g] for g in verifying or view.sensor_groups), default=0.0))


def repaired_components(view, events, new):
    """The components whose repair the next verifying reading judges, as the agent's board counts them (Board.repaired): the
    new ones, those of the agent's open work orders, and those of every prediction whose citing orders have all closed and
    for which no reading at or after the last close verifies them yet (Board.pending_verifications with the reading of
    CausalLoopRunner._verifying_reading), all read from the released events."""
    created = {e["id"]: e for e in events if e.get("event") == "work_order_created" and e.get("id")}
    closed = {e["id"]: int(e["t_end"]) for e in events if e.get("event") == "work_order_closed" and e.get("id") in created}
    repaired = set(new) | {o["component"] for i, o in created.items() if i not in closed}
    indicating = view.told_indicating_channels(events)
    reads = [(e["sensor_group"], max(int(o["t"]) for o in e["served"])) for e in events if e.get("event") == "read" and e.get("served")]
    by_prediction = {}
    for order_id, order in created.items():
        if order.get("prediction_id"):
            by_prediction.setdefault(order["prediction_id"], []).append(order_id)
    for ids in by_prediction.values():
        if any(i not in closed for i in ids):
            continue
        components = sorted({created[i]["component"] for i in ids})
        last = max(closed[i] for i in ids)
        if not any(t >= last and verifies(view.sensor_groups.get(g, ()), components, indicating, view.sensor_groups) for g, t in reads):
            repaired |= set(components)
    return sorted(repaired)


class CausalTools:
    """The four tools over the fits of one episode (Models.without). With released_path None the server serves no episode, and
    with models None it was launched without the block its fits leave out; every tool then answers with an error."""

    def __init__(self, models, released_path=None, ranking_model=DEFAULT_RANKING_MODEL, abstention_rule="decision", abstention_cut=DEFAULT_ABSTENTION_CUT,
                 alpha=DEFAULT_ALPHA, samples=MONTE_CARLO_SAMPLES, seed=MONTE_CARLO_SEED, e10_margin=0.0, batch_cap=None, reserve_plans=1):
        if ranking_model not in RANKING_MODELS:
            raise ValueError(f"ranking_model must be one of {RANKING_MODELS}")
        if abstention_rule not in ABSTENTION_RULES:
            raise ValueError(f"abstention_rule must be one of {ABSTENTION_RULES}")
        if not 0 < float(abstention_cut) <= 1:
            raise ValueError("abstention_cut must be a probability in (0, 1]")
        self.models = models
        self.model = diagnostics_backend(EffectDiagnostics(models, block=ranking_model != "joint"))
        self.released_path = None if released_path is None else Path(released_path)
        self.ranking_model = ranking_model
        self.abstention_rule = abstention_rule
        self.abstention_cut = float(abstention_cut)
        self.alpha = alpha
        self.samples = samples
        self.seed = seed
        # the server fields of a decision threshold card: launch options that launcher.py casts from key=value arguments, absent
        # on every row, so that each keeps the default above
        self.e10_margin, self.batch_cap, self.reserve_plans = e10_margin, batch_cap, reserve_plans

    # ----- the episode and the belief -----
    def _released(self):
        if self.released_path is None:
            raise episode_reading.NoEpisode("this server serves no episode; start one through the action server of the environment")
        if not self.released_path.exists():
            raise episode_reading.NoEpisode("no released evidence yet; the episode has not been started")
        if self.models is None:
            raise episode_reading.NoEpisode("this server was launched without the block its fits leave out, which the driver names at launch")
        data = self.released_path.read_bytes()
        transcript.note(released_sha256=transcript.bytes_sha256(data))  # the evidence this call read, for the line of the call
        raw = json.loads(data.decode("utf-8"))
        if raw.get("format") != RELEASED_FORMAT:
            raise episode_reading.NoEpisode(f"the released evidence has format {raw.get('format')}; this server reads format {RELEASED_FORMAT}")
        # released.json names its asset beside the rules (write_released), so the view is the profile of that asset, the
        # default asset when the file names none
        rules = dict(raw["rules"], asset_id=raw["asset_id"]) if "asset_id" in raw else raw["rules"]
        return episode_reading.Released(raw, profile_view(rules), parse_released(raw), raw["status"], raw["events"])

    def _belief(self, released):
        """The filter over the released evidence and the belief now, the masses replaced by the urn model's under
        discriminant_urn."""
        flt = self.model.replay(released.view, released.episode)
        post = flt.posterior()
        if self.ranking_model == "discriminant_urn":
            masses = urn_probabilities(self.models.urn, self.models.space, released.episode.readings, released.episode.closures)
            post.override([masses.get(h, 0.0) for h in post.candidates])
        return flt, post

    def _model_assumptions(self, view):
        """The six notes of every result, one text for every start, so that no note tells the block the fits leave out."""
        notes = [tool_texts.A_RULE, tool_texts.A_E1,
                 (f"E2: the belief covers every combination of the profile ({len(view.combinations)} here) with a prior uniform over them "
                  "times the likelihood of the released fix history under the public history rule once the agent has asked for it; for "
                  "every cycle the clock ran, the predict step applies the rule cards in the clock's order (the closures released for the "
                  "cycle, the expiry of emergency holds, the scheduled or stochastic degradation), a repair "
                  "taking only when every component it depends on is healthy, and every reading updates the belief by the likelihood of "
                  "the effect model where the rig served it, a reading of a read of zero duration and a closure settled while the clock "
                  "stood at a cycle before the rule cards of that cycle, in the order they were released; the server never reads a seed "
                  "or the split, and no result names the start"),
                 tool_texts.sources_text(view), tool_texts.A_DEDUPE]
        notes.append({"joint_block": tool_texts.A_BLOCK, "joint": tool_texts.A_JOINT, "discriminant_urn": tool_texts.A_URN}[self.ranking_model])
        return notes

    def _predictions_path(self):
        return self.released_path.with_name(episode_reading.PREDICTIONS_NAME)

    def _predictions(self):
        path = self._predictions_path()
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

    @staticmethod
    def _window(released, readings=None):
        readings = released.episode.readings if readings is None else readings
        start = min((r.t for r in readings), default=released.episode.time)
        return TimeWindow(start=start, end=released.episode.time, unit=episode_reading.UNIT)

    # ----- get_episode_evidence -----
    def get_episode_evidence(self, sensor_group):
        try:
            released = self._released()
        except episode_reading.NoEpisode as e:
            return ErrorResult(error=str(e))
        view = released.view
        group = sensor_group.strip()
        if group == "all":
            channels = None
        elif group in view.sensor_groups:
            channels = set(view.sensor_groups[group])
        else:
            sensor = view.sensor_of(group)
            if sensor is None:
                example = next(view.sensor_of(f) for g in sorted(view.sensor_groups) for f in view.group_fields(g))
                return ErrorResult(error=f"unknown sensor_group {group}: use all, {', '.join(sorted(view.sensor_groups))}, a channel such as {example} "
                                         f"or a field such as {view.channel_name(example)}")
            channels = {view.channel_name(sensor)}
        items = []
        for reading in released.episode.readings:
            anomaly = self.models.effect.anomaly_scores(reading.values)
            for field, value in sorted(reading.values.items()):
                if channels is not None and channel_field(field) not in channels:
                    continue
                sensor = view.sensor_of(field)
                unit, rate = view.unit_and_rate(sensor)
                band = view.band(field)
                items.append(EvidenceItem(evidence_id=f"obs/{reading.n}/{field}", field=field, channel=sensor, statistic=statistic_of(field),
                                          value=value, unit=unit, rate=rate, n=reading.n, t=reading.t, sensor_group=reading.group,
                                          normal_low=None if band is None else band["low"], normal_high=None if band is None else band["high"],
                                          in_normal_range=None if band is None else bool(band["low"] <= value <= band["high"]),
                                          anomaly_score=anomaly.get(field)))
        closures = released.episode.closures
        closed = [ClosedWorkOrder(id=c.id, component=c.component, source=c.source, t=c.t) for c in closures]
        outside = sum(1 for i in items if i.in_normal_range is False)
        scored = [i.anomaly_score for i in items if i.anomaly_score is not None]
        return EpisodeEvidence(
            sensor_group=group, n_observations=len(released.episode.readings), items=items, work_orders_closed=closed,
            evidence_ids=[i.evidence_id for i in items] + episode_reading.order_ids(closures), time_window=self._window(released),
            assumptions=[tool_texts.A_BANDS, tool_texts.A_E3, tool_texts.A_TIME, tool_texts.A_SIX_DIGITS],
            uncertainty=Uncertainty(kind="largest_anomaly_score", value=max(scored) if scored else 0.0),
            message=(f"{len(items)} items from {len(released.episode.readings)} observation(s); {outside} channel means outside their public "
                     f"normal range; {len(closed)} work order(s) closed"))

    # ----- rank_root_causes -----
    def rank_root_causes(self, top_k=DEFAULT_TOP_K, branch_detail=DEFAULT_BRANCH_DETAIL):
        started = time.perf_counter()
        result = self._rank_root_causes(top_k, branch_detail)
        rank_root_causes.TIMING.info("rank_root_causes answered in %.2f s", time.perf_counter() - started)
        return result

    @staticmethod
    def _set_masses(view, post, positive):
        """One row per faulty set with positive mass, summed over every combination of the set in the belief, by
        mass and then by the sorted names, so the set a reader judges a reading on is the same whatever top_k is."""
        mass = {}
        for k in positive:
            key = tuple(sorted(view.faulty(post.candidates[k])))
            mass[key] = mass.get(key, 0.0) + float(post.probabilities[k])
        return [SetMass(faulty_set=list(s), mass=m) for s, m in sorted(mass.items(), key=lambda item: (-item[1], item[0]))]

    def _rank_root_causes(self, top_k, branch_detail):
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            return ErrorResult(error="top_k must be a positive integer")
        # an empty string is the default, so a caller that fills the argument with nothing still gets the compact branches
        detail = branch_detail.strip() or DEFAULT_BRANCH_DETAIL if isinstance(branch_detail, str) else None
        if detail not in BRANCH_DETAILS:
            return ErrorResult(error=f"branch_detail must be one of {', '.join(BRANCH_DETAILS)}")
        try:
            released = self._released()
        except episode_reading.NoEpisode as e:
            return ErrorResult(error=str(e))
        view, episode = released.view, released.episode
        flt, post = self._belief(released)
        # the urn model has no transition model, so a reading is predicted from the belief now
        ahead = post if self.ranking_model == "discriminant_urn" else flt.next_reading().posterior()
        cost = costs(view, episode.time, released.status.get("steps_left"))
        serving = left_in_episode(view, episode, LIVE_WINDOW)  # what the row can still serve to a reading of each group, on every row
        unpriced = cost.unit == "calls"
        # decide gives E8 to E10 with the margin of the card and the count of every row, so a group the row can serve nothing more
        # draws nothing on both cost units; its flag and its test are those of the block e10_rule of the long forms
        decision = decide(post, cost, self.samples, self.seed, ahead=ahead, margin=self.e10_margin, serving=serving)
        order = post.order()
        current, levels = post.current_marginals(), post.level_probability()
        read_first, best, separating, abstention_text, read = self._rule_abstention(view, post, cost, decision, order, levels, serving, released)
        compact = detail == "compact"  # the default reply: what an agent that follows the advice reads, and nothing it does not
        attribution = None if compact else post.attribution(order[0])[0]
        stability = post.stability()
        # the first top_k candidates with positive mass, by mass and then by the combination in component order (post.order); the
        # rest is counted in more_candidates, and the advice and the reserve below still read every candidate
        positive = [k for k in order if post.probabilities[k] > 0]
        returned = positive[:top_k]
        more = MoreCandidates(count=len(positive) - len(returned), mass=float(sum(post.probabilities[k] for k in positive[top_k:])))
        set_masses = self._set_masses(view, post, positive)
        ranked = [Candidate(rank=i + 1, combination=dict(zip(view.components, post.candidates[k])),
                            current_combination=dict(zip(view.components, post.candidates[k])),
                            faults=[Fault(component=c, level=v, code=view.level_code(c, v)) for c, v in zip(view.components, post.candidates[k]) if v != view.healthy[c]],
                            probability=float(post.probabilities[k]), evidence_ids=self._bearing_ids(view, post.candidates[k], episode, newest=compact))
                  for i, k in enumerate(returned)]
        rate = expected_downtime_rate(post.candidates, post.probabilities, cost)
        test_cycles = read[0][1] if read else 1  # the one read takes every cycle of the group the rule names
        test = SeparatingTest(sensor_group=best, cycles=test_cycles, price=cost.test_price[best], duration=cost.test_duration[best] * test_cycles,
                              cost=decision.test_cost[best], expected_loss=decision.expected_loss_test[best]) if read_first else None
        top = post.candidates[order[0]]
        if self.abstention_rule == "fixed_cut":
            verdict_text = (f"read first under the fixed cut {self.abstention_cut:g}: {best} for {tool_texts.cycles_text(test_cycles)}" if read else
                            f"read first under the fixed cut {self.abstention_cut:g}: no reading is left" if read_first else
                            f"no reading first under the fixed cut {self.abstention_cut:g}")
        elif unpriced:
            verdict_text = (f"read first: {best} for {tool_texts.cycles_text(test_cycles)}, the recorded cycles that no reading of {best} has shown" if read_first
                            else "a reading is left, but its call does not fit the calls left beside the diagnosis, the orders and the finish"
                            if any(left.cycles for left in serving.values()) else "nothing left to read: the row can serve no reading that shows a new summary")
        else:
            verdict_text = (f"read first: {best} for {tool_texts.cycles_text(test_cycles)}, the recorded cycles that no reading of {best} has shown, which the "
                            f"budget pays beside the repair of every component" if read_first else
                            "nothing left to read that the budget pays beside the repair of every component")
        predicted = {}

        def predicted_for(components):
            key = tuple(components)
            if key not in predicted:
                predicted[key] = self._predicted_action(view, flt, post, list(components))
            return predicted[key]

        def register(components, rank):
            action = intervention_effect.REPLACE + ",".join(components)
            stored = self._stored_branch(action, released)
            return stored if stored is not None else self._store_prediction(released, action, components, predicted_for(components), branch=rank)

        branches = branch_previews(view, post, ahead, cost, decision, self.samples, self.seed, register,
                                   lambda components, rank: predicted_for(components), self.alpha, detail, serving=serving,
                                   level_reachable=self.models.calibration.level_reachable(self.alpha), returned=returned)
        scores = eliminations(ahead, decision)
        units, unit = read_units(decision, cost)
        ee_rate = per_cost({g: s.elimination for g, s in scores.items()}, units)
        ig_rate = per_cost({g: s.information_gain for g, s in scores.items()}, units)
        if compact:  # the edge cutting draws readings under every combination with mass, and compact reports none
            cuts = ec2_rate = {g: None for g in scores}
        else:
            cuts = ec2_cuts(ahead, cost, self.seed)
            cuts = {g: cut if serving[g].cycles > 0 else 0.0 for g, cut in cuts.items()}  # a reading the row cannot serve cuts nothing
            ec2_rate = per_cost(cuts, units)
        elimination = [Elimination(sensor_group=g, cycles=1, price=cost.test_price[g], duration=cost.test_duration[g], cost=units[g], cost_unit=unit,
                                   effective_before=scores[g].effective_before, expected_effective_after=scores[g].effective_after,
                                   expected_elimination=scores[g].elimination, elimination_per_cost=ee_rate[g], standard_error=scores[g].standard_error,
                                   information_gain=scores[g].information_gain, ig_per_cost=ig_rate[g], ec2=cuts[g], ec2_per_cost=ec2_rate[g])
                       for g in sorted(scores)]
        batch = self._reads_of_rule(view, post, order, cost, released, serving, read, decision.expected_loss_test)  # the reads of the rule in order

        def batch_read(group, cycles):
            return BatchRead(sensor_group=group, cycles=cycles, price=cost.test_price[group], cost=units[group], cost_unit=unit,
                             elimination_per_cost=ee_rate[group], information_gain=scores[group].information_gain, ig_per_cost=ig_rate[group])

        batch_reads = [batch_read(g, n) for g, n in batch]
        advice = self._advice(view, released, post, order, read_first, best, batch)
        # the long forms add the earlier rule, the one part of a reply that runs the search of E22
        e10_rule = None if detail == "compact" else self._e10_rule(view, post, ahead, cost, decision, order, released, batch_read)
        if unpriced:
            e22_text = (f"E22: read_batch holds the reads of the rule in order, the first through the group that can still show the most "
                        f"summaries over every recorded cycle it can still be shown and each later one through the group that still shows a "
                        f"summary no read before it holds, one read where the sensor groups are nested, empty when the first call does not fit the calls left after the calls of the top "
                        f"candidate's repair plan and the {ACT_CALLS} calls of the diagnosis and the finish, or its cycles do not leave that "
                        f"plan's cycles before the horizon")
        else:
            e22_text = ("E22: read_batch holds the reads of the record that the rule names in order, each through the sensor group that can "
                        "still show the most summaries not yet shown after the reads before it and that the budget pays beside the repair of "
                        "every component, one read where the sensor groups are nested, for the cycles the row "
                        "can still serve it and the money left pays, empty when no such read is left or its call or its cycles do not fit "
                        "beside that repair")
        if e10_rule is not None:
            e22_text += ", and e10_rule.read_batch is the batch of the earlier rule, which the text of e10_rule states"
        branch_text = (f"candidates: the first {top_k} combinations with positive mass (top_k), by mass and then by the combination in "
                       f"component order; more_candidates counts the combinations with positive mass left out and sums their mass. "
                       f"branches: one per distinct faulty set among the returned candidates, each walked forward once and listing in ranks the "
                       f"ranks of its returned candidates; its probability is the mass of every combination of the set, those not returned included, and its confirm test is the sensor "
                       f"group whose one cycle reading moves the set's mass apart the most between readings drawn under its candidates, "
                       f"weighted by their mass, and readings drawn under the mixture of the others ({self.samples} draws per group and set, seed {self.seed} plus "
                       f"the set's rank), with the two step pair reported when its gap beats the best single gap by more than {SEPARATION} standard "
                       f"errors; the repair plan creates one work order per faulty component, one after another, so one technician visit repairs "
                       f"them back to back in priority order after the longest lead time, priced from the public parts, repair charge and one "
                       f"visit charge, with the waiting cost at the expected downtime rate {rate:.3g} until the technician arrives and the "
                       f"stopped rate over the repair downtimes, the cycles to restored the last close plus one cycle for the verifying read, "
                       f"and every component a step depends on listed with whether the plan repairs it; a prediction is registered per step "
                       f"and, for a plan of several steps, one after all of them (plan_prediction_id) that every work order of the plan cites; "
                       f"the predicted outcome runs the predict step through the plan's closures (E13) and predicts the summaries with the "
                       f"conformal intervals of E14 at alpha {self.alpha:g}; expected_loss_if_acted is E8 of acting on the set's most probable candidate, the one a diagnosis "
                       f"submits, its faulty set repaired and that candidate diagnosed, and "
                       f"expected_loss_if_tested is E9 after the confirm test with its cost; act_now is the plan's price and downtime plus "
                       f"the expected loss beyond the outlay of the plan for the same candidate, the faults it leaves and a wrong diagnosis, since the outlay holds "
                       f"every replacement of the plan, and test_first is the cost of the confirm test plus, over its drawn readings, the "
                       f"price and downtime of the plan of the combination first after the reading and that reading's expected loss beyond "
                       f"the outlay; both are decision aids from the public prices, durations, downtime rates and the belief, not the cost "
                       f"the environment charges; branch_detail summary omits predicted_outcome.predicted and "
                       f"confirm_test.gap_by_group and full returns them")
        elimination_text = (f"E21: the effective number of hypotheses N(b) = exp(H(b)) is {post.effective_hypotheses():.3g} now; the information "
                            f"gain of one cycle of a group is the entropy of b_a minus the expected entropy of b_a updated with the reading, in "
                            f"nats, from the draws of E9, with b_a the belief the reading meets, and ig_per_cost, an informed heuristic without an "
                            f"approximation guarantee that ranks the reads, divides it by {tool_texts.UNIT_TEXT[unit]}; the expected elimination, N(b_a) "
                            f"minus the expected N after the reading, and its rate are reported beside the gain and turn negative for a read "
                            f"that tells a concentrated belief's top combination from the others; {e22_text}; "
                            f"ec2 is one step edge cutting over repair plan classes, after the EC2 objective of Golovin, Krause and Ray, "
                            f"not their algorithm: the expected weight of the edges between the plans of the repair loss of E8 that one "
                            f"reading of a group cuts, over {EC2_DRAWS} readings drawn under every combination with positive mass (seed "
                            f"{self.seed}), the outcome of a reading the plan of least expected repair loss after it and the class of a "
                            f"drawn hypothesis that plan given its outcomes on every group, ec2_per_cost divides it by the same cost, and it "
                            f"is reported for an agent that ranks reads by it and ranks nothing here")
        unsep_text = f"; unseparated: {', '.join(separating)}" if separating else ""
        if cost.unit == "calls":
            left = ", ".join(f"{g} {serving[g].cycles}" for g in sorted(serving))
            terms = (f"E7 to E10 in calls, since the rules price nothing on this row: a test costs one call, a repair plan one call per order, "
                     f"and no downtime is charged. A diagnosis off the truth in any level, a work order on a healthy component and a fault "
                     f"left unrepaired each void the pass, so acting on a combination that is not the true one costs the {cost.remaining} "
                     f"calls left before the step cap, what a failed episode costs, once, and the best act is the most probable combination. "
                     f"The tests are the readings the row can still serve, for every sensor group the recorded cycles that no reading has "
                     f"shown it ({left}), and a group with none has gain zero")
        else:
            terms = (f"E7 to E10: an unnecessary replacement costs the part plus the repair charge plus the stopped downtime rate over the repair "
                     f"downtime ({', '.join(f'{c} {v:g}' for c, v in cost.unnecessary.items())}); a missed fault costs its downtime rate over "
                     f"the remaining horizon of {cost.remaining} cycles; a test costs its read price for one cycle, and one cycle of the "
                     f"expected downtime rate {rate:.3g} besides only where the read plays its cycle, and a wrong diagnosis, a combination "
                     f"off the truth in any level, costs "
                     f"{cost.misdiagnosis:g}, the price prices.misdiagnosis of the rules, zero when they do not carry it")
        if self.abstention_rule == "fixed_cut":
            cost_text = (f"{terms}; the expected losses of E8 and E9 follow from these costs and the belief through the expectation of E9 over "
                         f"the next reading with {decision.samples} Monte Carlo draws per test (seed {decision.seed}), and the flag is the fixed "
                         f"cut of the abstention text")
        elif unpriced:
            cost_text = (f"{terms}; the flag is true while the row can still serve a reading and its one read fits the calls left, since on "
                         f"this row a reading lowers no outcome that the verifier scores, and the separating test is the reading that releases "
                         f"the most summaries not yet shown. The expected losses of E8 and E9 are reported beside it with {decision.samples} "
                         f"Monte Carlo draws per test (seed {decision.seed})")
        else:
            cost_text = (f"{terms}; the flag follows the rule of a row with prices: one read of the record through the richest sensor group "
                         f"that the budget pays beside the repair of every component, then the diagnosis of the first candidate; the expected "
                         f"losses of E8 and E9 are reported beside it with {decision.samples} Monte Carlo draws per test (seed {decision.seed})")
        state_text = (f"the filter carried {post.n_states} augmented states at cycle {episode.time} and at most {flt.peak_states} during the "
                      f"replay, with {flt.n_resampled} resampled fits for E6, every registered one, so the work of a call is fixed by the "
                      f"evidence and every replay answers alike")
        result = RootCauses(
            ranking_model=self.ranking_model, n_candidates=len(post.candidates), n_states=post.n_states, resampled_fits=flt.n_resampled,
            belief_time=episode.time, effective_hypotheses=post.effective_hypotheses(), candidates=ranked, more_candidates=more, set_masses=set_masses, current_fault_probability=current,
            level_probability=None if compact else {c: {str(lv): p for lv, p in levels[c].items()} for c in view.components},
            attribution=None if compact else [ComponentAttribution(component=c, current_fault_probability=current[c],
                                                                   explained_anomaly=attribution[c]["explained_anomaly"],
                                                                   channels=[ChannelShare(**item) for item in attribution[c]["channels"]])
                                              for c in view.components],
            stability=None if stability is None or compact else Stability(**stability), read_first=read_first,
            separating_test=test, expected_saving=decision.expected_saving, expected_loss_act=decision.expected_loss_act,
            expected_loss_test=decision.expected_loss_test, unseparated=sorted(separating, key=view.components.index),
            separating_channels={c: separating[c] for c in sorted(separating, key=view.components.index)},
            elimination=elimination, read_batch=batch_reads, left_to_read={g: serving[g].cycles for g in sorted(serving)}, advice=advice, e10_rule=e10_rule,
            monte_carlo=None if compact else MonteCarlo(samples=decision.samples, seed=decision.seed, standard_error=decision.standard_error),
            costs={"unit": cost.unit, "unnecessary_replacement": cost.unnecessary, "remaining_horizon": cost.remaining, "test_cost": decision.test_cost,
                   "expected_downtime_rate": rate},
            branches=branches,
            evidence_ids=episode_reading.observation_ids(episode.readings) + episode_reading.order_ids(episode.closures), time_window=self._window(released),
            assumptions=[tool_texts.A_COMPACT] if compact else self._model_assumptions(view) + [state_text, tool_texts.A_PREDICT_STEP, tool_texts.A_E3, tool_texts.A_E4, tool_texts.A_E6, cost_text,
                                                                                    abstention_text, elimination_text, branch_text, tool_texts.A_TIME],
            uncertainty=Uncertainty(kind="top_marginal_resampling_interval", value=stability["top_marginal"] if stability else float(max(current.values())),
                                    detail=None if stability is None else {"low": stability["interval_low"], "high": stability["interval_high"],
                                                                           "top_share": stability["top_share"]}),
            message=(f"{len(post.candidates)} combinations, effective number {post.effective_hypotheses():.3g} at cycle {episode.time}; top "
                     f"{tool_texts.combo_text(view, top)} at {post.probabilities[order[0]]:.3f}; fault probabilities "
                     + ", ".join(f"{c} {current[c]:.3f}" for c in view.components) + f"; {verdict_text}{unsep_text}"
                     + (f"; read batch {', '.join(g for g, _n in batch)}" if batch else "")
                     + (f"; advice: {advice.text}" if advice is not None else "")))
        if compact:
            result._left_out = rank_root_causes.COMPACT_LEFT_OUT
        return result

    def _reserve_terms(self, view, post, ranked, events):
        """The price, the calls (plan_completion) and the cycles to a restored rig kept for the repairs before a read is bought, the repair plan of the top
        combination and, under reserve_plans 2, also the cheapest repair plan of another combination of the credible set, the plan
        backtracking turns to. The price holds the read that verifies each repair as the agent's plan check prices it from what
        the released events show it was told (verifying_read_price)."""
        top = tuple(view.faulty(post.candidates[ranked[0]]))
        price, cycles = plan_terms(view, list(top))
        price += verifying_read_price(view, repaired_components(view, events, top), events)
        calls = plan_completion(view, list(top)).calls
        if self.reserve_plans < 2:
            return price, calls, cycles
        others = [(plan_terms(view, list(s))[0] + verifying_read_price(view, repaired_components(view, events, s), events), plan_terms(view, list(s))[1],
                   plan_completion(view, list(s), commit=False).calls - 1)  # its finish is the top plan's
                  for s in {tuple(view.faulty(post.candidates[k])) for k in ranked[1:]} - {(), top}]
        if not others:
            return price, calls, cycles
        extra_price, extra_cycles, extra_calls = min(others)
        return price + extra_price, calls + extra_calls, cycles + extra_cycles

    def _e10_rule(self, view, post, ahead, cost, decision, order, released, batch_read):
        """The block e10_rule of the long forms, the earlier rule that the arms of the agent's ablations read: read_first is E10
        with the e10_margin of the decision threshold card, the separating test its test for one cycle, and read_batch the batch
        of E22 with the batch_cap and the reserve of reserve_plans of the card, as the base computed them. These three fields of
        the card act on this block alone."""
        best = decision.separating_test
        read_first = bool(decision.read_first)
        test = SeparatingTest(sensor_group=best, cycles=1, price=cost.test_price[best], duration=cost.test_duration[best], cost=decision.test_cost[best],
                              expected_loss=decision.expected_loss_test[best]) if read_first else None
        batch = []
        if read_first:
            reserve_price, reserve_calls, reserve_cycles = self._reserve_terms(view, post, credible_set(post.probabilities, order, self.alpha), released.events)
            budget_left, steps_left = released.status.get("budget_left"), released.status.get("steps_left")
            if budget_left is None:  # the free base row has no budget, and the calls left before the step cap bound the batch
                budget_left = cost.remaining if cost.unit == "calls" else float("inf")
            batch = read_batch(ahead, cost, decision, budget_left, view.rules["horizon"] - released.episode.time, reserve_price, reserve_cycles,
                               self.seed, cap=self.batch_cap, reserve_calls=reserve_calls,
                               calls_left=view.rules["max_steps"] if steps_left is None else steps_left)
        text = (f"e10_rule, the earlier rule: read_first is true when the best test lowers the expected loss below acting now by more than "
                f"{self.e10_margin:g} times its cost, the e10_margin of the decision threshold card, over the groups the row can still "
                f"serve, and the separating test is that test for one cycle; read_batch starts with the affordable group of largest "
                f"ig_per_cost above {IG_EPSILON:g} and takes a further group only when it has the largest ig_per_cost, above "
                f"{IG_EPSILON:g}, under each of {BATCH_OUTCOMES} simulated outcomes of the reads before it ({BATCH_INNER_SAMPLES} draws "
                f"per group each), so that it ends where those outcomes name different groups or no group gains, "
                f"{'' if self.batch_cap is None else f'at most {self.batch_cap} reads, '}while the read prices fit the "
                f"budget left, or on the free base row the "
                f"calls left, after the price of the top branch's repair plan "
                f"{'' if self.reserve_plans < 2 else 'and of the cheapest plan of another combination of the credible set '}"
                f"and the reads leave that plan's cycles before the horizon; the belief is held fixed between the reads of a batch")
        return E10Rule(read_first=read_first, margin=float(self.e10_margin), separating_test=test, read_batch=[batch_read(g, 1) for g in batch],
                       batch_cap=None if self.batch_cap is None else int(self.batch_cap), reserve_plans=int(self.reserve_plans), text=text)

    def _rule_read(self, view, post, order, cost, released, serving, group, spent=(0.0, 0, 0)):
        """The one read of the rule: on a row whose rules price nothing the group over every cycle the row can still serve it, when
        its call fits the calls left beside the repair plan of the top candidate, the diagnosis and the finish and its cycles leave
        that plan's cycles before the horizon (a top candidate with no fault keeps the calls of the act alone); on a row with
        prices the read of record_read, the group that can still show the most summaries and that the budget pays beside the
        repair of every component, for the cycles the row serves and the money pays. group names the read of a row that prices
        nothing and is not read on a row with prices. spent, (money, calls, cycles), is what the reads of the rule before this
        one spend (_reads_of_rule). [] when no such read is left."""
        cycles_left = view.rules["horizon"] - released.episode.time - spent[2]
        if cost.unit == "calls":
            if group is None or serving[group].cycles == 0:
                return []
            top = list(view.faulty(post.candidates[order[0]]))
            reserve = plan_completion(view, top)
            return one_read(group, serving[group].cycles, cost, cost.remaining - spent[1], cycles_left, reserve.calls, plan_terms(view, top)[1])
        reserve_price, reserve_cycles = plan_terms(view, list(view.components))
        budget_left, steps_left = released.status.get("budget_left"), released.status.get("steps_left")
        return record_read(serving, cost, (float("inf") if budget_left is None else budget_left) - spent[0],
                           (reserve_price, plan_completion(view, list(view.components)).calls, reserve_cycles), (view.rules["max_steps"] if steps_left is None else steps_left) - spent[1],
                           cycles_left)

    def _reads_of_rule(self, view, post, order, cost, released, serving, read, losses):
        """read_batch, the reads that the rule names one after another (elimination.reads_in_turn): read, the read of the rule,
        first, then each read that the rule names on what the reads before it leave the row to serve and to spend, the group of a
        row that prices nothing by rule_group over the expected losses. On nested groups the first read leaves every group closed,
        so the batch is that read. Under the fixed cut of the ablation of the stop rule the batch is the one read."""
        if not read or self.abstention_rule != "decision":
            return read

        def first(left, spent):
            if not any(spent):
                return read
            group = rule_group(left, losses) if cost.unit == "calls" else None
            return self._rule_read(view, post, order, cost, released, left, group, spent)

        def spend(one):
            group, cycles = one
            return 0.0 if cost.unit == "calls" else cost.test_price[group] * cycles, 1, cycles * cost.test_duration[group]

        return reads_in_turn(first, view, serving, spend)

    def _rule_abstention(self, view, post, cost, decision, order, levels, serving, released):
        """The flag, the separating test, the unseparated components, the text and the one read of the rule of the server. Under
        the decision rule the flag is true exactly when the rule names its read (_rule_read), so that the flag, the test, the
        batch and the advice say one thing, and the unseparated components are E12. Under the fixed cut of the ablation of the
        stop rule the flag is true when the top mass lies below the cut, the read of the rule is taken while it is, and when no
        read is left the test is the cheapest sensor group the row can still serve, which a declaration names; a component is
        unseparated when its most probable level lies below the cut. Returns (flag, test, unseparated, text, [(group, cycles)])."""
        if self.abstention_rule == "decision":
            separating = unseparated(post, cost, decision)
            group = rule_group(serving, decision.expected_loss_test) if cost.unit == "calls" else None
            read = self._rule_read(view, post, order, cost, released, serving, group)
            if cost.unit == "calls":
                text = ("abstention_rule decision: on this row the flag is true while the row can still serve a reading and its one read "
                        "fits the calls left, the separating test is the reading that releases the most summaries not yet shown, and the "
                        "unseparated components are E12")
            else:
                text = ("abstention_rule decision: on this row the flag is true while the rule names one read of the record that the budget "
                        "pays beside the repair of every component, the separating test is that read, and the unseparated components are E12")
            return bool(read), read[0][0] if read else decision.separating_test, separating, text, read
        served = [g for g in cost.test_price if serving[g].cycles > 0]
        cheapest = min(served or cost.test_price, key=lambda g: (cost.test_price[g], -serving[g].summaries, g))
        read_first = bool(post.probabilities[order[0]] < self.abstention_cut)
        read = self._rule_read(view, post, order, cost, released, serving, cheapest) if read_first else []
        separating = {c: list(view.group_fields(cheapest)) for c in view.components if max(levels[c].values()) < self.abstention_cut}
        text = (f"abstention_rule fixed_cut with cut {self.abstention_cut:g}: read_first is true when the top mass is below the cut, "
                f"the one read of the rule is taken while it is, and when no read is left the separating test is the cheapest sensor group "
                f"the row can still serve ({cheapest}); a component is unseparated when its most probable level is below the cut, as the v1 "
                f"server did; the expected losses of E8 to E10 are reported beside it and decide nothing here")
        return read_first, read[0][0] if read else cheapest, separating, text, read

    def _advice(self, view, released, post, order, read_first, best, batch):
        """The step that the scoring of the episode favors before a diagnosis, by the rule of the server: the one read of the
        batch while there is one, else the diagnosis of the most probable combination with one work order per faulty component,
        and no work order when it has no fault. Under the fixed cut of the ablation a declaration stands where the top mass lies
        below the cut and no read is left. Every advice carries, per measure, the chance of a start that both separability
        labels mark inseparable above which a declaration would score more than the diagnosis, None where it never does. None
        once a diagnosis or a declaration is among the released events."""
        if any(event.get("event") in COMMIT_EVENTS for event in released.events):
            return None
        top = post.candidates[order[0]]
        above = declare_above(float(post.probabilities[order[0]]))
        if batch:
            group, cycles = batch[0]
            reads = ", then ".join(f"{g} for {tool_texts.cycles_text(n)}" for g, n in batch)
            return Advice(action="read", sensor_group=group, cycles=cycles, declare_above=above, text=f"read {reads} before a diagnosis")
        if self.abstention_rule == "fixed_cut" and read_first:
            return Advice(action="declare", sensor_group=best, cycles=1, declare_above=above,
                          text=(f"declare insufficient evidence with {best} for 1 cycle, citing a reading: the top mass lies below the fixed "
                                f"cut {self.abstention_cut:g} and no reading is left"))
        faulty = view.faulty(top)
        orders = (f"create one work order for each of its faulty components ({', '.join(faulty)})" if faulty else
                  "create no work order, since it holds no faulty component")
        return Advice(action="diagnose", rank=1, declare_above=above,
                      text=(f"diagnose rank 1, {tool_texts.combo_text(view, top)}, and {orders}. A declaration scores zero on the diagnosis and on "
                            f"the pass."))

    @staticmethod
    def _bearing_ids(view, combo, episode, newest=False):
        """The released summaries whose channel indicates a faulty component of the combination, and the closed orders. With
        newest, which the compact reply asks for, only the newest reading that released each such summary."""
        faulty = set(view.faulty(combo))
        bearing = [(r.n, field) for r in episode.readings for field in sorted(r.values) if faulty & set(view.indicates(view.sensor_of(field)))]
        if newest:
            last = {field: n for n, field in sorted(bearing)}
            bearing = [(n, field) for n, field in bearing if last[field] == n]
        return [f"obs/{n}/{field}" for n, field in bearing] + episode_reading.order_ids(episode.closures)

    # ----- estimate_intervention_effect -----
    def estimate_intervention_effect(self, action):
        try:
            released = self._released()
        except episode_reading.NoEpisode as e:
            return ErrorResult(error=str(e))
        view, status = released.view, released.status
        action = action.strip()
        if action == intervention_effect.MONITOR:
            component = None
        elif action.startswith(intervention_effect.REPLACE) and action[len(intervention_effect.REPLACE):] in view.components:
            component = action[len(intervention_effect.REPLACE):]
        else:
            return ErrorResult(error=f"action must be {intervention_effect.MONITOR} or {intervention_effect.REPLACE}<component> with the component one of {', '.join(view.components)}")
        rules = view.rules
        flt, post = self._belief(released)
        cost = costs(view, released.episode.time, status.get("steps_left"))
        rate = expected_downtime_rate(post.candidates, post.probabilities, cost)
        components = [] if component is None else [component]
        if component is None:
            price, duration = 0.0, rules["durations"]["advance_per_cycle"]
            downtime = rate * duration
        else:
            price = rules["prices"]["parts"][component] + rules["prices"]["repair_charge"] + view.visit_charge()
            duration = view.lead_time(component) + view.repair_downtime(component)
            stopped = 0.0 if cost.unit == "calls" else rules["downtime_rate"][STOPPED]  # the free base row charges no downtime
            downtime = rate * view.lead_time(component) + stopped * view.repair_downtime(component)
        predicted = self._predicted_action(view, flt, post, components)
        addresses, recovery, low, high, band = predicted.addresses, predicted.recovery, predicted.low, predicted.high, predicted.band
        risk_before = 1 - post.current_posterior().get(view.healthy_combination(), 0.0)
        risk_after = 1 - recovery
        # the range of R fits stands for the declared level only from fits_for_level(alpha) fits on, so below that no dead band exists
        dead_band = band if band is not None and predicted.fits >= fits_for_level(self.alpha) else None
        if dead_band is not None and risk_after < risk_before - dead_band:
            direction = "lower"
        elif dead_band is not None and risk_after > risk_before + dead_band:
            direction = "higher"
        else:
            direction = "no change" if dead_band is not None else "undetermined"
        reachable = self.models.calibration.level_reachable(self.alpha)
        prediction_id = self._store_prediction(released, action, components, predicted)
        notes = self._model_assumptions(view) + [
            ("E13: the prediction is for a standard order, and no emergency order is predicted; the belief after the action runs the "
             "predict step through the order's lead time and repair downtime with its close on the cycle schedule_visit gives one order alone, so the replacement takes only when every component it depends on is healthy "
             "at the close and the rule cards act meanwhile; for monitor it is the belief one cycle ahead; the recovery probability is the "
             "mass of the healthy combination at the first reading after the action, and addresses_fault is the mass of the combinations in "
             "which the component is faulty now"),
            tool_texts.A_PREDICT_STEP, tool_texts.A_E14, tool_texts.A_E6,
            (f"E15: risk is one minus the mass of the healthy combination before and after the action, and the direction uses a dead band "
             f"equal to the half width of the resampling interval of the recovery probability when that interval is a range over at least "
             f"{fits_for_level(self.alpha)} resampled fits, the number from which a range stands for alpha {self.alpha:g}, and is "
             f"undetermined below it"),
            f"the price is the part plus the repair charge plus one visit charge of the public price list, the duration the lead time plus "
            f"the repair downtime, and the downtime during the action the expected downtime rate {rate:.3g} over the lead time plus the "
            f"stopped rate over the repair downtime; for {intervention_effect.MONITOR} the duration is one cycle at the expected downtime rate, since no "
            f"observation is free",
            f"alpha = {self.alpha:g} is the declared error level of the intervals"]
        if component is not None and any(o.get("component") == component for o in status.get("open_work_orders", [])):
            notes.append("a work order on this component is open; the environment refuses a second one")
        if component is not None and not any(e.get("event") in COMMIT_EVENTS for e in released.events):
            notes.append("the environment accepts a work order only after a diagnosis or a declaration of insufficient evidence")
        if status["episode_over"]:
            notes.append("the episode is over, so the action can no longer be taken")
        return InterventionEffect(
            prediction_id=prediction_id, action=action, component=component, addresses_fault=addresses, recovery=recovery,
            recovery_interval_low=low, recovery_interval_high=high, price=price, duration=duration, downtime_during_action=downtime,
            predicted=predicted.summaries, alpha=self.alpha, level_reachable=reachable, risk_before=risk_before, risk_after=risk_after,
            direction=direction, dead_band=dead_band, resampled_fits=predicted.fits,
            evidence_ids=episode_reading.observation_ids(released.episode.readings) + episode_reading.order_ids(released.episode.closures) + [prediction_id],
            time_window=TimeWindow(start=released.episode.time, end=released.episode.time + duration, unit=episode_reading.UNIT), assumptions=notes,
            uncertainty=Uncertainty(kind="recovery_probability", value=recovery, detail=None if band is None else {"low": low, "high": high}),
            message=(f"{action}: addresses a fault with probability {addresses:.3f}, predicted recovery {recovery:.3f}"
                     + (f" (resampling interval {low:.3f} to {high:.3f} over {predicted.fits} fits)" if band is not None else "")
                     + f", price {price:g}, duration {duration} {'cycle' if duration == 1 else 'cycles'}; risk {risk_before:.3f} now, predicted risk {risk_after:.3f} after the action, {direction}; "
                     f"{len(predicted.summaries)} summaries predicted for {tool_texts.combo_text(view, predicted.combination)}, the predicted combination, "
                     f"the most probable after the action, with mass {predicted.mass:.3f}, with {1 - self.alpha:g} conformal intervals as {prediction_id}"
                     + ("" if reachable else f"; the calibration blocks cannot carry alpha {self.alpha:g}, so no interval carries its level")))

    def _predicted_action(self, view, flt, post, components):
        """E13 and E14 of one action under the belief: monitor (no component) is the belief one cycle ahead, the
        replacement of one or more components the belief at the first reading after the last close of their visit. The
        recovery interval is the range of the healthy combination's mass under the resampled fits of E6, which move
        through the same predict step."""
        healthy = view.healthy_combination()
        grid = np.array(post.candidates)
        replaced = [view.components.index(c) for c in components]
        addresses = float(post.probabilities[np.any(grid[:, replaced] != np.array([view.healthy[c] for c in components]), axis=1)].sum()) if components else 0.0
        low = high = band = None
        fits = 0
        if self.ranking_model == "discriminant_urn":
            after = moved_masses(view, post.candidates, post.probabilities, components)
        else:
            then = self.model.after_repair(flt, components, view).posterior()
            after = then.current_posterior()
            if then.resampled_probabilities is not None:
                resampled = np.clip(then.resampled_probabilities[:, post.candidates.index(healthy)], 0.0, 1.0)
                low, high = float(resampled.min()), float(resampled.max())
                band = (high - low) / 2
                fits = len(resampled)
        return intervention_effect.Predicted(after, min(1.0, addresses), after.get(healthy, 0.0), low, high, band, predict_summaries(self.models, after, self.alpha),
                         *most_probable(after, self.models.space), fits)

    def _stored_branch(self, action, released):
        """The identifier of a branch prediction of the same action stored for the same released evidence, or None: the
        prediction is the same whichever branch asks for it and a ranking is replayed many times in an episode."""
        return next((p["id"] for p in self._predictions() if "branch" in p and p["action"] == action
                     and p["events"] == len(released.events) and p["observations"] == len(released.episode.readings)), None)

    def _store_prediction(self, released, action, components, predicted, branch=None):
        """Append the prediction to predictions.json and return its identifier: pred/{n} for the intervention tool, pred/b{n}
        for a repair step or a repair plan of a ranking's branch, so that the branches never renumber the predictions the
        tool issues; the stored branch is the rank that registered it first. A record with several components is a plan,
        which verify_outcome compares with the first reading after the last of its replacements closed. The read of the
        file, the numbering and the write hold a lock on a file beside it, so that two server processes of one episode that
        store at once keep both records under two identifiers, and a branch prediction of the same action that another
        process stored meanwhile for the same released evidence is returned instead of a second record."""
        record = {"action": action, "component": components[0] if len(components) == 1 else None, "events": len(released.events),
                  "observations": len(released.episode.readings), "time": released.episode.time, "addresses": predicted.addresses,
                  "recovery": predicted.recovery, "recovery_interval": None if predicted.band is None else [predicted.low, predicted.high],
                  "alpha": self.alpha, "ranking_model": self.ranking_model, "after": {combo_key(h): p for h, p in predicted.after.items()},
                  "predicted": {p.field: [p.mean, p.low, p.high, p.quantile] for p in predicted.summaries},
                  "predicted_combination": combo_key(predicted.combination), "predicted_mass": predicted.mass}
        if len(components) > 1:
            record["components"] = list(components)
        with locked(self._predictions_path()):
            if branch is not None:
                stored = self._stored_branch(action, released)
                if stored is not None:
                    return stored
            predictions = self._predictions()
            if branch is None:
                prediction_id = f"pred/{sum(1 for p in predictions if 'branch' not in p) + 1}"
            else:
                record["branch"] = branch
                prediction_id = f"pred/b{sum(1 for p in predictions if 'branch' in p) + 1}"
            predictions.append({"id": prediction_id, **record})
            write_json_atomically(self._predictions_path(), predictions, indent=1)
        return prediction_id

    def _released_then(self, released, prediction):
        """The released evidence as the estimate of a stored prediction read it: the events and the readings released before
        it, the cycle of the estimate, and the work orders open then with the cycles at which they close."""
        raw, events = released.raw, released.raw["events"][:prediction["events"]]
        closed = {e.get("id") for e in events if e.get("event") == "work_order_closed"}
        opened = [{"id": e["id"], "component": e["component"], "mode": e.get("mode") or "standard", "expected_close_t": e["expected_close_t"]}
                  for e in events if e.get("event") == "work_order_created" and e.get("id") not in closed and e.get("expected_close_t") is not None]
        then = dict(raw, events=events, observations=[o for o in raw["observations"] if o["n"] <= prediction["observations"]],
                    status=dict(raw["status"], time=prediction["time"], open_work_orders=opened))
        return episode_reading.Released(then, released.view, parse_released(then), then["status"], events)

    def _retimed(self, released, prediction, components, created):
        """The stored prediction made again for the cycle at which its work orders were created, from the evidence its estimate
        held, the rule cards running through the cycles in between, so that the closures are those of the orders as created."""
        then = self._released_then(released, prediction)
        flt, post = self._belief(then)
        predicted = self._predicted_action(then.view, flt.ahead(created - prediction["time"]), post, components)
        return {**prediction, "retimed": created, "recovery": predicted.recovery,
                "recovery_interval": None if predicted.band is None else [predicted.low, predicted.high],
                "predicted": {s.field: [s.mean, s.low, s.high, s.quantile] for s in predicted.summaries},
                "predicted_combination": combo_key(predicted.combination), "predicted_mass": predicted.mass}

    # ----- verify_outcome -----
    def verify_outcome(self, prediction_id):
        try:
            released = self._released()
        except episode_reading.NoEpisode as e:
            return ErrorResult(error=str(e))
        prediction = next((p for p in self._predictions() if p["id"] == prediction_id.strip()), None)
        if prediction is None:
            return ErrorResult(error=f"unknown prediction_id {prediction_id}; estimate_intervention_effect returns one such as pred/1")
        view, events, first = released.view, released.events, prediction["events"]
        components = prediction.get("components") or ([] if prediction["component"] is None else [prediction["component"]])
        later = [r for r in released.episode.readings if r.n > prediction["observations"]]
        closures = {c.id: c for c in released.episode.closures}
        if not components:
            since, recovered_text = prediction["time"], "not applicable: monitor repairs nothing"
            # a monitor prediction speaks of the cycle it was made at, and a later read can release an older recorded cycle
            later = [r for r in later if r.t >= since]
        else:
            # a plan is observable once the last of its replacements closed, and its reading shows the state after all of them
            closings = []
            for component in components:
                closing = next((e for e in events[first:] if e.get("event") == "work_order_closed" and e.get("component") == component), None)
                if closing is None:
                    refusals = [e for e in events[first:] if e.get("event") == "refused" and isinstance(e.get("action"), dict)
                                and e["action"].get("type") == "create_work_order" and e["action"].get("component") == component]
                    if refusals:
                        status = f"action refused: {refusals[-1].get('reason')}"
                    elif any(o.get("component") == component for o in released.status.get("open_work_orders", [])):
                        status = "not yet observable: the work order is open and has not closed"
                    else:
                        status = "not yet observable: no work order on the component has closed since the prediction"
                    return self._verification(view, prediction, status, None, [], None, "not observable: the repair has not happened")
                closings.append(closures[closing["id"]])
            since = max(c.t for c in closings)
            later = [r for r in later if all(shows(r, c) for c in closings)]
            # the reading that verifies a plan holds an indicating channel of every repaired component, the one rule of the agent
            # (servers/public_rules/repair_completion.verifies): a reading of other channels leaves the plan not yet observable
            indicating = view.told_indicating_channels(events)  # what the agent was told, as its board knows it
            later = [r for r in later if verifies({channel_field(f) for f in r.values}, components, indicating, view.sensor_groups)]
            recovered_text = ""
            # orders created after the estimate, after thinking time or other calls, close later than the prediction assumed
            created = max((int(e["t_end"]) for e in events[first:] if e.get("event") == "work_order_created"
                           and e.get("id") in {c.id for c in closings}), default=prediction["time"])
            if created > prediction["time"] and self.ranking_model != "discriminant_urn":
                prediction = self._retimed(released, prediction, components, created)
        if not later:
            return self._verification(view, prediction, "not yet observable: no reading since the action", None, [], None,
                                      recovered_text or "not observable: no reading since the repair closed")
        observed = min(later, key=lambda r: (r.t, r.n))
        others = [closures[e["id"]] for e in events[first:] if e.get("event") == "work_order_closed" and e.get("component") not in components]
        others = [c for c in others if shows(observed, c)]
        if others:
            names = ", ".join(f"replace:{c.component}" for c in others)
            return self._verification(view, prediction, f"not comparable: {names} closed between the prediction and the reading", None, [], None,
                                      recovered_text or "not observable: another repair changed the combination")
        checks, combined, p_value, consistent = self.model.outcome_check(prediction, observed.values)
        channels = [ChannelCheck(field=f, predicted=m, observed=o, low=lo, high=hi, score=s) for f, m, o, lo, hi, s in checks]
        recovery_posterior = None
        if components:
            # the belief now holds every released reading and closure, the readings before the repair included, and is the
            # belief that a ranking on the same evidence reports
            _flt, now = self._belief(released)
            recovery_posterior = float(now.probabilities[now.candidates.index(view.healthy_combination())])
            recovered_text = (f"predicted recovery {prediction['recovery']:.3f}, mass of the healthy combination in the belief now "
                              f"{recovery_posterior:.3f}")
        return self._verification(view, prediction, "verified", observed, channels, (combined, p_value, consistent), recovered_text, recovery_posterior, since)

    def _verification(self, view, prediction, status, observed, channels, verdict, recovery_status, recovery_posterior=None, since=None):
        interval_ = prediction.get("recovery_interval")
        reachable = self.models.calibration.level_reachable(prediction["alpha"])
        n = len(channels)
        combined, p_value, consistent = (None, None, None) if verdict is None else verdict
        ids = [prediction["id"]] + ([f"obs/{observed.n}/{c.field}" for c in channels] if observed else [])
        window = TimeWindow(start=since if since is not None else prediction["time"], end=observed.t if observed else prediction["time"], unit=episode_reading.UNIT)
        if consistent is None:
            verdict_text = ""
        else:
            verdict_text = (f"; {'consistent' if consistent else 'inconsistent'} at alpha {prediction['alpha']:g} for "
                            f"{tool_texts.combo_text(view, parse_combo(prediction['predicted_combination']))}, the predicted combination, the most probable "
                            f"after the action, with mass {prediction['predicted_mass']:.3f}: the largest standardized deviation over {n} channels is "
                            f"{combined:.2f}, conformal p value {p_value:.3f}")
        return OutcomeVerification(
            prediction_id=prediction["id"], action=prediction["action"], status=status, observation=observed.n if observed else None,
            observed_t=observed.t if observed else None, channels=channels, n_channels=n, alpha=prediction["alpha"], level_reachable=reachable,
            combined_score=combined,
            p_value=p_value, consistent=consistent, recovery_predicted=prediction["recovery"],
            recovery_interval_low=None if interval_ is None else interval_[0], recovery_interval_high=None if interval_ is None else interval_[1],
            recovery_posterior=recovery_posterior, predicted_combination=dict(zip(view.components, parse_combo(prediction["predicted_combination"]))),
            predicted_mass=prediction["predicted_mass"], recovery_status=recovery_status, evidence_ids=ids, time_window=window,
            assumptions=self._model_assumptions(view) + [tool_texts.A_E14, tool_texts.A_E16, f"alpha = {prediction['alpha']:g} is the declared error level of the verdict", tool_texts.A_TIME],
            uncertainty=Uncertainty(kind="conformal_p_value", value=1.0 if p_value is None else p_value),
            message=(f"{prediction['id']} ({prediction['action']}): {status}{verdict_text}; recovery {recovery_status}"
                     + (f"; made again for cycle {prediction['retimed']}, when its work orders were created, from the evidence its "
                        "estimate held" if "retimed" in prediction else "")
                     + ("" if reachable else f"; the calibration blocks cannot carry alpha {prediction['alpha']:g}, so no reading can be "
                        f"found inconsistent")))
