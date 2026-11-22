"""Troubleshooting plans of rank_root_causes, E20: each top combination walked forward to a restored rig (version 7,
CausalDiagnosis/docs/causal-diagnosis-mcp.md, section Candidate plans).

The rig cannot be repaired once per candidate to see which repair works, so the walk forward happens in the server. For
each distinct faulty set among the returned candidates (the first top_k by mass), the ranking reports the sensor group whose next reading would confirm or refute the
set and by how much, the repair plan of the set, priced and timed as one technician visit (one visit
charge, the parts arriving after the longest lead time, the repairs back to back in priority order, every component the
plan's repairs depend on listed with whether the plan repairs it), the summaries predicted after those repairs, and the
expected loss of acting on the set against testing it first, both for the one combination a diagnosis submits: acting now the
set's most probable candidate, testing first the combination on top after the reading. The predicted outcome runs the predict step of the
filter through the plan's cycles with the plan's closures, so a repair whose dependency stays degraded predicts no
recovery and a hidden drop during the lead time is predicted as the rule cards make it. The cost of acting now counts the
outlay of the plan once and adds what acting can cost beyond it, and the cost of testing first is that of reading the
confirm test and then acting on the combination first after the reading. Every quantity is a posterior
mass, a price, a duration times a downtime rate, or an expected loss of E8 or E9, computed with the server's sample count
and a seed that adds the rank, so a replayed call returns the same branches. The previews are decision aids over the
public prices and the belief, not the cost the environment charges.
"""
from collections import namedtuple
from itertools import combinations_with_replacement

import numpy as np

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import STOPPED, cost_unit
from CausalDiagnosis.utils.public_rules.repair_completion import Rules, completion
from CausalDiagnosis.utils.public_rules.visits import visit_schedule as schedule_of_visit

from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.conformal_intervals import DEFAULT_ALPHA
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.decision_rule import decision_loss, expected_downtime_rate, expected_loss_act, loss_beyond_outlay, open_to
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tool_result_schemas import Branch, ConfirmTest, CostToRestored, Fault, PredictedOutcome, RepairPlan, RepairStep, TwoStepTest

SEPARATION = 2  # standard errors by which a two step pair must beat the best single test before it is reported
# compact, the default, holds what an agent that follows the advice reads, with the plans of summary; summary and full add the
# block e10_rule of the earlier rule and the keys compact leaves out, and summary leaves out the predicted summaries and the gap
# of every group, which dominate the size
BRANCH_DETAILS = ("compact", "summary", "full")
DEFAULT_BRANCH_DETAIL = "compact"

Separation = namedtuple("Separation", "mass_if_true mass_if_false gap standard_error")
VisitStep = namedtuple("VisitStep", "component lead_time repair_downtime close_offset")


def visit_schedule(view, components, mode="standard"):
    """The repairs of one visit whose orders are created now one after another, by the visit rule of the environment
    (servers/public_rules/visits.py): the technician arrives after the longest lead time, and the repairs follow back to back
    in priority order, ties in the given order. Returns the steps with the cycle after creation at which each closes."""
    components = list(components)
    plan = schedule_of_visit([(0, view.lead_time(c, mode), view.repair_downtime(c), view.priority(c)) for c in components])
    return [VisitStep(components[i], view.lead_time(components[i], mode), view.repair_downtime(components[i]), close) for i, _execute, close in plan]


def completion_rules(view):
    """The public values of the rule of finishing a repair (servers/public_rules/repair_completion.py) on the row of the view:
    a row that reads a record serves the cycles before the clock, so its verifying read comes one cycle after the close, takes no
    time and follows a wait whose change summary needs no call; a cycle runs after a close on the rig and, on such a row, where
    its tool table lists the clock. Where the table lists no wait, the verifying read is a forward read that plays the cycles to
    the close at the declared read duration, or none where that duration is 0; where it lists the change summary, the summary is a
    call (the agent's rule, CausalDiagnosis/agents/Causal_Diagnosis_Agent/situation_board/ Board.rules)."""
    durations, prices = view.rules["durations"], view.rules["prices"]
    recorded = view.reads_record
    listed = {row.get("name") for row in view.rules.get("tools", ())}
    clock_runs = not recorded or "get_current_time" in listed
    wait_served = not recorded or "wait_for_notification" in listed
    return Rules(lead_time={c: view.lead_time(c) for c in view.components}, repair_downtime={c: view.repair_downtime(c) for c in view.components},
                 priority={c: view.priority(c) for c in view.components},
                 order_price={c: (prices.get("parts") or {}).get(c, 0) + (prices.get("repair_charge") or 0) for c in view.components},
                 visit_charge=view.visit_charge(), read_duration=0 if recorded else durations.get("read_per_cycle", 0),
                 first_offset=1 if recorded else 0, summary_call=not recorded or "summarize_changes_since" in listed, clock_runs=clock_runs,
                 wait_served=wait_served, forward_read=0 if wait_served or not clock_runs else durations.get("read_per_cycle", 0))


def plan_completion(view, components, commit=True):
    """What finishing a repair of the components needs from now, by the one rule of the agent and the server: the calls with the
    diagnosis while none is made (commit), the orders, the waits, the verifying read and the finish, and the cycles to the end of
    that read. The server prices the repair without its verifying read (read_price=0, the cost to a restored rig as reported)."""
    return completion(completion_rules(view), 0, new=list(components), commit=commit, read_price=0)


def plan_terms(view, components):
    """The price and the cycles to a restored rig of one visit on the components: the parts, the repair charges and one
    visit charge in money, one call per order on the free base row, and the cycles of plan_completion to the end of the
    verifying read; nothing for an empty plan."""
    visit = visit_schedule(view, components)
    if not visit:
        return 0.0, 0
    cycles = plan_completion(view, components).cycles
    if cost_unit(view.rules["prices"]) == "calls":
        return float(len(visit)), cycles
    prices = view.rules["prices"]
    price = sum(prices["parts"][s.component] + prices["repair_charge"] for s in visit) + view.visit_charge()
    return float(price), cycles


def plan_money(view, faulty, cost, rate):
    """The money of the repair plan of a faulty set, one visit: its price, its waiting cost at the expected downtime rate
    until the technician arrives, and its stopped cost over the repair downtimes, which a row whose rules price nothing
    does not charge. Nothing for an empty plan."""
    visit = visit_schedule(view, faulty)
    if not visit:
        return 0.0, 0.0, 0.0
    stopped = 0.0 if cost.unit == "calls" else view.rules["downtime_rate"][STOPPED]
    price, _cycles = plan_terms(view, faulty)
    return price, float(rate * max(s.lead_time for s in visit)), float(sum(stopped * s.repair_downtime for s in visit))


def restored(view, combination, components):
    """Whether the repairs of one visit on the components restore the combination when nothing else happens: each repair,
    in priority order, takes when every component it depends on is healthy at its close."""
    levels = dict(zip(view.components, combination))
    for step in visit_schedule(view, components):
        if all(levels[d] == view.healthy[d] for d in view.dependencies(step.component)):
            levels[step.component] = view.healthy[step.component]
    return all(levels[c] == view.healthy[c] for c in view.components)


def moved_masses(view, combinations, probabilities, components):
    """{combination: mass} after the repairs of one visit on the components move every combination by the repair rule,
    without the hidden processes, for a ranking model that has no transition model."""
    after = {}
    for h, p in zip(combinations, probabilities):
        if p <= 0:
            continue
        levels = dict(zip(view.components, h))
        for step in visit_schedule(view, components):
            if all(levels[d] == view.healthy[d] for d in view.dependencies(step.component)):
                levels[step.component] = view.healthy[step.component]
        moved = tuple(levels[c] for c in view.components)
        after[moved] = after.get(moved, 0.0) + float(p)
    return after


def mass_after(post, members, weights, sequence, samples, rng):
    """The mass the combinations of members hold together after each of the hypothetical readings of the sequence drawn from
    the mixture `weights` over the combinations' predictive distributions, by the update of E2 solved in one batch."""
    return post.update_with(sequence, post.sample_next(sequence, samples, rng, weights=weights))[:, list(members)].sum(axis=1)


def mixture_of_others(probabilities, members):
    """The belief renormalized over the combinations outside members, or None when they hold no mass, as when the members
    hold all of it or when there is no other combination."""
    weights = np.array(probabilities, dtype=float)
    weights[list(members)] = 0.0
    if weights.sum() > 0:
        return weights / weights.sum()
    return None


def mixture_of(probabilities, members):
    """The belief renormalized over the combinations of members, or None when they hold no mass."""
    weights = np.zeros(len(probabilities))
    weights[list(members)] = np.asarray(probabilities, dtype=float)[list(members)]
    return weights / weights.sum() if weights.sum() > 0 else None


def separation(post, members, others, sequence, samples, rng):
    """How far one cycle of each group in sequence moves the mass of the combinations of members: their expected mass when
    the readings come from them, weighted by their mass, their expected mass when the readings come from the mixture of the
    others, the gap between the two, and the standard error of the gap over the draws. For one member this is the test of
    one combination."""
    weights = mixture_of(post.probabilities, members)
    if weights is None:  # the combinations cannot be the state the reading meets, so no reading confirms them
        return Separation(0.0, 0.0, 0.0, 0.0)
    if_true = mass_after(post, members, weights, sequence, samples, rng)
    if_false = mass_after(post, members, others, sequence, samples, rng) if others is not None else np.zeros(samples)
    error = float(np.sqrt((if_true.var(ddof=1) + if_false.var(ddof=1)) / samples)) if samples > 1 else 0.0
    return Separation(float(if_true.mean()), float(if_false.mean()), float(if_true.mean() - if_false.mean()), error)


def confirm_tests(post, members, cost, decision, samples, rng, detail, serving=None):
    """The confirm test of the combinations of members (the candidates of one faulty set), the single group with the largest gap, and the two step pair of groups when its
    gap beats the best single gap by more than SEPARATION standard errors of the difference, else None. Groups are visited
    cheapest first, so that a tie falls to the cheaper read and the draws keep one order for replay. serving is the count of
    serving.left_to_serve where the caller knows what the row can still serve: a group it can serve nothing more confirms
    nothing, the mass of the combinations staying where it is, and a pair is offered only when the row can serve both of its
    reads. Without it every group counts as open."""
    groups = sorted(cost.test_price, key=lambda g: (cost.test_price[g], g))
    others = mixture_of_others(post.probabilities, members)
    mass = float(np.asarray(post.probabilities, dtype=float)[list(members)].sum())
    unmoved = Separation(mass, mass, 0.0, 0.0)
    singles = {g: separation(post, members, others, [post.space.groups[g]], samples, rng) if open_to(serving, g) else unmoved for g in groups}
    best = max(groups, key=lambda g: singles[g].gap)
    confirm = ConfirmTest(sensor_group=best, cycles=1, price=cost.test_price[best], duration=cost.test_duration[best], cost=decision.test_cost[best],
                          mass_if_true=singles[best].mass_if_true, mass_if_false=singles[best].mass_if_false, gap=singles[best].gap,
                          standard_error=singles[best].standard_error, gap_by_group={g: singles[g].gap for g in groups} if detail == "full" else None)
    pairs = {pair: separation(post, members, others, [post.space.groups[g] for g in pair], samples, rng) for pair in combinations_with_replacement(groups, 2)
             if serving is None or all(serving[g].cycles >= pair.count(g) for g in pair)}
    if not pairs:
        return confirm, None
    pair = max(pairs, key=lambda p: pairs[p].gap)
    error = float(np.hypot(pairs[pair].standard_error, singles[best].standard_error))
    if pairs[pair].gap - singles[best].gap <= SEPARATION * error:
        return confirm, None
    two_step = TwoStepTest(sensor_groups=list(pair), cycles=2, price=sum(cost.test_price[g] for g in pair), duration=sum(cost.test_duration[g] for g in pair),
                           cost=sum(decision.test_cost[g] for g in pair), mass_if_true=pairs[pair].mass_if_true, mass_if_false=pairs[pair].mass_if_false,
                           gap=pairs[pair].gap, standard_error=pairs[pair].standard_error)
    return confirm, two_step


def repair_plan(view, combination, faulty, cost, rate, rank, register_prediction):
    """One work order per component of the combination's faulty set, created one after another so that one visit repairs them:
    the steps in the order the technician repairs them, each with its prices, its own lead time and repair downtime, the
    cycle after creation at which it closes, its stopped downtime cost and the components it depends on; the plan with one
    visit charge, the waiting cost at the expected downtime rate until the technician arrives, and the cycles to a
    restored rig, the last close plus one cycle for the verifying read. Every step registers the prediction of its own
    replacement and a plan of several steps one more after all of them, the one every work order of the plan cites."""
    prices = view.rules["prices"]
    stopped = 0.0 if cost.unit == "calls" else view.rules["downtime_rate"][STOPPED]  # the free base row charges no downtime
    visit = visit_schedule(view, faulty)
    steps = []
    for s in visit:
        steps.append(RepairStep(component=s.component, part_price=prices["parts"][s.component], repair_charge=prices["repair_charge"],
                                lead_time=s.lead_time, repair_downtime=s.repair_downtime, close_offset=s.close_offset,
                                downtime_cost=stopped * s.repair_downtime, depends_on=list(view.dependencies(s.component)),
                                dependencies_in_plan=[d for d in view.dependencies(s.component) if d in faulty],
                                prediction_id=register_prediction([s.component], rank)))
    if len(steps) > 1:
        plan_prediction_id = register_prediction([s.component for s in visit], rank)
    else:
        plan_prediction_id = steps[0].prediction_id if steps else None
    price, waiting, stopped_cost = plan_money(view, faulty, cost, rate)
    _price, cycles = plan_terms(view, faulty)
    return RepairPlan(steps=steps, plan_prediction_id=plan_prediction_id, visits=1 if steps else 0, visit_charge=view.visit_charge() if steps else 0.0,
                      price=price, lead_time=max((s.lead_time for s in visit), default=0), waiting_cost=waiting, downtime_cost=float(waiting + stopped_cost),
                      cycles_to_restored=cycles)


def predicted_outcome(predicted, alpha, detail, restored_if_true, level_reachable):
    """E13 through the plan's cycles and closures, then E14 on the result: the recovery if acted is the mass of the healthy
    combination at the first reading after the plan, which the predict step computed. The summaries are returned in the
    full detail only. level_reachable says whether the calibration blocks carry alpha."""
    return PredictedOutcome(predicted=predicted.summaries if detail == "full" else None, alpha=alpha, level_reachable=level_reachable,
                            recovery_if_acted=predicted.recovery, restored_if_true=restored_if_true)


def credible_set(probabilities, order, alpha):
    """The fewest candidates of the order, first to last, whose mass reaches 1 - alpha, among those with positive mass."""
    chosen, mass = [], 0.0
    for k in order:
        if probabilities[k] <= 0 or mass >= 1 - alpha:
            break
        chosen.append(int(k))
        mass += float(probabilities[k])
    return chosen


def faulty_sets(view, returned, post):
    """[(faulty set, [candidate indices])] of the returned candidates (the first top_k of the ranking, in rank order), one entry
    per distinct faulty set, in the order of the set's most probable returned candidate, each list in rank order."""
    sets = {}
    for k in returned:
        sets.setdefault(tuple(view.faulty(post.candidates[k])), []).append(k)
    return list(sets.items())


def branch_previews(view, post, ahead, cost, decision, samples, seed, register_prediction, predict_plan, alpha=DEFAULT_ALPHA,
                    detail=DEFAULT_BRANCH_DETAIL, serving=None, *, level_reachable, returned):
    """One Branch per distinct faulty set among the returned candidates (faulty_sets; returned is the candidate indices the
    reply lists, in rank order), keyed by the set: its rank is that of the set's most probable candidate and ranks lists the
    returned candidates of the set. The members of the set are every combination of the set with positive mass, returned or
    not, so that the branch's probability is the set's mass as set_masses gives it and nothing in the branch moves with top_k.
    ahead is the belief the
    next reading meets, on which the confirm tests are drawn. register_prediction(components, rank) stores the prediction of
    a plan through the adapter and returns its identifier; predict_plan(components, rank) returns the predicted outcome of
    the plan, one per set of components. detail is one of BRANCH_DETAILS, serving is what the row can still serve, which
    confirm_tests reads, and level_reachable whether the calibration blocks carry alpha, which every predicted outcome
    states. The confirm test moves the mass of the set. The expected loss of acting and the loss beyond the outlay are those of
    the combination the agent submits when it acts on the branch, the set's most probable candidate, so that acting now and
    testing first, which prices the combination on top after the reading, stand on one footing (E20);
    the members share the faulty set and so the repair, and differ only in the price of a wrong diagnosis."""
    loss = decision_loss(post.candidates, cost)
    beyond = loss_beyond_outlay(post.candidates, cost)
    rate = expected_downtime_rate(post.candidates, post.probabilities, cost)
    money = {}

    def outlay(k):
        """The price and the downtime cost of the repair plan of combination k, one plan_money per distinct faulty set."""
        faulty = tuple(view.faulty(post.candidates[k]))
        if faulty not in money:
            price, waiting, stopped = plan_money(view, list(faulty), cost, rate)
            money[faulty] = price + waiting + stopped
        return money[faulty]

    tested_first = {}

    def test_first(group):
        """The cost of reading the group and then acting on the combination first after the reading: the test's cost plus,
        over the drawn readings of E9, the outlay of that combination's plan and the reading's belief times its row of the
        loss beyond the outlay."""
        if group not in tested_first:
            after = decision.masses_after[group]
            tops = after.argmax(axis=1)
            acting = np.array([outlay(int(top)) for top in tops]) + (after * beyond[tops]).sum(axis=1)
            tested_first[group] = float(decision.test_cost[group] + acting.mean())
        return tested_first[group]

    ranks = {int(k): r for r, k in enumerate(post.order(), start=1)}
    everyone = {}
    for k in post.order():
        if post.probabilities[k] > 0:
            everyone.setdefault(tuple(view.faulty(post.candidates[k])), []).append(int(k))
    branches = []
    for faulty_key, shown in faulty_sets(view, returned, post):
        members = everyone[faulty_key]
        k, rank = shown[0], ranks[shown[0]]
        rng = np.random.default_rng(seed + rank)
        combination = post.candidates[k]
        faulty = list(faulty_key)
        confirm, two_step = confirm_tests(ahead, members, cost, decision, samples, rng, detail, serving)
        plan = repair_plan(view, combination, faulty, cost, rate, rank, register_prediction)
        acted = expected_loss_act(post.probabilities, loss, k)
        tested = decision.expected_loss_test[confirm.sensor_group]
        current = dict(zip(view.components, combination))
        branches.append(Branch(
            rank=rank, ranks=[ranks[m] for m in shown], combination=current, current_combination=current,
            faults=[Fault(component=c, level=v, code=view.level_code(c, v)) for c, v in current.items() if v != view.healthy[c]],
            probability=float(np.asarray(post.probabilities, dtype=float)[members].sum()), faulty_set=faulty, confirm_test=confirm,
            two_step=two_step, repair_plan=plan,
            predicted_outcome=predicted_outcome(predict_plan(faulty, rank), alpha, detail, restored(view, combination, faulty), level_reachable),
            expected_loss_if_acted=acted,
            expected_loss_if_tested=tested,
            cost_to_restored=CostToRestored(act_now=plan.price + plan.downtime_cost
                                            + expected_loss_act(post.probabilities, beyond, k),
                                            test_first=test_first(confirm.sensor_group))))
    return branches
