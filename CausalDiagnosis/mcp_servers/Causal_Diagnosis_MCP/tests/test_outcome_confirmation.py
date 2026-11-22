"""The confirmation of a verified repair by verify_outcome: a verdict is confirmed when it is consistent, the belief now puts at least
the confirmation mass on the predicted combination, and the verifying readings do not favor its strongest rival, whose
likelihood ratio the answer names beside the most probable combination of the belief with its mass; an inconsistent verdict refutes
whatever the mass and the ratio, and a verdict on no reading after the repair, on a reading that another closure changed or on a
monitor prediction carries no confirmation. The belief is the one that a ranking on the same evidence reports, on the fits of the
belief, and the belief before the readings is the replay of the same fits without them. No test states a value computed from the rig
data: the thresholds are set at the mass and at the ratio that the call reports and one step above them, and the verdict is made
consistent or inconsistent by construction."""
import asyncio
import math
import sys
from pathlib import Path

import pytest

from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import COMPONENTS, build_blocks, default_facts_path, load_cycles
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.effect_model import parse_combo
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.main import build_server
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_filter import parse_released
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.discriminant_urn_ranking import urn_probabilities
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools import CausalTools, episode_reading, intervention_effect, tool_texts
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tool_result_schemas import VerificationConcise, VerificationDetailed
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.fixture_episode import FixtureEpisode, episode_models

if not Path(default_facts_path()).exists():
    pytest.skip("no data file in this clone, as ASSETS_WITHOUT_DATA=1 states", allow_module_level=True)

from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tests.test_answer_levels import served  # noqa: E402  the call of a tool through the server

START = (100, 80, 0, 90)  # a start with two faulty components, the valve and the accumulator
CONFIRMATION_FIELDS = ("confirmed", "predicted_posterior", "belief_top_combination", "belief_top_mass", "reading_ratio")


@pytest.fixture(scope="module")
def world():
    cycles = load_cycles(default_facts_path())
    return cycles, build_blocks(cycles)


@pytest.fixture
def episode(tmp_path, world):
    cycles, blocks = world
    return FixtureEpisode(tmp_path, cycles, blocks, START, seed=0)


def tools_for(episode):
    return CausalTools(episode_models(episode.start), episode.released_path)


def repaired(episode, tools):
    """The accumulator replaced on its own prediction after a diagnosis, its order closed and one reading taken after the close: the
    identifier of the prediction, which verify_outcome can now verify."""
    episode.read("tier3")
    effect = tools.estimate_intervention_effect("replace:accumulator")
    episode.submit_diagnosis(dict(zip(COMPONENTS, START)))
    episode.create_work_order("accumulator", prediction_id=effect.prediction_id)
    episode.advance(effect.duration)
    episode.read("tier3")
    return effect.prediction_id


def force_verdict(tools, monkeypatch, consistent):
    """The outcome check of the tools with its verdict set to consistent, every other value of the check kept."""
    check = tools.model.outcome_check
    monkeypatch.setattr(tools.model, "outcome_check", lambda prediction, values: (*check(prediction, values)[:3], consistent))


def combination_of(answer):
    return tuple(answer[c] for c in COMPONENTS)


def masses_of(posterior):
    return dict(zip(posterior.candidates, map(float, posterior.probabilities)))


def without_the_verifying_readings(tools, released, first_after):
    """The belief of the fits that the released evidence built, replayed without the readings numbered first_after and above: the
    masses before the readings that verify the repair."""
    kept = [r for r in released.episode.readings if r.n < first_after]
    return masses_of(tools._belief_model(released).replay(released.view, released.episode._replace(readings=kept)).posterior())


def named(combination):
    return ", ".join(f"{c} {v}" for c, v in combination.items())


def odds_ratio(before, after, predicted, rival):
    return (after[predicted] / after[rival]) / (before[predicted] / before[rival])


@pytest.mark.parametrize("consistent", [True, False])
def test_a_repair_is_confirmed_exactly_when_its_verdict_is_consistent_and_the_belief_now_puts_the_threshold_on_the_prediction(
        episode, monkeypatch, consistent):
    """The mass of the predicted combination and the most probable combination with its mass are those of the ranking on the same
    evidence. With the threshold at that mass a consistent verdict is confirmed, since the rule asks for at least the threshold, and
    one step above it no verdict is; an inconsistent verdict is confirmed under neither and keeps its status, so it still refutes."""
    tools = tools_for(episode)
    prediction_id = repaired(episode, tools)
    force_verdict(tools, monkeypatch, consistent)
    first = tools.verify_outcome(prediction_id)
    assert first.status == "verified" and first.consistent is consistent
    ranking = tools.rank_root_causes(len(tools.models.space.combinations))
    masses = {combination_of(c.combination): c.probability for c in ranking.candidates}
    mass = first.predicted_posterior
    assert mass == masses.get(combination_of(first.predicted_combination), 0.0)
    assert first.belief_top_combination == ranking.candidates[0].combination and first.belief_top_mass == ranking.candidates[0].probability
    # the belief of the confirmation is on the fits of the belief, which leave out the served blocks, not on the start-only fits
    released = tools._released(note=False)
    assert tools._belief_model(released) is not tools.model
    start_only = tools.model.replay(released.view, released.episode).posterior()
    assert mass != dict(zip(start_only.candidates, map(float, start_only.probabilities)))[combination_of(first.predicted_combination)]
    top = ", ".join(f"{c} {v}" for c, v in first.belief_top_combination.items())
    for threshold, reached in ((mass, True), (math.nextafter(mass, 2.0), False)):
        monkeypatch.setattr(intervention_effect, "CONFIRMATION_MASS", threshold)
        again = tools.verify_outcome(prediction_id)
        assert (again.status, again.consistent, again.predicted_posterior) == ("verified", consistent, mass)
        assert again.confirmed is (consistent and reached)
        if again.confirmed:
            assert f"; confirmed: the belief now puts {mass:.3f} on the predicted combination" in again.message
        else:
            assert "; not confirmed: " in again.message
            assert f"{again.belief_top_mass:.3f} on its most probable combination ({top})" in again.message
    assert tool_texts.A_CONFIRM in first.assumptions


def test_no_reading_after_the_repair_a_second_closure_and_a_monitor_prediction_carry_no_confirmation(episode):
    """A verdict that is not yet observable, one that another repair made not comparable and the verdict on a monitor prediction,
    which repairs nothing, leave every field of the confirmation empty."""
    tools = tools_for(episode)
    episode.read("tier3")
    effect = tools.estimate_intervention_effect("replace:accumulator")
    pending = tools.verify_outcome(effect.prediction_id)
    episode.submit_diagnosis(dict(zip(COMPONENTS, START)))
    episode.create_work_order("accumulator", prediction_id=effect.prediction_id)
    episode.create_work_order("valve")
    episode.advance(10)
    episode.read("tier3")
    other = tools.verify_outcome(effect.prediction_id)
    watch = tools.estimate_intervention_effect("monitor")
    episode.read("tier1")
    watched = tools.verify_outcome(watch.prediction_id)
    assert pending.status.startswith("not yet observable") and other.status.startswith("not comparable") and watched.status == "verified"
    for answer in (pending, other, watched):
        assert all(getattr(answer, field) is None for field in CONFIRMATION_FIELDS), answer.status
        assert "confirmed" not in answer.message


def test_every_level_of_the_answer_carries_the_confirmation(episode):
    """Through the server: raw and detailed hold the four fields, concise keeps them as its schema lists them, and headline, the
    message, says whether the repair is confirmed; the assumption of the confirmation is its sentence in raw and its key elsewhere."""
    tools = tools_for(episode)
    prediction_id = repaired(episode, tools)
    server = build_server(episode.released_path, left_out=START)
    answers = {level: asyncio.run(served(server, "verify_outcome", {"prediction_id": prediction_id, "response_format": level}))[1]
               for level in ("raw", "detailed", "concise", "headline")}
    raw = answers["raw"]
    assert raw["status"] == "verified" and raw["confirmed"] in (True, False) and tool_texts.A_CONFIRM in raw["assumptions"]
    for level, schema in (("detailed", VerificationDetailed), ("concise", VerificationConcise)):
        kept = schema.model_validate(answers[level]).model_dump(mode="json")
        assert {f: kept[f] for f in CONFIRMATION_FIELDS} == {f: raw[f] for f in CONFIRMATION_FIELDS}, level
        assert "confirmed" in answers[level]["assumptions"]
    said = "; confirmed: " if raw["confirmed"] else "; not confirmed: "
    assert said in answers["headline"]["headline"] and said in raw["message"]


def test_the_ratio_is_the_odds_of_the_predicted_combination_against_its_strongest_rival_after_the_readings_over_the_odds_before_them():
    """The strongest rival is the combination other than the predicted one with the most mass after the readings, a tie going to the
    first combination of the belief, and the ratio is the posterior odds of the predicted combination against it divided by their
    odds before the readings. A mass of zero leaves the ratio undefined, which no repair is confirmed on. The rule asks for at least
    the threshold of the mass and at least the threshold of the ratio, and for a consistent verdict."""
    mass, minimum = intervention_effect.CONFIRMATION_MASS, intervention_effect.CONFIRMATION_RATIO
    predicted, near, far = (1,), (2,), (3,)
    before = {near: 0.30, predicted: 0.50, far: 0.20}
    after = {near: 0.03, predicted: 0.96, far: 0.01}
    rival, ratio = intervention_effect.reading_ratio(before, after, predicted)
    assert rival == near and ratio == pytest.approx((0.96 / 0.03) / (0.50 / 0.30))
    assert intervention_effect.reading_ratio(before, {near: 0.02, predicted: 0.96, far: 0.02}, predicted)[0] == near
    assert intervention_effect.reading_ratio(before, {far: 0.02, predicted: 0.96, near: 0.02}, predicted)[0] == far
    for zeroed in ({**before, predicted: 0.0}, {**before, near: 0.0}):
        assert intervention_effect.reading_ratio(zeroed, after, predicted) == (near, None)
    assert intervention_effect.reading_ratio(before, {near: 0.0, predicted: 1.0, far: 0.0}, predicted) == (near, None)
    extreme = {near: 1e-300, predicted: 1.0, far: 1e-300}, {near: 1.0, predicted: 1e-300, far: 1e-300}  # odds that no float holds
    assert intervention_effect.reading_ratio(extreme[0], extreme[1], predicted) == (near, 0.0)
    assert intervention_effect.reading_ratio(extreme[1], extreme[0], predicted) == (near, sys.float_info.max)
    cases = ((True, mass, minimum, True), (True, mass, math.nextafter(minimum, 0.0), False), (True, math.nextafter(mass, 0.0), 2 * minimum, False),
             (False, 1.0, 2 * minimum, False), (True, 1.0, None, False))
    assert [intervention_effect.confirms(consistent, m, r) for consistent, m, r, _ in cases] == [reached for *_, reached in cases]


def test_the_ratio_compares_the_belief_now_with_the_same_fits_replayed_without_the_verifying_readings(episode):
    """The strongest rival and the masses after the readings are those of a ranking on the same evidence, the masses before them those of
    the fits that the released evidence built, replayed without the verifying readings, so that no change of fits enters the
    ratio. The control: the fits built again for the evidence without those readings leave out other blocks and put the ratio
    elsewhere, so a verification that rebuilt them would fail this test."""
    tools = tools_for(episode)
    prediction_id = repaired(episode, tools)
    answer = tools.verify_outcome(prediction_id)
    released = tools._released(note=False)
    combinations = tools.models.space.combinations
    ranking = tools.rank_root_causes(len(combinations))
    after = {**{h: 0.0 for h in combinations}, **{combination_of(c.combination): c.probability for c in ranking.candidates}}
    predicted = combination_of(answer.predicted_combination)
    rival = next(combination_of(c.combination) for c in ranking.candidates if combination_of(c.combination) != predicted)
    before = without_the_verifying_readings(tools, released, first_after=answer.observation)
    assert answer.reading_ratio == pytest.approx(odds_ratio(before, after, predicted, rival), rel=1e-9)
    kept = {**released.raw, "observations": [o for o in released.raw["observations"] if o["n"] < answer.observation]}
    reduced = episode_reading.Released(kept, released.view, parse_released(kept), released.status, released.events)
    source = tools.models.source
    assert source.served_blocks(kept["observations"]) != source.served_blocks(released.raw["observations"])
    rebuilt = masses_of(tools._belief(reduced)[1])
    assert odds_ratio(rebuilt, after, predicted, rival) != pytest.approx(answer.reading_ratio, rel=1e-6)


def test_a_verification_builds_the_fits_of_the_belief_as_often_as_a_ranking_on_the_same_evidence_does(episode, monkeypatch):
    """The belief before the readings is a replay of the fits built for the released evidence, so the ratio adds no build and no pad:
    the builds of a verification are those of a ranking on the same evidence, the reference build and the build for the readings
    released."""
    builds = []
    timed_build = CausalTools._timed_build

    def counting(self, source, blocks):
        builds.append(sorted(map(tuple, blocks)))
        return timed_build(self, source, blocks)

    monkeypatch.setattr(CausalTools, "_timed_build", counting)
    prediction_id = repaired(episode, tools_for(episode))
    builds.clear()
    tools_for(episode).verify_outcome(prediction_id)
    verification = list(builds)
    builds.clear()
    tools_for(episode).rank_root_causes(3)
    assert verification == builds and len(builds) == 2


def test_a_repair_whose_verdict_and_mass_confirm_is_refused_by_the_ratio_alone_when_the_readings_favor_its_strongest_rival(episode, monkeypatch):
    """With the threshold of the mass at the mass that the call reports and a consistent verdict, a repair is confirmed when the
    threshold of the ratio is zero or the ratio that the call reports, and one step above that ratio the same answer is refused by the
    ratio alone: the message says that the verifying readings favor the strongest rival, names it with the ratio and the
    threshold, and still names the most probable combination of the belief with its mass. A confirmed repair says that the readings do
    not favor its rival."""
    tools = tools_for(episode)
    prediction_id = repaired(episode, tools)
    first = tools.verify_outcome(prediction_id)
    assert first.status == "verified" and first.consistent is True
    monkeypatch.setattr(intervention_effect, "CONFIRMATION_MASS", first.predicted_posterior)
    monkeypatch.setattr(intervention_effect, "CONFIRMATION_RATIO", math.inf, raising=False)
    assert tools.verify_outcome(prediction_id).confirmed is False  # no ratio reaches the threshold, whatever the mass and the verdict
    assert first.reading_ratio > 0
    predicted = combination_of(first.predicted_combination)
    ranking = tools.rank_root_causes(len(tools.models.space.combinations))
    rival = next(c.combination for c in ranking.candidates if combination_of(c.combination) != predicted)
    above = math.nextafter(first.reading_ratio, math.inf)
    answers = []
    for threshold in (0.0, first.reading_ratio, above):
        monkeypatch.setattr(intervention_effect, "CONFIRMATION_RATIO", threshold, raising=False)
        answers.append(tools.verify_outcome(prediction_id))
    free, at, refused = answers
    assert [a.confirmed for a in answers] == [True, True, False]
    assert len({(a.status, a.consistent, a.predicted_posterior, a.reading_ratio, a.belief_top_mass) for a in answers}) == 1
    assert "; confirmed: " in at.message and "do not favor its strongest rival, likelihood ratio" in at.message
    assert "; not confirmed: " in refused.message
    assert (f"the verifying readings favor its strongest rival ({named(rival)}) over it, likelihood ratio {refused.reading_ratio:.3g}, "
            f"below {above:g}") in refused.message
    assert f"{refused.belief_top_mass:.3f} on its most probable combination ({named(refused.belief_top_combination)})" in refused.message


def test_a_repair_refused_by_the_ratio_is_confirmed_when_asked_again_after_a_reading_that_favors_the_predicted_combination(episode, monkeypatch):
    """The ratio counts every verifying reading, so a later reading that favors the predicted combination raises it. With the
    mass asking nothing and the threshold of the ratio a little above the ratio of the first ask, which the test computes from the masses
    of the fits of the belief, the repair is refused, and asked again after one more reading of the repaired block it is confirmed
    with a larger ratio. The belief before the readings leaves out every verifying reading, the one of the first ask and the
    later one."""
    tools = tools_for(episode)
    prediction_id = repaired(episode, tools)
    left_out = []
    belief = tools._belief

    def spy(released, *args, **kwargs):
        left_out.append(sorted(kwargs.get("without", args[0] if args else ())))
        return belief(released, *args, **kwargs)

    def ratio_of_the_verifying_readings():
        released = tools._released(note=False)
        predicted = parse_combo(next(p for p in tools._predictions() if p["id"] == prediction_id)["predicted_combination"])
        after = masses_of(belief(released)[1])
        rival = max((h for h in after if h != predicted), key=after.get)
        return odds_ratio(without_the_verifying_readings(tools, released, first_after=2), after, predicted, rival)

    monkeypatch.setattr(tools, "_belief", spy)
    monkeypatch.setattr(intervention_effect, "CONFIRMATION_MASS", 0.0)
    first_ratio = ratio_of_the_verifying_readings()
    monkeypatch.setattr(intervention_effect, "CONFIRMATION_RATIO", 1.001 * first_ratio, raising=False)
    refused = tools.verify_outcome(prediction_id)
    episode.read("tier3")
    later_ratio = ratio_of_the_verifying_readings()
    later = tools.verify_outcome(prediction_id)
    assert later_ratio > 1.001 * 1.001 * first_ratio  # the later reading favors the predicted combination
    assert (refused.confirmed, later.confirmed) == (False, True)
    assert later.reading_ratio == pytest.approx(later_ratio, rel=1e-9) and later.reading_ratio > refused.reading_ratio
    assert [numbers for numbers in left_out if numbers] == [[2], [2, 3]]


def test_a_ratio_that_is_undefined_leaves_the_repair_not_confirmed_and_says_so(episode, monkeypatch):
    """A ratio that is undefined, since a mass is zero, confirms nothing: the answer carries no ratio, is not confirmed whatever the mass and the
    verdict, and says in the same sentence that the ratio is undefined, with the most probable combination of the belief and its mass."""
    tools = tools_for(episode)
    prediction_id = repaired(episode, tools)
    first = tools.verify_outcome(prediction_id)
    assert first.consistent is True
    monkeypatch.setattr(intervention_effect, "CONFIRMATION_MASS", first.predicted_posterior)
    ratio = intervention_effect.reading_ratio
    monkeypatch.setattr(intervention_effect, "reading_ratio", lambda before, after, predicted: (ratio(before, after, predicted)[0], None))
    undefined = tools.verify_outcome(prediction_id)
    assert (undefined.confirmed, undefined.reading_ratio, undefined.predicted_posterior) == (False, None, first.predicted_posterior)
    assert "; not confirmed: " in undefined.message and "the likelihood ratio of the verifying readings is undefined, a mass being zero" in undefined.message
    assert f"{undefined.belief_top_mass:.3f} on its most probable combination ({named(undefined.belief_top_combination)})" in undefined.message


def test_under_the_urn_model_the_ratio_is_taken_over_the_masses_of_the_urn(episode):
    """The belief before the readings goes through the path of the belief now, so a ranking model that replaces the masses replaces
    both: under discriminant_urn the ratio is that of the urn's masses with and without the verifying readings."""
    tools = CausalTools(episode_models(episode.start), episode.released_path, ranking_model="discriminant_urn")
    prediction_id = repaired(episode, tools)
    answer = tools.verify_outcome(prediction_id)
    released, models = tools._released(note=False), tools.models

    def urn(readings):
        masses = urn_probabilities(models.urn, models.space, readings, released.episode.closures)
        return {h: masses.get(h, 0.0) for h in models.space.combinations}

    after = urn(released.episode.readings)
    before = urn([r for r in released.episode.readings if r.n < answer.observation])
    predicted = combination_of(answer.predicted_combination)
    rival = max((h for h in after if h != predicted), key=after.get)
    assert answer.reading_ratio == pytest.approx(odds_ratio(before, after, predicted, rival), rel=1e-9)
