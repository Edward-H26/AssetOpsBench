"""CausalDiagnosis/experiments/episode_runners/run_reference_policies.py: the reference policies in process on seeded episodes, read back from the
CSV, every row verified, and the ordering check of RFC 008."""
import csv
import fnmatch
import importlib.util
from pathlib import Path

import pytest

from CausalDiagnosis.utils.paths import preset_files  # noqa: E402
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path

if not Path(default_facts_path()).exists():
    pytest.skip("no data file in this clone, as ASSETS_WITHOUT_DATA=1 states", allow_module_level=True)
if importlib.util.find_spec("CausalDiagnosis.environment.Causal_Diagnosis_Environment.verifier") is None:
    pytest.skip("needs the verifier", allow_module_level=True)

from CausalDiagnosis.experiments.episode_runners.run_configuration import CONFIGS, load_config, load_world
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import ROOT  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import DEFAULT_ASSET  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv  # noqa: E402
from CausalDiagnosis.environment.reference_policies.policies import profile_of  # noqa: E402
from CausalDiagnosis.environment.reference_policies.diagnosability_label import start_grade  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.start_state_seeds import EVALUATION_POOL, REPLICATES, combination_seeds, sampled_combination_seeds  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.verifier import reward as verifier_reward  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.run_reference_policies import POLICIES, default_out, main as run_policies_main, ordering_holds, play, policy_table  # noqa: E402

SEEDED_ROWS = sorted(path.stem for path in preset_files(CONFIGS).values() if path.stem != "anchored")  # the anchored preset anchors its own scenarios
# the rows of the ladder of the environment table in the order of their layers: the last, every layer on with the
# stochastic hazard and the failure card, is the full environment, and the scheduled row is the other mode of the
# deterioration layer
LADDER = ("base", "priced", "real_time", "advanced_real_time", "repair_rules", "deterioration")
FULL_ENVIRONMENT = LADDER[-1]
CEILING_ROWS = (*LADDER, "scheduled")
LABELLED_ROWS = ("real_time", "advanced_real_time")  # rows with starts that the label and E18 both mark inseparable, unlike the free base row
_PLAYED = {}


def inseparable(env):
    """A start that the label and the server's E18 grade both mark inseparable, where a declaration earns its credit."""
    return {"separable": False, "cost": None, "e18": {"separable": False}}


@pytest.fixture(scope="module")
def world():
    _facts, cycles, blocks = load_world()
    return cycles, blocks, policy_table(cycles, blocks)


def starts_of(world, row):
    """The configuration of a row and the smallest seed of every start combination."""
    cycles, blocks, _table = world
    config = load_config(row)[1]
    return config, combination_seeds(HydraulicRigEnv(cycles, blocks, config), 1)


def played_on_the_pool_and_the_test_set(world, row):
    """Every reference policy on every seed of the evaluation pool of a row, the smallest EVALUATION_POOL seeds of every start,
    and on every seed of the REPLICATES replicates of the test set, some of which lie beyond the pool, under the credit rule of a declaration,
    which earns its credit only on a start that the label and E18 both mark inseparable: per seed the verdict and the reward of each policy, played once per row for the tests below."""
    if row not in _PLAYED:
        cycles, blocks, table = world
        config = load_config(row)[1]
        env = HydraulicRigEnv(cycles, blocks, config)
        seeds = sorted(set(combination_seeds(env, EVALUATION_POOL)) | set(sampled_combination_seeds(env, replicates=REPLICATES)))
        _PLAYED[row] = {seed: {name: play(cycles, blocks, config, table[name], seed, start_grade) for name in POLICIES} for seed in seeds}
    return _PLAYED[row]


class DeclaresAtOnce:
    """A declaration before any reading, which earned the most of any constant policy before a correct declaration had to cite a reading."""

    def start(self, obs):
        self.declared = False

    def act(self, obs):
        if self.declared:
            return {"type": "finish", "summary": "Insufficient evidence."}
        self.declared = True
        return {"type": "declare_insufficient_evidence", "separating_test": {"sensor_group": "tier1", "cycles": 1}, "evidence_ids": []}


class WrongAnswer:
    """Reads the cheapest group, submits the healthy combination, which is wrong on every start, citing the reading, and finishes."""

    def start(self, obs):
        self.profile, self.stage = profile_of(obs), 0

    def act(self, obs):
        self.stage += 1
        if self.stage == 1:
            prices = self.profile["prices"]["read"]
            return {"type": "read_sensors", "sensor_group": min(prices, key=lambda g: (prices[g], g)), "cycles": 1}
        if self.stage == 2:
            cited = [f"obs/{served['n']}/{field}" for served in obs["result"]["served"] for field in served["values"]]
            return {"type": "submit_diagnosis", "diagnosis": dict(self.profile["healthy"]), "evidence_ids": cited}
        return {"type": "finish", "summary": "Nothing is wrong."}


@pytest.mark.parametrize("row", SEEDED_ROWS)
def test_the_oracle_beats_a_correct_declaration_on_every_start_of_every_row(world, row):
    """The reward keeps the oracle as the best of the five reference policies under the rule of a correct declaration: even with
    every start marked inseparable, so that the declaration after reading earns its whole credit."""
    cycles, blocks, table = world
    config, seeds = starts_of(world, row)
    for seed in seeds:
        declared, declared_reward = play(cycles, blocks, config, table["declare"], seed, inseparable)
        assert declared["abstention_correct"] and declared["success"] and not declared["pass"], seed
        oracle, oracle_reward = play(cycles, blocks, config, table["oracle"], seed, inseparable)
        assert oracle["pass"] and oracle_reward > declared_reward, seed


@pytest.mark.parametrize("row", LABELLED_ROWS)
def test_a_policy_that_knows_the_label_cannot_beat_the_mean_of_the_oracle(world, row):
    """It declares after reading wherever the label and E18 both mark the start inseparable, the starts where a declaration
    earns its credit, and plays the oracle elsewhere."""
    cycles, blocks, table = world
    config, seeds = starts_of(world, row)
    oracle_rewards, aware_rewards = [], []
    for seed in seeds:
        declared, declared_reward = play(cycles, blocks, config, table["declare"], seed, start_grade)
        _oracle, oracle_reward = play(cycles, blocks, config, table["oracle"], seed, start_grade)
        oracle_rewards.append(oracle_reward)
        aware_rewards.append(declared_reward if declared["abstention_correct"] else oracle_reward)
    assert aware_rewards != oracle_rewards  # both labels mark some start inseparable, so the policy did declare
    assert sum(aware_rewards) < sum(oracle_rewards)


@pytest.mark.parametrize("row", CEILING_ROWS)
def test_the_oracle_earns_at_least_every_reference_policy_on_every_start_of_the_ladder_and_both_deterioration_modes(world, row):
    """The oracle as the best of the five reference policies in the environment table: on every seed of the evaluation pool and of every test
    replicate the oracle, the perfect information policy, earns at least the reward of every other reference policy, on every
    row of the ladder and under both modes of the deterioration layer, the stochastic hazard of the full environment and the
    fixed schedule of level drops."""
    for seed, played in played_on_the_pool_and_the_test_set(world, row).items():
        rewards = {name: reward for name, (_verdict, reward) in played.items()}
        assert all(rewards["oracle"] >= reward for name, reward in rewards.items() if name != "oracle"), (seed, rewards)


def test_on_the_full_environment_doing_nothing_scores_below_zero_on_every_seed_of_the_pool_and_the_test_set(world):
    """On the full environment, every layer on: an
    agent that does nothing scores below zero, and the oracle, which repairs, above zero, on every seed of the evaluation pool
    and of every test replicate. The declaring policy earns its credit exactly on the seeds whose start the label and E18 both
    mark inseparable."""
    for seed, played in played_on_the_pool_and_the_test_set(world, FULL_ENVIRONMENT).items():
        assert played["do_nothing"][1] < 0 < played["oracle"][1], (seed, played["do_nothing"][1], played["oracle"][1])
        declared = played["declare"][0]
        assert declared["abstention_correct"] is (not declared["separable"] and not declared["separable_e18"]), seed


@pytest.mark.parametrize("row", CEILING_ROWS)
def test_doing_nothing_scores_below_every_policy_that_passes_on_every_seed_of_the_ladder_and_both_deterioration_modes(world, row):
    """The floor check of the environment table, on the rows of the test of the oracle above: on every seed of the evaluation pool
    and of every test replicate of every row of the ladder and under both modes of the deterioration layer, every reference
    policy whose episode the verifier passes earns more than doing nothing. A policy that recovers the rig without a pass, or
    fails, may earn less, since its outcome term is zero while it pays its costs."""
    for seed, played in played_on_the_pool_and_the_test_set(world, row).items():
        idle = played["do_nothing"][1]
        passing = {name: reward for name, (verdict, reward) in played.items() if name != "do_nothing" and verdict["pass"]}
        assert passing and all(reward > idle for reward in passing.values()), (seed, idle, passing)


@pytest.mark.parametrize("row", SEEDED_ROWS)
def test_a_declaration_with_no_reading_earns_no_outcome_on_any_start(world, row):
    cycles, blocks, _table = world
    config, seeds = starts_of(world, row)
    for seed in seeds:
        verdict, reward = play(cycles, blocks, config, lambda env, seed: DeclaresAtOnce(), seed, inseparable)
        assert verdict["abstained"] and not verdict["abstention_correct"] and not verdict["success"], seed
        assert verdict["failure_class"] == "abstention_without_reading" and reward <= 0.0, seed
        assert verifier_reward(verdict, config.horizon, w_c=0.0, w_t=0.0) == 0.0, seed


def test_on_one_start_of_the_real_time_row_a_repair_beats_a_declaration_after_reading_and_that_a_wrong_answer(world):
    cycles, blocks, table = world
    config, seeds = starts_of(world, "real_time")
    seed = next(s for s in seeds if play(cycles, blocks, config, table["declare"], s, start_grade)[0]["abstention_correct"])
    rewards = [play(cycles, blocks, config, make, seed, start_grade)[1] for make in (table["oracle"], table["declare"], lambda env, seed: WrongAnswer())]
    assert rewards[0] > rewards[1] > rewards[2]


@pytest.fixture(scope="module")
def policy_csv(tmp_path_factory):
    out = tmp_path_factory.mktemp("runs") / "policies_static.csv"
    run_policies_main(["--config", "static", "--seeds", "5", "--out", str(out)])
    with open(out, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_every_policy_plays_every_seed_and_is_verified(policy_csv):
    for name in POLICIES:
        assert sorted(int(r["seed"]) for r in policy_csv if r["policy"] == name) == list(range(5))
    assert all(r["verified"] == "True" for r in policy_csv)
    assert {r["policy"] for r in policy_csv} == set(POLICIES)


def test_the_oracle_diagnoses_on_a_start_the_label_marks_inseparable_and_the_verdict_counts_it_a_success():
    _facts, cycles, blocks = load_world()
    config = load_config("static")[1]
    table = policy_table(cycles, blocks)
    verdict, on_inseparable = play(cycles, blocks, config, table["oracle"], 0, label_of=inseparable)
    assert verdict["pass"] and verdict["success"] and verdict["commitment_on_inseparable"] and not verdict["abstained"]
    verdict, committed = play(cycles, blocks, config, table["oracle"], 0)
    assert verdict["pass"] and verdict["success"] and not verdict["abstained"]
    idle, idle_reward = play(cycles, blocks, config, table["do_nothing"], 0, label_of=inseparable)
    assert not idle["success"] and on_inseparable == pytest.approx(committed) and committed > idle_reward


def test_the_oracle_succeeds_on_every_seed_and_outranks_the_do_nothing_policy(policy_csv):
    oracle = [r for r in policy_csv if r["policy"] == "oracle"]
    assert all(r["success"] == "True" and r["recovered"] == "True" and r["separable"] in ("True", "False") for r in oracle)
    # it diagnoses on every start, whatever the independent label says
    assert all(r["pass"] == "True" and r["diagnosis_all_correct"] == "True" and r["abstained"] == "False" for r in oracle)
    rows = [{"policy": r["policy"], "seed": r["seed"], "reward": float(r["reward"]) if r["reward"] else None} for r in policy_csv]
    assert ordering_holds(rows)


def test_on_the_base_row_the_oracles_calls_normalize_the_cost_so_every_reward_is_a_number():
    """On the base row every price is zero and the cost runs in calls, so the normalizer is the verifier's
    cost of the oracle episode, as the drivers and the gym take it, and every policy gets a reward, the oracle's above the
    do nothing policy's."""
    _facts, cycles, blocks = load_world()
    config = load_config("base")[1]
    table = policy_table(cycles, blocks)
    rewards = {name: play(cycles, blocks, config, table[name], 0)[1] for name in POLICIES}
    assert all(isinstance(reward, float) for reward in rewards.values()), rewards
    assert rewards["oracle"] > rewards["do_nothing"]


def test_without_out_the_rows_go_to_an_ignored_runs_folder_of_their_asset():
    """The rows are measured values, so without --out they go to the runs folder of their asset, which a rule of .gitignore leaves out of
    the repository, the rig's and a port's alike, and never beside the generated tables of the results folder."""
    rules = ["CausalDiagnosis/" + line.strip().rstrip("/")
             for line in (ROOT / "CausalDiagnosis" / ".gitignore").read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.startswith(("#", "!"))]
    for asset_id in (DEFAULT_ASSET, "a_port"):
        folder = default_out(asset_id, "static").parent.relative_to(ROOT).as_posix()
        assert folder == "CausalDiagnosis/results/runs" or folder.startswith("CausalDiagnosis/results/runs/"), folder
        assert any(fnmatch.fnmatch(folder, rule) or folder.startswith(rule + "/") for rule in rules), folder
