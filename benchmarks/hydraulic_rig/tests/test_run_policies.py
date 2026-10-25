"""benchmarks/hydraulic_rig/run_policies.py: 100 seeded episodes of each reference policy, timed and read back from its CSV;
every row reruns on its own, and the oracle bounds every policy."""
import csv
import time
from pathlib import Path

import pytest

from envs.hydraulic_rig.classifiers import TierClassifiers
from envs.hydraulic_rig.data import build_blocks, default_facts_path, load_cycles
from envs.hydraulic_rig.env import TinyHydraulicEnv
from envs.hydraulic_rig.policies import RandomPolicy, run_episode
from envs.hydraulic_rig.scoring import score
from run_policies import main as run_policies_main, policy_table

if not Path(default_facts_path()).exists():
    pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH", allow_module_level=True)

CYCLES = load_cycles(default_facts_path())
BLOCKS = build_blocks(CYCLES)


@pytest.fixture(scope="module")
def policy_csv(tmp_path_factory):
    """Acceptance test: benchmarks/hydraulic_rig/run_policies.py with 100 seeded episodes per policy, timed, read back from its CSV."""
    out = tmp_path_factory.mktemp("runs") / "v0_scores.csv"
    started = time.perf_counter()
    run_policies_main(["--episodes", "100", "--out", str(out)])
    elapsed = time.perf_counter() - started
    with open(out, newline="") as f:
        return list(csv.DictReader(f)), elapsed


def test_run_policies_completes_100_seeded_episodes_of_each_policy_in_well_under_a_minute(policy_csv):
    rows, elapsed = policy_csv
    assert elapsed < 60
    for name in ("random", "oracle", "greedy"):
        assert sorted(int(r["seed"]) for r in rows if r["policy"] == name) == list(range(100))
    # score() returns the same fields in the same order for every log
    assert list(rows[0]) == ["policy", "seed", *score(run_episode(TinyHydraulicEnv(CYCLES, BLOCKS), RandomPolicy(0), 0))]


def test_every_row_of_the_policy_csv_reruns_on_its_own(policy_csv):
    rows, _ = policy_csv
    classifiers = TierClassifiers(CYCLES, BLOCKS)
    assert {r["policy"] for r in rows} == set(policy_table(TinyHydraulicEnv(CYCLES, BLOCKS), classifiers))
    for row in reversed(rows):  # last row first, each in a fresh environment, built by the table run_policies.py uses
        env, seed = TinyHydraulicEnv(CYCLES, BLOCKS), int(row["seed"])
        s = score(run_episode(env, policy_table(env, classifiers)[row["policy"]](seed), seed))
        assert {k: "" if v is None else str(v) for k, v in s.items()} == {k: row[k] for k in s}, (row["policy"], seed)


def test_the_oracle_bounds_every_policy_of_the_policy_csv(policy_csv):
    rows, _ = policy_csv
    oracle = {r["seed"]: r for r in rows if r["policy"] == "oracle"}
    for row in rows:
        best = oracle[row["seed"]]
        assert row["initial"] == best["initial"] and best["recovered"] == best["diagnosis_all_correct"] == "True"
        if row["recovered"] == "True":
            assert int(row["cost"]) >= int(best["cost"]), (row["policy"], row["seed"])
