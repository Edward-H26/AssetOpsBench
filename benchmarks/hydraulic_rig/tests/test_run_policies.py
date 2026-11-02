"""benchmarks/hydraulic_rig/run_policies.py: the reference policies in process on seeded episodes, read back from the
CSV, every row verified, and the ordering check of RFC 008."""
import csv
import importlib.util
from pathlib import Path

import pytest

from envs.hydraulic_rig.data import default_facts_path

if not Path(default_facts_path()).exists():
    pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH", allow_module_level=True)
if importlib.util.find_spec("envs.hydraulic_rig.verifier") is None:
    pytest.skip("waiting for the verifier", allow_module_level=True)

from run_policies import POLICIES, main as run_policies_main, ordering_holds  # noqa: E402


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


def test_the_oracle_passes_every_seed_and_outranks_the_do_nothing_policy(policy_csv):
    oracle = [r for r in policy_csv if r["policy"] == "oracle"]
    assert all(r["pass"] == "True" and r["diagnosis_all_correct"] == "True" and r["recovered"] == "True" for r in oracle)
    rows = [{"policy": r["policy"], "seed": r["seed"], "reward": float(r["reward"]) if r["reward"] else None} for r in policy_csv]
    assert ordering_holds(rows)
