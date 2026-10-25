"""Determinism audit, the pre-push check: the reference policies reproduce their episode logs exactly across fresh
interpreter processes.
usage: python3 -I benchmarks/hydraulic_rig/determinism_check.py [--facts PATH]

Runs the four policies of benchmarks/hydraulic_rig/run_policies.py (random, oracle, greedy_prior_1to2, greedy) on episode seeds 0 to 9 in
two fresh interpreter processes, each building its own world and classifiers, and compares the SHA-256 of every
episode's log written as sorted-key JSON. Prints OK when all 40 pairs match; otherwise lists the episodes that differ
and exits with status 1. Fresh processes matter because string hashing is salted per process (python3 -I ignores
PYTHONHASHSEED), so a result that depends on the iteration order of a set of strings can differ between processes
while two runs inside one process agree.

The reproducibility tests that already run inside one process:
- src/envs/hydraulic_rig/tests/test_env.py::test_reset_is_reproducible_and_hidden_state_is_not_observable, a seed gives the same hidden
  combination and the same first measurement;
- src/envs/hydraulic_rig/tests/test_env.py::test_a_seed_and_an_action_sequence_reproduce_the_log_exactly_including_the_draws_after_a_repair,
  the environment draws only from its own generator, also after a repair, and a seeded random policy reproduces its log;
- src/servers/hydraulic_rig/tests/test_server.py::test_every_call_reaches_env_step_and_replay_reproduces_the_log, replaying the calls
  recorded by the tool interface rebuilds the environment's log;
- src/envs/hydraulic_rig/tests/test_greedy.py::test_greedy_episodes_are_deterministic_legal_and_keep_budget_for_repairs, the greedy scores of
  20 seeds repeat."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # the fork's src/, also outside an editable install
from envs.hydraulic_rig.classifiers import TierClassifiers  # noqa: E402
from envs.hydraulic_rig.data import build_blocks, default_facts_path, load_cycles  # noqa: E402
from envs.hydraulic_rig.env import TinyHydraulicEnv  # noqa: E402
from envs.hydraulic_rig.policies import run_episode  # noqa: E402
from run_policies import policy_table  # noqa: E402

SEEDS = range(10)


def episode_hashes(facts):
    """{"<policy> seed <n>": SHA-256 of the episode's log} for every policy of run_policies.py and every seed."""
    cycles = load_cycles(facts)
    blocks = build_blocks(cycles)
    env = TinyHydraulicEnv(cycles, blocks)
    hashes = {}
    for name, make in policy_table(env, TierClassifiers(cycles, blocks)).items():
        for seed in SEEDS:
            log = run_episode(env, make(seed), seed)
            hashes[f"{name} seed {seed}"] = hashlib.sha256(json.dumps(log, sort_keys=True).encode()).hexdigest()
    return hashes


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default=str(default_facts_path()))
    ap.add_argument("--emit", action="store_true", help="one run: print the hashes as JSON (what the two processes do)")
    a = ap.parse_args(argv)
    if a.emit:
        print(json.dumps(episode_hashes(a.facts)))
        return 0
    command = [sys.executable, "-I", str(Path(__file__).resolve()), "--emit", "--facts", a.facts]
    first, second = (json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout) for _ in range(2))
    differing = sorted(k for k in first.keys() | second.keys() if first.get(k) != second.get(k))
    if differing:
        print(f"{len(differing)} of {len(first)} episodes differ between the two runs:")
        for k in differing:
            print(f"  {k}: {first.get(k)} against {second.get(k)}")
        return 1
    print(f"OK: {len(first)} episodes (4 policies on seeds 0 to {SEEDS[-1]}) give the same log SHA-256 in two fresh processes")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
