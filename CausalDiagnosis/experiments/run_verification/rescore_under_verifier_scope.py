"""Scores finished gym runs again under the verifier scope "all" from the logs they recorded, for the verifier scope
ablation of the paper (En4), whose arm is the runs of the five main rows scored again (CausalDiagnosis/docs/experiments.md).

Every episode keeps its hidden log, its released evidence, its predictions, its grade, the cost of the oracle policy and
its judge record. The script adds the sensor reads of the greedy policy's episode on the same seed and configuration,
which run_episodes.greedy_reference computes as the driver does for a run started with --verifier-scope all, and runs
the verifier under that scope, so a rescored episode carries the verdict such a run would have written while no agent
plays again. The oracle policy's cost has to equal the one the run recorded, or the code that plays it changed since the
run and the script stops.

The run folder stays as it is. Its rescored copy, the run folder's name with _scope_all, sits beside it and holds every
file of the run, with run.json, scores.csv, summary.md and the hidden oracle, verdict and score files of every episode
written again, and the run's ledger with one event that names the run it came from and every file the rescoring changed.
The copy is built in a hidden folder beside the run and takes its name only when every episode is scored. A run that has
not finished, a run without a ledger or whose files differ from its ledger, a run of another mode than gym, a run already
scored under the scope all and a run under a table whose means never read the start's block, which is a line of En15 and no main
row, are refused.

With --own-scope the script scores a run again under its own verifier scope with the verifier of this tree, the class scoring
of the registered analysis. The copy is the run folder's name with _own_scope. Its block rescored names the run's own scope and
own_scope true, and its code record holds the run's lists with the list scoring of this tree and the identity computed again from
them, so results_index reads the copy in place of the run when that scoring list is the one of the runs of record it joins
(results_index.rescored_refusal). A run that records no lists of its code is refused. Under either scope the judge cell of an
episode comes from its hidden/judge.json when the judge wrote one, summarized again by judge.summary_of, so a pass of the judge
after the run, as judge_finished_runs.py makes, is kept.

usage: python -m CausalDiagnosis.experiments.run_verification.rescore_under_verifier_scope [--own-scope] RUN_FOLDER [RUN_FOLDER ...]
"""
import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

from CausalDiagnosis.evaluation.metrics import judge_records  # noqa: E402
from CausalDiagnosis.utils import frameworks  # noqa: E402
from CausalDiagnosis.experiments.run_verification import run_ledger as ledger  # noqa: E402
from CausalDiagnosis.experiments.episode_runners import run_episodes  # noqa: E402
from CausalDiagnosis.experiments.run_verification import verify_results_folder as verify_results  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import COLUMNS, HIDDEN, PREDICTIONS_FILE, RELEASED_FILE, SCORE_COLUMNS, episode_dirs, episode_folder, episode_reward, read_rows  # noqa: E402
from CausalDiagnosis.utils.csv_rows import flat, write_rows  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.run_configuration import load_world  # noqa: E402
from CausalDiagnosis.utils.json_files import read_json, write_json  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.audit_reports import run_summary  # noqa: E402
from CausalDiagnosis.datasets.dataset_loader import recordings as records  # noqa: E402
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import verifier  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig  # noqa: E402

SCOPE = "all"
SUFFIX = "_scope_all"
OWN_SUFFIX = "_own_scope"
# one loaded world per asset and data file for the process: main takes several runs, and the classifiers of run_episodes are cached by
# the identity of the cycles object, so every run of a call shares the cycles and blocks of the first, and the data file is hashed once
WORLDS = {}
TOOL = "rescore_scope.py"
# the code that makes the change, hashed into the ledger event: this script, the driver's references and score row, the
# two reference policies and the environment they play, the verifier, and the module of every declared judge, whose summary of a
# record the copy recomputes
CODE = ("CausalDiagnosis/experiments/run_verification/rescore_under_verifier_scope.py", "CausalDiagnosis/experiments/episode_runners/run_episodes/__init__.py", "CausalDiagnosis/experiments/episode_runners/run_episodes/agent_kinds.py", "CausalDiagnosis/experiments/episode_runners/run_episodes/model_backends.py", "CausalDiagnosis/experiments/episode_runners/run_episodes/reference_policy_episodes.py", "CausalDiagnosis/experiments/episode_runners/run_episodes/episode_scoring.py", "CausalDiagnosis/experiments/episode_runners/run_episodes/static_agent.py", "CausalDiagnosis/experiments/episode_runners/run_episodes/run_record.py", "CausalDiagnosis/experiments/episode_runners/run_episodes/play_one_episode.py", "CausalDiagnosis/experiments/episode_runners/run_episodes/harness_rule.py", "CausalDiagnosis/experiments/episode_runners/run_episodes/agent_runs.py", "CausalDiagnosis/experiments/episode_runners/run_episodes/planned_episodes.py", "CausalDiagnosis/experiments/episode_runners/run_episodes/start_check.py", "CausalDiagnosis/experiments/episode_runners/run_episodes/runner_setup.py", "CausalDiagnosis/experiments/episode_runners/run_episodes/command_line.py", "CausalDiagnosis/experiments/episode_runners/run_episodes/__main__.py", "CausalDiagnosis/environment/reference_policies/policies.py",
        "CausalDiagnosis/environment/reference_policies/greedy_policy.py", "CausalDiagnosis/environment/reference_policies/tier_classifiers.py", "CausalDiagnosis/environment/Causal_Diagnosis_Environment/env.py", "CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/__init__.py", "CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/episode_log.py", "CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/episode_costs.py", "CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/episode_reward.py", "CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/failure_class.py", "CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/summary_consistency.py", "CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/episode_checks.py", "CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/delays.py", "CausalDiagnosis/environment/Causal_Diagnosis_Environment/verifier/verdict.py",
        *(entry["entry"].partition(":")[0].replace(".", "/") + ".py" for entry in frameworks.judges().values() if "entry" in entry))


def refusal(run_dir, own_scope=False):
    """Why a run folder cannot be rescored, or None. A run under the scope all can be scored again under its own scope."""
    meta = read_json(run_dir / "run.json")
    if not meta or not meta.get("ended"):
        return "it holds no run.json with an end time, so it has not finished"
    if ledger.run_kind(meta) is not None or meta.get("mode") != "gym":
        return f"it is a {meta.get('kind') or meta.get('mode')} run, and only a gym run has the logs the verifier reads"
    if not own_scope and (meta.get("config") or {}).get("verifier_scope") == SCOPE:
        return f"it was scored under the scope {SCOPE} already"
    if own_scope and not (meta.get("code") or {}).get("files"):
        return "it records no lists of its code, so a copy cannot name the scoring list of this tree beside them"
    state, problems = verify_results.check_ledger(run_dir)
    if state != "checked":
        return f"its ledger cannot be checked ({state})"
    if problems:
        return f"its files differ from its ledger ({problems[0]})"
    return None


def world(meta, config):
    """The cycles and blocks of the asset that the reference policies play, refused when the facts file here is not the one
    the run read. The data file is the one the loader of the asset reads, and its world is loaded and hashed once per process."""
    facts_path = Path(records.default_facts_path(get_profile(config.asset_id)))
    key = (config.asset_id, str(facts_path))
    if key not in WORLDS:
        _, cycles, blocks = load_world(facts_path, config.asset_id)
        WORLDS[key] = (cycles, blocks, ledger.sha256_of(facts_path))
    cycles, blocks, digest = WORLDS[key]
    if not digest.startswith(meta.get("facts_sha256", "")):
        raise SystemExit(f"{facts_path} is not the facts file the run read, so its reference policies cannot be played again")
    return cycles, blocks


def judge_of(hidden, score):
    """The judge cell of an episode: the record of hidden/judge.json with its summary computed again over its calls when the
    judge wrote one, which a pass of the judge after the run may have written, else the judge of score.json."""
    record = read_json(hidden / judge_records.JUDGE_NAME)
    if not record:
        return score.get("judge")
    return {**record, **judge_records.summary_of(record["calls"], record["calls_planned"])}


def rescore_episode(ep_dir, episode_config, cycles, blocks, seed, scope=SCOPE):
    """Writes the verdict under the scope of one recorded episode over its hidden oracle, verdict and score files and returns
    the cells of scores.csv it changes, or None for an episode the verifier could not read when the run scored it."""
    hidden = ep_dir / HIDDEN
    stored, score = read_json(hidden / "verdict.json"), read_json(hidden / "score.json")
    if not (stored or {}).get("verified") or score is None:
        return None
    cost_oracle = run_episodes.oracle_reference(cycles, blocks, episode_config, seed)
    if cost_oracle != stored["cost_oracle"]:
        raise SystemExit(f"{ep_dir}: the oracle policy costs {cost_oracle} here and cost {stored['cost_oracle']} in the run, "
                         "so the code that plays it changed since the run")
    oracle = read_json(hidden / "oracle.json")
    if scope == SCOPE:  # the scope all compares an agent's sensor reads with those of the greedy policy
        oracle = {**oracle, "reads": run_episodes.greedy_reference(cycles, blocks, episode_config, seed)}
    verdict = verifier.verify(read_json(hidden / "log.json"), read_json(ep_dir / RELEASED_FILE, {}), oracle, cost_oracle,
                              grade=stored.get("grade"), scope=scope, predictions=read_json(ep_dir / PREDICTIONS_FILE))
    judge = judge_of(hidden, score)
    row = run_episodes.verdict_row(verdict, episode_reward(verdict, episode_config.horizon), cost_oracle, judge)
    write_json(hidden / "oracle.json", oracle)
    write_json(hidden / "verdict.json", verdict)
    write_json(hidden / "score.json", {**row, "scope": scope, "grade_source": score.get("grade_source"), "judge": judge})
    return {**{column: flat(row[column]) for column in SCORE_COLUMNS}, "refusals": flat(verdict.get("refusals"))}


def scoped_row(row, cells, name, scope=SCOPE):
    """A row of the copy's scores.csv: the run's row with the cells its episode changed under the scope, the copy's name and
    the scope."""
    return {**row, **cells, "run": name, "verifier_scope": scope}


def scoped_meta(meta, source, episodes, scope=SCOPE, own_scope=False):
    """run.json of the copy: the run's under the scope, with the run it came from, the SHA-256 of that run's ledger and whether
    the scope is the run's own. A copy under the run's own scope holds the run's lists with the list scoring of this tree and
    the identity computed again from them."""
    copied = {**meta, "verifier_scope": scope, "config": {**meta["config"], "verifier_scope": scope},
              "rescored": {"from": source.name, "from_ledger_sha256": ledger.sha256_of(source / ledger.LEDGER), "scope": scope,
                           "own_scope": own_scope, "episodes": episodes, "tool": f"CausalDiagnosis/{TOOL}", "at": ledger.now()}}
    if own_scope:
        files = {**meta["code"]["files"], "scoring": ledger.code_identity.tree_files().get("scoring", {})}
        copied["code"] = {**meta["code"], "files": files, "identity": ledger.identity_of(files, meta.get("agent"))}
    return copied


def rescore_copy(copy, source, meta, scope=SCOPE, own_scope=False):
    """Rescores every episode of a copy of the run in place and returns the number rescored."""
    config = EpisodeConfig.from_json({**meta["config"], "verifier_scope": scope})
    cycles, blocks = world(meta, config)
    plan = {name: (seed, episode_config) for name, seed, _, episode_config in run_episodes.episode_plan(config, meta["seeds"], cycles, blocks)}
    before = ledger.snapshot(copy)
    cells = {}
    for ep_dir in episode_dirs(copy):
        if ep_dir.name not in plan:
            raise SystemExit(f"{source.name}/{ep_dir.name} is no episode of the run's plan")
        seed, episode_config = plan[ep_dir.name]
        changed = rescore_episode(ep_dir, episode_config, cycles, blocks, seed, scope)
        if changed is not None:
            cells[ep_dir.name] = changed
    rows = [scoped_row(row, cells.get(episode_folder(copy, row).name, {}), copy.name, scope) for row in read_rows(copy)]
    write_json(copy / "run.json", scoped_meta(meta, source, len(cells), scope, own_scope))
    write_rows(copy / "scores.csv", rows, COLUMNS)
    (copy / "summary.md").write_text(run_summary(copy), encoding="utf-8")
    reads = "the greedy policy's sensor reads and " if scope == SCOPE else ""
    ledger.record_changes(copy, before, TOOL, f"copy of the run {source.name}, every episode verified again under the verifier "
                          f"scope {scope} from the log it recorded, with {reads}the run's grade, cost of the oracle policy, "
                          "predictions and judge records", CODE)
    return len(cells)


def rescore_run(run_dir, own_scope=False):
    """The rescored copy of one run beside it and the number of episodes rescored, under the scope all, or with own_scope under
    the run's own scope."""
    run_dir = Path(run_dir).resolve()
    why = refusal(run_dir, own_scope)
    if why:
        raise SystemExit(f"{run_dir.name} is not rescored: {why}")
    scope = read_json(run_dir / "run.json")["config"]["verifier_scope"] if own_scope else SCOPE
    target = run_dir.with_name(run_dir.name + (OWN_SUFFIX if own_scope else SUFFIX))
    staging = run_dir.with_name(f".{target.name}.partial")
    for path in (target, staging):
        if path.exists():
            raise SystemExit(f"{path} exists already, so {run_dir.name} is not rescored again")
    copy = staging / target.name
    shutil.copytree(run_dir, copy)
    try:
        n = rescore_copy(copy, run_dir, read_json(copy / "run.json"), scope, own_scope)
    except BaseException:
        print(f"the rescoring of {run_dir.name} stopped, and its unfinished copy stays in {staging}", file=sys.stderr)
        raise
    copy.rename(target)
    staging.rmdir()
    return target, n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path, help="finished gym run folders, scored under the scope writes unless --own-scope")
    ap.add_argument("--own-scope", action="store_true", help="score each run again under its own scope with the verifier of this tree")
    a = ap.parse_args(argv)
    for run_dir in a.runs:
        target, n = rescore_run(run_dir, a.own_scope)
        print(f"{run_dir.name}: {n} episodes rescored into {target}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
