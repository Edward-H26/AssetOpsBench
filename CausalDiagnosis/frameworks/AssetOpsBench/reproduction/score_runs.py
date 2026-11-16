"""Score AssetOpsBench reproduction runs two ways (CausalDiagnosis/frameworks/AssetOpsBench/reproduction/README.md): the paper's rubric judge through the fork's
Evaluator under two protocols, and the efficiency counts the driver recorded.

usage, from the fork root with the fork's virtual environment first on PATH:
  python -m CausalDiagnosis.frameworks.AssetOpsBench.reproduction.score_runs RUN_DIR [RUN_DIR ...] [--scenarios FILE] [--judge-model NAME --judge-base-url URL]
      [--judge-protocols paper single] [--workers 4] [--rejudge-failed]

Judge. The scorer is AssetOpsBench's rubric judge on the whole trajectory (full_trajectory_scorer.py beside the rig's
drivers), run through the fork's own Evaluator on the trajectory record the runner
persisted and on the pinned scenario file, which holds the question and the characteristic form the judge compares against.
Every episode is judged under two protocols, the paper's (section 5.2), five calls at temperature 0.3 whose average it
reports, and this reproduction's, one call at temperature 0, the setting of the project's rig judge. Every call returns the six criteria
of the fork's rubric, whose first three, task completion, data retrieval accuracy and result verification, are the paper's
Figure 5 scores. Per episode and protocol, a criterion's rate is its share of true answers over the planned calls. A call
that failed (a backend error, among them an overflow of the judge's context, or a reply without a parsable review) counts as false on every criterion with the denominator
kept, as the paper-era tally did, and the failed calls are counted apart. The majority of the calls is kept beside the mean,
since the paper-era Agent-As-Tool code took a majority vote where the paper's text says average. An episode whose run raised
has no trajectory record and scores false on every criterion without a judge call. judge_paper.json and judge_single.json
in the episode folder hold every call. An episode already judged by the same model under a protocol is not judged again
under it, unless --rejudge-failed is given and its verdict records failed calls, as after a judge crash, in which case that
verdict is replaced. The verdict of another judge is renamed to judge_<protocol>.<model>.json before the new one is written.
The judge is reached through the OpenAI compatible endpoint that --judge-base-url names, with the key in API_KEY, as the
agent is, so that no trajectory leaves for a default endpoint, and must be another
model than the agent's. The request fields the judge backend adds, its thinking switch, are kept in every verdict and in
scores.csv. scores.csv also holds per criterion the agreement of the parsed calls, the share of their pairs that answered
alike, since the paper prints an agreement over five judge runs (Table 14) without a formula.

Efficiency. The steps and the wall clock of every episode, as the driver recorded them.

The output is scores.csv in every run folder, one row per episode with both.
"""
import argparse
import itertools
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from CausalDiagnosis.frameworks.AssetOpsBench.reproduction.run_folder_names import EPISODE_FILE, RUN_FILE, SCORES_FILE, episode_dirs
from CausalDiagnosis.frameworks.AssetOpsBench.reproduction.scenario_file import sha256_of

from CausalDiagnosis.frameworks.AssetOpsBench.LLM_Judges import episode_judge as rig_judge  # noqa: E402  the hydraulic rig's judge module: FixedTemperature, PROTOCOLS, router_backend
from CausalDiagnosis.utils.json_files import read_json, write_json
from CausalDiagnosis.utils.csv_rows import write_rows  # noqa: E402
from CausalDiagnosis.experiments.run_verification import run_ledger as ledger  # noqa: E402  the rig's run ledger, whose events record every scoring with the scorer code
from CausalDiagnosis.frameworks.AssetOpsBench.reproduction.ledger_fields import SCORER_CODE  # noqa: E402
from CausalDiagnosis.experiments.run_verification.public_record import public_argv, read_private  # noqa: E402  the public form of an event, the private file of a run
from CausalDiagnosis.datasets.dataset_loader.recordings import resolve_recorded_path  # noqa: E402  a public path resolved on this machine
from evaluation.evaluator import Evaluator  # noqa: E402
from CausalDiagnosis.frameworks.AssetOpsBench.LLM_Judges import full_trajectory_scorer  # noqa: E402  the released rubric judge on the whole trajectory, beside the rig's drivers
from evaluation.scorers.llm_judge import _RUBRIC_KEYS as CRITERIA  # noqa: E402  the six criteria of the fork's rubric

judge_backend = rig_judge.router_backend  # the judge's OpenAI compatible backend, replaced by the scorer tests
PROTOCOLS = rig_judge.PROTOCOLS  # paper: five calls at temperature 0.3, single: one call at temperature 0
EPISODE_FIELDS = ("episode", "agent", "scenario_id", "family", "task_kind", "repeat", "error", "runtime_s", "steps",
                  "tool_calls", "plan_steps", "model_calls", "input_tokens", "output_tokens")
JUDGE_FIELDS = ("judge_model", "judge_request", "judge_status", "judge_failed", *CRITERIA, *(f"{key}_majority" for key in CRITERIA),
                *(f"{key}_agreement" for key in CRITERIA), "fork_pass")
ROW_FIELDS = ("run", *EPISODE_FIELDS,
              *(f"{protocol}_{field}" for protocol in sorted(PROTOCOLS) for field in JUDGE_FIELDS))


def judge_file(protocol):
    return f"judge_{protocol}.json"


# ----- judge -----

def judge_call(record, scenarios, scenario_id, judge_model):
    """One call of the installed judge through the fork's Evaluator on one trajectory record."""
    report = Evaluator(default_scorer="llm_judge", judge_model=judge_model).evaluate(
        trajectories_path=Path(record), scenarios_paths=[Path(scenarios)], scenario_ids=[str(scenario_id)])
    if not report.results:
        raise SystemExit(f"the Evaluator joined no scenario to {record}")
    score = report.results[0].score
    details = score.details or {}
    return {"criteria": {key: details.get(key) is True for key in CRITERIA}, "passed": bool(score.passed), "score": score.score,
            "failed": not any(key in details for key in CRITERIA), "rationale": score.rationale}


def summarize(calls, planned):
    """Per criterion the share of true answers over the planned calls and whether they are the majority. A failed call, or
    a call not made because the run left no record, counts as false."""
    counts = {key: sum(c["criteria"][key] for c in calls) for key in CRITERIA}
    return {"failed": sum(c["failed"] for c in calls), "mean": {key: counts[key] / planned for key in CRITERIA},
            "majority": {key: counts[key] > planned / 2 for key in CRITERIA},
            "fork_pass": sum(c["passed"] for c in calls) / planned}


def judge_run(episodes, scenarios, backend, protocol, base_url, pool, rejudge_failed=False, probe=None):
    """Writes judge_<protocol>.json for every episode not yet judged by this model under this protocol, and with
    rejudge_failed also for every episode whose verdict by this model records failed calls, whose scorer, with the
    protocol's temperature, the caller has installed. probe is the health probe of the judge's endpoint that the caller names
    (judge.endpoint_probe): a failed attempt is sent again while it answers, and the first episode with a failed call after it
    found the endpoint gone ends the scoring with ServiceDown before its verdict is written, since the scorer keeps the
    ServiceDown of a call as a failed review that would count false on every criterion."""
    planned, temperature = PROTOCOLS[protocol]["calls"], PROTOCOLS[protocol]["temperature"]
    pending = []
    for ep_dir, episode in episodes:
        path = ep_dir / judge_file(protocol)
        old = read_json(path, {})
        same_judge = old.get("judge_model") == backend.model_id
        if same_judge and not (rejudge_failed and old.get("failed")):
            continue
        if old and not same_judge:  # the verdict of another judge stays beside the new one under its own name
            path.rename(ep_dir / f"judge_{protocol}.{old['judge_model'].replace('/', '_')}.json")
        record = ep_dir / episode["trajectory_record"] if episode.get("trajectory_record") else None
        futures = [] if record is None else [pool.submit(judge_call, record, scenarios, episode["scenario_id"], backend.model_id)
                                             for _ in range(planned)]
        pending.append((ep_dir, record, futures))
    for ep_dir, record, futures in pending:
        calls = [future.result() for future in futures]
        if probe is not None and probe.down and any(call["failed"] for call in calls):
            from CausalDiagnosis.utils.model_request_retry import ServiceDown  # here and not at the top: the agent package loads its runners on import

            pool.shutdown(wait=False, cancel_futures=True)
            raise ServiceDown(f"the judge endpoint {base_url} no longer answers its health probe, so {ep_dir.name} and the "
                              "episodes after it are not scored")
        verdict = {"format": 1, "judge_model": backend.model_id, "base_url": None,
                   "judge_request": getattr(backend, "extra_body", None), "protocol": protocol,
                   "temperature": temperature, "calls_planned": planned, "status": "judged" if record else "no_record",
                   "calls": calls, **summarize(calls, planned)}
        write_json(ep_dir / judge_file(protocol), verdict)
        print(f"{ep_dir.name} {protocol}: task completion {verdict['mean']['task_completion']:.2f} over {planned} calls, "
              f"{verdict['failed']} failed, {verdict['status']}", flush=True)


# ----- rows -----

def pairwise_agreement(calls, key):
    """The share of pairs of parsed calls that answered alike on one criterion, or None below two parsed calls."""
    pairs = list(itertools.combinations((c["criteria"][key] for c in calls if not c["failed"]), 2))
    return sum(a == b for a, b in pairs) / len(pairs) if pairs else None


def judge_columns(verdict):
    request = verdict.get("judge_request")  # absent from verdicts written before the field existed
    return {"judge_model": verdict["judge_model"], "judge_request": json.dumps(request, sort_keys=True) if request else "",
            "judge_status": verdict["status"], "judge_failed": verdict["failed"],
            **verdict["mean"], **{f"{key}_majority": value for key, value in verdict["majority"].items()},
            **{f"{key}_agreement": pairwise_agreement(verdict["calls"], key) for key in CRITERIA},
            "fork_pass": verdict["fork_pass"]}


def score_row(meta, ep_dir, episode):
    row = {"run": meta["run"], **{key: episode.get(key) for key in EPISODE_FIELDS}, "error": episode.get("error") or ""}
    for protocol in PROTOCOLS:
        verdict = read_json(ep_dir / judge_file(protocol))
        if verdict:
            row.update({f"{protocol}_{key}": value for key, value in judge_columns(verdict).items()})
    return row


def load_run(run_dir, scenarios_arg):
    """(folder, run.json, scenario file, [(episode folder, episode.json)]). The scenario file must be the run's own."""
    run_dir = Path(run_dir)
    meta = read_json(run_dir / RUN_FILE)
    if meta is None:
        raise SystemExit(f"{run_dir} holds no {RUN_FILE}")
    recorded = ((read_private(run_dir).get("starts") or [{}])[0].get("data") or {}).get("scenarios")
    scenarios = Path(scenarios_arg or recorded or resolve_recorded_path(meta["scenarios"]["path"]) or meta["scenarios"]["path"])
    if not scenarios.exists() or sha256_of(scenarios) != meta["scenarios"]["sha256"]:
        raise SystemExit(f"{run_dir.name} ran on the scenario file with SHA-256 {meta['scenarios']['sha256']}. "
                         f"Pass that file with --scenarios ({scenarios} is absent or differs)")
    return run_dir, meta, scenarios, [(d, read_json(d / EPISODE_FILE)) for d in episode_dirs(run_dir)]


def main(argv):
    ap = argparse.ArgumentParser(description="score AssetOpsBench reproduction runs: judge and efficiency")
    ap.add_argument("runs", nargs="+", help="run folders written by run_scenarios.py")
    ap.add_argument("--scenarios", help="the scenario file of the runs, when the path in run.json is not on this machine")
    ap.add_argument("--judge-model", help="the judge model at the endpoint. Without it only efficiency is scored")
    ap.add_argument("--judge-base-url", help="OpenAI compatible endpoint of the judge, required with --judge-model")
    ap.add_argument("--judge-protocols", nargs="+", choices=sorted(PROTOCOLS), default=sorted(PROTOCOLS))
    ap.add_argument("--workers", type=int, default=4, help="judge calls in flight at once")
    ap.add_argument("--rejudge-failed", action="store_true",
                    help="judge again every episode whose verdict by this judge records failed calls, and replace that verdict")
    a = ap.parse_args(argv)
    if a.judge_model and not a.judge_base_url:
        ap.error("--judge-model needs --judge-base-url")
    runs = [load_run(run, a.scenarios) for run in a.runs]
    before = {run_dir: ledger.snapshot(run_dir) for run_dir, *_ in runs}
    if a.judge_model:
        backend = judge_backend(a.judge_model, a.judge_base_url)
        for _, meta, _, _ in runs:
            if backend.model_id.split("/", 1)[-1] == meta["model"]:
                raise SystemExit(f"self judging: the judge model {backend.model_id} is the agent model {meta['model']}")
        try:
            with rig_judge.endpoint_probe(a.judge_base_url) as probe, ThreadPoolExecutor(max_workers=a.workers) as pool:
                for protocol in a.judge_protocols:
                    # the scorer calls generate(prompt) without a temperature, so each protocol fixes its own on the backend
                    full_trajectory_scorer.install(rig_judge.FixedTemperature(backend, PROTOCOLS[protocol]["temperature"]))
                    for _, _, scenarios, episodes in runs:
                        judge_run(episodes, scenarios, backend, protocol, a.judge_base_url, pool, a.rejudge_failed, probe)
        except BaseException:
            # the verdicts written before the error enter the ledger now, else the next scoring takes them into its state before
            for run_dir, *_ in runs:
                ledger.record_changes(run_dir, before[run_dir], "score.py", "judged before an error stopped the scoring with the arguments "
                                      + " ".join(public_argv(argv)), code=SCORER_CODE)
            raise
    scored = {}
    for run_dir, meta, _, episodes in runs:
        rows = [score_row(meta, d, e) for d, e in episodes]
        write_rows(run_dir / SCORES_FILE, rows, ROW_FIELDS)
        ledger.record_changes(run_dir, before[run_dir], "score.py", "scored with the arguments " + " ".join(public_argv(argv)),
                              code=SCORER_CODE)
        print(f"{run_dir.name}: {len(rows)} episodes scored into {SCORES_FILE}", flush=True)
        scored[run_dir.name] = rows
    return scored


if __name__ == "__main__":
    main(sys.argv[1:])
