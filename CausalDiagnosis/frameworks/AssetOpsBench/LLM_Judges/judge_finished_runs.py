"""The AssetOpsBench judge over the episodes of finished runs whose judge did not answer, with scores.csv refreshed.

A run's judge column stays empty when the judge could not run at all while the episode ended (run_episodes.run_judge records
the failure instead of failing the episode), or when the run started without the in-run judge, and it reads False when every
call of the judge failed, as when the endpoint was down, since a failed call counts as false (judge.summary_of). This script
judges every such episode of a gym run of a model agent with episode_judge.py, which writes hidden/judge.json, and rewrites the
judge_pass cell of its row, so that the A1 table counts it. A run of the static agent (mode static, Sc3) is judged the same way in
the judge's static mode, over its answer.txt, so that table A3 counts it; a static episode without an answer stays unjudged, as in the
run. A run without run.json (not finished) or of a reference policy is skipped, and an episode whose record holds a parsed review
keeps its pass. A run whose run.json names no judge model, one started without the in-run judge as every grid line is, is
judged only under the overrides --judge-model and --judge-endpoint, which then judge every run. The run's ledger gets
one event with the old and the new hash of every file the judging changed, the served name of the judge and its weights commit.

The judge is pinned: with the endpoint file of its serving job (HYD_JUDGE_ENDPOINT_FILE) every record names the serving identity
of the endpoint, and a run of record is refused, every reason named with both values, when the judge differs from the judge block
of the tracked serving file (ledger.serving_of_record): the served name, and every field the file declares against its line of the
endpoint file (ledger.serving_differences); the GPU card is compared only when env.sh sets GPU_OF_RECORD (ledger.card_differences).
A development run is judged without the check.

Each episode is judged inside a guard: an episode the judge cannot judge, one whose runner left no trajectory.json among them,
is named on the output and in the ledger event, and the pass goes on, while a judge whose endpoint no longer answers its health
probe (ServiceDown) or gives no model listing (judge.ContextUnknown) ends the pass with that reason. scores.csv is written after
every judged episode, so a pass that stops keeps what it judged, and the ledger event is written on the way out.
--rejudge-failed judges again every record that holds a failed call. --workers N judges N episodes at once, in threads that
share the one backend, since every call of a pass installs the same scorer, one backend at the one temperature of the protocol.
A run whose agent model is the judge model, compared on the bare names (judge.same_model), is refused.

--count judges nothing and prints, over the runs a pass reads under the same options, "left judged failed calls failed_calls
not_judged refused": the episodes without a judge record that a pass can judge, those with a record, those whose record holds a
failed call, the calls of the records and the failed ones, the episodes without a record that no pass can judge (a static episode
without an answer, a gym episode without trajectory.json) and the episodes of the runs the pass refuses. --names prints those
episodes and runs by name. CausalDiagnosis/cluster_jobs/3_run/run_judge.sh reads both, so the judge job and the pass select the same episodes.

    python -m CausalDiagnosis.frameworks.AssetOpsBench.LLM_Judges.judge_finished_runs runs/<run folder> --model Qwen3.8-27B-FP8 --base-url http://host:port/v1
    python -m CausalDiagnosis.frameworks.AssetOpsBench.LLM_Judges.judge_finished_runs runs/<run folder> --judge-model Qwen3.8-27B-FP8 --judge-endpoint http://host:port/v1
"""
from CausalDiagnosis.utils.paths import SRC
import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(SRC)]
from CausalDiagnosis.frameworks.AssetOpsBench.LLM_Judges import episode_judge as judge  # noqa: E402
from CausalDiagnosis.experiments.run_verification import run_ledger as ledger  # noqa: E402
from CausalDiagnosis.evaluation.result_tables import pooling_rules  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import COLUMNS, episode_folder
from CausalDiagnosis.utils.json_files import read_json
from CausalDiagnosis.utils.csv_rows import write_rows  # noqa: E402

GYM_MODEL_AGENTS = ("plan_execute", "agent_as_tool", "causal")
STATIC_AGENT = "static"


def _calls(folder):
    return (read_json(folder / "hidden" / judge.JUDGE_NAME, {}) or {}).get("calls") or []


def unreviewed(folder):
    """Whether the judge record of an episode holds no parsed review, every call failed, which reads False under the rule of
    judge.summary_of and is judged again here, since the failures were the judge's."""
    calls = _calls(folder)
    return bool(calls) and not any(judge.reviewed(call) for call in calls)


def holds_failed_call(folder):
    """Whether the judge record of an episode holds a failed call, which --rejudge-failed judges again."""
    return any(not judge.reviewed(call) for call in _calls(folder))


def answered(folder):
    """Whether a static episode holds an answer for the judge, the run's own rule: an empty answer is not judged."""
    path = folder / judge.ANSWER_FILE
    return path.exists() and path.read_text(encoding="utf-8") != ""


def judgeable(folder, static):
    """Whether a pass can judge an episode that holds no judge record: a static episode with an answer, a gym episode whose runner
    left trajectory.json (judge.judge_episode raises NotJudged without it)."""
    return answered(folder) if static else (folder / judge.TRAJECTORY_FILE).is_file()


def selected(row, folder, static, failed):
    """Whether the pass judges an episode: its folder exists, its judge cell is empty or its record holds no parsed review, or
    with failed holds a failed call, and a static episode holds an answer."""
    if not folder.exists() or (static and not answered(folder)):
        return False
    return row.get("judge_pass") in ("", None) or unreviewed(folder) or (failed and holds_failed_call(folder))


def guarded(judged, folder, *args, **mode):
    """(the judge record, None) of one episode, or (None, why it failed): a failure of one episode never stops the pass, while a
    judge that cannot be asked, whose endpoint no longer answers its health probe (ServiceDown) or gives no model listing
    (judge.ContextUnknown), ends it with the reason, as run_episodes.run_judge does."""
    from CausalDiagnosis.utils.model_request_retry import ServiceDown  # here and not at the top: the agent package loads its runners on import

    try:
        return judged(folder, *args, **mode), None
    except (ServiceDown, judge.ContextUnknown):
        raise
    except Exception as error:  # noqa: BLE001, every other failure is named and the pass goes on
        return None, f"{type(error).__name__}: {error}"


def judgements(tasks, workers):
    """(row, record, failure) for every (row, call) task, judged one after the other, or by that many threads with workers above
    1, in the order they end. A pass that stops cancels the episodes not yet started."""
    if workers <= 1:
        for row, call in tasks:
            yield (row, *call())
        return
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {pool.submit(call): row for row, call in tasks}
        for future in as_completed(futures):
            yield (futures[future], *future.result())
    finally:
        pool.shutdown(cancel_futures=True)


def endpoint_identity(endpoint):
    """The serving identity of the judge endpoint that ledger.served met, from the endpoint file of its serving job, or None
    without that file."""
    return ledger.serving_identity(endpoint) if (endpoint or {}).get("endpoint_file") else None


def judge_refusal(meta, backend, endpoint):
    """Why the pass may not judge a run of record with this judge, every reason with both values, or None, and None for any run
    that is not of record: the judge block of the tracked serving file, its served name, the fields it declares against the
    endpoint file, and the GPU card when env.sh sets GPU_OF_RECORD."""
    if pooling_rules.kind_of(meta) != "record":
        return None
    try:
        declared = ledger.serving_endpoints.serving_of_record()["judge"]
    except (OSError, ValueError, KeyError) as error:
        return f"the serving file of record names no judge ({error})"
    if not (endpoint or {}).get("endpoint_file"):
        return "the judge endpoint has no endpoint file, so its serving cannot be compared with the serving file of record"
    reasons, name = [], judge.served_name(backend)
    if name != declared.get("served_name"):
        reasons.append(f"the judge {name} is not {declared.get('served_name')}, the judge of the serving file of record")
    reasons += [f"the {field} {seen} of the judge endpoint is not the {field} {wanted} of the serving file of record"
                for field, wanted, seen in ledger.serving_endpoints.serving_differences(endpoint, "judge")]
    reasons += ledger.serving_endpoints.card_differences(endpoint)
    return "; ".join(reasons) or None


def run_mode(meta):
    """static for a run of the static agent in mode static, gym for a gym run of a model agent, None for every other run, which no
    pass judges."""
    if meta.get("agent") == STATIC_AGENT and meta.get("mode") == "static":
        return "static"
    if meta.get("agent") in GYM_MODEL_AGENTS and meta.get("mode") == "gym":
        return "gym"
    return None


def read_by_pass(meta, unrecorded):
    """Whether a pass reads a finished run: a run it judges (run_mode) whose run.json names a judge model, or any such run under
    the overrides (unrecorded)."""
    return run_mode(meta) is not None and bool(meta.get("judge_model") or unrecorded)


def run_refusal(meta, backend, endpoint):
    """Why a pass refuses a run it reads, or None: its agent model is the judge (judge.same_model), or it is a run of record whose
    judge differs from the serving file of record (judge_refusal)."""
    agent_model, judge_model = meta.get("model_id") or meta.get("model"), getattr(backend, "model_id", None)
    if judge.same_model(judge_model, agent_model):
        return f"the judge {judge_model} is the agent model {agent_model}"
    return judge_refusal(meta, backend, endpoint)


def rejudge_run(run_dir, backend, protocol, base_url, judged=judge.judge_episode, unrecorded=False, failed=False, workers=1, endpoint=None):
    """Judge the episodes of one run whose judge_pass is empty or whose judge record holds no parsed review, or with failed
    every record that holds a failed call, and return the number judged: a gym run of a model agent, or a run of the static
    agent in the judge's static mode. A run whose run.json names no judge model is judged only when unrecorded is set, as the
    overrides of the command line set it, and a run whose agent model is the judge is refused, as is a run of record whose
    judge differs from the serving file of record (judge_refusal). endpoint is the judge endpoint as ledger.served met it, whose identity
    every record names. Every failure is named, the row of every judged episode is written at once, and workers above 1 judge
    that many episodes at once."""
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if not read_by_pass(meta, unrecorded):
        return 0
    static = run_mode(meta) == "static"
    agent_model, judge_model = meta.get("model_id") or meta.get("model"), getattr(backend, "model_id", None)
    if (refusal := run_refusal(meta, backend, endpoint)) is not None:
        print(f"{run_dir.name}: refused, {refusal}", flush=True)
        return 0
    identity = endpoint_identity(endpoint)
    mode = {**({"static": True} if static else {}), **({"serving": identity} if identity else {})}
    with open(run_dir / "scores.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    watched = [run_dir / "scores.csv", *(episode_folder(run_dir, row) / "hidden" / judge.JUDGE_NAME for row in rows)]
    before = {path: ledger.sha256_of(path) for path in watched}
    tasks = []
    for row in rows:
        folder = episode_folder(run_dir, row)
        if selected(row, folder, static, failed):
            tasks.append((row, partial(guarded, judged, folder, backend, protocol, agent_model, base_url, **mode)))
    count, failures = 0, []
    try:
        for row, record, failure in judgements(tasks, workers):
            name = episode_folder(run_dir, row).name
            if failure is not None:
                failures.append(f"{name} ({failure})")
                print(f"{run_dir.name}/{name}: not judged, {failure}", flush=True)
            elif record.get("pass") is not None:
                row["judge_pass"] = record["pass"]
                count += 1
                write_rows(run_dir / "scores.csv", rows, COLUMNS)
    finally:
        after = {path: ledger.sha256_of(path) for path in watched}
        changed = {path.relative_to(run_dir).as_posix(): (before[path], after[path]) for path in watched if after[path] != before[path]}
        if changed or failures:
            served = judge.served_name(backend) if backend is not None else judge_model
            what = (f"judged {count} episodes whose judge cell was empty{' or whose record held a failed call' if failed else ''}, "
                    f"judge {served} (weights commit {(identity or {}).get('weights_commit')}) at {base_url}, protocol {protocol}")
            ledger.append_event(run_dir, "rejudge.py", what + (f"; not judged: {', '.join(failures)}" if failures else ""), changed)
    return count


def tally(roots, backend, endpoint, unrecorded):
    """What is left to judge under the roots, as a pass with the same options selects it: {"left", "judged", "failed",
    "not_judged": [episode folders], "refused": [(run folder, reason, episodes)], "calls", "failed_calls": counts}. An episode with a
    judge record is judged, and failed when the record holds a failed call (holds_failed_call, the rule of --rejudge-failed); one
    without is left when a pass can judge it (judgeable) and not judged otherwise."""
    out = {"left": [], "judged": [], "failed": [], "not_judged": [], "refused": [], "calls": 0, "failed_calls": 0}
    for root in roots:
        for meta_path in sorted(Path(root).glob("*/run.json")):
            run_dir, meta = meta_path.parent, json.loads(meta_path.read_text(encoding="utf-8"))
            if not read_by_pass(meta, unrecorded):
                continue
            with open(run_dir / "scores.csv", newline="", encoding="utf-8") as f:
                folders = [folder for folder in (episode_folder(run_dir, row) for row in csv.DictReader(f)) if folder.exists()]
            if (refusal := run_refusal(meta, backend, endpoint)) is not None:
                out["refused"].append((run_dir, refusal, len(folders)))
                continue
            for folder in folders:
                calls = _calls(folder)
                if (folder / "hidden" / judge.JUDGE_NAME).is_file():
                    out["judged"].append(folder)
                    out["calls"] += len(calls)
                    out["failed_calls"] += sum(1 for call in calls if not judge.reviewed(call))
                    if holds_failed_call(folder):
                        out["failed"].append(folder)
                else:
                    out["left" if judgeable(folder, run_mode(meta) == "static") else "not_judged"].append(folder)
    return out


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("roots", nargs="+", help="folders that hold run folders")
    ap.add_argument("--model", help="the judge model, for the runs whose run.json names one")
    ap.add_argument("--base-url", help="the judge endpoint that goes with --model")
    ap.add_argument("--judge-model", help="override: the judge model of every run, including a run whose run.json names none")
    ap.add_argument("--judge-endpoint", help="override: the judge endpoint that goes with --judge-model")
    ap.add_argument("--protocol", choices=sorted(judge.PROTOCOLS), default="paper",
                    help="the paper's five calls at temperature 0.3, the default, or single, one call at temperature 0 for development")
    ap.add_argument("--rejudge-failed", action="store_true", help="judge again every episode whose judge record holds a failed call")
    ap.add_argument("--workers", type=int, default=1, help="episodes judged at once, in threads that share the judge backend")
    ap.add_argument("--count", action="store_true", help="judge nothing and print the counts of what a pass leaves (the docstring)")
    ap.add_argument("--names", action="store_true", help="judge nothing and print the episodes no pass can judge, those with a failed "
                                                         "call and the runs a pass refuses")
    a = ap.parse_args(argv)
    override = bool(a.judge_model or a.judge_endpoint)
    model, endpoint = (a.judge_model, a.judge_endpoint) if override else (a.model, a.base_url)
    if not (model and endpoint):
        ap.error("give --model with --base-url, or the overrides --judge-model with --judge-endpoint")
    backend = judge.router_backend(model, endpoint)
    served = ledger.served(model, endpoint, "HYD_JUDGE_ENDPOINT_FILE")  # the endpoint file of the serving job of the judge, once
    if a.count or a.names:
        left = tally(a.roots, backend, served, override)
        if a.count:
            print(len(left["left"]), len(left["judged"]), len(left["failed"]), left["calls"], left["failed_calls"], len(left["not_judged"]),
                  sum(n for _run, _why, n in left["refused"]))
        if a.names:
            print("".join(f"JUDGE_NOT_JUDGED_EPISODE {e}\n" for e in left["not_judged"]) + "".join(f"JUDGE_FAILED_EPISODE {e}\n" for e in left["failed"])
                  + "".join(f"JUDGE_REFUSED_RUN {run} {why}\n" for run, why, _n in left["refused"]), end="")
        return
    with judge.endpoint_probe(endpoint):
        for root in a.roots:
            for meta_path in sorted(Path(root).glob("*/run.json")):
                n = rejudge_run(meta_path.parent, backend, a.protocol, endpoint, unrecorded=override, failed=a.rejudge_failed,
                                workers=a.workers, endpoint=served)
                print(f"{meta_path.parent.name}: judged {n} episode(s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
