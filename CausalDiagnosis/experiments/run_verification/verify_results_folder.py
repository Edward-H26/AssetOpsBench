"""Checks that the tables of a results folder follow from their run folders, and writes VERIFY.md beside INDEX.json
(CausalDiagnosis/docs/experiments.md).

The ledger check recomputes the SHA-256 of every file of every run that INDEX.json lists and compares it with the run's
ledger, where a change counts only when an event of the ledger records it. The replay check (--replay) plays every episode
of every run again through the servers of this tree, whatever the mode of the run, by feeding the transcript of the episode to
them under the recorded clock, and compares every answer, the files the episode leaves, the verdict and the score cells of
scores.csv with those of the run (replay_check/). The table check runs tables_from_runs/
and leaderboard.py again on a scratch root of links to the runs, with the parameters, the cutoffs and the runs in
progress that the index recorded and the published subsets of the results folder, and compares every CSV file byte for
byte, every Markdown file apart from its date line, the LaTeX form of the main tables, and the index apart from the folders left unread, which the scratch
root does not hold. VERIFY.md names runs, files, rows and columns and never a value, so it may sit beside the tables.
--certify replays one run folder alone, with no results folder, and writes its replay certificate into it when the replay
passes on the store kind and the platform that the run recorded, on a clone of the run's own commit (replay_check/ certify).

usage: python -m CausalDiagnosis.experiments.run_verification.verify_results_folder --runs ROOT [ROOT ...] [--results CausalDiagnosis/results]
       [--run RUN_FOLDER] [--replay] [--out VERIFY.md]
       python -m CausalDiagnosis.experiments.run_verification.verify_results_folder --certify RUN_FOLDER
"""
from CausalDiagnosis.utils.paths import RESULTS
import argparse
import csv
import json
import os
import re
import sys
import tempfile
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent

from CausalDiagnosis.evaluation.result_tables import tables_from_runs as aggregate  # noqa: E402
from CausalDiagnosis.evaluation.result_tables import leaderboard  # noqa: E402
from CausalDiagnosis.experiments.run_verification import run_ledger as ledger  # noqa: E402
from CausalDiagnosis.experiments.run_verification import replay_check  # noqa: E402
from CausalDiagnosis.evaluation.result_tables import index_file, pooling_rules  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import DEFAULT_RUNS
from CausalDiagnosis.utils.json_files import read_json  # noqa: E402

# the ledger records itself through its events, which a synced copy keeps partly beside it, and no ledger records the names
# and folders that ledger.ignored() leaves out
IGNORED_NAMES = ledger.IGNORED_NAMES | {ledger.LEDGER, ledger.LOCAL_EVENTS}
IGNORED_DIRS = ledger.IGNORED_DIRS
# the states of a run that a replay does not play, with the words of the report
NOT_REPLAYED = {"no transcript": "not replayed, no transcript", "static": "not replayed, a static run calls no server"}
OTHER_KIND = "not a run of the rig"  # the state of the replay of a run of a declared kind, whose words its kind gives


# ----- the runs -----

def find_runs(names, roots):
    """({name: folder}, problems) of the run folders under the roots, looked up at a root and up to two levels below it (a
    grid folder, an archive of grid folders), the shallowest depth first. A name found twice at one depth or nowhere is a
    problem."""
    found, problems = {}, []
    for name in names:
        for depth in range(3):
            pattern = "/".join(["*"] * depth + [name])
            places = sorted({p.resolve() for root in roots for p in Path(root).glob(pattern) if (p / "run.json").exists()})
            if len(places) == 1:
                found[name] = places[0]
                break
            if places:
                problems.append(f"run {name}: {len(places)} folders of that name at one depth under the roots")
                break
        else:
            problems.append(f"run {name}: not found under the roots")
    return found, problems


def current_files(run_dir):
    """{path relative to the run folder: SHA-256} of every file a ledger covers."""
    run_dir = Path(run_dir)
    out = {}
    for path in sorted(run_dir.rglob("*")):
        relative = path.relative_to(run_dir)
        if path.is_file() and path.name not in IGNORED_NAMES and not IGNORED_DIRS & set(relative.parts):
            out[relative.as_posix()] = ledger.sha256_of(path)
    return out


def expected_files(record):
    """{path: SHA-256} that a ledger expects: the files of its episodes and of the run, then every event in order, which
    sets the new hash of a file it changed (None for a file it removed). Events on data files outside the run folder name
    them by the path of the data record (public_record.public_path; an absolute path in a ledger written before) and stay out."""
    files = {f"{name}/{path}": entry["sha256"] for name, episode in record["episodes"].items() for path, entry in episode["files"].items()}
    files.update({path: entry["sha256"] for path, entry in record["run_files"].items()})
    data_paths = {entry["path"] for entry in (record.get("data") or {}).values() if isinstance(entry, dict) and entry.get("path")}
    for event in record["events"]:
        for path, change in event["files"].items():
            if Path(path).is_absolute() or path in data_paths:
                continue
            if change["new"] is None:
                files.pop(path, None)
            else:
                files[path] = change["new"]
    return files


# ----- the four checks -----

def check_ledger(run_dir):
    """(state, problems) of one run: checked, no ledger (a run made before the ledger existed) or not finished."""
    run_dir = Path(run_dir)
    if not (run_dir / ledger.LEDGER).exists():
        return "no ledger", []
    record = ledger.load(run_dir)
    if record.get("status") != "finished":
        return "not finished", [f"{run_dir.name}: the ledger has the status {record.get('status')}"]
    want, have = expected_files(record), current_files(run_dir)
    problems = [f"{run_dir.name}/{path}: {'missing' if path not in have else 'changed'} without a recorded event"
                for path in sorted(want) if have.get(path) != want[path]]
    problems += [f"{run_dir.name}/{path}: added without a recorded event" for path in sorted(set(have) - set(want))]
    return "checked", problems


def check_replay(run_dir):
    """(state, problems) of the replay of every episode of one run through the servers of this tree (replay_check.replay_run), a
    loop episode of GEPA and a sampled episode of fine tuning alike, with the score cells of scores.csv compared for a run that has
    the file. A replay that passed on another store kind or platform than the run's says so in its state. A replay whose only
    differences are generated tables counts as identical, as replay_check.certify counts it, and its state names the tables."""
    result = replay_check.replay_run(run_dir)
    if result["state"] in NOT_REPLAYED:
        return NOT_REPLAYED[result["state"]], []
    if result["state"] == OTHER_KIND:
        return ledger.run_kind(read_json(Path(run_dir) / "run.json", {}) or {}).not_replayed, []
    problems = [] if result["problems"] == result["table_problems"] else list(result["problems"])
    tables = [name for name in result["data_differences"] if name not in replay_check.RECORDED_DATA] if not problems else []
    drift = code_drift(read_json(Path(run_dir) / "run.json", {}) or {})
    if problems and drift:
        problems.append(f"{Path(run_dir).name}: {len(drift)} of the code files the run recorded differ from the files here, so a difference may come "
                        f"from the code ({', '.join(drift[:5])}{', ...' if len(drift) > 5 else ''})")
    advisory = f"; advisory: {'; '.join(result['advisory'])}" if result["advisory"] else ""
    computed = f"; generated tables computed again with every answer and score cell equal: {', '.join(tables)}" if tables else ""
    return f"{result['episodes']} episodes replayed{computed}{advisory}", problems


def code_drift(meta):
    """The code files that a run recorded, in full under the lists of code.files or as the SHA-256 prefixes of a run before the
    lists, whose file here differs or is gone (replay_check.changed_files)."""
    recorded = ledger.recorded_code(meta)
    return sorted(path for paths in replay_check.changed_files(meta).values() for path in paths if path in recorded)


def csv_differences(name, ours, theirs):
    """Where two CSV files differ, by row and column and never by value."""
    def rows(path):
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.reader(f))
    a, b = rows(ours), rows(theirs)
    if not a or not b or a[0] != b[0]:
        return [f"{name}: the header differs"]
    if len(a) != len(b):
        return [f"{name}: {len(a) - 1} rows against {len(b) - 1} in the rebuild"]
    return [f"{name}: row {i} column {a[0][j]} differs" for i in range(1, len(a)) for j in range(len(a[0])) if a[i][j] != b[i][j]]


def markdown_lines(path, kind):
    """The lines of a Markdown file that must match: aggregate.md without its title line, which carries the date, and with
    the folder of the runs blanked, and LEADERBOARD.md without the line that carries the date."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if kind == "aggregate.md":  # the title follows the lines that open every page of a development or a registered root
        title = next((i for i, line in enumerate(lines) if line.startswith("# ")), 0)
        return [re.sub(r"^Run folders under `[^`]*`", "Run folders under `ROOT`", line) for line in lines[:title] + lines[title + 1:]]
    if kind == "LEADERBOARD.md":  # the date, and the runs left out, which the rebuild on the index's inputs alone cannot list
        return [line for line in lines if not line.startswith(("- Written on ", "- Left out of every table: "))]
    return lines


def rebuild(index, results, found, scratch):
    """tables_from_runs/ and leaderboard.py again in a scratch folder, on links to the runs, as the index recorded them."""
    root = scratch / index["runs_root"]
    root.mkdir()
    for entry in index["inputs"]:
        os.symlink(found[entry["run"]], root / entry["run"])
    p = index["parameters"]
    # the share of the test set that the tables read stands in the index, so the rebuild reads the runs as the tables did
    if p.get("fraction") is None:
        raise SystemExit("INDEX.json names no share of the test set that its tables read (parameters.fraction), so no rebuild "
                         "can read the runs as they did: build the tables again with aggregate.py")
    out = scratch / "results"
    aggregate.aggregate(root, out / "tables", p["resamples"], p["alpha"], p["seed"], out / "README.md", Path(results) / "published_runs", p["cluster"],
                        out / "INDEX.json", departure=bool(p.get("departure")), commit=p.get("commit"), fraction=p["fraction"])
    if index.get("leaderboard"):
        section = index["leaderboard"]
        runs = leaderboard.read_runs(out / "tables")
        (out / "LEADERBOARD.md").write_text(leaderboard.page(runs, section["since"], section["running"]) + "\n", encoding="utf-8")
        index_file.add_section(out / "INDEX.json", "leaderboard", leaderboard.index_section(runs, section["since"], section["running"]))
    return out


def check_tables(index, results, found):
    """Problems of the rebuild against the results folder."""
    results = Path(results)
    with tempfile.TemporaryDirectory() as scratch:
        out = rebuild(index, results, found, Path(scratch))
        # the runs of the index that the rebuild could not pool, a commit or a line that changed since the index was built
        problems = [f"{run}: left out of the rebuild, {reason}"
                    for run, reason in sorted(pooling_rules.pooling_left_out(Path(scratch) / index["runs_root"]).items())]
        names = sorted({p.name for p in (results / "tables").glob("*.csv")} | {p.name for p in (out / "tables").glob("*.csv")})
        for name in names:
            ours, theirs = results / "tables" / name, out / "tables" / name
            if not ours.exists() or not theirs.exists():
                problems.append(f"tables/{name}: {'missing in the results folder' if not ours.exists() else 'not written by the rebuild'}")
            elif ours.read_bytes() != theirs.read_bytes():
                problems += csv_differences(f"tables/{name}", ours, theirs)
        for name, kind in (("tables/aggregate.md", "aggregate.md"), ("README.md", "README.md"), ("LEADERBOARD.md", "LEADERBOARD.md"),
                           ("MAIN_TABLES.md", "MAIN_TABLES.md"), ("main_tables.tex", "main_tables.tex")):
            ours, theirs = results / name, out / name
            if ours.exists() != theirs.exists():
                problems.append(f"{name}: {'missing in the results folder' if not ours.exists() else 'not written by the rebuild'}")
            elif ours.exists() and markdown_lines(ours, kind) != markdown_lines(theirs, kind):
                problems.append(f"{name}: the text differs")
        rebuilt = json.loads((out / "INDEX.json").read_text(encoding="utf-8"))
        for section in sorted(set(index) | set(rebuilt)):
            if section != "excluded" and index.get(section) != rebuilt.get(section):
                problems.append(f"INDEX.json: the section {section} differs")
    return problems


# ----- the report -----

def verify(results, roots, run=None, replay=False):
    """Every check over a results folder, as (runs, sections): per run its ledger and replay state with the problems, and
    per check its summary and problems."""
    results = Path(results)
    index = json.loads((results / "INDEX.json").read_text(encoding="utf-8"))
    names = [entry["run"] for entry in index["inputs"]]
    if run and run not in names:
        raise SystemExit(f"--run {run} is no run of {results / 'INDEX.json'}, so nothing would be checked")
    found, missing = find_runs(names, roots)
    # every run of a registered root holds its ledger and the transcripts of its episodes, so a copy without them is a partial copy
    registered = index.get("root_kind") == "record"
    runs = {}
    for name in names:
        if name not in found or (run and name != run):
            continue
        state, problems = check_ledger(found[name])
        if registered and state == "no ledger":
            problems = [f"{name}: no ledger, which every run of a registered root holds"]
        replayed, replay_problems = check_replay(found[name]) if replay else ("not asked", [])
        if registered and replayed == NOT_REPLAYED["no transcript"]:
            replay_problems = [f"{name}: no episode holds a transcript, which every run of a registered root holds"]
        runs[name] = {"ledger": state, "ledger_problems": problems, "replay": replayed, "replay_problems": replay_problems}
    table_problems = check_tables(index, results, found) if not missing else ["the rebuild needs every run of the index"]
    ledger_problems = [p for r in runs.values() for p in r["ledger_problems"]]
    replay_problems = [p for r in runs.values() for p in r["replay_problems"]]
    states = [r["ledger"] for r in runs.values()]
    sections = {
        "runs": (f"{len(found)} of {len(names)} runs of the index found", missing),
        "ledger": (f"{states.count('checked')} runs checked, {states.count('no ledger')} without a ledger", ledger_problems),
        "replay": ("asked" if replay else "not asked, pass --replay", replay_problems),
        "tables": ("tables_from_runs.py and leaderboard.py run again on the same runs", table_problems),
    }
    return runs, sections


def report(results, runs, sections):
    index_hash = ledger.sha256_of(Path(results) / "INDEX.json")
    lines = ["# Verification of the results", "",
             f"Checked on {date.today().isoformat()} by `CausalDiagnosis/experiments/run_verification/verify_results_folder.py` against `INDEX.json` "
             f"(SHA-256 {index_hash}).", "", "| Check | What ran | Result |", "|---|---|---|"]
    lines += [f"| {name} | {summary} | {'identical' if not problems else f'{len(problems)} differences'} |" for name, (summary, problems) in sections.items()]
    lines += ["", "| Run | Ledger | Replay | Result |", "|---|---|---|---|"]
    lines += [f"| `{name}` | {r['ledger']} | {r['replay']} | {'identical' if not r['ledger_problems'] + r['replay_problems'] else 'differences below'} |"
              for name, r in sorted(runs.items())]
    lines += ["", "## Differences", ""]
    differences = [f"- {name}: {problem}" for name, (_, problems) in sections.items() for problem in problems]
    return "\n".join(lines + (differences or ["None."])) + "\n"


def main(argv):
    ap = argparse.ArgumentParser(description="check that the tables of a results folder follow from their run folders")
    ap.add_argument("--runs", nargs="+", default=[str(DEFAULT_RUNS)], help="folders that hold the run folders of the index")
    ap.add_argument("--results", default=str(RESULTS), help="the results folder with INDEX.json and the tables")
    ap.add_argument("--run", default=None, help="check the ledger and the replay of this run folder only, the tables still in full")
    ap.add_argument("--replay", action="store_true", help="replay every episode and score it again")
    ap.add_argument("--out", default=None, help="the report (default: VERIFY.md beside INDEX.json)")
    ap.add_argument("--certify", default=None, metavar="RUN_FOLDER",
                    help="replay one run folder through the servers of this tree and write its certificate into it when the replay passes and "
                         "counts: exit 0 with the certificate, 1 with a difference or a run that cannot be replayed, 3 when the replay passed on "
                         "another store kind or platform than the run's and counts nowhere, 5 when it passed on a tree that is no clone of the "
                         "run's commit, such as an upload, where no certificate can be written")
    a = ap.parse_args(argv)
    if a.certify:
        status, result, path = replay_check.certify(Path(a.certify))
        for line in result["problems"] + [f"advisory: {reason}" for reason in result["advisory"]]:
            print(line)
        print(f"{Path(a.certify).name}: {result['state']}, {result['episodes']} episodes, {result['lines']} transcript lines compared, "
              + (f"wrote {path}" if path else "no certificate"))
        return status
    runs, sections = verify(a.results, a.runs, a.run, a.replay)
    out = Path(a.out) if a.out else Path(a.results) / "VERIFY.md"
    out.write_text(report(a.results, runs, sections), encoding="utf-8")
    failed = sum(len(problems) for _, problems in sections.values())
    for name, (summary, problems) in sections.items():
        print(f"{name}: {summary}, {'identical' if not problems else f'{len(problems)} differences'}")
    print(f"wrote {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
