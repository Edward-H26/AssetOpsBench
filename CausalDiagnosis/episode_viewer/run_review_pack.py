"""The fixed starts of the walkthroughs and the review pack of a run folder.

starts writes fixed_starts.json beside this file. For every rules key of the separability label
(CausalDiagnosis/environment/reference_policies/diagnosability_label.py) that a preset of configs/ uses, it holds two starts: the smallest seed of the
evaluation set (master seed 42, a pool of eight seeds per combination) whose start the label marks separable, and the
smallest it marks inseparable, each with the reserved seed of the same start combination. Every run under those rules
plays the same two starts on either set. No preset that the paper reports apart (OWN_TABLES of experiment_tables.py) has them, since
the anchored preset anchors its own scenarios and the control on the healthy start has a single start combination. The file holds
seeds, labels and the SHA-256 of the data file they were drawn on, and no reading.

pack writes one folder per run: INDEX.md, with the table rows and leaderboard lines that read the run (INDEX.json), its
ledger in short (job, node, code, model and judge with weights commit and engine, command line, configuration hash, seeds)
and its lines of VERIFY.md, beside the walkthrough pages of the two fixed starts, whose table of the files they read is
checked against the ledger. The pages quote sensor readings of single cycles, so a pack is written outside the
fork.

usage, from the fork root:
  python -m CausalDiagnosis.episode_viewer.run_review_pack starts
  python -m CausalDiagnosis.episode_viewer.run_review_pack pack RUN_FOLDER --results RESULTS_FOLDER --out PACKS_FOLDER [--runs ROOT ...]
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent

from CausalDiagnosis.utils.paths import preset_files  # noqa: E402
from CausalDiagnosis.experiments.run_verification import run_ledger as ledger  # noqa: E402
from CausalDiagnosis.evaluation.result_tables import pooling_rules  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.run_configuration import CONFIGS, DEFAULT_ASSET, get_profile, load_config, load_world, records
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import episode_name, refuse_page_output
from CausalDiagnosis.utils.json_files import read_json  # noqa: E402

FIXED_STARTS = HERE / "fixed_starts.json"
MASTER_SEED, SEED_POOL = 42, 8  # the evaluation set of run_episodes/ --starts sampled
CLASSES = ("separable", "inseparable")
FILES_HEADER = "| File | SHA-256 |"  # the table of the files a walkthrough page read, written by walkthrough/


# ----- the fixed starts -----

def pick(evaluation, development, separable_of):
    """{class: {"evaluation_seed", "development_seed"} or None}: per class the smallest evaluation seed whose start has the
    class, with the reserved seed of its start at the same place of the candidate order, since both lists follow that order.
    separable_of(seed) is True, False, or None for a start the label table does not hold."""
    chosen = {name: None for name in CLASSES}
    for seed, partner in sorted(zip(evaluation, development)):
        label = separable_of(seed)
        name = None if label is None else CLASSES[0] if label else CLASSES[1]
        if name and chosen[name] is None:
            chosen[name] = {"evaluation_seed": seed, "development_seed": partner}
    return chosen


def rules_key_of(config):
    from CausalDiagnosis.environment.reference_policies import diagnosability_label as separability

    return separability.rules_key(separability.rules_of(config))


def fixed_starts_of(config, cycles, blocks, labels_path=None):
    """The two fixed starts of a configuration under the separability label the verifier reads."""
    from CausalDiagnosis.environment.reference_policies import diagnosability_label as separability
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment import start_state_seeds as start_states
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv

    env = HydraulicRigEnv(cycles, blocks, config)
    evaluation = start_states.sampled_combination_seeds(env, MASTER_SEED, SEED_POOL)
    development = start_states.combination_seeds(env, 1)

    def separable_of(seed):
        env.reset(seed)
        label = separability.label_of(env, labels_path or separability.LABELS_PATH)
        return None if label is None else label["separable"]

    return pick(evaluation, development, separable_of)


def write_fixed_starts(path=FIXED_STARTS, labels_path=None):
    from CausalDiagnosis.experiments.table_definitions.experiment_tables import OWN_TABLES
    from CausalDiagnosis.environment.reference_policies.diagnosability_label import RULE

    _, cycles, blocks = load_world()
    presets, rules = {}, {}
    for preset in sorted(p.stem for p in preset_files(CONFIGS).values()):
        name, config = load_config(preset)
        if config.anchors is not None or name in OWN_TABLES:
            continue
        presets[name] = key = rules_key_of(config)
        if key not in rules:
            rules[key] = fixed_starts_of(config, cycles, blocks, labels_path)
    record = {"rule": RULE, "facts_sha256": cycles.facts_sha256, "master_seed": MASTER_SEED, "seed_pool": SEED_POOL, "presets": presets,
              "rules": rules}
    Path(path).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return record


def read_fixed_starts(path=FIXED_STARTS):
    """The fixed starts, or the empty record when the file does not exist. A file drawn on another data file than the one the
    asset reads now, or that names none, is refused."""
    path = Path(path)
    if not path.exists():
        return {"presets": {}, "rules": {}}
    fixed = read_json(path, {})
    records.require_same_data(path, fixed.get("facts_sha256"), records.check_data(get_profile(DEFAULT_ASSET)),
                              "Write it again with `python -m CausalDiagnosis.episode_viewer.run_review_pack starts`.")
    return fixed


def starts_of_run(meta, fixed):
    """(the run's fixed starts, how they were found) by the rules key of its configuration, or by its preset when
    fixed_starts.json does not hold that key, as for a run of older code or of another price table, or (None, why not)."""
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig

    try:
        config = EpisodeConfig.from_json(meta["config"])
    except (KeyError, TypeError, ValueError):
        config = None
    if config is not None and config.anchors is not None:
        return None, "the run anchors its scenarios, so it has no fixed starts"
    key = rules_key_of(config) if config is not None else None
    if key in fixed["rules"]:
        return fixed["rules"][key], f"They are the starts of the run's rules key {key}."
    preset = meta.get("config_name")
    if fixed["presets"].get(preset) in fixed["rules"]:
        return fixed["rules"][fixed["presets"][preset]], (
            f"They are the starts of the run's preset {preset}, since fixed_starts.json does not hold the run's rules key {key}, "
            "so the class of each start follows the preset's current rules.")
    return None, f"fixed_starts.json holds neither the run's rules key {key} nor its preset {preset}"


# ----- the pack -----

def page_hashes(text):
    """{path from the run folder: SHA-256} of the table of the files a walkthrough page read, or None without one."""
    lines = text.splitlines()
    if FILES_HEADER not in lines:
        return None
    hashes = {}
    for line in lines[lines.index(FILES_HEADER) + 2:]:
        if not line.startswith("|"):
            break
        path, sha = (cell.strip() for cell in line.strip().strip("|").split("|"))
        hashes[path.strip("`")] = sha
    return hashes


def page_check(text, record):
    """A sentence on whether the files a page read are those the run's ledger states, naming any that differ."""
    from CausalDiagnosis.experiments.run_verification.verify_results_folder import expected_files

    hashes = page_hashes(text)
    if hashes is None:
        return "The page carries no table of the files it read, so it cannot be matched with the ledger."
    if record is None:
        return f"The page read {len(hashes)} files, and the run has no ledger to match them with."
    expected = expected_files(record)
    differing = sorted(path for path, sha in hashes.items() if expected.get(path) != sha)
    if not differing:
        return f"All {len(hashes)} files the page read have the SHA-256 the run's ledger states."
    return f"{len(differing)} of the {len(hashes)} files the page read differ from the run's ledger, {', '.join(differing)}."


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] + ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]


def served_row(role, served):
    endpoint = served.get("endpoint_file") or {}
    if not endpoint:
        source = "no endpoint file of a serving job"
    elif endpoint.get("lines", True) is None:
        source = f"the endpoint file {endpoint.get('file')} was missing"
    else:
        match = "matches" if endpoint.get("matches_endpoint") else "does not match"
        source = (f"the endpoint file of the serving job {match} the run's endpoint, weights commit {endpoint.get('weights_commit', 'not recorded')}, "
                  f"SGLang {endpoint.get('sglang', 'not recorded')}, image SHA-256 {endpoint.get('image_sha256', 'not recorded')}")
    return [role, f"{served['name']}{' at ' + served['endpoint'] if served.get('endpoint') else ''}, {source}"]


def program_launch(program, cwd):
    """The program of a recorded command as a person starts it again: a module of the package by python -m, any other
    program as the run recorded it."""
    path = Path(program) if Path(program).is_absolute() else Path(cwd) / program
    parts = path.with_suffix("").parts
    if path.suffix != ".py" or "CausalDiagnosis" not in parts:
        return program
    first = len(parts) - 1 - parts[::-1].index("CausalDiagnosis")
    return "-m " + ".".join(parts[first:])


def ledger_rows(record):
    """The run's ledger in short, as the rows of a two column table."""
    if record is None:
        return [["Ledger", "none, since the run predates the ledger"]]
    job, code, command = record.get("job") or {}, record["code"], record["command"]
    rows = [["Status", f"{record['status']}, started {record['started']}, ended {record['ended']}"]]
    if job.get("slurm_job_id"):
        rows.append(["Job", f"Slurm job {job['slurm_job_id']}, array job {job.get('slurm_array_job_id')} task {job.get('slurm_array_task_id')}, "
                            f"restarts {job.get('slurm_restart_count')}, node {job.get('host')}"])
    elif job:
        rows.append(["Job", f"no Slurm job, host {job.get('host')}"])
    else:
        rows.append(["Job", "named in the private file of the run alone, which this copy does not hold"])
    if code.get("commit"):
        rows.append(["Code", f"commit {code['commit']}, with {len(code.get('dirty') or [])} of the run's code files changed in the working tree"])
    else:
        rows.append(["Code", f"upload manifest SHA-256 {code.get('upload_manifest_sha256')}, with {len(code.get('differs_from_upload') or [])} "
                             "of the run's code files changed after the upload (upload_manifest.py match names the commit)"])
    rows += [served_row(role.capitalize(), record[role]) for role in ("model", "judge") if record.get(role)]
    rows.append(["Command", f"`{command['python']} {program_launch(command['program'], command['cwd'])} {' '.join(command['argv'])}` in `{command['cwd']}`"])
    if command.get("grid_file"):
        rows.append(["Grid", f"line {command.get('grid_line')} of `{command['grid_file']}`, SHA-256 {command.get('grid_file_sha256')}"])
    configuration, seeds = record["configuration"], record["seeds"]
    rows.append(["Configuration", f"{configuration['config_name']}, SHA-256 {configuration['config_sha256']}, "
                                  f"price table {configuration.get('prices') or 'of the preset'}"])
    drawn = f", master seed {seeds['master_seed']}, pool {seeds['seed_pool']}" if seeds["master_seed"] is not None else ""
    rows.append(["Seeds", f"starts {seeds['starts']}{drawn}, {len(seeds['seeds'] or [])} seeds"])
    rows += [[f"Data {name}", entry["sha256"]] for name, entry in record["data"].items()]
    rows += [[f"Event {n}", f"{event['at']}, {event['tool']}, {event['what']}, {len(event['files'])} files"]
             for n, event in enumerate(record["events"], 1)]
    return rows


def appearances(index, run_dir):
    """Sentences and rows on where the results read the run: its input entry, the table rows built from its episodes and
    the leaderboard's choice."""
    run, text = run_dir.name, []
    entry = next((e for e in index["inputs"] if e["run"] == run), None)
    excluded = next((e for e in index["excluded"] if e["folder"] == run), None)
    if entry and entry["ledger_sha256"] is None:
        text.append("tables_from_runs.py read the run without a ledger, since the run predates the ledger.")
    elif entry:
        same = "is" if entry["ledger_sha256"] == ledger.sha256_of(run_dir / ledger.LEDGER) else "is not"
        text.append(f"tables_from_runs.py read the run with the ledger SHA-256 {entry['ledger_sha256']}, which {same} the ledger now.")
    if excluded:
        text.append(f"tables_from_runs.py left the run out, {excluded['reason']}.")
    rows = [[name, entry["row"], ", ".join(f"{k} {v}" for k, v in entry["key"].items())]
            for name, entries in sorted(index["tables"].items()) for entry in entries
            if any(e.startswith(f"{run}/") for e in (entry["episodes"] or []) + (entry["reference"] or []))]
    board = index.get("leaderboard") or {}
    for line in board.get("lines") or []:
        if run in line["run"].split():  # a line of the test set pools the runs of its replicates
            rule = line["rule"]
            cutoff = f"with the cutoff {rule['cutoff']}" if rule["cutoff"] else "with no cutoff"
            others = ", ".join(line["candidates"]) or "no other run"
            text.append(f"The leaderboard line {line['line']} of {line['model'] or 'no model'} shows this run. The rule took {rule['choice']}, "
                        f"in the {rule['episode_set']} set and {cutoff}, over {others}.")
    text += [f"The leaderboard leaves this run out, {e['reason']}." for e in board.get("left_out") or [] if run in e["run"].split()]
    text += ["The leaderboard marks this run still running." for e in board.get("running") or [] if e["run"] == run]
    return text, rows


def verification(results, run, index_sha256):
    """The lines of VERIFY.md on the run: the INDEX.json it checked, the table of the checks, the run's row and the
    differences that name the run."""
    path = Path(results) / "VERIFY.md"
    if not path.exists():
        return ["The results folder holds no VERIFY.md yet."]
    lines = path.read_text(encoding="utf-8").splitlines()
    checked = next((line for line in lines if line.startswith("Checked on ")), "VERIFY.md does not name the INDEX.json it checked.")
    out = [checked, "", "That is the INDEX.json this pack read." if index_sha256 in checked
           else "That is not the INDEX.json this pack read, so VERIFY.md is older than the tables.", ""]
    first = lines.index("| Check | What ran | Result |")
    out += lines[first:lines.index("", first)] + [""]
    row = next((line for line in lines if line.startswith(f"| `{run}` |")), None)
    out += ["| Run | Ledger | Replay | Result |", "|---|---|---|---|", row, ""] if row else ["VERIFY.md has no row for this run.", ""]
    out += [line for line in lines[lines.index("## Differences"):] if line.startswith("- ") and (f"{run}/" in line or f"{run}:" in line)]
    while out[-1] == "":
        out.pop()
    return out


def build(run_dir, results, out, fixed=None, roots=None):
    """The pack folder of a run: INDEX.md and the walkthrough page of each fixed start the run played."""
    from CausalDiagnosis.episode_viewer import walkthrough

    run_dir, results, out = Path(run_dir).resolve(), Path(results).resolve(), Path(out).resolve()
    refuse_page_output(out)
    index = json.loads((results / "INDEX.json").read_text(encoding="utf-8"))
    if index.get("root_kind") == "development" or pooling_rules.stamped(read_json(run_dir / "run.json", {})):
        raise SystemExit(f"{run_dir} or the tables of {results} are development runs, which are not results, so no pack is built of them")
    index_sha256 = ledger.sha256_of(results / "INDEX.json")
    meta = read_json(run_dir / "run.json", {})
    record = ledger.load(run_dir) if (run_dir / ledger.LEDGER).exists() else None
    starts, how = starts_of_run(meta, fixed if fixed is not None else read_fixed_starts())
    pack = out / run_dir.name
    pack.mkdir(parents=True, exist_ok=True)
    pages = []
    for name in CLASSES:
        start = (starts or {}).get(name)
        seeds = [] if not start else [("development", start["development_seed"]), ("evaluation", start["evaluation_seed"])]
        played = next(((kind, seed) for kind, seed in seeds if (run_dir / episode_name(seed) / "episode.json").exists()), None)
        if played is None:
            reason = how if starts is None else "the rules have no such start" if not start else (
                f"the run played neither the reserved seed {seeds[0][1]} nor the evaluation seed {seeds[1][1]}")
            pages.append(f"- The {name} start has no page, {reason}.")
            continue
        kind, seed = played
        folder = run_dir / episode_name(seed)
        text = walkthrough.render(folder, f"The {name} fixed start of CausalDiagnosis/episode_viewer/fixed_starts.json, the {kind} seed of its start.")
        (pack / f"walkthrough_{name}.md").write_text(text, encoding="utf-8")
        pages.append(f"- The {name} start is episode `{folder.name}` in `walkthrough_{name}.md`. {page_check(text, record)}")
    said, rows = appearances(index, run_dir)
    roots_text = " ".join(str(Path(r).resolve()) for r in (roots or [run_dir.parent]))
    text = [f"# Review pack of {run_dir.name}", "",
            f"Built on {date.today().isoformat()} by `CausalDiagnosis/episode_viewer/run_review_pack.py` from `{results / 'INDEX.json'}` "
            f"(SHA-256 {index_sha256}) and the run folder `{run_dir}`.", "",
            "## Where the results read the run", "", *([" ".join(said), ""] if said else []),
            *(table(["Table", "Row", "Key"], rows) if rows else ["No table row reads this run."]), "",
            "## The run's ledger", "", *table(["Field", "Value"], ledger_rows(record)), "",
            "## Verification", "", *verification(results, run_dir.name, index_sha256), "",
            "A reader who doubts a cell checks the run with this command, from the fork root.", "", "```",
            f"python -m CausalDiagnosis.experiments.run_verification.verify_results_folder --runs {roots_text} --results {results} --run {run_dir.name} --replay", "```", "",
            "## Walkthroughs of the fixed starts", "", *([how, ""] if starts else []), *pages, ""]
    (pack / "INDEX.md").write_text("\n".join(text), encoding="utf-8")
    return pack


def main(argv):
    ap = argparse.ArgumentParser(description="the fixed starts of the walkthroughs and the review pack of a run folder")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("starts", help="write fixed_starts.json beside this file")
    pack = sub.add_parser("pack", help="write the review pack of a run folder")
    pack.add_argument("run", type=Path)
    pack.add_argument("--results", type=Path, required=True, help="the results folder with INDEX.json and VERIFY.md")
    pack.add_argument("--out", type=Path, required=True, help="the folder of the packs, outside the fork")
    pack.add_argument("--runs", nargs="*", default=None, help="the roots the verify command of the pack names (default: the run's parent)")
    a = ap.parse_args(argv)
    if a.command == "starts":
        record = write_fixed_starts()
        print(f"wrote {FIXED_STARTS}, {len(record['rules'])} rules keys for {len(record['presets'])} presets")
    else:
        print(f"wrote {build(a.run, a.results, a.out, roots=a.runs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
