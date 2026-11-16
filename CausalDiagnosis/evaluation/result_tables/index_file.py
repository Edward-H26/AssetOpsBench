from CausalDiagnosis.utils.paths import ROOT
import json
from pathlib import Path
from CausalDiagnosis.experiments.run_verification import run_ledger as ledger
from CausalDiagnosis.utils.json_files import read_json
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import read_rows
from CausalDiagnosis.experiments.table_definitions.training_variants import asset_of  # the one rule of the asset of a run, which the modules of the tables and the training share
from . import pooling_rules


FORMAT = 2
CODE_KINDS = ("analysis", "rendering")  # the lists of the code that computes the tables and of the code that renders them, recorded apart


def episode_name(episode):
    """run folder/episode folder of an episode row of tables_from_runs/, or the run folder with the seed when the folder is missing."""
    run = Path(episode["run_dir"]).name
    folder = episode.get("episode_dir")
    return f"{run}/{Path(folder).name}" if folder is not None else f"{run}/seed {episode.get('seed', '')}"


def row_key(columns, row):
    """The cells that name a row: its text cells (agent, model, config, the settings, a stratum or a switch), leaving the
    numbers to the CSV file."""
    return {c: row[c] for c in columns if isinstance(row.get(c), str) and row[c] != ""}


def record_of(runs_root):
    """{asset: {"commits", "lists", "serving"}} of the runs of record that the tables read under the root, empty for another root:
    the commits they played and the parts of their group key, which the page heads and INDEX.json name."""
    out = {}
    for run_dir in pooling_rules.folders_read(runs_root):
        meta = read_json(run_dir / "run.json", {})
        if pooling_rules.kind_of(meta) != "record":
            continue
        key, group = pooling_rules.group_key(meta), out.setdefault(asset_of(meta), {"commits": set(), "lists": None, "serving": None})
        group["commits"].add(str(pooling_rules.commit_of(meta) or "none")[:12])
        group["lists"] = key["lists"]
        group["serving"] = group["serving"] or key["serving"]
    return {asset: {**group, "commits": sorted(group["commits"])} for asset, group in out.items()}


def inputs(runs_root):
    """One entry per run folder that tables_from_runs/ reads: its ledger hash and status, its episodes, its set of episodes, its commit
    and the GPU card of its model endpoint."""
    entries = []
    for run_dir in pooling_rules.folders_read(runs_root):
        meta = read_json(run_dir / "run.json", {})
        record = ledger.load(run_dir) if (run_dir / ledger.LEDGER).exists() else None
        entries.append({"run": run_dir.name, "ledger_sha256": ledger.sha256_of(run_dir / ledger.LEDGER),
                        "ledger_status": record.get("status") if record else None, "episodes": len(read_rows(run_dir)),
                        "seeds": meta.get("seeds"), "episode_set": {**{k: meta.get(k) for k in ("starts", "master_seed", "seed_pool", "replicate")},
                                                        **({"fraction": meta["fraction"]} if meta.get("fraction") is not None else {})},
                        "asset_id": asset_of(meta), "identity": pooling_rules.identity_of(meta), "commit": pooling_rules.commit_of(meta), "stamp": meta.get("stamp"),
                        "card": pooling_rules.card_of(meta)})
    return entries


@pooling_rules.reads_once
def excluded(runs_root):
    """The folders directly under the root that tables_from_runs/ does not read, each with the reason."""
    root = Path(runs_root)
    if not root.exists():
        return []
    read = {p.name for p in pooling_rules.folders_read(root)}
    left_out = pooling_rules.pooling_left_out(root)
    out = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and p.name not in read):
        if (refusal := pooling_rules.refused(folder)) is not None:
            reason = refusal
        elif folder.name in left_out:
            reason = left_out[folder.name]
        elif folder.name.startswith("archive_invalid"):
            reason = "archived as invalid, moved and never deleted"
        elif any(child.is_dir() and (child / "run.json").exists() for child in folder.iterdir()):
            reason = "a folder of run folders, which tables_from_runs.py reads only when it is the root"
        elif (folder / "scores.csv").exists() or (folder / ledger.LEDGER).exists():
            reason = "no run.json, so the run has not finished"
        else:
            reason = "no run.json and no episode, not a run folder"
        out.append({"folder": folder.name, "reason": reason})
    return out


def table_entries(table, rows, traces):
    entries = []
    for number, (row, traced) in enumerate(zip(rows, traces), 1):
        episodes, reference = traced if traced else (None, None)
        entries.append({"row": number, "key": row_key(table.columns, row),
                        "episodes": None if episodes is None else sorted({episode_name(e) for e in episodes}),
                        "reference": None if reference is None else sorted({episode_name(e) for e in reference})})
    return entries


def build(runs_root, tables_rows, traces, parameters, judges_left_out=()):
    """The index of one run of tables_from_runs/: tables_rows as aggregate() returns them, traces the (episodes, reference) of
    every row in the same order (None for a row its builder did not trace), and the parameters of the run, with which
    verify_results_folder.py rebuilds the tables."""
    return {"format": FORMAT, "runs_root": Path(runs_root).name, "parameters": parameters, "inputs": inputs(runs_root), "excluded": excluded(runs_root),
            "tables": {f"{table.id}.csv": table_entries(table, rows, traced) for (table, rows), traced in zip(tables_rows, traces)},
            "leaderboard": None, "paper": None, "root_kind": pooling_rules.root_kind(runs_root), "judges_left_out": list(judges_left_out),
            "code": code_of_tables(), "record": record_of(runs_root)}


def code_of_tables():
    """{list: {path: SHA-256}} of the code that computes the tables and of the code that renders them, the lists analysis and
    rendering of design.CODE_LISTS, recorded apart."""
    files = ledger.code_identity.tree_files(ROOT)
    return {kind: files.get(kind, {}) for kind in CODE_KINDS}


def write(index, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def add_section(path, name, section):
    """Writes one section into an existing index, as leaderboard.py does with its choice of runs."""
    index = json.loads(Path(path).read_text(encoding="utf-8"))
    index[name] = section
    write(index, path)
