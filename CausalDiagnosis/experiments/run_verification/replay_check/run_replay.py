import tempfile
from pathlib import Path
from CausalDiagnosis.experiments.run_verification import run_ledger as ledger  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import SCORE_COLUMNS, episode_dirs, episode_folder, read_rows  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import ROOT  # noqa: E402
from CausalDiagnosis.utils.csv_rows import flat  # noqa: E402
from CausalDiagnosis.utils.json_files import read_json  # noqa: E402
from CausalDiagnosis.experiments.table_definitions.training_variants import asset_of  # noqa: E402  the asset of a run as the tables read it
from . import run_record_checks, episode_replay


JUDGE_CELLS = ("judge_pass",)  # the judge does not run again in a replay
STATIC_MODE = "static"  # the mode of run.json of the static agent, whose episodes call no server


# ----- one run -----
def score_problems(run_dir, ep_dir, row, label):
    """The score cells of an episode's replay that differ from its row of scores.csv, the judge's cells left out."""
    rows = {episode_folder(run_dir, recorded).name: recorded for recorded in read_rows(run_dir)}
    if not rows:
        return []
    recorded = rows.get(Path(ep_dir).name)
    if recorded is None:
        return [f"{label}: no row of scores.csv names the episode"]
    differing = [c for c in SCORE_COLUMNS if c not in JUDGE_CELLS and str(flat((row or {}).get(c))) != recorded.get(c, "")]
    return [f"{label}: the cells {', '.join(differing)} differ from scores.csv"] if differing else []


def replay_run(run_dir, scratch_root=None, root=ROOT):
    """The replay of every episode folder of a run: {"state", "episodes", "lines", "passed", "problems", "scoring_problems",
    "table_problems", "answers_equal", "advisory", "changed", "agent_files_equal", "loaded_changed", "listings_equal", "data_equal",
    "data_differences", "config_equal", "store", "platform", "transcripts"}. The state is replayed, no transcript for a run made
    before the transcript existed, static for a run whose episodes call no server, or not a run of the rig for a reproduction run of
    AssetOpsBench, which its own scorer judges. The advisory reasons say why a replay that passed counts nowhere: another store kind
    or another platform than the run's, or a run without the data record. data_differences names every entry of the data record
    whose hash differs from the run's. scoring_problems (the files that the verifier writes from what it reads and the score cells)
    and table_problems (the generated tables among those entries, every entry but RECORDED_DATA) lie among the problems, and
    answers_equal holds when the episodes replayed and no other problem stands: every answer, every file the episode wrote while it
    played, the inputs of the verifier, the listings, the configuration, the recordings and the scenario records equal the run's."""
    from CausalDiagnosis.experiments.episode_runners.run_configuration import load_world
    from CausalDiagnosis.mcp_servers.utils import transcript

    run_dir = Path(run_dir)
    meta = read_json(run_dir / "run.json", {}) or {}
    result = {"run": run_dir.name, "state": "replayed", "episodes": 0, "lines": 0, "passed": False, "problems": [], "scoring_problems": [],
              "table_problems": [], "answers_equal": False, "advisory": [],
              "changed": {}, "agent_files_equal": None, "loaded_changed": {}, "listings_equal": None, "data_equal": None,
              "data_differences": [], "config_equal": None, "store": {"run": run_record_checks.store_kind(meta), "replay": None}, "platform": None, "transcripts": {}}
    if ledger.run_kind(meta) is not None:
        return {**result, "state": "not a run of the rig"}
    if meta.get("mode") == STATIC_MODE:
        return {**result, "state": "static"}
    episodes = episode_dirs(run_dir)
    if not any((ep_dir / transcript.NAME).exists() for ep_dir in episodes):
        return {**result, "state": "no transcript"}
    facts_path, cycles, blocks = load_world(None, asset_of(meta))
    config = run_record_checks.config_problem(run_dir, meta, blocks)
    result["config_equal"] = config is None
    result["problems"] += [config] if config else []
    store, store_reason = run_record_checks.store_of(meta)
    result["store"]["replay"] = (store or {}).get("kind")
    result["advisory"] += [reason for reason in (store_reason, run_record_checks.platform_reason(meta, root)) if reason]
    differing, note = run_record_checks.data_differences(meta, facts_path)
    result["data_equal"], result["data_differences"] = not differing, differing
    data = {name: f"{run_dir.name}: the {'data file' if name in run_record_checks.RECORDED_DATA else 'generated table'} {name} differs from the one the run recorded"
            for name in differing}
    result["problems"] += list(data.values())
    result["table_problems"] = [problem for name, problem in data.items() if name not in run_record_checks.RECORDED_DATA]
    result["advisory"] += [note] if note else []
    listings = None
    with tempfile.TemporaryDirectory(dir=scratch_root) as scratch:
        for ep_dir in episodes:
            replayed = episode_replay.replay_episode(ep_dir, meta, Path(scratch) / run_dir.name, store, (facts_path, cycles, blocks), listing=listings is None)
            if listings is None and replayed["listings"]:
                listings = replayed["listings"]
            result["problems"] += replayed["problems"]
            result["scoring_problems"] += replayed["scoring_problems"]
            if replayed["row"] is not None:  # an episode that could not be replayed has its one problem already
                cells = score_problems(run_dir, ep_dir, replayed["row"], f"{run_dir.name}/{ep_dir.name}")
                result["problems"] += cells
                result["scoring_problems"] += cells
            result["lines"] += replayed["lines"]
            result["episodes"] += 1
            result["transcripts"][ep_dir.name] = transcript.file_sha256(ep_dir / transcript.NAME)
    recorded_listings = meta.get("tool_listings") or {}
    if not recorded_listings:
        result["problems"].append(f"{run_dir.name}: run.json holds no tool listings")
    if listings is not None:  # no episode replayed far enough to build the servers, which its own problem names
        differing = sorted(server for server in set(recorded_listings) | set(listings) if recorded_listings.get(server) != listings.get(server))
        result["problems"] += [f"{run_dir.name}: the tool listing of the server {server} differs from run.json" for server in differing]
        result["listings_equal"] = bool(recorded_listings) and not differing
    result["changed"] = run_record_checks.changed_files(meta, root)
    result["agent_files_equal"] = ((meta.get("code") or {}).get("files") or {}).get(run_record_checks.agent_list(meta), {}) == ledger.code_identity.tree_files(root).get(run_record_checks.agent_list(meta), {})
    result["loaded_changed"] = run_record_checks.loaded_changed(meta, result["changed"])
    result["platform"] = ledger.platform_of(root)
    result["passed"] = not result["problems"]
    result["answers_equal"] = len(result["problems"]) == len(result["scoring_problems"]) + len(result["table_problems"])
    return result
