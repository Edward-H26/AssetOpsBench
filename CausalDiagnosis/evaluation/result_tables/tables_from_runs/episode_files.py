import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import episode_dirs, episode_folder, read_rows  # noqa: E402
from CausalDiagnosis.utils.json_files import read_json  # noqa: E402
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import DEFAULT_ASSET  # noqa: E402
from CausalDiagnosis.evaluation.result_tables import pooling_rules  # noqa: E402
from CausalDiagnosis.experiments.table_definitions.training_variants import REGISTERED, TRAINING_SPLIT_KEY  # noqa: E402
from CausalDiagnosis.experiments.table_definitions.experiment_tables import AS_RELEASED, TRAINING_FOLD_KEY, TRAINING_VARIANT_KEY, asset_of  # noqa: E402


# ----- episodes -----
def load_episodes(runs_root):
    """Every row of every scores.csv of the run folders read under the root, the episodes of episodes_of. A table set is the
    table set of one asset, so a root whose runs play two assets stops the tool by name."""
    episodes = [episode for run_dir in pooling_rules.folders_read(runs_root) for episode in episodes_of(run_dir)]
    # no cell of a table holds the asset, so the runs of two assets in one root would pool in one cell
    assets = sorted({episode["asset_id"] for episode in episodes})
    if len(assets) > 1:
        raise SystemExit(f"the runs under {runs_root} play the assets {', '.join(assets)}, and a table set is the table set of one "
                         "asset: give each asset a root of its own (--runs)")
    return episodes


def episodes_of(run_dir):
    """The episodes of one run folder: every row of its scores.csv with the run's name and folder and the per episode folder
    beside it, so a builder can open the hidden files of an episode when a table needs more than scores.csv holds, and the
    fields of run.json that the counting rule and the tables read, among them the step limit of the run's config (max_steps of
    run.json), the cap of the turns to a correct diagnosis."""
    meta = read_json(run_dir / "run.json", {})
    folders = {p.name: p for p in episode_dirs(run_dir)}
    tables_of_run = config_tables(meta.get("config") or {})
    episodes = []
    for row in read_rows(run_dir):
        episode = dict(row)
        episode["run"] = row.get("run") or run_dir.name
        episode["run_dir"] = run_dir
        episode["episode_dir"] = folders.get(episode_folder(run_dir, row).name)
        episode["mode"] = meta.get("mode", "gym")
        episode["abstention_rule"] = meta.get("abstention_rule", "decision")
        episode["training_variant"] = meta.get(TRAINING_VARIANT_KEY) or AS_RELEASED
        episode["lora_adapters"] = int(meta.get("lora_adapters") or 0)  # the serving setup, the adapters of the server of the model
        episode["abstention_cut"] = meta.get("abstention_cut") if meta.get("abstention_cut") is not None else ""  # the cut of the fixed-cut rule
        episode["prices_sha256"] = meta.get("prices_sha256") or ""  # the price table itself, beside its name
        episode["training_fold"] = meta.get(TRAINING_FOLD_KEY)  # the fold the model of a trained run holds out, none as released
        episode["asset_id"] = asset_of(meta)  # whose split names the folds
        episode["training_split"] = meta.get(TRAINING_SPLIT_KEY) or REGISTERED  # registered, or development in a trial on the development split
        episode["run_judge_model"] = meta.get("judge_model") or ""  # the judge the run names, under which an episode without a record is unjudged
        episode["episodes_planned"] = meta.get("episodes_planned")
        episode["replicate"] = meta.get("replicate")  # the replicate of a run of the test set, none off it
        episode["fraction"] = meta.get("fraction") or ""  # the share of a cut look (a row field), empty for a whole run
        episode["surface"] = pooling_rules.surface_of(meta)
        episode["max_steps"] = (meta.get("config") or {}).get("max_steps")
        episode["horizon"] = (meta.get("config") or {}).get("horizon")  # the cycles of the preset, the time to repair of a fault never restored
        episode["plan_length"] = (meta.get("causal_options") or {}).get("plan_length")  # the longest plan of the CausalDiagnosis agent of the run
        episode["run_tables"] = tables_of_run  # (components, healthy) of run_tables, read once per run
        episodes.append(episode)
    return episodes


def grade_record(episode):
    """The separability label that the verdict of an episode recorded, the grade cell of scores.csv, or None."""
    cell = episode.get("grade", "")
    return json.loads(cell) if cell else None


def e18_separable(episode):
    """The separability of the start of an episode under the server's E18 grade in the grade cell, None without one."""
    e18 = (grade_record(episode) or {}).get("e18")
    return None if e18 is None else e18["separable"]


def run_tables(episode):
    """(components, healthy) of an episode: the component order and the healthy levels of the configuration in its run's
    run.json, or of the default asset's profile for a run.json that names neither. episodes_of reads them once per run; an
    episode made without it reads its run.json here."""
    if "run_tables" in episode:
        return episode["run_tables"]
    return config_tables(read_json(Path(episode["run_dir"]) / "run.json", {}).get("config", {}) if episode.get("run_dir") else {})


def config_tables(config):
    """(components, healthy) of the configuration of a run.json, run_tables's rule."""
    if config.get("components") and config.get("healthy"):
        return tuple(config["components"]), dict(config["healthy"])
    profile = get_profile(config.get("asset_id") or DEFAULT_ASSET, engine=False)
    return profile.components, dict(profile.healthy)


def _levels(combination, components):
    """The levels of a ranked combination in component order, from the list of the servers before version 7 or from the
    {component: level} map of version 7."""
    return [combination[c] for c in components] if isinstance(combination, dict) else list(combination)


def _hidden_log(episode):
    folder = episode.get("episode_dir")
    return read_json(Path(folder) / "hidden" / "log.json", None) if folder is not None else None


def _trajectory(episode):
    folder = episode.get("episode_dir")
    return read_json(Path(folder) / "trajectory.json", None) if folder is not None else None


def _oracle(episode):
    folder = episode.get("episode_dir")
    return read_json(Path(folder) / "hidden" / "oracle.json", None) if folder is not None else None


def _answer(step):
    """The parsed JSON answer of a trajectory step, or an empty dict when the call failed or the text is no JSON object."""
    if step.get("error"):
        return {}
    try:
        answer = json.loads(step.get("response") or "")
    except (TypeError, ValueError):
        return {}
    return answer if isinstance(answer, dict) else {}


def ranking_answers(trajectory):
    """The parsed answers of the rank_root_causes calls of a trajectory that succeeded, in call order."""
    return [answer for step in trajectory or [] if step.get("tool") == "rank_root_causes" and (answer := _answer(step))]


def _verdict(episode):
    folder = episode.get("episode_dir")
    return read_json(Path(folder) / "hidden" / "verdict.json", None) if folder is not None else None
