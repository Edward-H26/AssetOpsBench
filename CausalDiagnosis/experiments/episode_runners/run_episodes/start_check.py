import os
from CausalDiagnosis.experiments.run_verification import run_ledger as ledger  # noqa: E402
from CausalDiagnosis.experiments.table_definitions.experiment_tables import ARM_PRESETS, LADDER, RUNS_IN_FLIGHT, AGENTS_PRESET, HEALTHY_CONTROL_PRESET, TRAINING_PRESET, TOKEN_PRESETS  # noqa: E402
from CausalDiagnosis.experiments.table_definitions.training_variants import DEVELOPMENT, adapter_of, block_of  # noqa: E402
from . import agent_kinds, run_record


# ----- the start check -----
START_SETS = ("seeds", "combinations", "sampled", "training", "development")  # the choices of --starts, which pending_grid_lines.py and precompute_grid_tables.py share
STAMP = "development"  # run.json stamp of a run that the start check does not register
REGISTERED_STARTS = ("sampled", "training")  # the start sets a registered run plays
SEED_STARTS = ("seeds", "combinations", "development")  # the start sets that count seeds with --seeds


def table_presets():
    """The presets that a table of the paper plays: the main table, the ladder, the arms of the ablations, training, the twins
    of the clock presets in the generated_tokens mode and the healthy control, whose table counts registered episodes and whose
    runs share the job of the main table."""
    return {AGENTS_PRESET, *LADDER, *ARM_PRESETS, TRAINING_PRESET, *TOKEN_PRESETS, HEALTHY_CONTROL_PRESET}


def runs_in_flight():
    """GRID_PARALLEL of the grid job as an integer, None outside a grid job."""
    value = os.environ.get("GRID_PARALLEL")
    return int(value) if value and value.isdigit() else None


def tree_reasons():
    """The reasons of the start check that the tree alone gives, rules 1, 2 and 4 (ledger.tree_record_reasons)."""
    return ledger.code_identity.tree_record_reasons()


def stamped_inputs(a, config):
    """The adapter and the block of the trained variant of a run that carry the stamp of the runs they were built from, which the
    data builder and the export of a loop write into their records, so that a run on them is not registered either."""
    adapter = adapter_of(a.agent, a.training_variant, a.training_fold, config.asset_id, split=a.training_split)
    _text, block = block_of(a.agent, a.training_variant, a.training_fold, config.asset_id, split=a.training_split)
    return [name for name, record in (("the adapter", adapter), ("the block", block)) if (record or {}).get("stamp")]


FILLER_REASON = "a filler run keeps the runs in flight of its grid job and is never registered"


def serving_reasons(record, role, what):
    """Rules 7 and 8 for one endpoint: its endpoint file equals the serving file of record in every field, and its card is the card
    of record when env.sh names one (ledger.serving_differences, ledger.card_differences)."""
    try:
        found = [f"{field} {declared} in the serving file and {value} at the endpoint" for field, declared, value in ledger.serving_endpoints.serving_differences(record, role)]
    except ledger.OpenServingFile as e:
        return [str(e)]
    reasons = [f"the {what} differs from the serving file of record: {'; '.join(found)}"] if found else []
    return reasons + [f"the {what}: {reason}" for reason in ledger.serving_endpoints.card_differences(record)]


def registration(a, config, config_name, facts_path):
    """The reasons why a run is not of record, in this order, and none for a run of record: the tree (rules 1, 2 and 4), the files
    the driver has loaded so far (rule 3, checked again at the end of the run), the data (rule 5; the run holds the generated tables
    read only, so a missing one stops it by name), the open values (rule 6), the serving of a model agent's endpoint and of a judge
    that judges in the run against the serving file of record, with the card of record (rules 7 and 8), the runs in flight of a preset
    with a clock (rule 9), the start set, the development split, the stamp of the adapter or the block that a trained variant
    plays (rule 10), and an agent whose declaration keeps its runs out of the tables. main() asks before any table could be written."""
    reasons = tree_reasons()
    if config.environment_mode is not None:
        reasons.append("explicit static/dynamic modes are development conditions, not registered table presets")
    filler = [FILLER_REASON] if getattr(a, "filler", False) else []
    # rule 3 asks git about the commit, so it is asked only of a tree that passed rules 1, 2 and 4
    if not reasons and (untracked := ledger.code_identity.untracked_loaded(run_record.loaded_files())):
        reasons.append(f"the driver loaded {len(untracked)} files that git does not track, first {untracked[0]}")
    reasons += ledger.code_identity.data_reasons(facts_path, config.asset_id) + ledger.code_identity.open_reasons()
    if a.agent in agent_kinds.MODEL_AGENTS:
        reasons += serving_reasons(ledger.served(a.model, a.base_url, "HYD_ENDPOINT_FILE"), "agent", "model endpoint")
    if a.judge == "local":
        judging = ledger.served(a.judge_model or a.model, a.judge_base_url or a.base_url, "HYD_JUDGE_ENDPOINT_FILE")
        reasons += serving_reasons(judging, "judge", "judge endpoint")
    if config.clock_mode == "generation_time":
        if runs_in_flight() != RUNS_IN_FLIGHT:
            reasons.append(f"the preset {config_name} has a clock and plays {runs_in_flight()} runs in flight, not the registered {RUNS_IN_FLIGHT}")
    if a.starts not in REGISTERED_STARTS or config_name not in table_presets():
        reasons.append(f"--starts {a.starts} on the preset {config_name} is no registered start set of a table")
    if a.training_split == DEVELOPMENT:
        reasons.append(f"--training-split {a.training_split} plays the development split of a trial before the runs of record")
    if stamped := stamped_inputs(a, config):
        reasons.append(f"--training-variant {a.training_variant} plays {' and '.join(stamped)} built from runs that carry the stamp {STAMP}")
    if not agent_kinds.DECLARED_AGENTS.get(a.agent, {}).get("registered", True):
        reasons.append(f"--agent {a.agent} is a check that no table reads")
    return reasons + filler


def refuse_registered_training(a, reasons, what):
    """Refuses by name, before its run folder exists, a run that plays the training starts of the registered split of its asset, a
    sampling run of fine tuning (--starts training) or a loop of GEPA, when the start check gives it reasons, and names the
    development split. The registered split is cross-fitted: the training starts of the model of a fold lie on test combinations
    of the other folds, at entry offsets that no replicate plays, so they are played only in a run of record, and a trial of the
    training path plays the development split. An evaluation run on the test starts and a run of record pass."""
    if reasons and a.training_split != DEVELOPMENT:
        raise SystemExit(f"{what} on the registered split plays the training starts of its folds, which lie on test combinations of "
                         f"the other folds, and only a run of record plays them, while this run is not of record: {'; '.join(reasons)}. "
                         "A trial of the training path plays the development split: --training-split development")
