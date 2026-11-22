"""The separability label of a start, computed without the CausalDiagnosis server.

Under one rules key, which holds the prices, durations, downtime rates, horizon and budget with the fault rule of the start
sampler, whether a clock runs on the row and the text of the rule below, the greedy reference policy of greedy_policy.py plays the static episode of a start on ten
seeds whose reset draws that start, the smallest after the evaluation pool of the start (start_states.EVALUATION_POOL), so
that no rollout of a label replays an evaluation or a development episode. Before its first action a rollout reads the
prehistory of its seed, the HISTORY_RECORDINGS recordings of the start that the AssetOpsBench rows record before cycle 0
(recorded_history_env.py), every channel through the sensor group that covers them and without charge, so the reference policy reads
what the rows serve, and it reads through the cross fitted per tier classifiers of tier_classifiers.py (8 folds over the 144
combinations at fold seed 0). Where a clock runs on the row as given the rollout then plays the policy over the live
recordings. A row without a clock serves none, so there the rollout takes the most probable combination of the reference
after the prehistory. The policy never reads the fault rule, which only picks the starts of a table and the seeds
of their rollouts. The start is separable when the policy submits the start combination, every level, in more than half of
the rollouts, the majority rule for a binary label. The label says what this policy did within the budget and not that the
recordings cannot tell the combination. The share of correct rollouts is kept beside the label in data/labels.json, one table
per rules key as in the server's grades.json but without a ranking model, so every condition is scored against one frozen
table. The label inherits the constants of the greedy policy, the confidence of 0.95 and the minimum gain of 0.05 bits per
unit price, which every table records. It shares no model with the server, whose E18 grade stays the second label, and
start_grade nests that grade under the label in the form the verifier reads, which credits a declaration only on a start
both mark inseparable. The file
is written under a lock, so the processes of one clone that fill it at once keep each other's tables, and
CausalDiagnosis/experiments/cluster_runs/precompute_grid_tables.py fills it for every rules key of a grid before the grid's lines start.

usage: python -m CausalDiagnosis.environment.reference_policies.diagnosability_label [--config PRESET.json]
       fills the table of a configuration, the default one without an argument.
"""
import argparse
import hashlib
import json
from pathlib import Path

from CausalDiagnosis.datasets.dataset_loader import recordings as records
from CausalDiagnosis.datasets.dataset_loader.dataset_packages import table_path
from CausalDiagnosis.datasets.dataset_loader.evidence_source import SEGMENT
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
from CausalDiagnosis.utils.atomic_writes_and_locks import files_digest, locked, refuse_writing, write_json_atomically
from CausalDiagnosis.utils.paths import DATASETS, PACKAGE

from CausalDiagnosis.environment.reference_policies.tier_classifiers import COMBOS, COMPONENTS, N_FOLDS, PROFILE, TierClassifiers
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv
from CausalDiagnosis.environment.reference_policies.greedy_policy import GreedyPolicy, ServedClassifiers
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.start_state_seeds import EVALUATION_POOL, combination_seeds, combo_key

LABELS_PATH = table_path(records.release_path(PROFILE).parent / "labels.json")  # derived from the recordings, so it stays out of git with the other data files
# 6 since a rollout plays the live recordings only where a clock runs on the row and the rules key holds whether one runs
LABELS_FORMAT = 6
ROLLOUTS = 10  # rollouts per start, as many as the server's E18 takes
FOLD_SEED = 0
HISTORY_RECORDINGS = SEGMENT  # the recordings of the start before cycle 0 that the AssetOpsBench rows record, which every rollout reads first
RULE_KEYS = ("prices", "durations", "downtime_rate", "horizon", "budget", "fault_rule", "max_steps", "failure", "sensor_groups", "asset_id")
ENGINE_ROOT = (PACKAGE / "environment", DATASETS)  # the engine and the assets, whose code a rollout runs
RULE = ("separable when the greedy reference, after reading without charge the recordings of the start that the AssetOpsBench rows "
        "record before cycle 0, and where a clock runs on the row the live recordings it then plays, submits the start combination, "
        "every level, in more than half of its rollouts on the seeds after the evaluation pool, its most probable combination after "
        "the record standing for its submission where no clock runs")
RELABEL = "Remove it and label again with `python -m CausalDiagnosis.experiments.cluster_runs.precompute_grid_tables`."
STATIC_KEYS = ("schedule", "hazard", "start", "allowed", "anchors", "question", "clock_mode")


def rules_of(config):
    """The fields of the configuration as given that a rollout reads, with whether a clock runs on it, which decides whether a
    rollout plays the live recordings after the record and which static_config would hide by resetting the clock mode."""
    data = config.to_json()
    return {**{key: data[key] for key in RULE_KEYS}, "clock": config.has_clock()}


def rules_key(rules):
    """One label table per price table, duration table, downtime table, horizon, budget, fault rule, step cap, failure card,
    sensor groups, asset, clock and rule text, every field that a rollout reads: the fault rule decides which seeds draw a start
    and so which recorded cycles its rollouts read, the step cap and the failure card can end a rollout, the sensor groups are
    what it reads, the clock decides whether it plays the live recordings, and the rule text says what a rollout reads before
    its first action and which seeds it plays."""
    return hashlib.sha256(json.dumps({**{key: rules[key] for key in RULE_KEYS}, "clock": rules["clock"], "rule": RULE},
                                     sort_keys=True).encode()).hexdigest()


def code_digest(asset_id, root=ENGINE_ROOT):
    """The digest of the code and the asset that write a label table, for any asset: the digest of every .py file of the
    engine package outside the tests folders and the folders of other assets, and the canonical JSON of the asset's profile.
    The digest covers the whole package and not a list of chosen files, which would have to follow every later import, so a
    table is read only by the code and the rules that wrote it, and a change of any engine file computes the tables again on
    first use."""
    profile = json.dumps(get_profile(asset_id, engine=False).to_json(), sort_keys=True).encode()
    return hashlib.sha256((files_digest(tuple(map(str, root)) if isinstance(root, tuple) else str(root), asset_id) + hashlib.sha256(profile).hexdigest()).encode()).hexdigest()


def static_config(config):
    """The configuration a label is computed in: the rules of the given one with every hidden process off, the declared clock
    and no anchored or restricted start, so that a timed or deterioration preset is labelled by its static counterpart."""
    default = get_profile(config.asset_id).default_config().to_json()
    return EpisodeConfig.from_json({**config.to_json(), **{key: default[key] for key in STATIC_KEYS}})


def history_group(config):
    """The sensor group through which a rollout reads the prehistory: the one whose channels are every channel of the asset,
    since a history read of the rows with no sensor list serves every field."""
    channels = set(get_profile(config.asset_id, engine=False).channels)
    return next(group for group, sensors in config.sensor_groups.items() if set(sensors) == channels)


def first_diagnosis(env, policy, seed, history_group=None, live=True):
    """Play the policy from the seed until it submits a diagnosis: the submitted combination, or None when the episode ends
    before one, which counts as a rollout that did not separate the start. With a history group the policy first reads the
    prehistory of the seed through it, the recordings of the start that the AssetOpsBench rows record before cycle 0. Without
    live, on a row without a clock, which serves no live recording, the rollout takes the most probable combination of the
    policy after the prehistory, with no read of its own."""
    obs = env.reset(seed)
    if history_group is not None:
        policy.start(obs)
        sensors = env.config.sensor_groups[history_group]
        policy.read_history(history_group, [env.source.served_values(index, sensors) for index in env.source.prehistory(env.initial)])
    if not live:
        top = dict(zip(COMPONENTS, COMBOS[max(range(len(COMBOS)), key=policy.belief.__getitem__)]))
        return tuple(top[c] for c in env.config.components)
    while not env.done:
        mark = len(env.log)
        obs, _cost, _done, _info = env.step(policy.act(obs))
        # the entry of the diagnosis among the entries of the step, found by its event, since in the instant mode the events of
        # the cycle of the submission follow it
        entry = next((e for e in env.log[mark:] if e["event"] == "submit_diagnosis"), None)
        if entry is not None:
            return tuple(entry["diagnosis"][c] for c in env.config.components)
    return None


def compute_table(config, cycles, blocks, rollouts=ROLLOUTS, classifiers=None):
    """The labels of every start combination of the configuration's fault rule under its rules key. classifiers is the
    observation model of the greedy policy, the cross fitted ServedClassifiers when None."""
    static = static_config(config)
    env = HydraulicRigEnv(cycles, blocks, static)
    if classifiers is None:
        classifiers = ServedClassifiers(TierClassifiers(cycles, blocks, n_folds=N_FOLDS, seed=FOLD_SEED), cycles)
    group = history_group(static)
    live = config.has_clock()  # of the row as given, since the static configuration declares the clock
    seeds = combination_seeds(env, rollouts, after=EVALUATION_POOL)
    labels = {}
    for i, start in enumerate(env.candidates):
        own = seeds[i * rollouts:(i + 1) * rollouts]
        correct = sum(first_diagnosis(env, GreedyPolicy(classifiers), seed, group, live) == tuple(start) for seed in own)
        labels[combo_key(start)] = {"separable": correct / rollouts > 0.5, "share": correct / rollouts, "seeds": own}
    reference = GreedyPolicy(classifiers)
    rules = rules_of(config)
    return {"rules_key": rules_key(rules), "rules": rules, "rollouts": rollouts, "seeds_after": EVALUATION_POOL,
            "history": {"recordings": HISTORY_RECORDINGS, "sensor_group": group}, "live": live, "confidence": reference.confidence,
            "min_gain_per_price": reference.min_gain_per_price, "folds": N_FOLDS, "fold_seed": FOLD_SEED, "rule": RULE,
            "code": code_digest(static.asset_id), "labels": labels}


def ensure_labels(config, cycles, blocks, path=None):
    """The label table of the configuration's rules key, computed on first use and kept in path. The read, the computation
    and the write run under a lock on the file, so two processes that fill it at once keep each other's tables and the one
    that waited reads the table the other wrote. Without a path the table is LABELS_PATH as the module holds it at the call,
    which the tests point at a copy. The file names the SHA-256 of the data file its tables were computed from, and a file
    computed from another data file is refused."""
    path = Path(LABELS_PATH if path is None else path)
    key = _table_key(config)
    facts_sha256 = getattr(cycles, "facts_sha256", None)  # None for recordings that came from no file, which get no table
    with locked(path):
        stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if stored.get("format") != LABELS_FORMAT:
            stored = {"format": LABELS_FORMAT, "facts_sha256": facts_sha256, "tables": {}}
        records.require_same_data(path, stored.get("facts_sha256"), facts_sha256, RELABEL)
        if _current_table(stored, key, config.asset_id, facts_sha256) is None:
            refuse_writing(f"the label table of the rules key {key[:12]}")
            stored["tables"][key] = compute_table(config, cycles, blocks)
            write_json_atomically(path, stored, indent=1)
    return stored["tables"][key]


def _table_key(config):
    """The rules key under which a label file keeps the table of the configuration."""
    return rules_key(rules_of(config))


def _current_table(stored, key, asset_id, facts_sha256):
    """The table of the rules key in a parsed label file when ensure_labels returns it without computing it and without a
    refusal: a file of the current format, built from the data file in use, that holds the key, written by the code of today.
    None in every other case."""
    if stored.get("format") != LABELS_FORMAT or facts_sha256 is None or stored.get("facts_sha256") != facts_sha256:
        return None
    table = stored["tables"].get(key)
    # a table written by other code is computed again, so a label never outlives a change of the engine
    return table if table is not None and table.get("code") == code_digest(asset_id) else None


def current_label_table(config, facts_sha256, path=None):
    """The stored label table of the configuration's rules key when it is current (the format, the data file and the code of
    today), else None, read without the lock and without computing anything: the check of the tables before a job."""
    return _current_table(_parsed(Path(LABELS_PATH if path is None else path)), _table_key(config), config.asset_id, facts_sha256)


_PARSED = {}  # path of a label file -> ((inode, size, modification time), the parsed file)


def _parsed(path):
    """The label file at path as it stands now, parsed again only when its inode, size or modification time changed, and {}
    without a file. It is read without the lock, since write_json_atomically replaces the file in one step, so a reader sees
    the old file or the new one whole."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {}
    stamp = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if path not in _PARSED or _PARSED[path][0] != stamp:
        _PARSED[path] = (stamp, json.loads(path.read_text(encoding="utf-8")))
    return _PARSED[path][1]


def label_of(env, path=None):
    """The label of the episode's start in the form the verifier reads, {"separable", "share", "rollouts", "source",
    "rules_key"}, or None when the table holds no such start (an anchored start outside the fault rule of the sampler). The
    table comes from the label file as it stands when ensure_labels would return it without computing it, and from
    ensure_labels in every other case."""
    path = Path(LABELS_PATH if path is None else path)
    table = _current_table(_parsed(path), _table_key(env.config), env.config.asset_id, env.facts_sha256)
    if table is None:
        cycles = records.Recordings(sorted(env.cycles.values(), key=lambda c: c.index), env.facts_sha256)
        table = ensure_labels(env.config, cycles, env.blocks, path)
    entry = table["labels"].get(combo_key(env.initial))
    if entry is None:
        return None
    return {"separable": entry["separable"], "share": entry["share"], "rollouts": table["rollouts"], "source": "greedy reference",
            "rules_key": table["rules_key"]}


def start_grade(env, rules=None):
    """The separability of the episode's start in the form the verifier reads: the label of label_of
    with the server's E18 grade of the start nested under "e18", {"separable", "commit_share", "first_share", "cost"}, read from
    the grade table of the rules key of the episode's released rules, rules or the public profile, whatever the row serves, so
    that one rule scores every row. E18 is separable under the majority rule of the label, and a start on which no rollout
    commits is not. None only for a start outside the fault rule of the sampler, such as an anchored day, which no label table
    holds by design, since absence is not inseparability. A start of the fault rule that the label table or the grade table
    lacks raises, as a failed lookup does, since scoring it as unknown would withhold or give the credit of a
    declaration for no reason the verdict shows. The models and the grade table are read only when they were built from
    the data file that the environment plays."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools.fitted_models import load_models
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.fit_models import MODELS_PATH, REFIT
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.diagnosability_grade import combo_key as grade_key, ensure_grades, separable

    label = label_of(env)
    if label is None:
        # the label table holds every start of the fault rule, so only a start outside it, such as an anchored day, has no label
        if tuple(env.initial) in {tuple(c) for c in env.candidates}:
            raise LookupError(f"the label table of the episode's rules holds no start {combo_key(env.initial)}")
        return None
    models = load_models()
    records.require_same_data(MODELS_PATH, models.raw["facts_sha256"], env.facts_sha256, REFIT)
    grades = ensure_grades(models, env.public_profile() if rules is None else rules)["grades"]
    key = grade_key(env.initial)
    if key not in grades:
        raise LookupError(f"the E18 grade table of the episode's rules holds no start {key}")
    entry = grades[key]
    if entry is None:  # no E18 rollout commits to the start within the budget
        e18 = {"separable": False, "commit_share": 0.0, "first_share": 0.0, "cost": None}
    else:
        e18 = {"separable": separable(entry), "commit_share": entry["commit_share"], "first_share": entry["first_share"], "cost": entry["cost"]}
    return {**label, "e18": e18}


def main(argv=None):
    ap = argparse.ArgumentParser(description="fill the separability label table of a configuration")
    ap.add_argument("--config", type=Path, default=None, help="a JSON file of an EpisodeConfig, the default configuration without it")
    ap.add_argument("--out", type=Path, default=LABELS_PATH)
    a = ap.parse_args(argv)
    config = EpisodeConfig.from_json(json.loads(a.config.read_text(encoding="utf-8"))) if a.config else EpisodeConfig()
    cycles = records.load_cycles(records.default_facts_path(PROFILE), PROFILE)
    table = ensure_labels(config, cycles, records.build_blocks(cycles), a.out)
    separable = sum(entry["separable"] for entry in table["labels"].values())
    print(f"{a.out}: rules key {table['rules_key'][:12]}, {separable} of {len(table['labels'])} starts separable")


if __name__ == "__main__":
    main()
