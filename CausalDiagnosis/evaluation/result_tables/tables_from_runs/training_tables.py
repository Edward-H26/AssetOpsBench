import json
from pathlib import Path
from CausalDiagnosis.utils.csv_rows import as_bool, as_float  # noqa: E402
from CausalDiagnosis.utils.json_files import read_json  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import DEFAULT_ASSET  # noqa: E402
from CausalDiagnosis.evaluation.result_tables import pooling_rules  # noqa: E402
from CausalDiagnosis.experiments.table_definitions.training_variants import REGISTERED, VISITS, loop_counts  # noqa: E402
from CausalDiagnosis.experiments.table_definitions.experiment_tables import ADAPTER_VARIANTS, AS_RELEASED, GATE_FIELDS, GATE_PATHS, TRAINING_BASELINES, TRAINING_COMPARISONS, TRAINING_MEASURES, TRAINING_PRESET, TRAINING_FOLD_KEY, TRAINING_RECORD_SOURCES, TRAINING_ROWS, TRAINING_VARIANT_KEY, fold_ids, fold_of, row_label  # noqa: E402
from CausalDiagnosis.evaluation.metrics.table_statistics import rounded as _r  # noqa: E402  the one rounding rule of a table cell, which mcp_metrics shares
from . import cell_statistics, family_correction, episode_selection, judge_tables


# ----- the online loop of GEPA (GEPA, GEPA_curve) -----
GEPA_ONLINE = "gepa_online"  # the mode of run.json of gepa_prompt_optimization/, whose runs enter only the tables that read ALL_EPISODES
GEPA_RESULT = "gepa_online.json"  # the result file of a run of the online loop


def gepa_runs(episodes):
    """The finished runs of the online loop among the episodes as (run folder, result file, {GEPA call: episode}), in
    folder order. A run whose result file is missing or says running enters no table until it finishes."""
    runs = []
    for _, eps in cell_statistics.groups([e for e in episodes if e.get("mode") == GEPA_ONLINE], "run_dir").items():
        run_dir = Path(eps[0]["run_dir"])
        result = read_json(run_dir / GEPA_RESULT, {})
        if result.get("status") == "finished":
            runs.append((run_dir, result, {str(e.get("gepa_call")): e for e in eps}))
    return runs


def mean_reward(episodes):
    """The mean of the verifier's reward over the episodes it scored, None without one, the score of a candidate in the
    loop, where an episode that the verifier could not score enters no mean."""
    rewards = [r for r in (as_float(e.get("reward")) for e in episodes) if r is not None]
    return sum(rewards) / len(rewards) if rewards else None


def _gepa_cells(run_dir, result, episodes):
    first = episodes[0] if episodes else {}
    return {"run": run_dir.name, "model": first.get("model", ""), "config": first.get("config", ""), "card": result.get("card", ""),
            "reflection_model": result.get("reflection_model", "")}


def table_gepa(episodes, ctx):
    """One row per finished run of the online loop with what the loop did: its card and reflection model, the candidates that
    played on the stream, the stream episodes, every member's validation on the blocks counted, those the loop left out, and the
    episodes played until the final candidate was first chosen. A loop plays training start states alone, so the comparison of
    a block with no block is TRAINING_pairs, from the evaluation runs that play the test set once."""
    rows = []
    for run_dir, result, by_call in gepa_runs(episodes):
        played, counts = list(by_call.values()), loop_counts(result)
        row = {**_gepa_cells(run_dir, result, played), "candidates_played": counts["candidates_played"],
               "stream_episodes": counts["stream_episodes"], "left_out": counts[VISITS], "rollouts_to_best": result.get("rollouts_to_best", "")}
        rows.append(cell_statistics.trace(ctx, row, played))
    return rows


def table_gepa_curve(episodes, ctx):
    """One row per round of every finished run of the online loop, the point of the online curve: the stream episodes
    played before the round, validation counted, and the mean reward of the incumbent fixed after the round before on
    the block of the round, scored before the reflection model saw it, over its episodes of the round that the loop did not leave out."""
    rows = []
    for run_dir, result, by_call in gepa_runs(episodes):
        base = _gepa_cells(run_dir, result, list(by_call.values()))
        for point in result.get("rounds") or []:
            calls = [e.get("call") for e in result.get("episodes") or []
                     if e.get("phase") == "stream" and e.get("round") == point.get("round") and e.get("candidate") == point.get("incumbent")
                     and not e["left_out"]]
            eps = [by_call[str(call)] for call in calls if str(call) in by_call]
            row = {**base, "round": point.get("round"), "rollouts": point.get("rollouts"), "mean_reward": _r(mean_reward(eps)),
                   "n": sum(1 for e in eps if as_float(e.get("reward")) is not None)}
            rows.append(cell_statistics.trace(ctx, row, eps))
    return rows


# ----- the trained agents (TRAINING, TRAINING_pairs) and what they trained on (TRAINING_RECORD) -----

# the subsets of the training tables: every test episode, each played by a model that never saw its fault combination, on which
# the training hypotheses decide, the single fault and the two fault starts, and in the hypotheses the episodes of each fold, all reported
# beside the decision
TRAINING_SUBSETS = ("all", *episode_selection.FAULT_COUNTS)
TRAINING_DECISION = "all"


def training_pair_subsets(asset_id=DEFAULT_ASSET):
    """The subsets of the training comparisons: those of TRAINING_SUBSETS and one per fold of the asset's registered split."""
    return (*TRAINING_SUBSETS, *(f"fold_{fold}" for fold in fold_ids(asset_id)))


def episode_fold(episode):
    """The fold of the fault combination an episode starts in, from its block cell of scores.csv and the split of its run, of
    its asset (training.fold_of). An episode without a start combination lies in no fold, and the error names its run and seed."""
    if not episode.get("block"):
        raise ValueError(f"run {episode.get('run')}, seed {episode.get('seed')}: the episode has no start combination, so it lies in no fold")
    return fold_of(json.loads(episode["block"]), episode.get("asset_id") or DEFAULT_ASSET, episode.get("training_split") or REGISTERED)


def in_subset(episode, subset):
    """Whether an episode belongs to a subset of the training tables: every episode, the starts of the faulty components that
    FAULT_COUNTS gives the subset, or the episodes of a fold of the cross-fitting, those of the combinations in that fold."""
    if subset in episode_selection.FAULT_COUNTS:
        return as_float(episode.get("n_faulty")) == episode_selection.FAULT_COUNTS[subset]
    return subset == "all" or episode_fold(episode) == int(subset.removeprefix("fold_"))


def cross_fitted(episodes):
    """The episodes after the check of the cross-fitting: a trained run names in run.json the fold its model holds out and
    plays only episodes of that fold, else the script stops and names the run, since its episodes would score a model on the
    combinations it trained on."""
    for e in episodes:
        if e["training_variant"] != AS_RELEASED and (e.get("training_fold") not in fold_ids(e.get("asset_id") or DEFAULT_ASSET,
                                                                                            e.get("training_split") or REGISTERED)
                                                     or episode_fold(e) != e["training_fold"]):
            raise ValueError(f"run {e['run']} of the variant {e['training_variant']} names the held-out fold {e.get('training_fold')!r} and plays an "
                             f"episode of fold {episode_fold(e)}, while a trained run plays only the fold its model holds out")
    return episodes


def training_row(episode):
    """The row of design.TRAINING_ROWS or design.TRAINING_BASELINES that an episode plays, or None."""
    key = (episode.get("agent"), as_bool(episode.get("diagnostics_server")))
    return next((row for row in (*TRAINING_ROWS, *TRAINING_BASELINES) if (row_label(row, episode["model"]), row[1]) == key), None)


def baselines_on_the_same_episodes(episodes):
    """The episodes after the check of the baselines of the training table: the baseline row of an agent, the agent without the MCP on the top rung of the ladder, and
    its untrained row with the MCP play the same seeds of the test set, else the script stops and names the agent, since the training
    table reads the baseline beside that row as a cell of the same episodes."""
    seeds = {}
    for e in episodes:
        if e["training_variant"] == AS_RELEASED:
            row = training_row(e)
            seeds.setdefault((e["model"], row[0], row[1]), set()).add(str(e.get("seed")))
    for (model, prefix, server), played in seeds.items():
        with_mcp = seeds.get((model, prefix, True))
        if not server and with_mcp is not None and played != with_mcp:
            raise ValueError(f"the baseline row of {row_label((prefix, server), model)} without the MCP plays {len(played)} seeds and its untrained row "
                             f"with the MCP {len(with_mcp)}, {len(played ^ with_mcp)} of them not in both, while the training table reads both on the "
                             "same test episodes")
    return episodes


def adapter_markers(runs_root):
    """The marker runs under the runs root: per model, agent label, diagnostics server, variant and training fold, the run folder
    whose run.json says that no adapter was trained for that agent and fold, one entry however many replicate lines wrote a marker."""
    markers = {}
    for run_dir in pooling_rules.folders_read(Path(runs_root)) if runs_root else []:
        meta = read_json(run_dir / "run.json", {})
        if (meta.get("adapter") or {}).get("trained") is False:
            key = (meta.get("model", ""), meta.get("agent_label", ""), as_bool(meta.get("diagnostics_server")), meta.get(TRAINING_VARIANT_KEY),
                   meta.get(TRAINING_FOLD_KEY))
            markers.setdefault(key, run_dir)
    return markers


def aliased(kept, markers):
    """The episodes of the training tables with the folds that trained no adapter filled in: for each marker, the episodes of no training of its agent on
    the fold its model is tested on, under the marker's variant, fold and run, marked aliased. A marker beside an evaluation run of the
    same variant and fold stops the script, since a fold either trained its adapter or did not."""
    played = {(e["model"], e["agent"], as_bool(e.get("diagnostics_server")), e["training_variant"], e.get("training_fold")) for e in kept}
    out = list(kept)
    for (model, agent, server, variant, fold), run_dir in sorted(markers.items(), key=lambda item: str(item[0])):
        if variant not in ADAPTER_VARIANTS:
            raise ValueError(f"the marker run {run_dir.name} names the variant {variant!r}, while only {', '.join(ADAPTER_VARIANTS)} need an adapter")
        if (model, agent, server, variant, fold) in played:
            raise ValueError(f"the marker run {run_dir.name} says that {agent} trained no adapter on fold {fold}, while an evaluation run of {variant} "
                             "on that fold exists")
        out += [{**e, "training_variant": variant, "training_fold": fold, "run": run_dir.name, "run_dir": run_dir, "aliased": True} for e in kept
                if e["training_variant"] == AS_RELEASED and (e["model"], e["agent"], as_bool(e.get("diagnostics_server"))) == (model, agent, server)
                and episode_fold(e) == fold]
    return out


def training_episodes(episodes, ctx):
    """The episodes of the training tables: the three rows with the MCP (design.TRAINING_ROWS) on the preset of training at the main
    level of every setting but the training variant, which the tables vary, and the untrained episodes of the three agents without the
    MCP on that preset (design.TRAINING_BASELINES), their reference lines, which the training table prints as baselines, and
    the folds that trained no adapter read through the runs of no training. Every row reads one serving setup (one_setup). The
    selection is made once per episode list and kept in ctx, since four tables read it and each pass reads the marker runs."""
    cache = ctx.setdefault("training_episodes", {})
    if id(episodes) not in cache:
        cache[id(episodes)] = (episodes, _training_episodes(episodes, ctx))  # the list stays referenced, so its id names it while the entry lives
    return cache[id(episodes)][1]


def _training_episodes(episodes, ctx):
    kept = [e for e in episode_selection.agent_episodes(episodes) if e.get("config") == TRAINING_PRESET
            and episode_selection.at_main_levels({**e, "training_variant": AS_RELEASED, "lora_adapters": episode_selection.MAIN_LEVELS["lora_adapters"]})
            and (training_row(e) in TRAINING_ROWS or (training_row(e) in TRAINING_BASELINES and e["training_variant"] == AS_RELEASED))]
    return baselines_on_the_same_episodes(cross_fitted(aliased(one_setup(kept), adapter_markers(ctx.get("runs")))))


def on_adapters(episode):
    """Whether an episode played on a server that holds LoRA adapters beside the model (run.json lora_adapters above 0)."""
    return int(episode.get("lora_adapters") or 0) > 0


def one_setup(episodes):
    """The episodes of the training tables on one serving setup per model, tool surface and row: on a server with LoRA adapters when
    any episode of the row played there, since a variant that plays an adapter needs that server and every comparison of the family
    reads its variants on one server, else on the server without. So the runs without training that the part environment plays on
    the server of the agents table never pair with a trained variant, and the copies that the part training plays beside the
    adapters do."""
    rows = {}
    for e in episodes:
        rows.setdefault((e.get("model"), e.get("surface"), e.get("agent"), as_bool(e.get("diagnostics_server"))), []).append(e)
    return [e for eps in rows.values() for e in eps if on_adapters(e) == any(on_adapters(x) for x in eps)]


def table_training(episodes, ctx):
    """The verified outcome after training: per model, tool surface, row with the MCP, training variant of design.TRAINING_VARIANTS
    and subset of TRAINING_SUBSETS, every episode and the single fault and two fault starts, the pass and the top-1 accuracy with
    intervals from a bootstrap clustered by block, and the three scores of the judge under the five call protocol over the episodes
    of the row, an episode without a judge record false on every criterion as in JUDGE (without_record, counting.unjudged) and one
    judged under another protocol out of the _n cells, with the counts that the page states above its tables. A row without a judge
    record of that protocol leaves the scores empty."""
    rows = []
    for (model, surface, agent, server, variant), eps in cell_statistics.groups(training_episodes(episodes, ctx), "model", "surface", "agent", "diagnostics_server",
                                                               "training_variant").items():
        for subset in TRAINING_SUBSETS:
            kept = [e for e in eps if in_subset(e, subset)]
            row = {"model": model, "config": TRAINING_PRESET, "surface": surface, "agent": agent, "diagnostics_server": server,
                   "training_variant": variant, "subset": subset, "n": len(kept)}
            for m in (*TRAINING_MEASURES, "unmatched_evidence"):
                row.update(cell_statistics.summary(kept, m, ctx))
            row.update(judge_tables.judge_kappa(kept, ctx))
            judged = [e for e in kept if e.get("judge_record_protocol") == cell_statistics.PAPER_PROTOCOL]
            unjudged = judge_tables.without_record(kept)
            row.update({"judged": len(judged), "aliased": sum(1 for e in kept if e.get("aliased")), "episodes_without_record": len(unjudged),
                        "failed_calls": sum(e["judge_failed_calls"] for e in judged), "declared": sum(1 for e in kept if as_bool(e.get("abstained")))})
            for criterion in cell_statistics.JUDGE_CRITERIA:
                row.update(cell_statistics.summary(judged + unjudged if judged else [], criterion, ctx))
            rows.append(cell_statistics.trace(ctx, row, kept))
    return rows


def table_training_pairs(episodes, ctx):
    """The training hypotheses: per model, tool surface and row with the MCP, the comparisons of design.TRAINING_COMPARISONS, each
    trained variant against the agent as released, GEPA against GEPA on the score alone and fine tuning against its control, fine-tuning on
    unfiltered trajectories, on the
    paired differences of pass and top-1, matched on seed, scenario and every factor, with intervals from a bootstrap clustered by
    block. Each comparison has a row per subset of training_pair_subsets: all, the decision on every episode of the test set, with
    Holm over the tests of those rows of a model and tool surface, 30 when every variant has run, and the single fault and the two fault
    starts and the episodes of each fold, printed beside the decision with their intervals and no test. The decision of a comparison
    whose rows are incomplete or refused is not run (training_status) and enters the Holm at p = 1; every row of it prints the status."""
    rows = []
    for (model, surface), eps in cell_statistics.groups(training_episodes(episodes, ctx), "model", "surface").items():
        family = []
        # the folds of the registered split of the asset the episodes played, so that a table of training is laid out per asset
        asset = next((e["asset_id"] for e in eps if e.get("asset_id")), DEFAULT_ASSET)
        for row in TRAINING_ROWS:
            arm = episode_selection._arm(eps, row_label(row, model), row[1])
            for variant, reference, name in TRAINING_COMPARISONS:
                level = [e for e in arm if e.get("training_variant") == variant]
                against = [e for e in arm if e.get("training_variant") == reference]
                if not level or not against:
                    continue
                cells = {"model": model, "config": TRAINING_PRESET, "surface": surface, "agent": row_label(row, model), "diagnostics_server": row[1],
                         "training_variant": variant, "reference_variant": reference, "comparison": name,
                         "status": training_status(level, arm, ctx) or training_status(against, arm, ctx)}
                for subset in training_pair_subsets(asset):
                    kept, kept_against = [e for e in level if in_subset(e, subset)], [e for e in against if in_subset(e, subset)]
                    if cells["status"] and subset == TRAINING_DECISION:
                        kept, kept_against = [], []
                    paired = cell_statistics._paired_cells(kept, kept_against, "training_variant", TRAINING_MEASURES, ctx)
                    if subset != TRAINING_DECISION:
                        paired.update({f"{m}_p": "" for m in TRAINING_MEASURES})
                    rows.append(cell_statistics.trace(ctx, {**cells, "subset": subset, **paired}, kept, kept_against))
                    if subset == TRAINING_DECISION:
                        family.append(rows[-1])
        for row in family_correction.holm_over(family, TRAINING_MEASURES, ctx["alpha"], family_correction.registered_size(ctx, "training")):
            row["reading"] = family_correction.reading(row, TRAINING_MEASURES)
    return rows


def training_status(side, arm, ctx):
    """The status of one side of a training comparison (row_status): its episodes that are no alias, and, when it holds the aliased
    episodes of a fold that trained no adapter, the episodes of the same row as released, whose completeness those copies share."""
    plain = [e for e in side if not e.get("aliased")]
    released = [e for e in arm if e["training_variant"] == AS_RELEASED and not e.get("aliased")]
    return cell_statistics.row_status(plain, ctx) or (cell_statistics.row_status(released, ctx) if len(plain) < len(side) else "")


def record_value(meta, path):
    """The value at a path of keys of run.json, or None where a key is missing."""
    for key in path:
        meta = meta.get(key) if isinstance(meta, dict) else None
    return meta


def table_training_record(episodes, ctx):
    """What each agent trained on (the budget table): per model, row with the MCP and fold of the cross-fitting, the fields of
    design.TRAINING_RECORD_SOURCES from run.json of its trained evaluation runs, where the training pipeline records the adapter and the
    task block that the model of a fold plays. A field that no run of its variants carries stays empty, and runs that disagree on a field
    stop the script, since the replicate runs of a variant and fold play one adapter and one block. A row traces to the runs it read."""
    rows = []
    trained = [e for e in training_episodes(episodes, ctx) if e["training_variant"] != AS_RELEASED]
    for (model, agent, _, fold), eps in cell_statistics.groups(trained, "model", "agent", "diagnostics_server", "training_fold").items():
        variants = {Path(e["run_dir"]): e["training_variant"] for e in eps}
        metas = {run_dir: read_json(run_dir / "run.json", {}) for run_dir in variants}
        row = {"agent": agent, "model": model, "fold": fold}
        for field, (carriers, path) in TRAINING_RECORD_SOURCES.items():
            found = {run_dir.name: record_value(metas[run_dir], path) for run_dir, variant in sorted(variants.items())
                     if variant in carriers}
            found = {name: value for name, value in found.items() if value is not None}
            if len({json.dumps(value, sort_keys=True) for value in found.values()}) > 1:
                raise ValueError(f"the runs of {agent} on fold {fold} disagree on {field} of the budget table: "
                                 + ", ".join(f"{name} {value}" for name, value in found.items()))
            row[field] = next(iter(found.values()), "")
        gates = {}
        for run_dir, variant in sorted(variants.items()):
            meta = metas[run_dir]
            gate = next((value for value in (record_value(meta, path) for path in GATE_PATHS) if value), None)
            if variant in ADAPTER_VARIANTS and gate:
                gates[run_dir.name] = {field: gate.get(field, "") for field in GATE_FIELDS}
        if len({json.dumps(gate, sort_keys=True) for gate in gates.values()}) > 1:
            raise ValueError(f"the runs of {agent} on fold {fold} disagree on the gate of fine tuning: "
                             + ", ".join(f"{name} {gate}" for name, gate in gates.items()))
        row.update({f"gate_{field}": value for field, value in next(iter(gates.values()), dict.fromkeys(GATE_FIELDS, "")).items()})
        rows.append(cell_statistics.trace(ctx, row, eps))
    return rows
