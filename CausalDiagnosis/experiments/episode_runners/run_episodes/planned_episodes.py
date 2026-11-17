import dataclasses
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import episode_name  # noqa: E402
from CausalDiagnosis.datasets.splits.development_split import DEVELOPMENT_SEED, coverage_order, development_starts, faulty_levels, on_starts  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import start_state_seeds as start_states  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.run_configuration import scenario_set  # noqa: E402
from CausalDiagnosis.experiments.table_definitions.training_variants import AS_RELEASED, DEVELOPMENT, TRAINED_AGENTS, asset_training  # noqa: E402
from . import play_one_episode


# ----- the plan of a run -----
def episode_plan(config, seeds, cycles, blocks, scenario_ids=None):
    """[(folder name, seed, scenario or None, config of the episode)]. The anchored preset plays every scenario of the scenario
    set that a framework declares for the asset of the preset (run_setup.scenario_set), or those of scenario_ids alone (named_scenarios), under every seed, anchored at the scenario's date with the scenario text as the
    question, the scenario's day as the day of cycle 0 and the asset id that the texts name as another id of the asset (anchors),
    and stops with the command that fetches the file when it is missing; every other config plays the seeds as they are."""
    if config.anchors is None:
        return [(episode_name(seed), seed, None, config) for seed in seeds]
    anchors = scenario_set(config.asset_id)
    date0 = config.anchors.get("date0", anchors.DATE0)
    plan = []
    for scenario in named_scenarios(anchors.load_scenarios()["scenarios"], scenario_ids):
        anchored = anchors.anchor(scenario["date"], cycles, blocks, date0)
        episode_config = dataclasses.replace(config, start=anchors.start_of(anchored), question=scenario["text"],
                                             anchors={"date0": scenario["date"], "asset_alias": anchors.ASSET_ALIAS})
        for seed in seeds:
            plan.append((episode_name(seed, scenario["id"]), seed, {**scenario, "anchor": anchored}, episode_config))
    return plan


def named_scenarios(scenarios, scenario_ids):
    """The scenarios of a scenario file in its order, those whose ids --scenario-ids names alone when it names any, refused by name
    for an id that the file does not hold."""
    if scenario_ids is None:
        return scenarios
    held = [scenario["id"] for scenario in scenarios]
    unknown = [i for i in scenario_ids if i not in held]
    if unknown:
        raise SystemExit(f"--scenario-ids names {'/'.join(map(str, unknown))}, which the scenario file does not hold; its ids are "
                         f"{'/'.join(map(str, held))}")
    return [scenario for scenario in scenarios if scenario["id"] in scenario_ids]


def chosen_seeds(a, config, cycles, blocks):
    """The seeds a run plays: --seeds of them from --first-seed, with --starts combinations the --seeds smallest seeds
    that draw each start combination of the config, with --starts sampled the seeds of one replicate of the test set,
    one per combination, the replicate of replicate_of, for a trained variant only those of the combinations of the fold
    its model holds out, a cut look (--fraction) those seeds of replicate 1 that start_states.cut_seeds keeps, or with --starts
    training the seeds of the training sample of that model (the anchored preset anchors its scenarios and keeps the plain seeds)."""
    if a.starts in ("combinations", "development") and config.anchors is None:
        return start_states.combination_seeds(HydraulicRigEnv(cycles, blocks, config), a.seeds)
    if (replicate := replicate_of(a, config)) is not None:
        env = HydraulicRigEnv(cycles, blocks, config)
        if a.training_fold is not None:  # the model of a fold plays the test episodes of the fold it holds out alone
            # the split is read only where a fold is named, so an asset without training files plays its runs without training
            split = asset_training(config.asset_id, cycles.facts_sha256, split=a.training_split).split
            seeds = start_states.fold_replicate_seeds(env, split, replicate, a.training_fold, a.master_seed, a.seed_pool)
        else:
            seeds = start_states.replicate_seeds(env, replicate, a.master_seed, a.seed_pool)
        return seeds if getattr(a, "fraction", None) is None else start_states.cut_seeds(env, seeds, a.fraction, a.master_seed)
    if a.starts == "training" and config.anchors is None:
        split = asset_training(config.asset_id, cycles.facts_sha256, split=a.training_split).split
        return start_states.training_seeds(HydraulicRigEnv(cycles, blocks, config), split, training_sample_of(a), a.training_fold,
                                           a.master_seed, a.seed_pool)
    return list(range(a.first_seed, a.first_seed + a.seeds))


def training_sample_of(a):
    """The training sample a run of --starts training plays, --training-sample or the first, or None for a run of other starts."""
    return (a.training_sample or 1) if a.starts == "training" else None


def training_checks(a, config):
    """Refuses the training options where they do not apply: a trained variant or the training starts belong to the three agents
    of the training table, the sampling of one round plays the agent without training, the samples name the training starts,
    a trained variant or a sampling run names its fold, which the agent without training has not, and a trained variant plays the
    test set, --starts sampled, whose episodes of the fold its model holds out it never trained on."""
    if a.training_variant != AS_RELEASED and a.agent not in TRAINED_AGENTS:
        raise SystemExit(f"--training-variant {a.training_variant} trains one of {', '.join(TRAINED_AGENTS)}, not {a.agent}")
    if a.training_sample is not None and a.starts != "training":
        raise SystemExit("--training-sample names a sample of the training starts, so it needs --starts training")
    if a.starts == "training" and (a.agent not in TRAINED_AGENTS or a.training_variant != AS_RELEASED or config.anchors is not None):
        raise SystemExit(f"--starts training samples one of {', '.join(TRAINED_AGENTS)} without training on a seeded preset, the one round of "
                         "rejection sampling that configs/training/<asset_id>/rft.json registers")
    trained = a.training_variant != AS_RELEASED or a.starts == "training"
    if a.training_fold is not None and a.training_fold not in asset_training(config.asset_id, split=a.training_split).fold_ids:
        raise SystemExit(f"--training-fold {a.training_fold} is no fold of the {a.training_split} split of {config.asset_id}, whose folds are "
                         f"{asset_training(config.asset_id, split=a.training_split).fold_ids}")
    if a.starts == "development" and (trained or config.anchors is not None):
        raise SystemExit("--starts development plays the development starts without training on a seeded preset")
    if trained and a.training_fold is None:
        raise SystemExit("a trained variant and the training starts belong to one fold of the registered split, so they need --training-fold")
    if not trained and a.training_fold is not None:
        raise SystemExit("--training-fold names the fold of a trained model or of a sampling run, and the agent without training plays "
                         "every test episode in one line")
    if a.training_variant != AS_RELEASED and a.starts != "sampled":
        raise SystemExit(f"--training-variant {a.training_variant} plays the test episodes of the fold its model holds out, so it needs "
                         f"--starts sampled, and --starts {a.starts} would play combinations its model trained on")


# the start sets that choose the starts of a run by a search of seeds, which a pinned start would leave with one start to find
CHOOSING_START_SETS = ("combinations", "sampled", "training")


def start_word(start):
    """A field start in the form of --start-state: the levels joined by /, then @ and the entry offset."""
    return "/".join(map(str, start["combination"])) + f"@{start['entry_offset']}"


def selector_checks(a, config):
    """Refuses a selector where it cannot play what it names: --start-state, which fixes the start of every episode, beside a start
    set that chooses the starts itself, on the development split and on a preset whose scenarios carry their own starts, and
    --scenario-ids on a preset that anchors no scenario. A preset whose field start pins a start is refused alike beside a start set
    that chooses the starts by a search of seeds, before the search, since every seed would draw the pinned start, unless the run
    plays the development split, whose widening drops the pin (development_config)."""
    if config.start is not None and a.starts in CHOOSING_START_SETS and a.training_split != DEVELOPMENT:
        raise SystemExit(f"the preset {a.config} pins the start state {start_word(config.start)}, and --starts {a.starts} chooses the "
                         "starts itself: play a pinned start with --starts seeds")
    if a.start_state is not None:
        if a.starts != "seeds":
            raise SystemExit(f"--start-state fixes the start state of every episode, so it needs --starts seeds, and --starts {a.starts} "
                             "chooses the starts itself")
        if a.training_split == DEVELOPMENT:
            raise SystemExit("--start-state plays the start state it names on the preset, and --training-split development plays the "
                             "development starts")
        if config.anchors is not None:
            raise SystemExit("--start-state fixes the start state of every episode, and the anchored preset takes the start of each "
                             "scenario from its date: --scenario-ids names the scenarios it plays")
    if a.scenario_ids is not None and config.anchors is None:
        raise SystemExit("--scenario-ids names scenarios of the scenario set of the anchored preset, and this preset anchors none")


def development_config(config, blocks, cap=None):
    """The configuration of a run of --starts development or --training-split development: the preset with its fault rule and its
    allowed starts replaced by the range of fault counts and the starts of the development starts of its asset, or of cap of them
    (capped_starts), so that reset draws them alone, disjoint from the test set, and run.json records them. A start that the preset
    pins in its field start names the start of the line's own play, so the widened configuration drops it and every seed draws a
    development start, in a filler run and in the tables of the development starts alike."""
    try:
        development = development_starts(play_one_episode.get_profile(config.asset_id), set(blocks))
    except ValueError as e:  # development_split.SplitRefused
        raise SystemExit(f"--starts development plays the development starts of the asset, and the asset {config.asset_id} has none: {e}") from None
    if cap is not None:
        development = capped_starts(development, cap, [config.healthy[c] for c in config.components])
    return on_starts(dataclasses.replace(config, start=None), development)


def played_config(a, config, blocks):
    """The configuration a run plays, chosen before any seed: for --starts development and for --training-split development the
    preset widened to the development starts of its asset (development_config), with --start-state the preset whose field start
    names that start state (started_config), else the preset itself."""
    if a.starts == "development" or a.training_split == DEVELOPMENT:
        return development_config(config, blocks, a.start_cap)
    if a.start_state is not None:
        return started_config(a, config)
    return config


def start_state_of(a, config):
    """The field start of EpisodeConfig that --start-state names, {"combination": the levels in component order, "entry_offset":
    the offset}, or None without the option. The option writes the level of every component of the asset in their order, each as
    the asset file writes it, joined by /, then @ and the entry offset, one shell word that a grid line and GRID_EXTRA carry as it
    is: 3/100/0/130@2 on the rig."""
    text = getattr(a, "start_state", None)
    if text is None:
        return None
    levels, at, offset = text.rpartition("@")
    words = levels.split("/") if at else []
    if len(words) != len(config.components) or not offset.isdigit():
        raise SystemExit(f"--start-state {text}: write the level of each component in the order {'/'.join(config.components)}, joined "
                         "by /, then @ and the entry offset")
    combination = []
    for component, word in zip(config.components, words):
        level = next((level for level in config.levels[component] if str(level) == word), None)
        if level is None:
            raise SystemExit(f"--start-state {text}: {word} is no level of {component}, whose levels are "
                             f"{'/'.join(map(str, config.levels[component]))}")
        combination.append(level)
    return {"combination": combination, "entry_offset": int(offset)}


def started_config(a, config):
    """The configuration of a run of --start-state: the preset with its field start set to the start state that the option names
    (start_state_of), as a preset whose field start names it, refused by name when the fault rule of the preset does not allow the
    combination. recorded_start refuses a combination without a recording at that offset once the data are loaded."""
    start = start_state_of(a, config)
    if not config.within_fault_rule(start["combination"]):
        rule = config.fault_rule
        raise SystemExit(f"--start-state {a.start_state}: the fault rule of the preset allows {rule['min_faulty']} to "
                         f"{rule['max_faulty']} faulty components, and the combination holds {config.n_faulty(start['combination'])}")
    return type(config).from_json({**config.to_json(), "start": start})


def recorded_start(a, config, cycles, blocks):
    """Refuses a run of --start-state, before its run folder exists, whose combination the data of the asset hold no recording of or
    whose entry offset they do not serve, by the check that the environment applies to a preset whose field start names a start."""
    try:
        HydraulicRigEnv(cycles, blocks, config)
    except ValueError as e:
        raise SystemExit(f"--start-state {a.start_state}: {e}") from None


def replicate_of(a, config):
    """The replicate of the test set that a run of --starts sampled plays, --replicate or the first, one run per replicate so
    that a line of the grid is start_states.REPLICATES runs and a job cut short at its time limit leaves whole runs,
    or None for a run of other starts."""
    return (a.replicate or 1) if a.starts == "sampled" and config.anchors is None else None


def fraction_checks(a):
    """Refuses --fraction where it plays no cut look: off the test set, a whole share of replicates, which a grid plays as those
    replicates with no flag, a share of neither kind (start_states.fraction_plan) and a replicate other than the first."""
    if a.fraction is None:
        return
    if a.starts != "sampled":
        raise SystemExit("--fraction cuts the test set, so it needs --starts sampled")
    try:
        kind, value = start_states.fraction_plan(a.fraction)
    except ValueError as e:
        raise SystemExit(f"--fraction: {e}") from None
    if kind == "whole":
        raise SystemExit(f"--fraction {a.fraction} is {value} whole replicates, which a grid plays as --replicate 1 to {value} "
                         "(pending_lines.py --fraction), and a run of a whole replicate records no share")
    if (a.replicate or 1) != 1:
        raise SystemExit("a cut look plays replicate 1 alone, so --fraction takes no other --replicate")


def start_combination(config, seed, cycles, blocks):
    """The start combination that the reset of an episode of the configuration draws under the seed."""
    env = HydraulicRigEnv(cycles, blocks, config)
    env.reset(seed)
    return tuple(env.initial)


def capped_starts(starts, cap, healthy, seed=DEVELOPMENT_SEED):
    """cap of the starts, in the order of starts: in an order that the seed shuffles, first every start that holds a faulty level
    of a component that no start taken before holds, then more in the same order up to cap. A cap below what covers every faulty
    level of the starts is refused with the number the coverage needs."""
    if cap < 1:
        raise SystemExit("--start-cap needs at least one start")
    covering, rest = coverage_order(starts, lambda start: faulty_levels(start, healthy), seed)
    if len(covering) > cap:
        raise SystemExit(f"--start-cap {cap} is below the {len(covering)} starts that cover every faulty level of the development starts")
    kept = set((covering + rest)[:cap])
    return [start for start in starts if start in kept]


def anchored_note(config, plan):
    """The two assumptions of the anchored preset and the anchor of every scenario, for run.json and the reports."""
    if config.anchors is None:
        return None
    anchors = scenario_set(config.asset_id)
    return {"assumptions": anchors.ASSUMPTIONS, "date0": config.anchors.get("date0", anchors.DATE0),
            "anchors": {name: {k: scenario["anchor"][k] for k in ("date", "cycle_index", "stable", "anchored_cycle_index", "block_position")}
                        for name, _, scenario, _ in plan}}
