from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from CausalDiagnosis.experiments.episode_runners.episode_launchers import DEFAULT_ABSTENTION_RULE, DEFAULT_RANKING_MODEL  # noqa: E402
from CausalDiagnosis.utils.csv_rows import as_bool  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.run_configuration import load_config  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.start_state_seeds import REPLICATES  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import start_state_seeds as start_states  # noqa: E402  the fractions of the test set (fraction_plan, fraction_text, fraction_value)
from CausalDiagnosis.evaluation.result_tables import pooling_rules  # noqa: E402
from CausalDiagnosis.experiments.table_definitions.experiment_tables import ABLATION_PLACES, ARM_PRESETS, AS_RELEASED, HEALTHY_CONTROL_PRESET, PARALLEL_OFF_SUFFIX, OURS, POLICY_TABLE_POLICIES, REFERENCE_POLICIES, AGENTS_PRESET, AGENT_ROWS, row_label  # noqa: E402
from . import cell_statistics


# ----- the read at one fraction of the test set -----
def fraction_share(kind, amount):
    """The share of the test set that a fraction of start_states.fraction_plan plays, as an exact fraction."""
    return Fraction(amount, REPLICATES) if kind == "whole" else Fraction(amount)


def read_of(fraction, plan):
    """The read of the tables at one fraction of the test set: its text in the one written form (start_states.fraction_text),
    which INDEX.json records and a rebuild passes back as it stands; its kind, whole (the replicates 1 to r) or cut (a prefix of
    the first replicate); the tag of a cut look in its one stored form; and partial when it lies below the fraction of record of
    the registered file, a read that runs no registered test."""
    record = str(plan["counting"]["fraction_of_record"])
    text = start_states.fraction_text(start_states.fraction_value(fraction))
    kind, amount = start_states.fraction_plan(text)
    return {"text": text, "kind": kind, "replicates": amount if kind == "whole" else 1,
            "tag": start_states.fraction_value(text) if kind == "cut" else None,
            "partial": fraction_share(kind, amount) < fraction_share(*start_states.fraction_plan(record))}


def in_read(episode, read):
    """Whether the read takes an episode: one outside the replicates of the test set always; at a whole fraction an untagged episode
    of the replicates 1 to r, so that every row is read on the same cells; at a cut look an episode tagged with exactly that fraction."""
    if episode.get("replicate") in (None, ""):
        return True
    tagged = episode.get("fraction") not in (None, "")
    if read["kind"] == "whole":
        return not tagged and int(episode["replicate"]) <= read["replicates"]
    return tagged and float(episode["fraction"]) == read["tag"]


def at_read(episodes, read):
    """(the episodes the read takes, {run folder: the reason it is left out}) for INDEX.json."""
    kept, left_out = [], {}
    for episode in episodes:
        if in_read(episode, read):
            kept.append(episode)
        else:
            left_out[Path(episode["run_dir"]).name] = f"outside the read at the fraction {read['text']}"
    return kept, left_out


# ----- the six rows of the agents and the CausalDiagnosis MCP (AGENTS, AGENTS_pairs, A13_A17, CALLS, STRATA) -----
AGENT_KEY = ("agent", "diagnostics_server")  # the factors a comparison between agents leaves out of the pairing key
SWITCH_OF_SUFFIX = {"nobranch": "branches_off", "single": "single_hypothesis", "noexp": "expectations_off", PARALLEL_OFF_SUFFIX: "parallel_off"}
# the settings that a line of the grid varies besides the preset, the clock, the tool surface and the server, held fixed in
# a row of AGENTS, CALLS and STRATA with those four and the agent and the model, so that neither an ablation line, a sensitivity
# line of the clock nor a trained agent pools into a row
# the condition of a row, the fields of results_index.CONDITION_FIELDS that keep rows apart, the one list of the condition of a run
CONDITION = pooling_rules.ROW_FIELDS
SETTINGS = CONDITION[CONDITION.index("diagnostics_server") + 1:]
PAIR_SETTINGS = ("clock_mode", "surface", *SETTINGS)  # the settings of the condition that AGENTS_pairs names beside model and config
FAULT_COUNTS = {"single": 1, "two": 2}  # the starts of EFFICIENCY, in the place of the single-agent and multi-agent tasks of Table 3 of AssetOpsBench


@lru_cache(maxsize=None)
def preset_config(config_name):
    """The EpisodeConfig of a config preset (configs/<name>.json over the defaults of EpisodeConfig), or None for a
    config that is no preset of this folder."""
    try:
        return load_config(config_name)[1]
    except SystemExit:
        return None


def config_clock_mode(config_name):
    """The clock mode of a config preset, the mode in which the hypotheses on the agents are tested; None for a config
    that is no preset of this folder, whose episodes keep every mode."""
    config = preset_config(config_name)
    return None if config is None else config.clock_mode


def in_config_mode(episode):
    """Whether the episode ran in the clock mode of its config preset. The grid plays some agents under other clock modes
    too (the thinking cycles of CALLS), and a summary row of the agent tables would otherwise mix them with
    the mode the hypotheses are tested in."""
    mode = config_clock_mode(str(episode.get("config", "")))
    return mode is None or episode.get("clock_mode") == mode


# the settings that run_episodes/ plays without an override (leaderboard.py reads them from here), and the settings
# whose main level is the one its config preset sets
# the main level of the serving setup is a server without LoRA adapters, the server of every table but the training tables
MAIN_LEVELS = {"abstention_rule": DEFAULT_ABSTENTION_RULE, "prices": "default", "ranking_model": DEFAULT_RANKING_MODEL, "training_variant": AS_RELEASED,
               "lora_adapters": 0}
PRESET_LEVELS = ("guidance_level", "verifier_scope")


def at_main_settings(episode):
    """Whether the episode ran every setting but the clock mode at the level of the main comparison: the guidance level
    and verifier scope of its config preset and the driver's abstention rule, price table and ranking model, with the agent as
    released. A config that is no preset of this folder sets no preset level, as in in_config_mode."""
    config = preset_config(str(episode.get("config", "")))
    levels = {**MAIN_LEVELS, **({} if config is None else {name: getattr(config, name) for name in PRESET_LEVELS})}
    return all(str(episode.get(name, "")) == str(level) for name, level in levels.items())


def at_main_levels(episode):
    """Whether the episode ran every setting at the level of the main comparison, the clock mode of its config preset
    and the settings of at_main_settings. The grid plays the ablations En4 to En7 and En9 on the presets of the agents table and of
    training at other levels, and a pair of the agents family would otherwise pool them with the main runs."""
    return in_config_mode(episode) and at_main_settings(episode)


def at_main_levels_but(episode, varied):
    """Whether the episode ran every setting but the varied one at the level of the main comparison (at_main_levels), the episodes of
    a table that varies one setting, so that a row prints one configuration."""
    return at_main_levels({**episode, varied: MAIN_LEVELS[varied]})


def agent_episodes(episodes, every_clock_mode=False):
    """The episodes of the agents of the six rows of design.AGENT_ROWS: the labels plan_execute_<model>, agent_as_tool_<model> and
    causal_<model> (agent_label_of in run_episodes/) in the gym mode and, unless every_clock_mode, in the clock mode of
    their config preset. The ablation labels of the causal agent have their own table, the episodes of the online loop of
    GEPA (mode gepa_online) enter the GEPA tables alone, and the policies and the static agent are not agents of the
    comparison."""
    return [e for e in episodes if e.get("mode") == "gym" and e.get("model") and (every_clock_mode or in_config_mode(e))
            and e.get("config") != HEALTHY_CONTROL_PRESET and e.get("agent") in {row_label(row, e["model"]) for row in AGENT_ROWS}]


def causal_episodes(episodes):
    """The episodes of the causal agent and its ablations (the labels causal..._<model>) in the gym mode and in the clock
    mode of their config preset."""
    return [e for e in episodes if e.get("mode") == "gym" and e.get("model") and in_config_mode(e)
            and str(e.get("agent", "")).startswith(OURS) and str(e["agent"]).endswith(f"_{e['model']}")]


EN4_PRESET = AGENTS_PRESET  # the row of the six rows, whose runs rescore_under_verifier_scope.py scores again


def en4_episodes(episodes):
    """The episodes of A7: the six rows on the preset of the agents table at the main level of every setting but the verifier scope,
    which the table varies, so that both scopes read the same starts."""
    return [e for e in agent_episodes(episodes) if e.get("config") == EN4_PRESET and at_main_levels({**e, "verifier_scope": preset_config(EN4_PRESET).verifier_scope})]


def in_rows(episode, rows):
    """Whether an episode plays one of the rows of design.AGENT_ROWS given, by its agent label and its level of the MCP."""
    return (episode.get("agent"), as_bool(episode.get("diagnostics_server"))) in {(row_label(row, episode.get("model", "")), row[1]) for row in rows}


def placed_episodes(episodes, ablation, varied):
    """The episodes of the table of a supporting ablation: the gym episodes of the rows that design.ABLATION_PLACES gives it, on its
    preset, at the main level of every setting but the one the table varies."""
    preset, rows = ABLATION_PLACES[ablation]
    config = preset_config(preset)
    levels = {"clock_mode": config.clock_mode, **MAIN_LEVELS, **{name: getattr(config, name) for name in PRESET_LEVELS}}
    return [e for e in episodes if e.get("mode") == "gym" and e.get("model") and e.get("config") == preset and in_rows(e, rows)
            and all(str(e.get(name, "")) == str(level) for name, level in levels.items() if name != varied)]


SWITCH_ABLATIONS = ("L2", "L3", "L5")  # the switches of SWITCHES, while L1 has the table MISDIAGNOSIS_PRICE of its own, with no test


def arm_episodes(episodes, ablations):
    """The episodes of the arms of design.ARM_PRESETS whose ablation is among ablations: the rows that design.ABLATION_PLACES gives
    them, on their place and on every arm preset of theirs, each at the main level of every setting of its own preset, so that an
    arm whose preset sets another clock mode pairs against its place in its own mode."""
    arms = {preset: ablation for preset, (ablation, _) in ARM_PRESETS.items() if ablation in ablations}
    presets = {ABLATION_PLACES[ablation][0] for ablation in arms.values()} | set(arms)
    rows = {row for ablation in arms.values() for row in ABLATION_PLACES[ablation][1]}
    return [e for e in episodes if e.get("mode") == "gym" and e.get("model") and e.get("config") in presets and in_rows(e, rows) and at_main_levels(e)]


def switch_episodes(episodes):
    """The episodes of SWITCHES: the rows that design.ABLATION_PLACES gives the small switches of layers 2, 3 and 5, the three agents
    without the MCP, on the preset of training and on every arm preset of design.ARM_PRESETS of those switches, each at the main level
    of every setting of its own preset, so that an arm whose preset sets another clock mode pairs against the preset of training in its own mode."""
    return arm_episodes(episodes, SWITCH_ABLATIONS)


def switch_of(episode):
    """The switch of a causal row from its label, which CausalLoopOptions.label() writes as causal with one suffix per
    switch that differs from the default and the model appended (causal_nobranch_<model>): full without a suffix, else
    the switch names of rows A13 to A17, joined by a plus sign when a run turned several on at once."""
    label, model = str(episode.get("agent", "")), str(episode.get("model", ""))
    stem = label[: -len(model) - 1] if model and label.endswith(f"_{model}") else label
    names = [f"plan_length_{part[3:]}" if part.startswith("len") and part[3:].isdigit() else SWITCH_OF_SUFFIX.get(part, part)
             for part in stem.split("_")[1:]]
    return "+".join(names) or "full"


def _arm(episodes, label, server):
    """The episodes of one agent label with the diagnostics server on (True) or off (False)."""
    return [e for e in episodes if e.get("agent") == label and as_bool(e.get("diagnostics_server")) is server]


def main_level_groups(episodes):
    """The episodes of the six rows at the main level of every setting (at_main_levels) per model, config and the
    settings of PAIR_SETTINGS, the groups within which AGENTS_pairs and ELIM_pairs pair two rows."""
    return cell_statistics.groups([e for e in agent_episodes(episodes) if at_main_levels(e)], "model", "config", *PAIR_SETTINGS)


# ----- the realized elimination of the paid reads (ELIM, ELIM_pairs) -----
REFERENCE_LABELS = tuple(label for label, _ in REFERENCE_POLICIES)  # POLICY_AGENTS of run_episodes/, the agent column of their rows
POLICY_TABLE_LABELS = tuple(label for label, _ in POLICY_TABLE_POLICIES)  # the agent column of the rows of POLICIES alone


def elimination_episodes(episodes):
    """The episodes of ELIM: those of the six rows under every clock mode (agent_episodes) and those of the reference
    policies in the gym mode, measured by the same replay whether or not their run had the diagnostics server."""
    return agent_episodes(episodes, every_clock_mode=True) + [e for e in episodes if e.get("mode") == "gym" and e.get("agent") in REFERENCE_LABELS]
