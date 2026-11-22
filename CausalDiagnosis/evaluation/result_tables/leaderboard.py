"""The leaderboard of the hydraulic rig gym in the form of the Leaderboards section of the AssetOpsBench README: per model,
a table of the six rows of the agents and the CausalDiagnosis MCP on layer 0 with the oracle policy and the declaring policy below
it, the environment table with the three agents with the MCP on every row of the ladder, and a table of the supporting
ablations of the paper (CausalDiagnosis/docs/experiments.md). The presets, the rows and the places of the ablations come
from experiment_tables.py.

usage: python -m CausalDiagnosis.evaluation.result_tables.leaderboard [--aggregate CausalDiagnosis/results/tables]
       [--out CausalDiagnosis/results/LEADERBOARD.md] [--since AGENT=ISO ...] [--runs FOLDER ...]

Every number is a cell of RUNS.csv, the table of tables_from_runs/ with one row per run folder or per line of the test set over
the runs of its replicates, and every row of the page names its folders. A run counts when tables_from_runs/ read
it, which needs the run.json that a run writes when it ends, and when it
started at or after the cutoff that --since gives its agent, the rule of pending_grid_lines.py for runs of code that a later
fix replaced. The tables of a model use one set of episodes, the evaluation set once a counted run played it and the
combinations set of --starts combinations before, so that the rows pair episode by episode. A row of the main table, the oracle policy, the declaring policy or an
ablation without a counted run reads still running while a run folder of it under --runs has no run.json yet, judged by
its scores.csv and the start time in its name, stopped when that folder's ledger holds the stop of the harness rule, stalled
when its ledger was not updated for twice the wall limit of an episode, and not run otherwise. A page without a counted run states that no run
has been played on the current code, in place of the sentence on preliminary numbers. Under the counted runs the page lists
every run under --runs that tables_from_runs/ left out so that no two runs pool, a run of another commit than the latest start or
the earlier of two runs of one line on a shared seed, with the reason that INDEX.json carries (results_index.pooling_left_out).
Every agent label that
run_episodes/ writes maps to a row, an ablation or a reference that the page leaves out, and any other label, switch
or preset stops the script, as does a run that varies two elements at once or an element on a preset or a row where no
supporting ablation varies it (design.ABLATION_PLACES). A run of a preset that no row or arm of the paper plays (design.UNREPORTED), a trained agent and a
row without the MCP beside layer 0 outside the place of a supporting ablation are left out with their reason.
"""
from CausalDiagnosis.utils.paths import RESULTS
import argparse
import csv
import re
import sys
import time
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

HERE = Path(__file__).resolve().parent

from CausalDiagnosis.evaluation.result_tables import tables_from_runs as aggregate  # noqa: E402
from CausalDiagnosis.experiments.episode_runners import attempt_error_classes as failures  # noqa: E402
from CausalDiagnosis.experiments.run_verification import run_ledger as ledger  # noqa: E402
from CausalDiagnosis.evaluation.result_tables import index_file, pooling_rules  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import DEFAULT_RUNS, read_rows
from CausalDiagnosis.utils.csv_rows import as_bool, as_float
from CausalDiagnosis.experiments.episode_runners.run_configuration import load_config
from CausalDiagnosis.utils.json_files import read_json  # noqa: E402
from CausalDiagnosis.experiments.table_definitions.experiment_tables import ABLATION_PLACES, ARM_PRESETS, AS_RELEASED, EPISODE_WALL_LIMIT_S, HEALTHY_CONTROL_PRESET, LADDER, LADDER_NAMES, LADDER_ROWS, OWN_TABLES, REFERENCE_ARM, AGENTS_PRESET, AGENT_ROWS, TOKEN_PRESETS, UNREPORTED  # noqa: E402

# the rows of the environment table in the order of the ladder, each preset adding one big layer to the row above it,
# as (preset, name), on which the three agents with the MCP play
ENVIRONMENT_ROWS = tuple(zip(LADDER, LADDER_NAMES))
MAIN_ROWS = AGENT_ROWS  # (label prefix, CausalDiagnosis MCP, name), in the order of the tables of the paper
ABLATIONS = (  # identifier, family and short name of the supporting ablations of the paper
    ("Sc3", "Scoring", "the static answering"),
    ("En4", "Environment", "the verifier scope"), ("En5", "Environment", "the abstention rule"), ("En6", "Environment", "the price table"),
    ("En7", "Environment", "the guidance level"), ("En9", "Environment", "the ranking model"),
    ("L1", "Environment", "the misdiagnosis price of layer 1"), ("L2", "Environment", "the thinking time of layer 2"), ("L3", "Environment", "the action durations of layer 3"),
    ("L5", "Environment", "the deterioration and failure of layer 5"),
    ("Ag1", "Agent", "the candidate troubleshooting plans"), ("Ag2", "Agent", "the number of candidate troubleshooting plans"),
    ("Ag3", "Agent", "the plan length"), ("Ag7", "Agent", "the read the board advises"),
    ("Ag9", "Agent", "the schedule of candidate plans"), ("Ag10", "Agent", "the structured output of the replies"),
)
# label suffix of the CausalDiagnosis agent (CausalLoopOptions.label): ablation and arm, with len<N> as Ag3. The suffixes noexp
# and noback of the expectation checks and of replanning stay in the agent, but the experiments no longer play their rows, so a label with one of them stops the
# page like any switch that no ablation names
SWITCHES = {
    "nobranch": ("Ag1", "candidate troubleshooting plans off"), "single": ("Ag2", "candidate troubleshooting plans for the top candidate only"),
    "e10": ("Ag7", "the agent's own read, the discriminating test of E10 from the long form of the reply, against the advice of the MCP that "
                   "the registered agent follows"),
    "ec2": ("Ag7", "the agent's own read, one step edge cutting over repair plan classes from the long form of the reply, against the advice "
                   "of the MCP that the registered agent follows"),
    "seq": ("Ag9", "the schedule of candidate plans off, every read step alone with its own ranking"),
    "noschema": ("Ag10", "structured output off, the planning and the orientation replies decoded without the JSON schema of their form, as the two reimplemented designs decode"),
}
REFERENCES = ("oracle", "do_nothing", "greedy", "random", "declare", "scripted", "mcp_follower")  # labels without a model, of which the page shows ANCHORS
ANCHORS = ("oracle", "declare")  # the best of the five reference policies and the most a constant declaration can earn, under the main table and on no other line
STATIC_AGENT = ("Sc3", "one model call over the question, without the gym")
FACTORS = {  # setting of RUNS.csv and the ablation that varies it, its level in a row being that of the row's preset
    "clock_mode": "L2", "verifier_scope": "En4", "abstention_rule": "En5", "prices": "En6", "guidance_level": "En7", "ranking_model": "En9",
}
DRIVER_LEVELS = aggregate.MAIN_LEVELS  # run_episodes/'s defaults, the main levels at which AGENTS_pairs pairs
# the start time that run_episodes.run_label puts at the end of a run folder's name, before a collision suffix
STARTED_IN_NAME = re.compile(r"_(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})(?:-\d+)?$")
COLUMNS = ("Model", "Preset", "Episode set", "Episodes", "Pass rate (95% interval clustered by start)", "Judge success rate",
           "Cost / oracle policy cost", "Median delay in cycles (episodes with a delay)", "Run folder")
TRAINED = "a trained agent, which the tables of training in MAIN_TABLES.md read"
WITHOUT_MCP_BESIDE_LAYER_0 = ("a row without the CausalDiagnosis MCP on a later layer outside the place of a supporting ablation, which the "
                              "cumulative layer study plays with the MCP")


class Slot(NamedTuple):
    kind: str  # main, environment, ablation or one of ANCHORS
    row: int | None = None  # index into MAIN_ROWS, None for ANCHORS and the static agent
    ablation: str = ""
    arm: str = ""  # the arm of an ablation, or the name of a row of the environment table


# ----- where a run belongs -----

def framework_of(label, model):
    """(framework, (ablation, arm) of a switch of the CausalDiagnosis agent or None) of a label that
    run_episodes.agent_label_of writes."""
    if label in REFERENCES:
        return label, None
    stem = label[: -len(model) - 1] if model and label.endswith(f"_{model}") else ""
    if stem in {prefix for prefix, _, _ in MAIN_ROWS} | {"static"}:
        return stem, None
    parts = stem.split("_")
    if parts[0] != "causal":
        raise ValueError(f"unknown agent label {label!r} of model {model!r}: no row, ablation or reference of the leaderboard")
    switches = [switch_arm(label, part) for part in parts[1:]]
    if len(switches) > 1:
        raise ValueError(f"agent label {label!r} turns on several switches at once, which no ablation of the paper does")
    return "causal", (switches[0] if switches else None)


def switch_arm(label, part):
    if part in SWITCHES:
        return SWITCHES[part]
    if part.startswith("len") and part[3:].isdigit():
        return "Ag3", f"plan length {part[3:]}"
    raise ValueError(f"agent label {label!r} carries the switch {part!r}, which no ablation of the paper names")


@lru_cache(maxsize=None)
def preset_config(preset):
    return load_config(preset)[1]


def level(preset, name):
    """The level of a setting of FACTORS in a run of the preset without an override: the driver's default, or the preset's
    own clock mode, guidance level and verifier scope (configs/<preset>.json over EpisodeConfig)."""
    return DRIVER_LEVELS[name] if name in DRIVER_LEVELS else str(getattr(preset_config(preset), name))


def environment_row(run):
    """The index into ENVIRONMENT_ROWS of the row a run plays, or None for a preset outside the ladder."""
    return LADDER.index(run["config"]) if run["config"] in LADDER else None


def factor_arms(run):
    """(ablation, arm) of every setting of the run that differs from its level in the run's preset."""
    return [(ablation, f"{name.replace('_', ' ')} {run[name]}") for name, ablation in FACTORS.items() if str(run[name]) != level(run["config"], name)]


def on_preset(preset):
    """The words that name the row of an arm played beside layer 0."""
    if preset == AGENTS_PRESET:
        return ""
    names = [name for p, name in ENVIRONMENT_ROWS if p == preset]
    return f", on {f'the row {names[0]}' if names else ARM_PRESETS[preset][1]}"


def main_row(framework, run):
    if framework == "static":
        return None
    server = as_bool(run["diagnostics_server"])
    for i, (prefix, with_server, _) in enumerate(MAIN_ROWS):
        if (prefix, with_server) == (framework, server):
            return i
    raise ValueError(f"run {run['run']} plays {framework} with the CausalDiagnosis MCP {server}, which no row of the main table holds")


def trained(run):
    """Whether a run plays a trained agent, any training variant but the agent as released."""
    return (run.get("training_variant") or AS_RELEASED) != AS_RELEASED


def placed(run, agent_row, ablation, arm):
    """The slot of an arm of a supporting ablation, refused for a row of the main table that the design does not give it
    (design.ABLATION_PLACES)."""
    if agent_row is not None and MAIN_ROWS[agent_row] not in ABLATION_PLACES[ablation][1]:
        raise ValueError(f"run {run['run']} plays {ablation} with {MAIN_ROWS[agent_row][2]}, a row that no supporting ablation gives it (design.ABLATION_PLACES)")
    return Slot("ablation", agent_row, ablation, arm)


def reference_slot(run, agent_row):
    """The slot of a row as released on the place of a supporting ablation whose cell no line of the main table or of the
    environment table shows: the reference arm of the first ablation of that place, as grid_lines names the reference line, also
    for the rows without the MCP on all six big layers that grid_lines writes as the baselines of training, or None for a row that
    no place puts on the preset of the run."""
    preset, row = run["config"], MAIN_ROWS[agent_row]
    if preset == AGENTS_PRESET or (preset in LADDER and row in LADDER_ROWS):
        return None
    first = next((ablation for ablation, (place, rows) in ABLATION_PLACES.items() if place == preset and row in rows), None)
    return None if first is None else Slot("ablation", agent_row, first, REFERENCE_ARM + on_preset(preset))


def classify(run):
    """The slot of a run of RUNS.csv, or None for a run the page leaves out: the anchored runs, which the paper leaves out, the
    runs of the healthy control, which the table HEALTHY_CONTROL reports apart,
    the runs of a preset in design.UNREPORTED, the runs of a token twin of a clock preset, a second condition of the clock, a trained agent, a row without the MCP beside layer 0 outside the place of an ablation, the floor references, and
    ANCHORS beside layer 0. A run with every setting at the level of its preset is a line of the main table on layer 0, of the
    environment table on a later row of the ladder for an agent with the MCP, or an arm of design.ARM_PRESETS, a small switch
    of layers 2, 3 or 5 as a preset of its own, and a run that varies one element is an arm of that element's ablation. A
    supporting ablation plays where design.ABLATION_PLACES puts it, on its preset and with its rows, the clock mode only as
    its preset sets it, and the static agent, which plays no gym, is Sc3 on the preset of the agents table. A row as released on the place of
    a supporting ablation where no other line plays it is the reference arm of that place (reference_slot). RUNS.csv holds no run that
    results_index refuses, so no run of the retired mode gepa and no model agent on a row over the rig's action server
    reaches this function."""
    preset = run["config"]
    if preset in OWN_TABLES or preset in UNREPORTED or preset in TOKEN_PRESETS or trained(run):
        return None
    if preset not in ARM_PRESETS and preset not in LADDER:
        raise ValueError(f"run {run['run']} plays the preset {preset!r}, which no row or ablation of the leaderboard maps")
    framework, switch = framework_of(run["agent"], run["model"])
    row = environment_row(run)
    arms = ([STATIC_AGENT] if framework == "static" else []) + ([switch] if switch else []) + factor_arms(run)
    if framework in REFERENCES:
        return Slot(framework) if framework in ANCHORS and preset == AGENTS_PRESET and not arms else None
    if len(arms) > 1:
        raise ValueError(f"run {run['run']} varies {' and '.join(a for a, _ in arms)} at once, which no ablation of the paper does")
    agent_row = main_row(framework, run)
    if not arms:
        if preset == AGENTS_PRESET:
            return Slot("main", agent_row)
        if row is None:
            return placed(run, agent_row, *ARM_PRESETS[preset])
        if MAIN_ROWS[agent_row] in LADDER_ROWS:
            return Slot("environment", agent_row, "", ENVIRONMENT_ROWS[row][1])
        return reference_slot(run, agent_row)
    (ablation, arm), = arms
    if ablation == "L2":
        raise ValueError(f"run {run['run']} plays the clock mode {run['clock_mode']} on the preset {preset!r}, while the thinking time off is a preset of its own (design.ARM_PRESETS)")
    if preset != ABLATION_PLACES[ablation][0]:
        raise ValueError(f"run {run['run']} varies {ablation} on the preset {preset!r}, which no supporting ablation does (design.ABLATION_PLACES)")
    return placed(run, agent_row, ablation, arm + on_preset(preset))


def episode_set(run):
    """The set of episodes a run played, or None for seeds of its own (run.json starts and master_seed)."""
    if run["starts"] == "sampled":
        return f"evaluation, master seed {run['master_seed']}"
    return "combinations" if run["starts"] == "combinations" else None


def counts(run, since):
    """Whether a run started at or after the cutoff of its agent, as pending_lines.matches decides."""
    cutoff = since.get(framework_of(run["agent"], run["model"])[0])
    return cutoff is None or run["started"] >= cutoff


# ----- the runs -----

def read_runs(aggregate_dir):
    path = Path(aggregate_dir) / "RUNS.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found, so run tables_from_runs.py over the run folders first")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def runs_in_progress(roots):
    """The run folders under the roots without run.json, each as the settings of its first scores.csv row, the training variant
    of its ledger and the start time in its name. Mode and abstention rule are on record only in run.json, so a running run counts
    as a gym run under the decision rule, except a run of the online loop of GEPA, whose rows carry their GEPA call and which no line of
    the page counts. The training variant sits in run.json too, so the ledger, which the driver writes before the first episode,
    carries it, and a running run of a trained agent is left out as its finished run would be, and never read as the agent as
    released still running. A run of chosen starts, which the seeds block of its ledger names (pooling_rules.chosen_starts), is left
    out as its finished run is, so it marks no line still running."""
    found = []
    for root in roots:
        for scores in sorted(Path(root).glob("*/scores.csv")):
            run_dir, match = scores.parent, STARTED_IN_NAME.search(scores.parent.name)
            rows = [] if (run_dir / "run.json").exists() or not match else read_rows(run_dir)
            if not rows or rows[0].get("gepa_call") not in (None, ""):
                continue
            record = read_json(run_dir / ledger.LEDGER, {}) or {}
            if pooling_rules.chosen_starts(record.get("seeds") or {}):
                continue
            year, month, day, hour, minute, second = match.groups()
            configuration = record.get("configuration") or {}
            found.append({**rows[0], "run": run_dir.name, "mode": "gym", "abstention_rule": DRIVER_LEVELS["abstention_rule"],
                          "training_variant": configuration.get("training_variant") or AS_RELEASED,
                          "started": f"{year}-{month}-{day}T{hour}:{minute}:{second}", "state": progress_state(run_dir)})
    return found


def progress_state(run_dir):
    """The state of a run folder without run.json: stopped when its ledger holds the stop event of the harness rule, stalled when
    its ledger was not updated for longer than twice the wall limit of an episode, since a live run updates it after every
    episode, and still running otherwise."""
    path = Path(run_dir) / ledger.LEDGER
    record = read_json(path, {}) or {}
    if any(event.get("what") == failures.STOP_EVENT for event in record.get("events", [])):
        return "stopped"
    if path.exists() and time.time() - path.stat().st_mtime > 2 * EPISODE_WALL_LIMIT_S:
        return "stalled"
    return "still running"


def chosen_set(placed):
    """The evaluation set once a counted run of an agent played it, else the combinations set of --starts combinations."""
    sets = {episode_set(run) for slot, run in placed if slot.kind not in ANCHORS}
    evaluation = sorted(s for s in sets if s and s.startswith("evaluation"))
    return evaluation[0] if evaluation else "combinations"


def latest_per_slot(placed, wanted_set):
    best = {}
    for slot, run in placed:
        if episode_set(run) == wanted_set and (slot not in best or run["started"] > best[slot]["started"]):
            best[slot] = run
    return best


# ----- the page -----

def number(value, digits):
    x = as_float(value)
    return "none" if x is None else f"{x:.{digits}f}"


def cells(run):
    """The cells of COLUMNS after Model, Preset and Episode set for a counted run."""
    n = int(run["n"])
    judged = int(run.get("judge_verdicts") or 0)  # the episodes with a verdict of the judge, which RUNS counts
    pass_rate = "none" if as_float(run["scope_pass"]) is None else f"{number(run['scope_pass'], 3)} [{number(run['scope_pass_low'], 3)}, {number(run['scope_pass_high'], 3)}]"
    delay = "none" if not run["delay_median"] else f"{as_float(run['delay_median']):g} ({run['delay_n']} of {n})"
    return [str(n), pass_rate, number(run["judge_pass"], 3) if n and judged == n else "judge pending",
            number(run["cost_normalized"], 2), delay, folders(run)]


def folders(run):
    """The run folders of a row of RUNS, one or the replicate runs of a line, each as code."""
    return " ".join(f"`{name}`" for name in run["run"].split())


def in_folders(run):
    return f"in the run folder{'s' if len(run['run'].split()) > 1 else ''} {folders(run)}"


def table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]


def status(slot, in_flight):
    """The text of a line of the ablation table without a counted run."""
    return in_flight[slot].get("state", "still running") if slot in in_flight else "not run"


def facts(run):
    """A sentence of the cells of a counted run, for the two references under the main table."""
    n, pass_rate, _, cost, delay, _folders = cells(run)
    cost = "no cost ratio" if cost == "none" else f"a cost ratio of {cost}"
    delay = "no episode with a delay" if delay == "none" else f"a median delay of {delay}"
    return f"it has a pass rate of {pass_rate} over {n} episodes, {cost} and {delay}, {in_folders(run)}"


def declaring_facts(run):
    """A sentence of the cells of a counted run of the declaring policy, which never passes, so its success and its reward are what it shows."""
    return (f"it has a verifier success rate of {number(run['success'], 3)} [{number(run['success_low'], 3)}, {number(run['success_high'], 3)}] "
            f"and a mean reward of {number(run['reward'], 3)} over {run['n']} episodes, {in_folders(run)}")


IN_A_SENTENCE = {"still running": "it is still running", "not run": "it has not run", "stopped": "its run stopped after harness errors in a row",
                 "stalled": "its run stopped updating its ledger"}


def agent_of(slot):
    return "the model alone" if slot.row is None else MAIN_ROWS[slot.row][2]


def main_section(model, wanted, best, in_flight):
    rows = [[label, model, AGENTS_PRESET, wanted, *(cells(best[Slot("main", i)]) if Slot("main", i) in best else [status(Slot("main", i), in_flight), "", "", "", "", ""])]
            for i, (_, _, label) in enumerate(MAIN_ROWS)]
    oracle, declaring = best.get(Slot("oracle")), best.get(Slot("declare"))
    return [f"## Main comparison, {model or 'no model yet'}", "", *table(("Agent", *COLUMNS), rows), "",
            "The oracle policy, which reads the hidden state, is the best of the five reference policies "
            "and not a row of this table. On the same set " + (facts(oracle) if oracle else IN_A_SENTENCE[status(Slot("oracle"), in_flight)]) + ".", "",
            "The declaring policy, which reads the cheapest sensor set once, cites that reading and declares insufficient evidence on "
            "every start, shows the most a constant declaration can earn and is not a row of this table. On the same set "
            + (declaring_facts(declaring) if declaring else IN_A_SENTENCE[status(Slot("declare"), in_flight)]) + ".", ""]


def environment_section(model, wanted, best, in_flight):
    """One line per row of the ladder and agent with the MCP, in the paper's order, with layer 0 as one line that points at
    the rows with the MCP of the main table, which are its runs."""
    lines = []
    for preset, name in ENVIRONMENT_ROWS:
        if preset == AGENTS_PRESET:
            lines.append([name, "the rows with the CausalDiagnosis MCP of the main table above", model, preset, wanted, "", "", "", "", "", ""])
            continue
        for i, (_, _, label) in ((i, row) for i, row in enumerate(MAIN_ROWS) if row in LADDER_ROWS):
            slot = Slot("environment", i, "", name)
            lines.append([name, label, model, preset, wanted, *(cells(best[slot]) if slot in best else [status(slot, in_flight), "", "", "", "", ""])])
    return [f"## Environment table, {model or 'no model yet'}", "", *table(("Row of the environment table", "Agent", *COLUMNS), lines), ""]


def ablation_section(model, wanted, best, in_flight):
    """One line per arm and agent of every ablation with a counted run or a run in progress, and one line not run for any other
    ablation."""
    lines = []
    for ablation, family, name in ABLATIONS:
        head = [ablation, family, name]
        slots = sorted({s for s in [*best, *in_flight] if s.kind == "ablation" and s.ablation == ablation},
                       key=lambda s: (len(MAIN_ROWS) if s.row is None else s.row, s.arm))
        for s in slots:
            run = best.get(s) or in_flight[s]
            lines.append([*head, s.arm, agent_of(s), model, run["config"], wanted,
                          *(cells(run) if s in best else [status(s, in_flight), "", "", "", "", ""])])
        if not slots:
            lines.append([*head, "", "", model, "", wanted, "not run", "", "", "", "", ""])
    return [f"## Ablations, {model or 'no model yet'}", "", *table(("Id", "Family", "Ablation", "Arm", "Agent", *COLUMNS), lines), ""]


def choose(runs, since, running):
    """(the counted runs as (slot, run), {model: (episode set, {slot: chosen run}, {slot: run in progress}, the counted runs
    of the model)}), the choice that the page shows and the results index records."""
    placed = [(slot, run) for run in runs if (slot := classify(run)) and counts(run, since) and episode_set(run)]
    in_flight = {slot: run for run in running if (slot := classify(run)) and counts(run, since)}
    chosen = {}
    for model in sorted({run["model"] for slot, run in placed if run["model"]}) or [""]:
        mine = [(slot, run) for slot, run in placed if run["model"] in (model, "")]
        wanted = chosen_set(mine)
        best = latest_per_slot(mine, wanted)
        chosen[model] = (wanted, best, {slot: run for slot, run in in_flight.items() if run["model"] in (model, "") and slot not in best}, mine)
    return placed, chosen


def line_of(slot):
    """The name of a line of the page: a row of the main table, a reference under it, or an ablation with its arm and agent."""
    if slot.kind == "main":
        return MAIN_ROWS[slot.row][2]
    if slot.kind == "oracle":
        return "the oracle policy"
    if slot.kind == "declare":
        return "the declaring policy"
    if slot.kind == "environment":
        return f"{slot.arm}, {agent_of(slot)}"
    return f"{slot.ablation} {slot.arm}, {agent_of(slot)}"


def left_out(run, since, placed, chosen):
    """Why a run of RUNS.csv is on no line of the page, or None for a run that is."""
    if any(best.get(slot) is run for _, best, _, _ in chosen.values() for slot in best):
        return None
    framework = framework_of(run["agent"], run["model"])[0]
    if classify(run) is None:
        if run["config"] == HEALTHY_CONTROL_PRESET:
            return f"a run of the preset {run['config']}, which the table HEALTHY_CONTROL reports apart"
        if run["config"] in OWN_TABLES:
            return f"a run of the preset {run['config']}, which the paper leaves out"
        if run["config"] in UNREPORTED:
            return f"a run of the preset {run['config']}, which no table of the page reads since {UNREPORTED[run['config']]}"
        if run["config"] in TOKEN_PRESETS:
            return (f"a run of the preset {run['config']}, the twin of {TOKEN_PRESETS[run['config']]} in the generated_tokens mode, a second "
                    "condition of the clock that the page leaves out")
        if trained(run):
            return TRAINED
        if framework not in REFERENCES:
            return WITHOUT_MCP_BESIDE_LAYER_0
        return "a reference policy that the page leaves out"
    if not counts(run, since):
        return f"started {run['started']}, before the cutoff {since[framework]} of {framework}"
    if not episode_set(run):
        return "seeds of its own, outside the combinations set and the evaluation set"
    reasons = []
    for model, (wanted, best, _, mine) in chosen.items():
        slot = next((s for s, r in mine if r is run), None)
        if slot is None:
            continue
        if episode_set(run) != wanted:
            reasons.append(f"the {episode_set(run)} set, while the tables of {model or 'no model'} use the {wanted} set")
        else:
            reasons.append(f"an earlier start of the same line, {best[slot]['run']} counts")
    return ". ".join(reasons) or "no line of the page"


RUNNING_FIELDS = ("run", "agent", "model", "config", "mode", "diagnostics_server", "abstention_rule", "training_variant", *FACTORS, "started",
                  "state")  # what classify reads, and the state of the run, so that verify_results_folder.py writes the same page again


def index_section(runs, since, running):
    """The leaderboard section of INDEX.json: the run chosen for every line, the other counted candidates of the line, the
    rule that chose it, every run of RUNS.csv left out with the reason, and the settings of the runs then in progress,
    which the page marks still running and verify_results_folder.py needs to write the same page again."""
    placed, chosen = choose(runs, since, running)
    lines = []
    for model, (wanted, best, _, mine) in chosen.items():
        for slot, run in sorted(best.items(), key=lambda item: (item[0].kind, -1 if item[0].row is None else item[0].row, item[0].ablation, item[0].arm)):
            framework = framework_of(run["agent"], run["model"])[0]
            lines.append({"model": model, "line": line_of(slot), "run": run["run"],
                          "candidates": sorted(r["run"] for s, r in mine if s == slot and r is not run),
                          "rule": {"cutoff": since.get(framework), "episode_set": wanted, "choice": "the latest start among the counted runs of the set"}})
    return {"since": dict(sorted(since.items())), "lines": lines,
            "left_out": [{"run": run["run"], "reason": reason} for run in runs if (reason := left_out(run, since, placed, chosen))],
            "running": [{field: run.get(field, "") for field in RUNNING_FIELDS} for run in running]}


# the line under the title: the page counts runs and prints the verifier's pass of each line in its own order, while the tables of the
# paper, in their order and layout, are those that paper_tables/ writes beside it
STATUS_LINE = ("This page is a status board of the runs and no page of results, and the tables of the paper, in their order and layout, "
               "are those of `MAIN_TABLES.md` beside it.")


def page(runs, since, running, left_out=(), chosen_starts=()):
    """The Markdown of the leaderboard from the rows of RUNS.csv, the cutoffs, the runs in progress, the runs that tables_from_runs/
    left out so that no two runs pool, as (folder, reason), and the runs of chosen starts that no table reads, by folder."""
    placed, chosen = choose(runs, since, running)
    models = list(chosen)
    judges = sorted({run["judge_model"] for _, run in placed if run.get("judge_model")})
    cutoffs = ", ".join(f"{agent} from {start}" for agent, start in sorted(since.items())) or "every run"
    # a page without a counted run holds no number and must not read as if results existed
    opening = ("These numbers are preliminary. They come from the runs on Columbia's cluster, and every number traces to" if placed else
               "No run has been played on the current code, so the tables below hold no number. They stay empty until the test set is "
               "played on Columbia's cluster, and every number then traces to")
    out = ["# Leaderboard of the hydraulic rig gym", "", STATUS_LINE, "",
           f"{opening} the run folders its row "
           "names through `CausalDiagnosis/evaluation/result_tables/tables_from_runs.py`, whose table `RUNS.csv` holds one row per run folder, or per line of "
           "the test set over the runs of its replicates, and which "
           "`CausalDiagnosis/evaluation/result_tables/leaderboard.py` turns into this page. The run folders stay out of the repository, since they hold "
           "sensor readings of single cycles.", "",
           f"- Agent models: {', '.join(m for m in models if m) or 'none yet'}",
           "- Scoring: the verifier of the environment on the outcome in the hidden state, and the AssetOpsBench judge on the closing summary as a "
           f"second scorer (judge models: {', '.join(judges) or 'none yet'})",
           f"- Runs counted: {cutoffs}, by the start time of the run",
           f"- Written on {date.today().isoformat()} from {sum(len(run['run'].split()) for _, run in placed)} counted run folders",
           *(f"- Left out of every table: `{folder}`, {reason}" for folder, reason in left_out),
           *([f"- Left out of every table: {pooling_rules.chosen_starts_line(chosen_starts, quote='`')}"] if chosen_starts else []), "",
           "Pass is the verifier's pass with its 95 percent interval from a bootstrap that resamples start combinations, which on the lines of En4 also needs the read gate of the verifier scope all. The judge success rate is the share of episodes the judge "
           "passes, shown once every episode of the run has a verdict. The cost ratio is the mean over the episodes of the cost, prices "
           "plus downtime, over the cost of the oracle policy on the same seed. The delay of an episode is the larger of its "
           "detection and repair delay. The combinations set plays every start combination under its smallest seed, and the evaluation "
           "set plays each at five start states, one run per replicate, the first under a seed drawn with the master seed, so a line "
           "counts the episodes of every replicate played.", "",
           "The main table plays layer 0, the base preset, with the six rows of the agents and the CausalDiagnosis MCP, and the environment "
           "table plays the three agents with the MCP on every layer, each row adding one big layer to the row above it.", ""]
    for model, (wanted, best, still_running, _) in chosen.items():
        out += (main_section(model, wanted, best, still_running) + environment_section(model, wanted, best, still_running)
                + ablation_section(model, wanted, best, still_running))
    return "\n".join(out)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--aggregate", default=str(RESULTS / "tables"), help="the folder of the tables tables_from_runs.py wrote")
    ap.add_argument("--out", default=None, help="the page (default: LEADERBOARD.md in the parent of --aggregate)")
    ap.add_argument("--since", nargs="*", default=[], help="agent=ISO time: runs of that agent count only from that start on")
    ap.add_argument("--runs", nargs="*", default=[str(DEFAULT_RUNS)], help="folders of run folders, searched for runs in progress")
    ap.add_argument("--index", default=None, help="the results index that tables_from_runs.py wrote (default: INDEX.json in the parent of --aggregate)")
    ap.add_argument("--commit", default=None, help="the commit whose runs of record the page reads when a root holds more than one group "
                                                   "(default: the commit the index records)")
    a = ap.parse_args(argv)
    since = dict(item.split("=", 1) for item in a.since)
    index = Path(a.index) if a.index else Path(a.aggregate).parent / "INDEX.json"
    out = Path(a.out) if a.out else Path(a.aggregate).parent / "LEADERBOARD.md"
    recorded = pooling_rules.read_json(index, {}) if index.exists() else {}
    # the page names the runs left out of the same group as the tables it shows
    pooling_rules.select_commit(a.commit or (recorded.get("parameters") or {}).get("commit"))
    runs, running = read_runs(a.aggregate), runs_in_progress(a.runs)
    left_out = sorted((folder, reason) for root in a.runs for folder, reason in pooling_rules.pooling_left_out(root).items())
    chosen_starts = sorted(name for root in a.runs for name in pooling_rules.chosen_starts_left_out(root))
    # the kind of the root behind the tables as the index of tables_from_runs/ records it, else of the folders searched: a page of a
    # development root opens with the banner and never lands in the results folder
    kinds = {recorded.get("root_kind"), *(pooling_rules.root_kind(root) for root in a.runs)}
    kind = "development" if "development" in kinds else recorded.get("root_kind")
    pooling_rules.refuse_results_folder_of_kind(kind, out)
    text = "\n".join([*pooling_rules.head_lines(kind, recorded.get("record")), page(runs, since, running, left_out, chosen_starts)])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    if index.exists():
        index_file.add_section(index, "leaderboard", index_section(runs, since, running))
    print(f"wrote {out}" + (f" and the leaderboard section of {index}" if index.exists() else ""))


if __name__ == "__main__":
    main(sys.argv[1:])
