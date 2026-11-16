from CausalDiagnosis.utils.paths import SRC
import functools
import json
import math
import re
import statistics
import sys
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from . import run_health_checks


HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(SRC)]
from CausalDiagnosis.evaluation.result_tables import tables_from_runs as aggregate  # noqa: E402
from CausalDiagnosis.experiments.table_definitions import experiment_tables as design  # noqa: E402
from CausalDiagnosis.experiments.episode_runners import attempt_error_classes as failures  # noqa: E402
from CausalDiagnosis.experiments.cluster_runs import run_lines  # noqa: E402
from CausalDiagnosis.experiments.run_verification import run_ledger as ledger  # noqa: E402
from CausalDiagnosis.experiments.cluster_runs import pending_grid_lines as pending_lines  # noqa: E402
from CausalDiagnosis.experiments.table_definitions import registered_analysis as registered  # noqa: E402
from CausalDiagnosis.experiments.table_definitions import training_variants as training  # noqa: E402
from CausalDiagnosis.utils.csv_rows import as_float
from CausalDiagnosis.experiments.episode_runners.run_configuration import load_world
from CausalDiagnosis.utils.json_files import read_json  # noqa: E402
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile  # noqa: E402
from CausalDiagnosis.datasets.splits.development_split import development_starts  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import DEFAULT_ASSET  # noqa: E402
from CausalDiagnosis.evaluation.metrics.table_statistics import binomial_bar  # noqa: E402


DEVELOPMENT_STARTS = "development"  # the development starts of the split, of which a round and a pilot play the share of start_cap
FAILURE_END = aggregate.FAILURE_END  # the end reason of an episode that a failure of the failure card ended
ENDPOINT_KINDS = (failures.CONNECTION, failures.SERVER, failures.SETUP, failures.TIMEOUT)
CONTEXT_WORDS = re.compile(r"context[ _]length", re.IGNORECASE)  # how an OpenAI compatible server refuses a prompt longer than its context


def number(name):
    """A number of the block development of the registered file; an open number stops the check with its name."""
    return registered.value(f"development.{name}")


@functools.lru_cache(maxsize=None)
def development_count(asset_id):
    """The number of development starts of an asset, development_split.development_starts on its recorded blocks."""
    _facts, _cycles, blocks = load_world(None, asset_id)
    return len(development_starts(get_profile(asset_id), set(blocks)))


def start_cap(asset_id):
    """The --start-cap of a round and of a pilot on an asset: the registered share of its development starts, rounded up to a
    whole start, the share read as an exact fraction so that no rounding of its float moves the count."""
    return math.ceil(Fraction(str(number("start_share"))) * development_count(asset_id))


def _development(meta, row, preset):
    agent, with_mcp, _name = row
    wanted = {"agent": agent, "diagnostics_server": with_mcp, "config_name": preset, "starts": DEVELOPMENT_STARTS,
              "training_variant": training.AS_RELEASED}
    return all(meta.get(k) == v for k, v in wanted.items()) and meta.get("start_cap") == start_cap(meta.get("asset_id") or DEFAULT_ASSET)


def development_run(root, row, preset):
    """The one finished run of a row on a preset under the folder, on the registered share of the development starts and without
    training."""
    found = [(meta, run) for meta, run in pending_lines.finished_runs([root]) if _development(meta, row, preset)]
    if len(found) != 1:
        raise SystemExit(f"{root} holds {len(found)} finished development runs of {row[2]} on {preset} (--starts {DEVELOPMENT_STARTS} "
                         "with the --start-cap that cap prints), and the check reads exactly one")
    return found[0]


def rate(episodes, column):
    """The mean of a column over the episodes under the counting rule, an episode without a verdict at zero."""
    return sum(aggregate.value(e, column) or 0.0 for e in episodes) / len(episodes) if episodes else 0.0


def endpoint_or_context(text):
    """Whether an error text names an error of the endpoint, by the kinds of attempt_error_classes.py, or a prompt longer than the context."""
    if not text:
        return False
    kind = failures.kind_of(text)
    return kind in ENDPOINT_KINDS or (kind == failures.REQUEST and bool(CONTEXT_WORDS.search(text)))


def round_check(root):
    """A development round: per row of the main table on its preset, the unscored share against the registered limit of the counting
    rule, the share lost to the harness (class harness_error, a failed call or a cut at the wall limit) against the registered
    limit, and the longest episode against half the wall limit of an episode, with the share whose error names an error of the
    endpoint or a prompt longer than the context and the input tokens per model call beside them. One line per row, a ROUND_MISS
    line for each row that passes a limit, and exit 1 then."""
    unscored_limit, harness_limit = registered.value("counting.max_unscored_share"), number("max_harness_share")
    half = design.EPISODE_WALL_LIMIT_S / 2
    missed = 0
    for row in design.AGENT_ROWS:
        episodes = aggregate.episodes_of(development_run(root, row, design.AGENTS_PRESET)[1])
        n = len(episodes)
        unscored = aggregate.completeness(episodes, {})["unscored"]
        lost = sum(1 for e in episodes if e.get("error_class") == failures.HARNESS_ERROR or (as_float(e.get("failed_calls")) or 0) > 0
                   or (as_float(e.get("wall_limit")) or 0) > 0)
        longest = max((as_float(e.get("wall_s")) or 0.0 for e in episodes), default=0.0)
        errors = sum(1 for e in episodes if endpoint_or_context(e.get("error")))
        calls = sum(as_float(e.get("llm_calls")) or 0 for e in episodes)
        tokens = sum(as_float(e.get("input_tokens")) or 0 for e in episodes)
        per_call = f"{tokens / calls:.0f}" if calls else "none"
        print(f"ROUND {row[2]}: {n} episodes, unscored {unscored} against the share {unscored_limit}, lost to the harness {lost} against the "
              f"share {harness_limit}, longest episode {longest:.0f} s against {half:.0f} s, endpoint or context errors {errors}, "
              f"input tokens per model call {per_call}")
        passed = [limit for limit, over in (("unscored", unscored > unscored_limit * n), ("lost to the harness", lost > harness_limit * n),
                                            ("longest episode", longest > half)) if over]
        if passed:
            missed += 1
            print(f"ROUND_MISS {row[2]}: {', '.join(passed)}")
    print("ROUND_OK" if not missed else f"ROUND_MISS {missed} of {len(design.AGENT_ROWS)} rows")
    return 1 if missed else 0


def patience_of(meta):
    """The patience of the failure card that the configuration of a run records, None without one."""
    return ((meta.get("config") or {}).get("failure") or {}).get("patience")


def pilot_steps(root):
    """[(patience, {row: (run.json, folder)})] of the steps of the patience rule under the folder in rising patience: the finished
    development runs of the three agents with the MCP on the full environment, one step for each patience their configurations
    record, each step with exactly one run per agent."""
    found = {}
    for meta, run in pending_lines.finished_runs([root]):
        for row in design.TRAINING_ROWS:
            if _development(meta, row, design.TRAINING_PRESET):
                found.setdefault(patience_of(meta), {}).setdefault(row, []).append((meta, run))
    steps = []
    for patience in sorted(found, key=lambda p: (p is not None, p or 0)) or [None]:
        runs = found.get(patience, {})
        for row in design.TRAINING_ROWS:
            if len(runs.get(row, [])) != 1:
                raise SystemExit(f"{root} holds {len(runs.get(row, []))} finished development runs of {row[2]} on {design.TRAINING_PRESET} "
                                 f"with the patience {patience} (--starts {DEVELOPMENT_STARTS} with the --start-cap that cap prints), and the "
                                 "check reads exactly one")
        steps.append((patience, {row: runs[row][0] for row in design.TRAINING_ROWS}))
    return steps


def episode_refusal():
    """Why one episode cannot finish inside one job, None when it can: an attempt of an episode ends at design.EPISODE_WALL_LIMIT_S,
    an attempt that the infrastructure broke is played design.INFRASTRUCTURE_REPLAYS times more, and every attempt must fit the
    job limit design.JOB_LIMIT_S. Nothing else stops the pilot for time."""
    attempts = 1 + design.INFRASTRUCTURE_REPLAYS
    if design.EPISODE_WALL_LIMIT_S * attempts > design.JOB_LIMIT_S:
        return (f"{attempts} attempts of {design.EPISODE_WALL_LIMIT_S} s each pass the job limit of {design.JOB_LIMIT_S} s, so an episode "
                "may never finish inside one job")
    return None


def seconds_per_episode(meta, folder):
    """The seconds per planned episode of a pilot run: its elapsed seconds from started to ended of run.json, which hold every cost of
    its job between two episodes, or the mean wall seconds of its episodes for a run that a later job took up again, whose elapsed
    seconds hold the wait between the jobs."""
    events = (read_json(Path(folder) / ledger.LEDGER, {}) or {}).get("events") or []
    if any(event.get("tool") == ledger.RESUME_TOOL for event in events):
        seconds = [as_float(e.get("wall_s")) or 0.0 for e in aggregate.episodes_of(folder)]
        return sum(seconds) / len(seconds) if seconds else 0.0
    elapsed = (datetime.fromisoformat(meta["ended"]) - datetime.fromisoformat(meta["started"])).total_seconds()
    return elapsed / max(meta.get("episodes_planned") or 1, 1)


def projected_runs(runs):
    """(names, jobs): the agents whose registered run the pilot projects past design.PROJECTION_SHARE of design.JOB_LIMIT_S, and the
    jobs that the slowest projection needs at that share of a job each, with one line per agent: its seconds per planned episode
    at the runs in flight that its run.json records (1 outside a grid job), times the episodes of one registered run
    (grid_lines.STARTS), and the jobs of that run."""
    bound = float(design.PROJECTION_SHARE * design.JOB_LIMIT_S)
    slow, jobs = [], 1
    for row in design.TRAINING_ROWS:
        meta, folder = runs[row]
        per_episode = seconds_per_episode(meta, folder)
        projected = per_episode * run_lines.STARTS
        need = max(1, math.ceil(projected / bound))
        print(f"PILOT projected run, {row[2]}: {per_episode:.0f} s per planned episode at runs in flight {meta.get('runs_in_flight') or 1}, "
              f"times {run_lines.STARTS} episodes of a registered run, {projected / 3600:.1f} h against {bound / 3600:.1f} h, "
              f"{design.PROJECTION_SHARE} of the job limit of {design.JOB_LIMIT_S / 3600:.0f} h, {need} job{'s' if need > 1 else ''}")
        if projected > bound:
            slow.append(row[2])
        jobs = max(jobs, need)
    return slow, jobs


def pilot(root):
    """The pilot of the development steps on the full environment. The rate: per agent the median of the seconds of its model
    calls, and think_seconds_per_cycle is the median of the three medians, so no agent weighs by its number of calls. The token rate:
    seconds_per_token is the think_seconds_per_cycle it writes over the median of the three agents' medians of the generated tokens
    of their model calls, so a call of median length moves the clock of generated tokens as far as a call of median duration moves
    the wall clock, both measured on the same runs. The patience
    rule, on the last step: each agent closes the repair of its first order no later than the first failure, the
    deadline of the failure card, in the registered share of its episodes with an order due; else the patience is multiplied by
    the registered factor and the pilot is played again. The
    registered number of steps counts the patience values played, the first included, and no step may reach the horizon of the
    preset, where the failure rule never fires; past either the pilot stops and names no patience. Every agent also
    prints the share of its episodes that a failure ended before any order."""
    share, factor, limit = number("patience_share"), number("patience_factor"), number("patience_steps")
    steps = pilot_steps(root)
    patience, runs = steps[-1]
    medians = []
    for row in design.TRAINING_ROWS:
        seconds = run_health_checks.call_seconds(runs[row][1])
        if not seconds:
            raise SystemExit(f"the pilot run of {row[2]} under {root} records no model call")
        medians.append(statistics.median(seconds))
    value = statistics.median(medians)
    print(f"PILOT think_seconds_per_cycle {value:.2f}: the median of the medians of the seconds of the model calls of the {len(medians)} agents "
          f"with the {design.MCP} on {design.TRAINING_PRESET}, development starts")
    print(f'PILOT measured "think_seconds_per_cycle": {value:.2f}; every run of record keeps the value the asset file declares')
    tokens = []
    for row in design.TRAINING_ROWS:
        counts = run_health_checks.call_tokens(runs[row][1])
        if not counts:
            raise SystemExit(f"the pilot run of {row[2]} under {root} records no generated tokens of a model call")
        tokens.append(statistics.median(counts))
    generated = statistics.median(tokens)
    per_token = round(value, 2) / generated
    print(f"PILOT seconds_per_token {per_token:.6g}: think_seconds_per_cycle {value:.2f} over the median {generated:g} of the medians of the "
          f"generated tokens of the model calls of the {len(tokens)} agents, so a call of median length moves the clock of generated tokens "
          "as far as a call of median duration moves the wall clock")
    print(f'PILOT measured "seconds_per_token": {per_token:.6g}; every run of record keeps the value the clock of the asset file declares')
    slow, jobs = projected_runs(runs)
    if (refusal := episode_refusal()) is not None:
        print(f"PILOT_EPISODE_TOO_LONG {refusal}")
        return 1
    short = []
    for row in design.TRAINING_ROWS:
        facts = [aggregate.funnel_facts(e) for e in aggregate.episodes_of(runs[row][1])]
        due = [f for f in facts if f["orders_due"]]
        in_time = sum(1 for f in due if f["order_in_time"])
        print(f"PILOT first order, {row[2]}: {in_time} of {len(due)} episodes with an order due close one no later than the first failure, "
              f"{sum(1 for f in facts if f['failure_before_order'])} of {len(facts)} ended by a failure before any order")
        if in_time < share * len(due):
            short.append(row[2])
    if not short:
        if slow:
            print(f"PILOT_SLOW {', '.join(slow)}: the projected registered run passes {design.PROJECTION_SHARE} of the job limit at the runs in "
                  f"flight of the pilot and needs {jobs} jobs, each taking the run up from its last finished episode (run_episodes.py --resume); "
                  f"the runs in flight of record stay design.RUNS_IN_FLIGHT, {design.RUNS_IN_FLIGHT}, the value the pushed commit declares")
        print("PILOT_OK")
        return 0
    if patience is None:
        raise SystemExit(f"the pilot runs under {root} record no patience of the failure card, so the patience rule cannot lengthen it")
    played = len(steps)
    if played >= limit:
        print(f"PILOT_STOP {', '.join(short)} stay below the registered share {share} after {played} steps of the patience rule, the registered "
              "number, and the pilot names no patience")
        return 1
    longer = math.ceil(patience * factor)
    horizons = {(meta.get("config") or {}).get("horizon") for meta, _run in runs.values()} - {None}
    if horizons and longer >= min(horizons):
        print(f"PILOT_STOP {', '.join(short)} stay below the registered share {share} with the patience {patience}, and the next patience {longer} "
              f"reaches the horizon {min(horizons)} of {design.TRAINING_PRESET}, where the failure rule never fires, so the pilot names "
              "no patience")
        return 1
    print(f"PILOT_PATIENCE {longer}: {', '.join(short)} stay below the registered share {share} with the patience {patience}; play the pilot "
          f"again with the patience {longer} of the failure card, step {played + 1} of {limit}")
    return 1


def gate(root):
    """The gate of the development steps, read off the last round on the main table's preset, with three readings, one line per
    reading and row, and exit 1 on a miss. The graded Top-1 and the graded pass of the CausalDiagnosis agent without the MCP reach
    those of Plan-and-Execute without it. Every row without the MCP has a diagnosis accepted in at least the registered share of its
    episodes, and a reference design below it reads GATE_PORT_SUSPECT with its funnel, since a fault of the port is looked for
    first. The count of exact matches of the CausalDiagnosis agent without the MCP, under the counting rule, reaches the bar of the
    registered rule: the smallest count of its episodes that an exact one-sided binomial test at the registered level separates
    from the count of the constant answer on the same episodes (stats.binomial_bar). When no count separates, or the bar lies at
    or below the floor of the starts the round played, the check refuses the registration itself."""
    valid_share, alpha = number("valid_share"), number("exact_match_alpha")
    rows = [row for row in design.AGENT_ROWS if not row[1]]
    runs = {row: development_run(root, row, design.AGENTS_PRESET) for row in rows}
    if len({json.dumps(meta.get("seeds")) for meta, _run in runs.values()}) > 1:
        raise SystemExit(f"{', '.join(run.name for _meta, run in runs.values())} play other seeds, and the gate compares the rows on the same episodes")
    episodes = {row: aggregate.episodes_of(run) for row, (_meta, run) in runs.items()}
    ours = episodes[design.CD]
    n = len(ours)
    right = round(sum(aggregate.value(e, "faulty_set_right") or 0.0 for e in ours))
    constant = aggregate.exact_match_floor(ours, None, {})["faulty_set_right_baseline_k"] or 0
    bar = binomial_bar(n, constant, alpha)
    floor = max(aggregate.registered_floor({}), Fraction(constant, n)) if n else None
    if bar is None:
        print(f"GATE_REFUSED no count of the {n} episodes of {design.CD[2]} separates from the {constant} of the constant answer by an exact "
              f"one-sided binomial test at {alpha}, so the registration itself is refused")
        return 1
    if Fraction(bar, n) <= floor:
        print(f"GATE_REFUSED the bar {bar} of {n} of the registered rule lies at or below the floor {float(floor):.3f} of the starts the round "
              "played, so the registration itself is refused")
        return 1
    reached = True
    for name, column in (("graded Top-1", "graded_top1"), ("graded pass", "graded_pass")):
        a, b = rate(ours, column), rate(episodes[design.PE], column)
        reached = reached and a >= b
        print(f"GATE {name}: {design.CD[2]} {a:.3f} over {len(ours)}, {design.PE[2]} {b:.3f} over {len(episodes[design.PE])}")
    for row in rows:
        k, n_row = sum(1 for e in episodes[row] if aggregate.funnel_facts(e)["diagnosis_accepted"]), len(episodes[row])
        if k >= valid_share * n_row:
            print(f"GATE valid diagnoses, {row[2]}: {k} of {n_row}, at or above the registered share {valid_share}")
            continue
        reached = False
        if row == design.CD:
            print(f"GATE valid diagnoses, {row[2]}: {k} of {n_row}, below the registered share {valid_share}")
            continue
        funnel = aggregate.funnel_row(episodes[row])
        print(f"GATE_PORT_SUSPECT {row[2]}: {k} of {n_row} episodes have a diagnosis accepted, below the registered share {valid_share}, so a "
              "fault of the port is looked for first; funnel " + ", ".join(f"{step} {funnel[f'{step}_k']}" for step in aggregate.FUNNEL_STEPS))
    reached = reached and right >= bar
    print(f"GATE exact match, {design.CD[2]}: {right} of {n} against the bar {bar} of {n}, the smallest count that an exact one-sided binomial "
          f"test at {alpha} separates from the {constant} of the constant answer")
    print("GATE_PASS" if reached else "GATE_MISS")
    return 0 if reached else 1
