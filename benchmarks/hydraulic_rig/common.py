"""Shared pieces of the hydraulic rig drivers (docs/hydraulic_rig/benchmarks.md): the fork's import path, the two
servers with their tool sets, the configuration presets and price tables, the per episode launcher files, the run
folder files and the columns of scores.csv.

The agent runners spawn a server from its launcher path, as ``python -m <dotted path>`` from the repository root when the
path lies inside the repository and as ``python <launcher path>`` otherwise, with almost no environment variables, so a
launcher cannot rely on PYTHONPATH or on an installed package. Each episode folder therefore gets one small launcher file
per server that puts the fork's ``src`` on the import path itself and binds the server process to the folder's state
file, the same binding that ``servers/hydraulic_rig/launcher.py`` and ``servers/causal_diagnosis/launcher.py`` use.
"""
import csv
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
HERE = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from envs.hydraulic_rig.data import build_blocks, default_facts_path, load_cycles  # noqa: E402
from servers.json_files import write_json_atomically  # noqa: E402

CONFIGS = HERE / "configs"
PRICE_TABLES = CONFIGS / "prices"
DEFAULT_RUNS = HERE / "runs"

RIG_SERVER, CAUSAL_SERVER = "hydraulic_rig", "causal_diagnosis"
RIG_TOOLS = frozenset({"get_asset_profile", "read_sensors", "advance_time", "summarize_changes_since", "get_failure_modes",
                       "submit_diagnosis", "declare_insufficient_evidence", "create_work_order", "finish"})
CAUSAL_TOOLS = frozenset({"get_episode_evidence", "rank_root_causes", "estimate_intervention_effect", "verify_outcome"})
WRITE_TOOLS = frozenset({"submit_diagnosis", "declare_insufficient_evidence", "create_work_order", "finish"})
STATE_FILE, RELEASED_FILE, PREDICTIONS_FILE = "episode.json", "released.json", "predictions.json"
STATE_FILES = {RIG_SERVER: STATE_FILE, CAUSAL_SERVER: RELEASED_FILE}
HIDDEN = "hidden"  # the subfolder of an episode that no agent facing tool reads: oracle, verdict, score and log
HIDDEN_FILES = ("oracle.json", "verdict.json", "score.json", "log.json")
PUBLIC_FILES = (STATE_FILE, RELEASED_FILE, "trajectory.json", "llm_calls.json", "report.md", "question.txt")

# scores.csv: the run and episode identity, the factors of the ablations, the verifier's scores and the bookkeeping
META_COLUMNS = ("run", "seed", "scenario_id", "date", "config", "agent", "model", "clock_mode", "guidance_level",
                "diagnostics_server", "prices", "verifier_scope", "ranking_model", "block", "n_faulty", "wall_s", "error")
SCORE_COLUMNS = ("pass", "success", "separable", "commitment_on_inseparable", "diagnosis_all_correct", "recovered", "unnecessary", "cost",
                 "cost_normalized", "downtime_total",
                 "detection_delay", "repair_delay", "turns_to_correct_diagnosis", "abstained", "abstention_correct",
                 "prediction_verdict", "failure_class", "end_reason", "time_used", "steps", "provenance_ok", "reward",
                 "cost_oracle", "grade", "judge_pass", "static_faulty_set_correct", "static_repairs_match_oracle")
BOOKKEEPING_COLUMNS = ("tool_calls", "refusals", "causal_calls", "llm_calls", "input_tokens", "output_tokens", "plan_steps",
                       "replay_matches")
COLUMNS = (*META_COLUMNS, *SCORE_COLUMNS, *BOOKKEEPING_COLUMNS)

_LAUNCHER = '''"""Per-episode entry point of the {server} MCP server, written by the benchmark driver beside the episode's files."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # a server process writes nothing beside the episode files but its own state
sys.path.insert(0, {src!r})

from servers.{server}.main import main  # noqa: E402

main(Path(__file__).resolve().with_name({state!r}){extra})
'''
from servers.causal_diagnosis.posterior import DEFAULT_RANKING_MODEL, RANKING_MODELS  # noqa: E402  the driver and the server share one default
DEFAULT_ABSTENTION_RULE = "decision"


def load_world(facts_path=None):
    """The data file, the recorded cycles and the stable blocks per combination."""
    facts = Path(facts_path or default_facts_path())
    if not facts.exists():
        raise SystemExit(f"rig data file not found: {facts}; set HYD_FACTS_PATH (benchmarks/hydraulic_rig/README.md)")
    cycles = load_cycles(str(facts))
    return facts, cycles, build_blocks(cycles)


def contract_tools(server):
    """The tool names the contract lists for a server (servers.contract), or the plan's list while the contract module
    does not exist yet."""
    fallback = {RIG_SERVER: RIG_TOOLS, CAUSAL_SERVER: CAUSAL_TOOLS}[server]
    try:
        from servers import contract
    except ImportError:
        return fallback
    return frozenset(contract.tools_of(server)) or fallback


def _preset_file(spec, folder, kind):
    path = Path(spec)
    if path.suffix == ".json" and path.exists():
        return path
    candidate = folder / f"{spec}.json"
    if candidate.exists():
        return candidate
    names = sorted(p.stem for p in folder.glob("*.json"))
    raise SystemExit(f"unknown {kind} {spec!r}: give a path to a JSON file or one of {names}")


def load_config(spec):
    """(name, EpisodeConfig) of a preset name under configs/ or a JSON path: the preset's fields over the defaults of
    EpisodeConfig, validated by its constructor, so a preset lists only what it changes and run.json records the whole."""
    from envs.hydraulic_rig.config import EpisodeConfig

    path = _preset_file(spec, CONFIGS, "config preset")
    preset = json.loads(path.read_text(encoding="utf-8"))
    fields = {k: v for k, v in preset.items() if k not in ("name", "description")}
    base = EpisodeConfig().to_json()
    unknown = sorted(set(fields) - set(base))
    if unknown:
        raise SystemExit(f"{path}: fields {unknown} are not fields of EpisodeConfig")
    base.update(fields)
    return preset.get("name", path.stem), EpisodeConfig.from_json(base)


def _empty_paths(value, prefix=""):
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in _empty_paths(v, f"{prefix}{k}/")]
    return [prefix.rstrip("/")] if value is None else []


def load_price_table(spec):
    """A price table of configs/prices/ or a JSON path, refused while any value is empty (the sme.json template)."""
    path = _preset_file(spec, PRICE_TABLES, "price table")
    table = json.loads(path.read_text(encoding="utf-8"))
    sections = {k: table[k] for k in ("prices", "durations", "downtime_rate") if k in table}
    if "prices" not in sections:
        raise SystemExit(f"{path}: a price table needs a 'prices' section")
    empty = _empty_paths(sections)
    if empty:
        raise SystemExit(f"{path}: the table {table.get('name', path.stem)!r} has empty values at {empty}; "
                         "fill them after the session with the subject matter experts")
    return {"name": table.get("name", path.stem), "source": table.get("source", ""), **sections}


def apply_price_table(config, table):
    """The configuration with the table's prices (and its durations and downtime rates when the table gives them),
    validated again through the JSON round trip of EpisodeConfig."""
    replaced = dataclasses.replace(config, **{k: table[k] for k in ("prices", "durations", "downtime_rate") if k in table})
    return type(config).from_json(replaced.to_json())


def write_launchers(ep_dir, servers=(RIG_SERVER, CAUSAL_SERVER), ranking_model=DEFAULT_RANKING_MODEL, abstention=None):
    """The launcher files of an episode folder, as {server name: absolute path} for a runner's server_paths. The
    diagnostics launcher always names the ranking model (ablation A12), so that the model the run records is the model
    the server runs, whatever the server's own default is, and names the abstention rule with its cut (ablation A8,
    abstention = (rule, cut)) only when they differ from the default rule."""
    paths = {}
    for server in servers:
        extra = ""
        if server == CAUSAL_SERVER:
            extra += f", ranking_model={ranking_model!r}"
        if server == CAUSAL_SERVER and abstention and abstention[0] != DEFAULT_ABSTENTION_RULE:
            extra += f", abstention_rule={abstention[0]!r}, abstention_cut={float(abstention[1])!r}"
        path = Path(ep_dir) / f"{server}_server.py"
        path.write_text(_LAUNCHER.format(server=server, src=str(SRC), state=STATE_FILES[server], extra=extra), encoding="utf-8")
        paths[server] = path.resolve()
    return paths


def write_json(path, data):
    """An atomic JSON write; values json cannot encode (paths, tuples as keys) are written through str."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_json_atomically(path, json.loads(json.dumps(data, default=str)), indent=1)


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(run_dir):
    """The rows of a run's scores.csv, every cell a string."""
    path = Path(run_dir) / "scores.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows, fieldnames):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), restval="")
        writer.writeheader()
        writer.writerows(rows)


def flat(value):
    """A CSV cell: dictionaries and lists as JSON, None as an empty string, everything else as it is."""
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, separators=(",", ":"), default=str)
    return value


def as_bool(value):
    """A CSV cell read back as a flag; empty cells are None."""
    if value is None or value == "":
        return None
    return str(value).strip().lower() in ("true", "1", "yes")


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def episode_dirs(run_dir):
    """The episode folders of a run, in name order; a folder is an episode when it holds the state file."""
    return sorted(p for p in Path(run_dir).iterdir() if p.is_dir() and (p / STATE_FILE).exists())


def run_dirs(runs_root):
    """The run folders under a root, in name order; a folder is a run when it holds run.json."""
    root = Path(runs_root)
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "run.json").exists())
