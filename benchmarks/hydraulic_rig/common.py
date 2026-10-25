"""Shared pieces of the hydraulic-rig benchmark drivers: the fork's import path, the scenario classes, the episodes of a
class, episode folders and the per-episode launcher files of the two MCP servers.

The agent runners spawn a server from its launcher path, as ``python -m <dotted path>`` from the repository root when the
path lies inside the repository and as ``python <launcher path>`` otherwise, with almost no environment variables, so a
launcher cannot rely on PYTHONPATH or on an installed package. Each episode folder therefore gets two small launcher
files that put the fork's ``src`` on the import path themselves and bind the server process to the folder's state
file, the same binding that ``servers/hydraulic_rig/launcher.py`` and ``servers/causal_diagnosis/launcher.py`` use
under ``uv run``.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from envs.hydraulic_rig.data import build_blocks, default_facts_path, load_cycles  # noqa: E402
from scenario_anchors import CLASSES, class_combinations, start_state_table  # noqa: E402
from servers.causal_diagnosis.adapter import task_text  # noqa: E402
from servers.hydraulic_rig import episode  # noqa: E402

RIG_SERVER, CAUSAL_SERVER = "hydraulic_rig", "causal_diagnosis"
RIG_TOOLS = frozenset({"measure", "submit", "submit_insufficient_evidence", "replace", "finish"})
CAUSAL_TOOLS = frozenset({"get_asset_context", "query_fmea_paths", "get_symptom_report", "get_episode_evidence", "rank_root_causes",
                          "estimate_intervention_effect", "verify_outcome"})
SEEDED = "seeded"  # the environment's default rule: one or two faulty components, 47 candidate combinations drawn by seed
SCENARIOS = (*CLASSES, SEEDED)
STATE_FILES = {RIG_SERVER: "episode.json", CAUSAL_SERVER: episode.RELEASED_NAME}

_LAUNCHER = '''"""Per-episode entry point of the {server} MCP server, written by the benchmark driver beside the episode's files."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # a server process writes nothing beside the episode files but its own state
sys.path.insert(0, {src!r})

from servers.{server}.main import main  # noqa: E402

main(Path(__file__).resolve().with_name({state!r}))
'''


def load_world(facts_path=None):
    """The data file, the recorded cycles and the stable blocks per combination."""
    facts = Path(facts_path or default_facts_path())
    if not facts.exists():
        raise SystemExit(f"rig data file not found: {facts}; set HYD_FACTS_PATH (benchmarks/hydraulic_rig/README.md)")
    cycles = load_cycles(str(facts))
    return facts, cycles, build_blocks(cycles)


def scenario_env_kwargs(name):
    """Environment parameters of a scenario class (its exact fault count and its combinations), or the seeded default."""
    kwargs = episode.default_env_kwargs()
    if name != SEEDED:
        faults = CLASSES[name]
        kwargs.update(min_faulty=faults, max_faulty=faults, allowed=class_combinations(name))
    return kwargs


def scenario_task_text(name):
    """The task text an episode of the scenario opens with, worded from its public rule."""
    kwargs = scenario_env_kwargs(name)
    return task_text(kwargs["min_faulty"], kwargs["max_faulty"])


def scenario_episodes(name, cycles, blocks, per_record=None, episodes=5, first_seed=0, max_records=None):
    """The (seed, record) pairs of a run. For a class, every record (combination) contributes its first per_record start
    states, each from the smallest seed whose reset reaches it, or all of its start states when per_record is None, and
    max_records keeps only the first records in sorted order (quick checks); for the seeded default, `episodes` seeds from
    first_seed, whose records are drawn by the environment."""
    if name == SEEDED:
        return [(seed, None) for seed in range(first_seed, first_seed + episodes)]
    table = start_state_table(name, cycles, blocks)
    pairs = []
    for combo in sorted(table)[:max_records]:
        states = table[combo] if per_record is None else table[combo][:per_record]
        pairs += [(seed, combo) for seed, _window, _entry in states]
    return pairs


def start_episode(ep_dir, seed, env_kwargs, facts_path, task_text=None):
    """An episode folder with its state file and first released answers; returns the state file path."""
    ep_dir.mkdir(parents=True, exist_ok=True)
    state_path = ep_dir / STATE_FILES[RIG_SERVER]
    episode.start(state_path, seed, env_kwargs, facts_path, task_text=task_text)
    return state_path


def write_launchers(ep_dir, servers=(RIG_SERVER, CAUSAL_SERVER)):
    """The launcher files of an episode folder, as {server name: absolute path} for a runner's server_paths."""
    paths = {}
    for server in servers:
        path = Path(ep_dir) / f"{server}_server.py"
        path.write_text(_LAUNCHER.format(server=server, src=str(SRC), state=STATE_FILES[server]), encoding="utf-8")
        paths[server] = path.resolve()
    return paths


_CLAIMS = re.compile(r"\b(restored|recovered|recovery|repaired|fixed|resolved|back to (?:its )?(?:normal|healthy)|"
                     r"(?:is|are|now) healthy|fully operational|operating normally)\b", re.IGNORECASE)
_DENIES = re.compile(r"\b(?:not|no|never|cannot|could not|couldn't|wasn't|was not|isn't|is not|remains?|still)\W{1,3}"
                     r"(?:fully |completely |yet |been )?(?:restored|recovered|repaired|fixed|resolved|healthy|operational)\b",
                     re.IGNORECASE)


def record_weighted_mean(rows, value):
    """The mean of value over rows with every record weighted equally: records hold unequal numbers of start states, a
    record with a long stable block holding many, so a plain mean over rows mostly describes the largest records."""
    by_record = {}
    for r in rows:
        by_record.setdefault(r["record"], []).append(float(value(r)))
    return sum(sum(v) / len(v) for v in by_record.values()) / len(by_record) if by_record else None


def record_sizes_line(records):
    """One sentence on how many start states each record holds; records has one entry per start state."""
    sizes = {}
    for record in records:
        sizes[record] = sizes.get(record, 0) + 1
    low, high = min(sizes.values()), max(sizes.values())
    if low == high:
        return f"Every record holds {high} start states."
    largest = max(sizes, key=sizes.get)
    return f"Records hold {low} to {high} start states; the largest, {largest}, holds {high} of the {sum(sizes.values())}."


def answer_claims_recovery(text):
    """Heuristic reading of a closing answer: it states or implies that the rig was restored, with no denial beside it. The
    flag sorts answers for the run summary; the report shows the text, which the reader judges."""
    if not text:
        return False
    return bool(_CLAIMS.search(text)) and not _DENIES.search(text)
