"""CausalDiagnosis/docs/metrics.md, the cards of the metrics, states facts of the tree and nothing else: each count it names stands
in the sentence that names it, computed here from the code, and no other number appears in its prose or tables outside code
spans and links. Every card holds its name and the nine fields after it in their order, and the cards of the MCP name every cell
of the table MCP. Every name under Computed in stands in a module that the same field names, every table a card names exists,
every table of the tree is named by a card or stands in the list of tables that hold records, and every score and bookkeeping
column of scores.csv stands on the page. No test of this file checks a rule of writing."""
import collections
import inspect
import itertools
import re
from pathlib import Path

from CausalDiagnosis.evaluation.result_tables import tables_from_runs as aggregate
from CausalDiagnosis.experiments.episode_runners import run_folder_layout
from CausalDiagnosis.experiments.table_definitions import experiment_tables as design
from CausalDiagnosis.frameworks.AssetOpsBench.LLM_Judges import episode_judge as judge
from CausalDiagnosis.evaluation.result_tables import paper_tables as main_tables
from CausalDiagnosis.evaluation.metrics import mcp_metrics
from CausalDiagnosis.experiments.table_definitions import registered_analysis as registered
from CausalDiagnosis.evaluation.metrics import table_statistics as stats
from CausalDiagnosis.datasets.dataset_loader.evidence_source import SEGMENT
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import start_state_seeds as start_states, verifier
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.environment.reference_policies.diagnosability_label import ROLLOUTS
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP import tools as adapter
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline import diagnosability_grade as grade, offline_table
from CausalDiagnosis.experiments.episode_runners.run_configuration import load_config

from CausalDiagnosis.utils.paths import ROOT
DOC = ROOT / "CausalDiagnosis" / "docs" / "metrics.md"
MAIN_TABLES = ROOT / "CausalDiagnosis" / "evaluation" / "result_tables" / "paper_tables" / "__init__.py"
FIELDS = ("Class", "Definition", "Formula", "Inputs", "Computed in", "Sanity relations", "Gaming", "Hypothesis", "Table")  # after the name
WORDS = dict(enumerate(("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")))
JUDGE_COLUMNS = {"TC", "DRA", "RV"}  # the columns of the judge scores, the only capitalized names of a Table field that are no aggregate table
RECORDS = "## Tables that contain records and no metric"
# a number of the prose: digits not glued to a letter, a digit, an underscore, a dot, a hyphen or an at sign, so that hyd_1 and AC@1
# are names and not counts
NUMBER = re.compile(r"(?<![\w.\-@])\d+(?:,\d{3})*(?![\w])")
FILES = (".csv", ".json", ".jsonl", ".md", ".txt", ".tex")


def prose(text):
    """The text a reader reads as prose and tables: without fenced code, code spans, link targets and URLs."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"\]\([^)]*\)", "]", text)
    return re.sub(r"https?://\S+", " ", text)


def page():
    return DOC.read_text(encoding="utf-8")


def _listed(items):
    return f"{', '.join(map(str, items[:-1]))} and {items[-1]}"


def counts():
    """[(the sentence fragment of the page, the count it states)], every count computed from the code."""
    profile, config = get_profile("hyd_1", engine=False), EpisodeConfig()
    combinations = list(itertools.product(*(profile.levels[c] for c in profile.components)))
    starts = [k for k in combinations if config.within_fault_rule(k)]
    rule = config.fault_rule
    level = round(100 * inspect.signature(stats.cluster_proportion).parameters["level"].default)
    resamples = inspect.signature(stats.cluster_bootstrap).parameters["resamples"].default
    base = design.LADDER.index(design.AGENTS_PRESET)
    assert load_config(design.AGENTS_PRESET)[1].cost_unit() == "calls"  # the page says that layer 0 prices nothing
    hypotheses = len(design.HYPOTHESES) * len(design.HYPOTHESIS_MEASURES)
    training = len(design.TRAINING_ROWS) * len(design.TRAINING_COMPARISONS) * len(design.TRAINING_MEASURES)
    reward = inspect.signature(verifier.reward).parameters
    protocol = judge.PROTOCOLS[aggregate.PAPER_PROTOCOL]
    readings = offline_table.READINGS
    # the correction groups and the family sizes of the registered analysis, which the paragraph and the table of families state
    groups = {group["name"]: set(group["families"]) for group in registered.groups()}
    h1_top, h2_prime = registered.size("h1_top"), registered.size("h2_prime")
    full, parallel = registered.size("full_environment"), registered.size("parallel_time")
    assert groups["agents_environment"] == {"h1_top", "h2_prime"} and groups["judge"] == {"h0"} and groups["agents"] == {"agents"}
    assert registered.role("full_environment") == registered.role("parallel_time") == "secondary"
    assert tuple(registered.measures("h2_prime")) == design.H2_PRIME_MEASURES == ("pass", "reward")
    assert len(design.H2_PRIME) * len(design.H2_PRIME_MEASURES) == h2_prime
    assert ROLLOUTS == grade.ROLLOUTS and main_tables.OFFLINE_PREHISTORY == SEGMENT  # one count stands for each pair
    rows, replicates = len(design.AGENT_ROWS), start_states.REPLICATES

    def table_of(builder):
        return int(re.search(r"Table (\d+) of AssetOpsBench", builder.__doc__).group(1))

    return [
        (f"with a {level}% interval from a bootstrap", level),
        (f"a bootstrap of {resamples:,} resamples", resamples),
        (f"with the CausalDiagnosis MCP off and on, on layer {base}", base),
        (f"`J` contains the {len(profile.components)} components {_listed(list(profile.components))}, with "
         f"{_listed([len(profile.levels[c]) for c in profile.components])} levels", len(profile.components)),
        (f"`S` holds {len(combinations)} combinations, and the test set consists of {replicates} replicates of each of the {len(starts)} test "
         f"combinations with {rule['min_faulty']} or {rule['max_faulty']} faulty components, {replicates * len(starts)} episodes per configuration", len(starts)),
        (f"Layer {base} prices nothing", base),
        (f"The {len(design.REFERENCE_POLICIES)} policies of `REFERENCE_POLICIES`", len(design.REFERENCE_POLICIES)),
        (f"runs the {rows} configurations of `AGENT_ROWS`", rows),
        (f"reject over the {hypotheses} tests of `tab:hyp`", hypotheses),
        (f"lies below {round(100 * aggregate.registered_floor(None))}% is at floor", aggregate.registered_floor(None)),
        (f"names {WORDS[len(groups)]} primary groups", len(groups)),
        (f"Its {h1_top} tests share one Holm correction with the {h2_prime} tests of H2' in the group `agents_environment`", h1_top),
        (f"Its {h2_prime} tests share one Holm correction with the {h1_top} tests of `h1_top` in the group `agents_environment`, "
         f"{h1_top + h2_prime} tests in all", h1_top + h2_prime),
        (f"The family `full_environment` is secondary, with a Holm correction of its own over its {full} tests", full),
        (f"The family `parallel_time` is secondary, with {WORDS[parallel]} test", parallel),
        (f"contains the {rows + len(design.HYPOTHESES)} tests of the judge", rows + len(design.HYPOTHESES)),
        (f"two measures of the {training} training tests", training),
        (f"with Holm over the {len(design.LADDER) - 1} layers above the base", len(design.LADDER) - 1),
        (f"the other measure of the {training} training tests", training),
        (f"runs each of the {rows} configurations of the first experiment on the healthy combination at {replicates} entry offsets, "
         f"{rows * replicates} episodes", rows * replicates),
        (f"The weights `w_c` and `w_t` are {reward['w_c'].default} and {reward['w_t'].default}, the cap `c_max` is {reward['cost_cap'].default}, "
         f"the constant `COST_CAP`, and the credit `c_decl` is {verifier.ABSTENTION_CREDIT}, the constant `ABSTENTION_CREDIT`", verifier.ABSTENTION_CREDIT),
        (f"with {aggregate.BINS} calibration bins", aggregate.BINS),
        (f"The first of {len(verifier.FAILURE_CLASSES)} classes", len(verifier.FAILURE_CLASSES)),
        (f"A repeated malformed call is {verifier.REPEATED_REFUSALS} or more refused actions", verifier.REPEATED_REFUSALS),
        (f"its protocol of {protocol['calls']} calls at temperature {protocol['temperature']}", protocol["calls"]),
        (f"as Table {table_of(aggregate.table_judge_agreement)} of AssetOpsBench reports it", table_of(aggregate.table_judge_agreement)),
        (f"as Table {table_of(aggregate.table_plans)} of AssetOpsBench reports them", table_of(aggregate.table_plans)),
        (f"the declared share on {len(main_tables.STRATA_SHOWN)} strata", len(main_tables.STRATA_SHOWN)),
        (f"each count of live readings of `READINGS`, {_listed(list(readings))}", readings),
        (f"the prehistory contains {SEGMENT} recordings", SEGMENT),
        (f"The label runs {ROLLOUTS} rollouts per initial combination", ROLLOUTS),
        (f"how many candidate root causes an answer returns ({adapter.DEFAULT_TOP_K} when left out, with no maximum)", adapter.DEFAULT_TOP_K),
        # the counts in words that stand for a design count
        (f"on all {WORDS[len(design.LADDER)]} big layers", len(design.LADDER)),
        (f"pools the {WORDS[rows]} configurations of a model", rows),
    ]


def cards(text):
    """(heading, [the bullets]) of every card, the third level headings of the sections."""
    out = []
    for section in re.split(r"(?m)^## ", text)[1:]:
        for block in re.split(r"(?m)^### ", section)[1:]:
            lines = block.splitlines()
            out.append((lines[0], [line for line in lines[1:] if line.startswith("- ")]))
    return out


def field(bullets, name):
    return next(b for b in bullets if b.startswith(f"- **{name}.** "))


def modules(spans):
    """The text of every Python file that a field names, by its path from the root or by its file name under src and benchmarks."""
    paths = []
    for span in spans:
        if span.endswith(".py"):
            paths += [ROOT / span] if "/" in span else [p for top in ("src", "benchmarks") for p in (ROOT / top).rglob(span)]
    return [p.read_text(encoding="utf-8") for p in paths if p.is_file()]


def tables_of_the_tree():
    """(the labels of the paper layout of paper_tables/, the ids of the tables of tables_from_runs/), as the tree builds them."""
    built = main_tables.built_layout(main_tables.Sheet(collections.defaultdict(list), "m"))
    return {table.label for section in built for table in section}, {table.id for table in aggregate.tables()}


def test_each_count_of_the_page_stands_in_its_sentence():
    text = " ".join(page().split())
    assert [fragment for fragment, _count in counts() if fragment not in text] == []


def test_the_page_names_no_number_that_the_code_does_not_pin():
    """Every number of the prose and the tables stands inside one of the pinned sentences of counts(), so a number typed by
    hand fails even when its value equals a pinned count."""
    text = " ".join(prose(page()).split())
    covered = set()
    for fragment, _count in counts():
        fragment = " ".join(prose(fragment).split())
        covered |= {i for m in re.finditer(re.escape(fragment), text) for i in range(m.start(), m.end())}
    loose = [text[max(0, m.start() - 40):m.end() + 20] for m in NUMBER.finditer(text) if not set(range(m.start(), m.end())) <= covered]
    assert loose == []


def test_every_card_holds_its_name_and_nine_fields_in_their_order_and_the_measures_of_the_mcp_name_every_cell():
    found = cards(page())
    wrong = [heading for heading, bullets in found if [re.match(r"- \*\*(.+?)\.\*\* ", b).group(1) for b in bullets] != list(FIELDS)]
    assert found and wrong == []
    assert [cell for cell in mcp_metrics.CELLS if f"`{cell}`" not in page()] == []


def test_every_name_under_computed_in_stands_in_a_module_that_the_field_names():
    missing = []
    for heading, bullets in cards(page()):
        spans = re.findall(r"`([^`]+)`", field(bullets, "Computed in"))
        texts = modules(spans)
        if not texts:
            missing.append((heading, "names no module"))
        for span in spans:
            if span.endswith(".py") or span.endswith(FILES) or not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", span):
                continue
            if not all(any(re.search(rf"\b{re.escape(part)}\b", t) for t in texts) for part in span.split(".")):
                missing.append((heading, span))
    assert missing == []


def test_every_table_that_a_card_names_exists():
    labels, ids = tables_of_the_tree()
    named, unknown = set(), []
    for heading, bullets in cards(page()):
        text = field(bullets, "Table")
        named |= set(re.findall(r"`tab:(\w+)`", text))
        unknown += [(heading, w) for w in re.findall(r"\b[A-Z][A-Z0-9]+(?:_[A-Za-z0-9]+)*\b", prose(text)) if w not in ids | JUDGE_COLUMNS]
    assert unknown == [] and sorted(named - labels) == [] and {"mcpvalue", "mcpoffline"} <= named


def test_every_table_of_the_tree_is_named_by_a_card_or_listed_as_a_table_of_records():
    labels, ids = tables_of_the_tree()
    text = page()
    named_labels, named_ids = set(), set()
    for _heading, bullets in cards(text):
        cell = field(bullets, "Table")
        named_labels |= set(re.findall(r"`tab:(\w+)`", cell))
        named_ids |= set(re.findall(r"\b[A-Z][A-Z0-9]+(?:_[A-Za-z0-9]+)*\b", prose(cell))) & ids
    records = text.split(RECORDS)[1]
    listed_labels = set(re.findall(r"^\| `tab:(\w+)` \|", records, re.M))
    listed_ids = set(re.findall(r"^\| ([A-Z][A-Za-z0-9_]+) \|", records, re.M))
    assert sorted(labels - named_labels - listed_labels) == [] and sorted(ids - named_ids - listed_ids) == []


def test_every_score_column_has_a_card():
    """A verdict key or a bookkeeping column added to scores.csv later without its sentence on the page fails here."""
    text = page()
    assert [c for c in (*run_folder_layout.SCORE_COLUMNS, *run_folder_layout.BOOKKEEPING_COLUMNS) if f"`{c}`" not in text] == []
