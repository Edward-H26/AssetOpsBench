"""CausalDiagnosis/docs/dataset.md, the datasheet of the rig's data, states no number that the code, the committed files or the
recordings do not pin: each number stands in the sentence fragment that names it, built here from the code and the files, as
often as the page means it, and no other number appears in its prose or tables outside code spans and links. The table of
sources states the hashes and revisions that the code checks. The fragments that need the recordings are skipped without the
data file. The values that the page holds back come from one command of the fork, datasheet_statistics.py. The test holds no
value computed from the data, and it asserts facts of the tree alone, no writing rule."""
import inspect
import itertools
import json
import math
import re
from pathlib import Path
from statistics import NormalDist

import pytest

from CausalDiagnosis.evaluation.result_tables import datasheet_statistics as data_properties
from CausalDiagnosis.experiments.table_definitions import experiment_tables as design
from CausalDiagnosis.experiments.table_definitions import registered_analysis as registered
from CausalDiagnosis.evaluation.metrics import table_statistics as stats
from CausalDiagnosis.frameworks.AssetOpsBench.scenarios import download_scenarios as fetch_scenarios
from CausalDiagnosis.datasets.dataset_loader.evidence_source import BLOCK_CAP, SEGMENT
from CausalDiagnosis.datasets.dataset_loader import recordings as records
from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile
from CausalDiagnosis.datasets.splits.development_split import development_starts, starts_of_test_set
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import start_state_seeds as start_states
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path
from CausalDiagnosis.environment.reference_policies.diagnosability_label import ROLLOUTS

from CausalDiagnosis.utils.paths import ROOT
DOC = ROOT / "CausalDiagnosis" / "docs" / "dataset.md"
FOLDS = ROOT / "CausalDiagnosis" / "configs" / "training" / "hyd_1" / "folds.json"
FETCH = ROOT / "CausalDiagnosis" / "datasets" / "hyd_1" / "download_and_build.py"
PORTING = ROOT / "CausalDiagnosis" / "docs" / "adding-a-dataset.md"
CHECKSUMS = ROOT / "CausalDiagnosis" / "datasets" / "hyd_1" / "data" / "CHECKSUMS.txt"
HAS_DATA = Path(default_facts_path()).exists()
# a number of the prose: digits not glued to a letter, a digit, an underscore, a dot or a hyphen, so that E18, hyd_1 and SHA-256
# are names and not counts
NUMBER = re.compile(r"(?<![\w.\-])\d+(?:,\d{3})*(?:\.\d+)?(?![\w])")
POWER, DISCORDANCE = 0.8, (0.1, 0.2, 0.3)  # the design inputs of the table of what the counts can detect
# the constants that pin a dataset the tree reads, by the file that checks it; a name that a file of the tree does not hold pins
# nothing, and the archive of the recordings is pinned by its line of CHECKSUMS.txt
PINS = (("CausalDiagnosis/datasets/hyd_1/build_features_and_blocks.py", "EQA_FACTS_SHA256"),
        ("CausalDiagnosis/datasets/hyd_1/verify_data_facts.py", "FACTS_SHA256"),
        ("CausalDiagnosis/frameworks/AssetOpsBench/reproduction/scenario_file.py", "SCENARIO_REVISION"),
        ("CausalDiagnosis/frameworks/AssetOpsBench/reproduction/scenario_file.py", "SCENARIO_SHA256"),
        ("CausalDiagnosis/frameworks/AssetOpsBench/reproduction/fetch_public_servers_and_data.py", "FORMS_REVISION"),
        ("CausalDiagnosis/frameworks/AssetOpsBench/reproduction/fetch_public_servers_and_data.py", "FORMS_SHA256"),
        ("CausalDiagnosis/frameworks/AssetOpsBench/scenarios/download_scenarios.py", "REVISION"),
        ("CausalDiagnosis/frameworks/AssetOpsBench/scenarios/download_scenarios.py", "SOURCE_SHA256"))


def prose(text):
    """The text a reader reads as prose and tables: without fenced code, code spans, link targets and URLs."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"\]\([^)]*\)", "]", text)
    return re.sub(r"https?://\S+", " ", text)


def page():
    return " ".join(DOC.read_text(encoding="utf-8").split())


def faulty(profile, combination):
    return [c for c, level in zip(profile.components, combination) if level != profile.healthy[c]]


def smallest_n(tests, alpha):
    """The fewest clusters with a nonzero sum at which an exact two sided sign flip test can pass the first Holm step."""
    return math.ceil(1 + math.log2(tests / alpha))


def family_row(name, tests, clusters, episodes, alpha):
    """A row of the table of families: the minimum detectable effect at each discordance, per episode and per cluster mean."""
    z = NormalDist().inv_cdf(1 - alpha / (2 * tests)) + NormalDist().inv_cdf(POWER)
    cells = [f"{z * math.sqrt(p / (clusters * episodes)):.2f} to {z * math.sqrt(p / clusters):.2f}" for p in DISCORDANCE]
    return f"| {name} | {tests} | {clusters} | {episodes} | {smallest_n(tests, alpha)} | " + " | ".join(cells) + " |"


def correction_size(name):
    """The number of tests that Holm corrects together with the family `name` of the registered file: the sum of the sizes of
    the families of its primary group, or its own size for a secondary family."""
    for group in registered.groups():
        if name in group["families"]:
            return sum(registered.size(member) for member in group["families"])
    return registered.size(name)


def family_rows(test_clusters, single_clusters, alpha):
    """The table of families, each with the number of tests of its Holm correction in the registered file: H1 and H2 on the
    base layer, H1 again on the full environment with H2' in one primary group, H0, the layer study that pools the agents
    without the MCP over the big layers it adds, and the training family, also on the combinations with one fault."""
    r, agents = start_states.REPLICATES, len(design.LADDER_ROWS)
    training = correction_size("training")
    families = [("H1 and H2 on the base layer", correction_size("agents"), test_clusters, r),
                ("H1 and H2' on the full environment", correction_size("h2_prime"), test_clusters, r),
                ("H0", correction_size("h0"), test_clusters, r),
                ("layer study", correction_size("layers"), test_clusters, r * agents),
                ("training", training, test_clusters, r),
                ("training, combinations with one fault", training, single_clusters, r)]
    return [family_row(*family, alpha) for family in families], training


def listed(items):
    """Items joined as the page joins them: "a, b and c"."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def static_fragments():
    """[(fragment, the times the page states it)] that the code and the committed files fix."""
    profile, config = get_profile("hyd_1", engine=False), EpisodeConfig()
    grid = list(itertools.product(*(profile.levels[c] for c in profile.components)))
    starts = [k for k in grid if config.within_fault_rule(k)]
    faults = [len(faulty(profile, k)) for k in starts]
    two = [k for k in starts if len(faulty(profile, k)) == 2]
    rule, r = config.fault_rule, start_states.REPLICATES
    alpha = inspect.signature(stats.holm).parameters["alpha"].default
    uci = int(re.search(r"/static/public/(\d+)/", FETCH.read_text(encoding="utf-8")).group(1))
    folds = json.loads(FOLDS.read_text(encoding="utf-8"))
    ids = sorted({row["fold"] for row in folds["combinations"]})
    sizes = [sum(row["fold"] == f for row in folds["combinations"]) for f in ids]
    trains = {sum(f in row["models"] for row in folds["combinations"]) for f in ids}
    assert len(trains) == 1, trains  # the datasheet states one count for every model
    trains = trains.pop()
    atom = {(a["component"], a["level"]): a["fold"] for a in folds["atoms"]}
    spans = [len({atom[(c, level)] for c, level in zip(profile.components, k) if level != profile.healthy[c]}) for k in two]
    master = inspect.signature(start_states.sampled_combination_seeds).parameters["master_seed"].default
    levels = [str(len(profile.levels[c])) for c in profile.components]
    rest = BLOCK_CAP - r
    per = [sum(c in faulty(profile, k) for k in starts) for c in profile.components]
    deps = profile.config_fields()["dependencies"]
    ordered = [k for k in two if any(b in deps.get(a, []) for a in faulty(profile, k) for b in faulty(profile, k))]
    steps = re.findall(r"^(\d+)\. (.*)$", PORTING.read_text(encoding="utf-8"), flags=re.M)
    register = [n for n, step in steps if step.startswith("Register the training split")]
    assert len(register) == 1, register
    rows, training = family_rows(len(starts), faults.count(1), alpha)
    agents = len(design.AGENT_ROWS)
    per_component = [f"of the {c} in {n}" for c, n in zip(profile.components, per)]
    return [
        (f"dataset {uci} of the UCI", 2),
        (f"UCI dataset {uci}", 2),
        (f"a constant load cycle of {config.seconds_per_cycle:g} s", 1),
        (f"records {len(profile.channels)} sensor channels", 1),
        (f"The labels set {len(profile.components)} components", 1),
        (f"with {listed(levels)} levels", 1),
        (f"contains the {len(profile.channels)} channel means of the same cycles", 1),
        (f"The {fetch_scenarios.RECORDS} anchored scenarios", 1),
        (f"the first {BLOCK_CAP} stable cycles", 1),
        (f"splits them into {SEGMENT} cycles that the tools of AssetOpsBench show before the first step and {SEGMENT} cycles that "
         "the episode replays", 1),
        (f"after the {SEGMENT} cycles of an episode the same {SEGMENT} cycles repeat", 1),
        (f"The {BLOCK_CAP} starting offsets of a combination", 1),
        (f"the same {BLOCK_CAP} offsets", 1),
        (f"allows {rule['min_faulty']} to {rule['max_faulty']} faulty components", 1),
        (f"{len(starts)} are test combinations, {faults.count(1)} with one fault and {faults.count(2)} with two", 1),
        (f"Each test combination has {BLOCK_CAP} starting offsets, so the environment can draw {len(starts) * BLOCK_CAP} initial "
         "states of the test set", 1),
        (f"The test set consists of {r} replicates of every test combination, {r * len(starts)} episodes", 1),
        (f"from the {start_states.EVALUATION_POOL} smallest seeds", 1),
        (f"The smallest of the {start_states.EVALUATION_POOL}", 1),
        (f"under the master seed {master}", 1),
        (f"runs {ROLLOUTS} rollouts", 1),
        (f"Each of the {agents} agent configurations of the first experiment runs on the fault-free combination at {r} starting "
         f"offsets, {agents * r} episodes", 1),
        (f"partitions the {faults.count(1)} single faults into K = {folds['k']} folds from seed {folds['seed']}", 1),
        (f"The folds contain {sizes[0]} and {sizes[1]} test combinations", 1),
        (f"Each model trains on {trains} combinations", 1),
        (f"Of the {len(two)} combinations with two faults, {spans.count(1)} contain both faults in one fold and {spans.count(2)} contain "
         "one fault from each fold", 1),
        (f"on those {spans.count(2)} combinations, {spans.count(2) * r} test episodes", 1),
        (f"the {rest} initial states of each of its training combinations that no replicate uses, {trains * rest} per model", 1),
        (f"since a block provides only {BLOCK_CAP} cycles", 1),
        (f"as step {register[0]} of [adding-a-dataset.md](adding-a-dataset.md) describes", 1),
        (f"at alpha = {alpha}", 1),
        (f"at power {POWER}", 1),
        (f"the test set has {len(starts)} clusters of {r} paired episodes, {faults.count(1)} combinations with one fault and "
         f"{faults.count(2)} with two", 1),
        (f"A fault {per_component[0].replace(' in ', ' stands in ', 1)} test combinations, {listed(per_component[1:])}", 1),
        (f"the order of the repairs matters in {len(ordered)} of the {len(two)} combinations with two faults", 1),
        (f"The rule switches on with layer {design.LADDER.index('repair_rules')}", 1),
        (" | ".join(f"MDE at p_d {p}" for p in DISCORDANCE), 1),
        *((row, 1) for row in rows),
        (f"only the families over all {len(starts)} test combinations decide", 1),
        (f"has only {faults.count(1)} clusters, below the {smallest_n(training, alpha)} that its family needs", 1),
        (f"against the {agents} agent configurations under the base configuration", 1),
        (f"A combination provides {BLOCK_CAP} stable cycles, and the {SEGMENT} cycles of an episode repeat", 1),
        (f"so the {r} replicates of a combination are not independent rigs", 1),
        (f"a block of at least {BLOCK_CAP} stable cycles", 1),
        (f"the {fetch_scenarios.RECORDS} anchored scenarios", 1),
    ]


def data_fragments():
    """[(fragment, the times the page states it)] that need the recorded cycles: the recorded grid and the starts of the
    development set and of the test set as CausalDiagnosis.datasets.splits.development_split gives them."""
    profile, config = get_profile("hyd_1", engine=False), EpisodeConfig()
    blocks = records.build_blocks(records.load_cycles(str(default_facts_path()), profile))
    grid = list(itertools.product(*(profile.levels[c] for c in profile.components)))
    assert sorted(blocks) == sorted(grid)  # the datasheet says that every combination of the grid has a block
    starts = list(starts_of_test_set(profile, list(blocks)))
    development = list(development_starts(profile, list(blocks)))
    moved = len([k for k in blocks if config.within_fault_rule(k)]) - len(starts)
    numbers = sorted({len(faulty(profile, k)) for k in development})
    tested = sorted({len(faulty(profile, k)) for k in starts})
    return [
        (f"All {len(grid)} combinations of the label grid have a block", 1),
        (f"Of the {len(blocks)} recorded combinations", 1),
        (f"the {len(development)} recorded combinations with {numbers[0]} or {numbers[-1]} faults", 1),
        ("so no test combination moves" if moved == 0 else f"so {moved} test combinations move", 1),
        (f"a development episode starts with {numbers[0]} or {numbers[-1]} faults, and a test episode starts with {tested[0]} or "
         f"{tested[-1]}", 1),
    ]


def test_each_number_of_the_code_stands_in_its_sentence_as_often_as_the_page_means_it():
    text = page()
    assert [(fragment, times, text.count(fragment)) for fragment, times in static_fragments() if text.count(fragment) != times] == []


@pytest.mark.skipif(not HAS_DATA, reason="hydraulic rig facts file not present; set HYD_FACTS_PATH")
def test_each_number_of_the_recordings_stands_in_its_sentence():
    text = page()
    assert [(fragment, times, text.count(fragment)) for fragment, times in data_fragments() if text.count(fragment) != times] == []


@pytest.mark.skipif(not HAS_DATA, reason="hydraulic rig facts file not present; set HYD_FACTS_PATH")
def test_the_datasheet_names_no_number_that_the_code_does_not_pin():
    """Every number of the prose and the tables stands inside one of the pinned fragments, so a number typed by hand fails
    even when its value equals a pinned count."""
    text = " ".join(prose(DOC.read_text(encoding="utf-8")).split())
    covered = set()
    for fragment, _times in static_fragments() + data_fragments():
        fragment = " ".join(prose(fragment).split())
        covered |= {i for m in re.finditer(re.escape(fragment), text) for i in range(m.start(), m.end())}
    loose = [m.group(0) for m in NUMBER.finditer(text) if not set(range(m.start(), m.end())) <= covered]
    assert loose == []


def test_the_table_of_sources_states_the_hashes_and_revisions_the_code_checks():
    """Every hash and revision that a file of the tree pins for a dataset stands in the table of sources, the archive of the
    recordings by its line of the committed checksum file, and every SHA-256 of the table is such a pin, all read from the
    source texts so that no module is imported."""
    after = DOC.read_text(encoding="utf-8").split("### Data sources", 1)[1].splitlines()
    first = next(i for i, line in enumerate(after) if line.startswith("|"))
    table = "\n".join(itertools.takewhile(lambda line: line.startswith("|"), after[first:]))
    pins = {}
    for rel, name in PINS:
        values = re.findall(rf'^{name} = "([0-9a-f]+)"', (ROOT / rel).read_text(encoding="utf-8"), flags=re.M) if (ROOT / rel).is_file() else []
        assert len(values) <= 1, (rel, name, values)
        pins.update({(rel, name): value for value in values})
    archive = re.search(r'^ARCHIVE_NAME = "(.+)"', FETCH.read_text(encoding="utf-8"), flags=re.M).group(1)
    lines = [line.split() for line in CHECKSUMS.read_text(encoding="utf-8").splitlines() if line.strip()]
    archived = [words[0] for words in lines if words[-1].endswith(archive)]
    assert len(archived) == 1, archived
    pins[("CHECKSUMS.txt", archive)] = archived[0]
    assert [key for key, value in pins.items() if f"`{value}`" not in table and f"/{value}/" not in table] == []
    stated = set(re.findall(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", table))
    assert sorted(stated - set(pins.values())) == []


@pytest.mark.skipif(not HAS_DATA, reason="hydraulic rig facts file not present; set HYD_FACTS_PATH")
def test_the_held_values_come_from_one_command_of_the_fork():
    """datasheet_statistics.py gives every value that the page holds back, one object per value with its name and its group, in
    the order of the held sentences. Relations only, since no value enters the test."""
    found = data_properties.properties()
    assert [(p["name"], p["group"]) for p in found] == [("block_sizes", "C"), ("cycles", "C"), ("healthy_cycles", "C"),
                                                       ("healthy_left_out", "R"), ("drift", "R"), ("crossing", "R")]
    value = {p["name"]: p["value"] for p in found}
    assert 0 < value["block_sizes"]["smallest"] <= value["block_sizes"]["largest"] <= value["cycles"]
    assert value["block_sizes"]["smallest"] <= value["healthy_cycles"] <= value["block_sizes"]["largest"]
    assert 0 <= value["healthy_left_out"] <= value["healthy_cycles"]
    assert value["drift"]["pairs"] > 0 and value["drift"]["median"] <= value["drift"]["largest"] and value["drift"]["reference"] > 0
    crossed = value["crossing"]
    assert crossed["next"] <= crossed["crossed"] <= crossed["blocks"] and crossed["test"] <= crossed["crossed"]
