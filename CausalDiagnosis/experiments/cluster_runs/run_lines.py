from CausalDiagnosis.experiments.table_definitions import experiment_tables as design  # noqa: E402
from CausalDiagnosis.experiments.cluster_runs import pending_grid_lines as pending_lines  # noqa: E402
from CausalDiagnosis.experiments.table_definitions import training_variants as training  # noqa: E402


SERVERS = "--servers aob --store couchdb"  # every model run of the paper plays AssetOpsBench's servers with CouchDB
WITHOUT_MCP = " --no-diagnostics"
STARTS = len(training.FOLDS)  # the start combinations of the test set, every one in the registered split of the folds
REPLICATES = len(pending_lines.REPLICATES)  # the start states of every start that the test set plays, one run of a line each
# the arms of the supporting ablations that are switches of run_episodes/, as (arm, switch) under the id of the ablation.
# An ablation of design.ARM_PRESETS plays presets of its own and needs no switch here.
SWITCH_ARMS = {
    "Ag1": (("candidate troubleshooting plans off", "--causal-no-branches"),),
    "Ag2": (("candidate troubleshooting plans for the top candidate only", "--causal-single-hypothesis"),),
    "Ag3": (("plan length one against no count", "--causal-plan-length 1"),),
    "Ag7": (("the agent's own read, the discriminating test of E10 from the long form of the reply, against the advice of the MCP that "
             "the registered agent follows", "--causal-e10-reads"),
            ("the agent's own read, one step edge cutting over repair plan classes from the long form of the reply, against the advice "
             "of the MCP that the registered agent follows", "--causal-ec2-reads")),
    "Ag9": (("the schedule of candidate plans off, every read step alone with its own ranking, against the agent as released",
             "--causal-sequential"),),
    "En5": (("the fixed confidence cut against the decision rule of the MCP", "--abstention-rule fixed_cut"),),
    "En6": (("the price table that prices the high fidelity group far above the rest", "--prices full_tier_high"),),
    "En7": (("guidance level 1 against the level 2 of the main comparison", "--guidance 1"),
            ("guidance level 3 against the level 2 of the main comparison", "--guidance 3")),
    "En9": (("the joint ranking model against the joint block model of the main comparison", "--ranking-model joint"),
            ("the discriminant urn ranking model of the first version of the MCP against the joint block model",
             "--ranking-model discriminant_urn")),
}
# the ablations that play another agent than a row of the agents table, as (arm, agent), and those that play no line with their reason
AGENT_ARMS = {"Sc3": ("static answering, one model call over the question without the gym, judged", "static")}
NO_LINE = {"En4": "rescore_under_verifier_scope.py scores the runs of the six rows of the agents table again under the verifier scope all"}
LAYERS = {design.AGENTS_PRESET: "layer 0", design.PRICED_PRESET: "the priced layer", design.TRAINING_PRESET: "all six big layers"}
# the arm of a supporting ablation that stands beside the main table as its companion, by its switch, played in the part companion
COMPANION_SWITCHES = {"En9": f"--ranking-model {design.COMPANIONS['En9']}"}
# the start combinations of the healthy control that a replicate plays: the one combination with every component at its healthy
# level, whatever the asset
HEALTHY_STARTS = 1
WALL_CLOCK = "generation_time"  # the clock mode that charges the wall seconds between two tool calls as thinking time
REFERENCE_AGENTS = (*(label for label, _name in design.REFERENCE_POLICIES), "scripted")  # the policies of the environment table and the wiring check
# the parts that play on a server started with SERVE_ADAPTERS=1, beside the LoRA adapters of the training table (CausalDiagnosis/cluster_jobs/README.md); every
# other part plays on a server without adapters, and run.json records the adapters of the server (run_episodes.lora_adapters_of)
ADAPTER_PARTS = ("training",)


def run_line(preset, row, switch=""):
    """The run line of a row of design.AGENT_ROWS on a preset, with the switch of an arm when it plays one."""
    agent, with_mcp, _name = row
    return f"--config {preset} {SERVERS} --agent {agent}{'' if with_mcp else WITHOUT_MCP}{' ' + switch if switch else ''}"


def agents():
    """(comment, line) of the six rows of the main table, the first lines of the grid, whose runs the rescoring of En4 scores again."""
    count = len(design.AGENT_ROWS)
    return [(f"main table, row {i} of {count}, {row[2]}", run_line(design.AGENTS_PRESET, row)) for i, row in enumerate(design.AGENT_ROWS, 1)]


def healthy_control():
    """(comment, line) of the healthy control: the rows of the main table on the start with no faulty component, whose counts
    enter a table of their own and no family."""
    return [(f"healthy control, {row[2]}", run_line(design.HEALTHY_CONTROL_PRESET, row)) for row in design.HEALTHY_CONTROL_ROWS]


def main_part():
    """(comment, line) of the part main: the rows of the main table, then the healthy control."""
    return agents() + healthy_control()


def ladder_lines(presets):
    """(comment, line) of the rungs of the environment ladder on the given presets, each with the rows of design.LADDER_ROWS, the three
    agents with the MCP."""
    return [(f"cumulative layer study, layer {rung} of {len(design.LADDER)}, {name}, {row[2]}", run_line(preset, row))
            for rung, (preset, name) in enumerate(zip(design.LADDER, design.LADDER_NAMES), 1) if preset in presets
            for row in design.LADDER_ROWS]


def references():
    """(comment, line) of the three rows without the MCP on all six big layers (design.TRAINING_BASELINES), the baselines of the
    training table, the rows without the MCP of h1_top and h2_prime and the reference of the arms of design.ARM_PRESETS."""
    return [(f"reference without the {design.MCP} on all six big layers, {row[2]}", run_line(design.TRAINING_PRESET, row))
            for row in design.TRAINING_BASELINES]


def ladder():
    """(comment, line) of the rungs of the environment ladder between the first, whose runs are lines of the main table, and the
    last, whose runs are the part environment, then the reference lines without the MCP on all six big layers, in the place the top
    rung of the ladder held while the ladder played the rows without the MCP."""
    return layers() + references()


def layers():
    """(comment, line) of the rungs between layer 0 and all six big layers, the lines of the ladder that no table of the agents
    plays, since the agents table plays the first rung and its lines in the environment the last."""
    return ladder_lines([preset for preset in design.LADDER if preset not in (design.AGENTS_PRESET, design.TRAINING_PRESET)])


def token_twins():
    """{preset of the wall clock: its token twin} of design.TOKEN_PRESETS."""
    return {preset: twin for twin, preset in design.TOKEN_PRESETS.items()}


def layers_token():
    """(comment, line) of the part layers_token: every line of the part layers whose rung has a token twin, on the twin, so the layer
    study plays its rungs with a clock under the clock of generated tokens too."""
    twins = token_twins()
    return [(f"cumulative layer study under the clock of generated tokens, layer {rung} of {len(design.LADDER)}, {name}, {row[2]}",
             run_line(twins[preset], row))
            for rung, (preset, name) in enumerate(zip(design.LADDER, design.LADDER_NAMES), 1)
            if preset in twins and preset not in (design.AGENTS_PRESET, design.TRAINING_PRESET) for row in design.LADDER_ROWS]


def environment_token():
    """(comment, line) of the part environment_token: the six rows of the agents on the token twin of all six big layers, the rows of
    the tables of the agents in the environment under the clock of generated tokens."""
    twin = token_twins()[design.TRAINING_PRESET]
    return [(f"agents in the environment under the clock of generated tokens, {row[2]}", run_line(twin, row)) for row in design.AGENT_ROWS]


def ablation(name):
    """(comment, line) of every run of one supporting ablation: each arm on the preset and with the rows that
    design.ABLATION_PLACES gives the ablation, an arm of design.ARM_PRESETS on its own preset."""
    preset, rows = design.ABLATION_PLACES[name]
    place = LAYERS.get(preset, f"the preset {preset}")
    if name in NO_LINE:
        return []
    if name in AGENT_ARMS:
        arm, agent = AGENT_ARMS[name]
        return [(f"{name}, {arm}, {place}", f"--config {preset} {SERVERS} --agent {agent}")]
    arms = [(f"{arm}, {place}", preset, switch) for arm, switch in SWITCH_ARMS.get(name, ())]
    arms += [(f"{arm}, {place}", own, "") for own, (owner, arm) in design.ARM_PRESETS.items() if owner == name]
    if not arms:
        raise ValueError(f"the supporting ablation {name} of design.ABLATION_PLACES has no arm in SWITCH_ARMS, AGENT_ARMS, NO_LINE or design.ARM_PRESETS")
    return [(f"{name}, {arm}, {row[2]}", run_line(arm_preset, row, switch)) for arm, arm_preset, switch in arms for row in rows]


def is_companion(name, line):
    """Whether a line of an ablation is the arm that stands beside the main table, which the part companion plays."""
    switch = COMPANION_SWITCHES.get(name)
    return switch is not None and line.endswith(f" {switch}")


# the supporting ablations whose arms play as a part of their own at the end of grid_agents.txt, after every other line, so that
# the lines added to the grid leave the place of every earlier line unchanged
APPENDED_ABLATIONS = ("L1",)


def placed_lines():
    """(ablation, comment, line) of every supporting ablation in the order of design.ABLATION_PLACES, each preceded by the
    reference line of every row of its place whose cell no other line plays, once for the grid: the rule holds for any place, so
    an ablation moved to a preset that its rows play nowhere else brings the line its arms are compared with."""
    unplaced = sorted({*SWITCH_ARMS, *AGENT_ARMS, *NO_LINE, *(owner for owner, _arm in design.ARM_PRESETS.values())} - set(design.ABLATION_PLACES))
    if unplaced:
        raise ValueError(f"design.ABLATION_PLACES gives no place to {', '.join(unplaced)}")
    covered, out = design.played_cells(), []
    for name in design.ABLATION_PLACES:
        if name in APPENDED_ABLATIONS:
            continue
        entries = ablation(name)
        preset, rows = design.ABLATION_PLACES[name]
        if entries:
            place = LAYERS.get(preset, f"the preset {preset}")
            for row in rows:
                if (preset, row) not in covered:
                    covered.add((preset, row))
                    out.append((name, f"{name}, {design.REFERENCE_ARM}, {place}, {row[2]}", run_line(preset, row)))
        out += [(name, comment, line) for comment, line in entries]
    return out


def ablations():
    """(comment, line) of the part ablations: every supporting ablation with its reference lines, without the companion arm."""
    return [(comment, line) for name, comment, line in placed_lines() if not is_companion(name, line)]


def misdiagnosis_price():
    """(comment, line) of the part misdiagnosis_price: the arms of APPENDED_ABLATIONS, the ablation L1 with the misdiagnosis price
    at the budget on the place that design.ABLATION_PLACES gives it, whose rows the top rung of the ladder plays on that preset."""
    return [entry for name in APPENDED_ABLATIONS for entry in ablation(name)]


def environment():
    """(comment, line) of the part environment: the three rows with the MCP on all six big layers as released, the cells of the agents
    table on the full environment, of h1_top, h2_prime and full_environment, on the server of every other line of grid_agents.txt. The
    part training plays the same lines for the training table beside the adapters."""
    return [(f"agents in the environment, {row[2]}", run_line(design.TRAINING_PRESET, row)) for row in design.MCP_ROWS]


def companion():
    """(comment, line) of the part companion: the arm of an ablation that stands beside the main table, on the rows of its place."""
    return [(comment, line) for name, comment, line in placed_lines() if is_companion(name, line)]


def pilot():
    """(comment, line) of the optional pilot, which measures the thinking rate against the value the asset file declares: the three
    agents with the MCP on the full environment. No run of record reads what it measures."""
    return [(f"optional pilot of the thinking rate, {row[2]}", run_line(design.TRAINING_PRESET, row)) for row in design.TRAINING_ROWS]


def gate():
    """(comment, line) of the development gate: the CausalDiagnosis agent and the paper-era Plan-and-Execute, both without the MCP,
    on layer 0."""
    return [(f"development gate, {row[2]}", run_line(design.AGENTS_PRESET, row)) for row in (design.CD, design.PE)]


def sampling():
    """(comment, line) of the sampling runs of the fine tuning, one per agent, fold and sample of the registered count."""
    return [(f"training table, sampling, {row[2]}, fold {fold}, sample {sample} of {training.CONFIG['samples_per_state']}",
             run_line(design.TRAINING_PRESET, row, f"--training-fold {fold} --training-sample {sample}"))
            for row in design.TRAINING_ROWS for fold in training.held_out_folds.fold_ids()
            for sample in range(1, training.CONFIG["samples_per_state"] + 1)]


def second_round(agent, fold):
    """(comment, line) of the second round of sampling for one agent and fold, the samples that
    training.extension_samples(training.DEFAULT_ASSET) names after the registered first round, played only when the gate of the first round says
    second round due."""
    rows = [row for row in design.TRAINING_ROWS if row[0] == agent]
    if len(rows) != 1 or fold not in training.held_out_folds.fold_ids():
        raise SystemExit(f"second_round takes an agent of {', '.join(row[0] for row in design.TRAINING_ROWS)} and a fold of "
                         f"{', '.join(str(f) for f in training.held_out_folds.fold_ids())}, not {agent} and {fold}")
    return [(f"training table, second round of sampling, {rows[0][2]}, fold {fold}, sample {sample}",
             run_line(design.TRAINING_PRESET, rows[0], f"--training-fold {fold} --training-sample {sample}"))
            for sample in training.extension_samples(training.DEFAULT_ASSET)]


def evaluation():
    """(comment, line) of the evaluation runs of the training table: every agent without training in one line, and every trained variant
    in one line per fold, whose model plays the test episodes of the fold it holds out."""
    out = []
    for row in design.TRAINING_ROWS:
        out.append((f"training table, {row[2]}, {design.VARIANT_NAMES[training.AS_RELEASED]}",
                    run_line(design.TRAINING_PRESET, row, f"--training-variant {training.AS_RELEASED}")))
        out += [(f"training table, {row[2]}, {design.VARIANT_NAMES[variant]}, the model of fold {fold}",
                 run_line(design.TRAINING_PRESET, row, f"--training-variant {variant} --training-fold {fold}"))
                for variant in training.TRAINING_VARIANTS if variant != training.AS_RELEASED for fold in training.held_out_folds.fold_ids()]
    return out


PARTS = ("main", "companion", "ladder", "layers", "training", "ablations", "pilot", "gate", "sampling", "environment_token", "layers_token",
         "misdiagnosis_price", "environment")
_PART_LINES = {"main": main_part, "companion": companion, "ladder": ladder, "layers": layers, "training": evaluation, "ablations": ablations,
               "environment_token": environment_token, "layers_token": layers_token, "misdiagnosis_price": misdiagnosis_price,
               "environment": environment,
               "pilot": pilot, "gate": gate, "sampling": sampling}
# the parts whose lines play one run per replicate of the test set; the pilot and the gate play the development starts and a
# sampling line one run of its training starts
_REPLICATED = ("main", "companion", "ladder", "layers", "training", "ablations", "environment_token", "layers_token", "misdiagnosis_price",
               "environment")


def part(name):
    """(comment, line) of one part of the grid, in file order."""
    if name not in _PART_LINES:
        raise KeyError(name)
    return _PART_LINES[name]()


def replicates(name):
    """The runs that each line of a part plays: one per replicate of the test set, or one."""
    if name not in _PART_LINES:
        raise KeyError(name)
    return REPLICATES if name in _REPLICATED else 1


def row_grid(row_id):
    """(comment, line) of one development row: the row of design.ROWS_BY_ID on the preset of the main table."""
    row = design.ROWS_BY_ID[row_id]
    return [(f"development row {row_id}, {row[2]}", run_line(design.AGENTS_PRESET, row))]
