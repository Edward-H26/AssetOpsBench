"""The switches of the general agent (CausalDiagnosis/docs/general-agent-plan.md, section The prompt, paragraph Switches): Options with one field per
switch, and the registry SWITCHES, one record per switch with its name, its default, its label suffix, the flags of the driver
that set it and its owner, core or the advice kind of the module that owns it. A switch that a module owns is refused by name on
a run whose table lists no tool of that module's advice kind."""
from dataclasses import asdict, dataclass

AGENT_NAME = "causal_next"  # the name under which the driver knows the agent, the first word of its label
DEFAULT_EXPECTATION_TOLERANCE = 0.05  # a mass or a recovery this far below the expectation is not a surprise
DEFAULT_PLAN_LENGTH = None  # no count: the limits of the episode bound a plan
# the options that run.json records only where a run moves them from their default, so the run.json of a run that sets none keeps
# the keys of the present agent's
OMITTED_AT_DEFAULT = {"recall": True, "summary_turn": True, "structured_output": True, "binding": True}


@dataclass(frozen=True)
class Switch:
    """One switch: its name, its default, its label suffix where the label shows it, the flags of the driver that set it, and
    its owner. suffix is a text, with {} for the value where the value is a count, or a tuple of (value, suffix) pairs for a
    switch of several values; None where the label never shows the switch."""

    name: str
    default: object
    suffix: object
    flags: tuple
    owner: str


SWITCHES = (
    Switch("branches", True, "nobranch", ("--causal-no-branches",), "ranking"),
    Switch("single_hypothesis", False, "single", ("--causal-single-hypothesis",), "ranking"),
    Switch("plan_length", DEFAULT_PLAN_LENGTH, "len{}", ("--causal-plan-length",), "core"),
    Switch("expectations", True, "noexp", ("--causal-no-expectations",), "core"),
    Switch("reads", "gain", (("separating", "e10"), ("ec2", "ec2")), ("--causal-e10-reads", "--causal-ec2-reads"), "ranking"),
    Switch("backtracking", True, "noback", ("--causal-no-backtrack",), "core"),
    Switch("expectation_tolerance", DEFAULT_EXPECTATION_TOLERANCE, None, ("--causal-expectation-tolerance",), "core"),
    Switch("parallel", True, "seq", ("--causal-sequential",), "core"),
    Switch("recall", True, "norecall", (), "core"),
    Switch("summary_turn", True, "nosumturn", (), "core"),
    Switch("structured_output", True, "noschema", ("--causal-no-structured-output",), "core"),
    Switch("binding", True, "raw", ("--causal-raw-calls",), "core"),
)


@dataclass
class Options:
    """The switches of the agent, one field per record of SWITCHES and in its order."""

    branches: bool = True  # the section of the candidate plans of the ranking
    single_hypothesis: bool = False  # the first candidate plan alone
    plan_length: int | None = DEFAULT_PLAN_LENGTH  # the count of steps that a plan may hold; None: the limits of the episode bound it
    expectations: bool = True  # a failed expectation stops the plan; False records it and runs the plan on
    reads: str = "gain"  # the reads the advice names: gain, separating or ec2
    backtracking: bool = True  # a refuted repair is recorded as a set; False keeps it a surprise only
    expectation_tolerance: float = DEFAULT_EXPECTATION_TOLERANCE
    parallel: bool = True  # consecutive observe steps run as one batch, and the board shows the schedule
    recall: bool = True  # the own step recall and its block of the prompt
    summary_turn: bool = True  # the summary request at the compaction
    structured_output: bool = True  # the planning and the orientation requests carry the schema of their reply
    binding: bool = True  # the binding of the surface writes the arguments; False plans on the listed tools

    def label(self):
        """The agent label of the driver's tables: the agent's name, with one suffix per switch that differs from its default, in
        the order of SWITCHES, so that the ablation rows separate."""
        parts = [AGENT_NAME]
        for switch in SWITCHES:
            value = getattr(self, switch.name)
            if value == switch.default or switch.suffix is None:
                continue
            if isinstance(switch.suffix, tuple):
                parts += [suffix for named, suffix in switch.suffix if named == value]
            else:
                parts.append(switch.suffix.format(value))
        return "_".join(parts)

    def to_json(self):
        """The options in run.json, those of OMITTED_AT_DEFAULT only where a run moves them."""
        data = asdict(self)
        for name, default in OMITTED_AT_DEFAULT.items():
            if data[name] == default:
                del data[name]
        return data
