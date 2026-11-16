from dataclasses import asdict, dataclass


DEFAULT_EXPECTATION_TOLERANCE = 0.05  # a mass or a recovery this far below the expectation is not a surprise
DEFAULT_PLAN_LENGTH = None  # no count: the budget, the horizon and the step cap bound a plan (plan.check_plan)
# the switches that change what the diagnostics server advises or that need its ranking, so a run without the server refuses them
SERVER_SWITCHES = ("branches", "single_hypothesis", "rule_only", "reads")
# the options that run.json records only where a run moves them from their default, so the run.json of a run that sets none keeps
# its bytes: the recall step and the summary turn of the compaction
OMITTED_AT_DEFAULT = {"recall": True, "summary_turn": True}


@dataclass
class CausalLoopOptions:
    """The switches of the agent. Without the diagnostics server the plan length, the expectations, backtracking and the tolerance
    apply, while the others change what the server advises and are refused (SERVER_SWITCHES)."""

    branches: bool = True
    single_hypothesis: bool = False
    plan_length: int | None = DEFAULT_PLAN_LENGTH
    expectations: bool = True
    rule_only: bool = False  # an internal test tool: the rule plans every turn, and no driver flag sets it
    reads: str = "gain"  # the read the advice names (Ag7): gain (the server's batch by information gain per unit cost), separating (E10), ec2
    backtracking: bool = True  # a refuted repair set is recorded, marked and the cheapest other plan advised; False keeps the verdict a surprise only (Ag8)
    expectation_tolerance: float = DEFAULT_EXPECTATION_TOLERANCE
    parallel: bool = True  # the schedule of the candidate plans on the board and the batch of consecutive read steps, with or without the server; False plays every read alone
    recall: bool = True  # the plan step that puts collapsed rows back on the next board
    summary_turn: bool = True  # the summary of the earlier turns at the compaction trigger; False sends no summary request

    def label(self):
        """The agent label of the driver's tables: causal, with one suffix per switch that differs from the default, so that
        the ablation rows separate."""
        parts = ["causal"]
        if not self.branches:
            parts.append("nobranch")
        if self.single_hypothesis:
            parts.append("single")
        if self.plan_length != DEFAULT_PLAN_LENGTH:
            parts.append(f"len{self.plan_length}")
        if not self.expectations:
            parts.append("noexp")
        if self.rule_only:
            parts.append("rule")
        if self.reads == "separating":
            parts.append("e10")
        elif self.reads == "ec2":
            parts.append("ec2")
        if not self.backtracking:
            parts.append("noback")
        if not self.parallel:
            parts.append("seq")
        if not self.recall:
            parts.append("norecall")
        if not self.summary_turn:
            parts.append("nosumturn")
        return "_".join(parts)

    def to_json(self):
        """The options in run.json, those of OMITTED_AT_DEFAULT only where a run moves them, so the run.json of a run that sets none
        keeps its bytes."""
        data = asdict(self)
        for name, default in OMITTED_AT_DEFAULT.items():
            if data[name] == default:
                del data[name]
        return data
