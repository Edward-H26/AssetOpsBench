from dataclasses import dataclass, field


@dataclass
class ToolRecord:
    """One tool call, in the shape the audit report reads (report.py of the drivers)."""

    step: int
    server: str
    tool: str
    tool_args: dict
    response: str
    error: str = None
    duration_ms: float = None


@dataclass
class StepRecord:
    kind: str
    args: dict
    expect: dict
    status: str  # done, surprise, failed (a check that failed with the expectations off) or skipped
    observed: dict = field(default_factory=dict)
    note: str = ""


@dataclass
class PlanRecord:
    index: int
    source: str
    plan: dict
    candidates: list
    after_surprise: bool = False
    steps: list = field(default_factory=list)
    surprise: dict = None
    parse_errors: list = field(default_factory=list)
    check_problems: list = field(default_factory=list)
    schedule: dict = None  # the schedule that the board showed at the planning turn (candidate_plan_schedule.py), None with the switch off or without one


@dataclass
class CausalLoopResult:
    question: str
    answer: str
    plans: list
    trajectory: list
    model_calls: int
    surprises: list
    revisions: int
    fallbacks: int
    mode_of_first_plan: str
    differential: list
    board: dict
    end_reason: str
    error: str = None
    server: bool = True  # whether the diagnostics server was registered
    # why the runner stopped: finish (a finish step of a plan was accepted), episode_over (the environment ended the episode,
    # also at the no_call step of a turn without a call), cancelled (the caller cut the run) or error (the run raised). The
    # controller has no limit of its own.
    end: str = None
