"""The records of an episode of the general agent, which the driver, the viewer and the tables read
(CausalDiagnosis/docs/general-agent-plan.md, section The loop). A reader of these records ignores a key it does not know."""
from dataclasses import dataclass, field


@dataclass
class ToolRecord:
    """One tool call, in the shape that the audit report reads (report.py of the drivers). guard holds the class name of the
    exception where the last guard read the answer."""

    step: int
    server: str
    tool: str
    tool_args: dict
    response: str
    error: str = None
    duration_ms: float = None
    guard: str = None


@dataclass
class StepRecord:
    tool: str
    args: dict
    expect: dict
    # done, surprise, failed (a check that failed with the expectations off), skipped or standing (a step whose effect already
    # stood and that made no call, with the clause under observed["stands"])
    status: str
    observed: dict = field(default_factory=dict)
    note: str = ""


@dataclass
class PlanRecord:
    """One plan of an episode. A reader takes the steps of a plan from steps, and assumes neither that plan holds steps nor that
    they are not empty."""

    index: int
    source: str
    plan: dict
    candidates: list
    after_surprise: bool = False
    steps: list = field(default_factory=list)
    surprise: dict = None
    parse_errors: list = field(default_factory=list)
    check_problems: list = field(default_factory=list)
    schedule: dict = None  # the schedule that the board showed at the planning turn, None without one
    cut: dict = None  # {"step", "steps", "reasons"} of a plan cut at its first step that could not be used
    guard: str = None  # the class name of the exception where the last guard read the reply


@dataclass
class EpisodeResult:
    question: str
    answer: str
    plans: list
    trajectory: list
    model_calls: int
    surprises: list
    revisions: int
    fallbacks: int
    # test_first where the first step of the first plan that is no facts or knowledge read has the role observe, else the name of
    # that step's tool
    mode_of_first_plan: str
    differential: list
    board: dict
    end_reason: str
    error: str = None
    server: bool = True  # whether the diagnostics server was registered
    advice: tuple = ()  # the advice kinds of the modules that were active
    # why the runner stopped: finish (the write that ends the episode was accepted), episode_over (the environment ended the
    # episode, also at the step of a turn without a call), cancelled (the caller cut the run) or error (the run raised)
    end: str = None
