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
    # done, surprise, failed (a check that failed with the expectations off), standing (a step whose effect already stood on the
    # board and that made no call, with its clause under observed["stands"] and its reason as the note) or skipped, which nothing gives
    status: str
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
    # a plan cut at its first step that the controller cannot use: {"step": that step, "steps": the kinds of it and of every step
    # after it, which did not run, "reasons": why}; None for a plan that ran whole or not at all
    cut: dict = None


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
    # every model reply that only the last guard of the reader refused, {"model_call", "reader", "fault": the class of the exception},
    # a fault of the reader and never of the model (reply_parser.read_reply)
    reader_faults: list = field(default_factory=list)
    # every character of a summary of the model that UTF-8 cannot encode, replaced by U+FFFD where the summary enters the controller,
    # {"model_call", "summary": "closing" or "earlier turns", "characters": the count} (controller._kept_summary)
    replaced_characters: list = field(default_factory=list)
