"""A standing step: a step of a plan whose effect already stands on the board, so that a call for it could add nothing, since the
board holds its answer and the answer cannot have changed, or the board holds the commitment that the step would make. Such a step
makes no call and does not stop the plan, and its status is standing. The five clauses stand here alone, and both the plan check,
with the state of its walk, and the controller, with the record of the plan, ask them, so that the walk counts no call that the run
will not make:
- episode_answer: a read of the opening whose answer holds for the whole episode, for requests that the board holds answered;
- surface_answer: a read of the opening that the surface needs before a reading, answered, where no cycle passes in the episode;
- nothing_new: a read of a sensor group whose every recorded cycle the board holds (Board.new_cycles), where the record cannot have
  grown since the latest answer: no cycle passes, or cycles pass only inside calls that take time (Board.record_holds_between_calls);
- same_commitment: a diagnosis or a declaration equal to the commitment that stands;
- state_holds: a write of a public state whose value the state already has, for a kind that the board declares a state write
  (Board.state_writes), which no surface of this agent declares.
No other kind stands: a work order is a new act each time, a wait lets time pass, finish runs once and a recall lasts one request by
its own rule. A step that contradicts what stands, another commitment or an order on a component whose order is open, is refused by
the plan check, not stood. Every reason opens with the change that the model can make, then names what the board already holds."""
from dataclasses import dataclass, field

from CausalDiagnosis.agents.utils import core_texts
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.situation_board.board import KIND_WORDS, call_text, commitment_text
from . import plan_schema


EPISODE_ANSWER, SURFACE_ANSWER, NOTHING_NEW, SAME_COMMITMENT = "episode_answer", "surface_answer", "nothing_new", "same_commitment"
STATE_HOLDS = "state_holds"
RECALL_STEP = "recall"  # the step that brings collapsed rows back on the board
STANDS_STATE = "leave this step out, since {state} is already {value}"  # word for word from the plan page, which keeps it in the plan package


@dataclass(frozen=True)
class Standing:
    clause: str  # one of the five clauses
    reason: str  # the note of the step's record, which the next planning request names


@dataclass
class Walked:
    """What the walk of the plan check has done before the step that it asks about: the requests of the opening that earlier steps
    of the plan answer, the commitment that an earlier step of the plan makes (commitment_of), the candidates that a diagnosis
    names by rank, the cycles that the walk's read of this step pays for anew (ReadWalk.new) and the cycle that the walk has reached
    before it, and the public states that earlier state writes of the plan set."""
    answered: list = field(default_factory=list)
    commitment: tuple = None
    candidates: list = field(default_factory=list)
    new: int = None
    time: int = None
    states: dict = field(default_factory=dict)


def requests_of(step, board):
    """The keyword fields that the controller hands to the calls of a read of the opening: one request per component of the profile
    for a failure_modes step with an empty component, else one; None where no profile names the components yet."""
    if step.kind != "failure_modes":
        return [{}]
    if step.args.get("component"):
        return [{"component": step.args["component"]}]
    return None if board.profile is None else [{"component": c} for c in board.profile["components"]]


def submitted_levels(board, want):
    """The levels that a diagnosis of the planned candidate want submits: with the diagnostics server and a ranked candidate the
    current combination of the board's candidate with the same combination, else the candidate's own, since a diagnosis states
    the condition at the cycle of the write; for a candidate of the model's own, its combination."""
    if board.server and not want.get("own"):
        current = next((c for c in board.candidates() if c["combination"] == want["combination"]), None)
        return (current or {}).get("current_combination") or want.get("current_combination") or want["combination"]
    return want["combination"]


def commitment_of(step, board, want=None):
    """The commitment that a commit step would make, (kind, value): the levels of a diagnosis of want, the sensor group and the
    cycles of a declaration; None for a diagnosis whose rank names no candidate, which nothing can equal."""
    if step.kind == "declare":
        return "declare", (step.args["sensor_group"], step.args["cycles"])
    return None if want is None else ("diagnose", dict(submitted_levels(board, want)))


def standing_commitment(board, walked=None):
    """The commitment that stands, as commitment_of gives it: the board's, else the one that an earlier step of the plan makes."""
    if board.diagnosis is not None:
        return "diagnose", dict(board.diagnosis["diagnosis"])
    if board.declaration is not None:
        test = board.declaration["separating_test"]
        return "declare", (test["sensor_group"], test["cycles"])
    return None if walked is None else walked.commitment


def same_commitment(made, stands, board):
    """Whether two commitments are one answer to the environment: one kind, and the same level of every component of the profile
    or the same sensor group and cycles."""
    if made is None or stands is None or made[0] != stands[0]:
        return False
    if made[0] == "declare":
        return made[1] == stands[1]
    return all(made[1].get(c) == stands[1].get(c) for c in board.profile["components"])


def planned_candidate(step, candidates):
    """The candidate that a diagnosis names by its rank among candidates, or None."""
    rank = step.args.get("rank")
    return candidates[rank - 1] if rank is not None and 1 <= rank <= len(candidates) else None


def standing(step, board, record=None, walked=None):
    """The clause by which the step stands on the board as it is when the step is reached, with its reason, or None. The plan check
    asks with walked, the state of its walk, and the controller with record, the record of the plan, whose candidates a diagnosis
    names by rank."""
    if step.kind in board.state_writes:
        return _state_holds(step, board, walked)
    if step.kind in plan_schema.OPENING_KINDS:
        return _answered(step, board, walked)
    if step.kind == "read":
        return _nothing_new(step, board, walked)
    if step.kind in plan_schema.COMMIT_KINDS:
        candidates = walked.candidates if walked is not None else record.candidates if record is not None else []
        stands = standing_commitment(board, walked)
        if same_commitment(commitment_of(step, board, planned_candidate(step, candidates)), stands, board):
            return Standing(SAME_COMMITMENT, core_texts.STANDS_COMMITMENT.format(commitment=commitment_text(*stands)))
    return None


def _answered(step, board, walked):
    """Clauses episode_answer and surface_answer: every request of the step answered, by the board or by an earlier step of the plan,
    and for a read that a surface needs first, no cycle passing that could move its answer."""
    requests = requests_of(step, board)
    earlier = walked.answered if walked is not None else []
    if requests is None or not all(board.holds_answer(step.kind, r) or [step.kind, r] in earlier for r in requests):
        return None
    call = call_text(step.kind, step.args)
    if step.kind in plan_schema.EPISODE_KINDS:
        return Standing(EPISODE_ANSWER, core_texts.STANDS_EPISODE.format(call=call))
    if board.clock_runs:
        return None
    return Standing(SURFACE_ANSWER, core_texts.STANDS_SURFACE.format(call=call))


def _nothing_new(step, board, walked):
    """Clause nothing_new: the read would pay for no cycle anew, in the walk after the earlier reads of the plan or on the board, and
    the record cannot have grown since the latest answer, since no cycle passes in the episode or cycles pass only inside calls that
    take time (Board.record_holds_between_calls) and, in the walk, no earlier step of the plan took time where a clock runs, since
    such a step records cycles that the board does not hold yet. The reason names the recall step where rows of the group stand
    collapsed and the run offers the step, since a model may restate the read to see them again."""
    group, cycles = step.args["sensor_group"], step.args["cycles"]
    new = walked.new if walked is not None and walked.new is not None else board.new_cycles(group, cycles)
    moved = board.clock_runs and walked is not None and walked.time is not None and walked.time > board.time
    if new or moved or not board.record_holds_between_calls:
        return None
    collapsed = sorted(r["n"] for r in board.collapsed_rows if r["sensor_group"] == group)
    back = (core_texts.STANDS_COLLAPSED.format(first=collapsed[0], last=collapsed[-1], recall_step=RECALL_STEP)
            if collapsed and board.recall_offered else "")
    if board.clock_runs:
        return Standing(NOTHING_NEW, f"{core_texts.STANDS_NOTHING_RECORDED.format(**KIND_WORDS, name=group)}{back}")
    return Standing(NOTHING_NEW, f"{core_texts.STANDS_NOTHING_NEW.format(**KIND_WORDS, name=group)}{back}")

def _state_holds(step, board, walked):
    """Clause state_holds: the write sets its state to the value that the state already has, after the earlier writes of the plan or
    on the board. While the state is unknown, the write does not stand."""
    state, value = board.state_writes[step.kind]
    states = {**board.states, **(walked.states if walked is not None else {})}
    if state not in states or states[state] != value:
        return None
    return Standing(STATE_HOLDS, STANDS_STATE.format(state=state, value=value))
