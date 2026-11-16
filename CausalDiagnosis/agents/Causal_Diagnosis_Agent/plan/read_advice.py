from CausalDiagnosis.agents.Causal_Diagnosis_Agent.candidate_plan_schedule import Needs, schedule
from . import repair_budget


def _ec2_read(board):
    """The read with the largest expected cut of the E8 repair plan classes per unit cost (one step edge cutting after the EC2 objective of Golovin, Krause and Ray, not their algorithm), or none when no
    reading can change the plan on its own, which the server reports as an ec2 of zero on every row."""
    rows = [e for e in board.elimination() if (e.get("ec2_per_cost") or 0) > 0]
    if not rows:
        return []
    best = max(rows, key=lambda e: e["ec2_per_cost"])
    return [(best["sensor_group"], int(best.get("cycles") or 1))]


DEFAULT_CHOICES = {"read": ("read_advice", "gain")}  # the option of each choice of the agent that has arms, and its default


def follows_advice(board, choice):
    """Whether a choice of the agent, the read or the repair plan, follows the advice of the server: the ranking carries advice
    and the option of the choice holds its default. A run that sets an arm keeps the choice of that arm. The repair plan has
    no arm, so it follows the advice whenever the ranking carries one."""
    if board.server_advice() is None:
        return False
    if choice not in DEFAULT_CHOICES:
        return True
    option, default = DEFAULT_CHOICES[choice]
    return getattr(board, option) == default


def ranking_detail(board):
    """The form of the ranking a run asks for: compact when every choice holds its default and so follows the advice, and the
    long form summary under an arm, whose lines read the earlier rule in e10_rule."""
    return "compact" if all(getattr(board, option) == default for option, default in DEFAULT_CHOICES.values()) else "summary"


def named_reads(board, reads="gain"):
    """(source, [(sensor group, cycles)]) of the reads that the ranking names while it says read first, after a
    commitment too, or None while it does not: the server's read batch under gain (E21 and E22), the one step edge cutting
    read under ec2, and the separating test of E10 under separating or whenever the policy proposes nothing, the source then
    "separating"."""
    if board.ranking is None or not board.read_first():
        return None
    source, candidates = {"gain": ("batch", board.read_batch()), "ec2": ("ec2", _ec2_read(board))}.get(reads, ("separating", []))
    if not candidates:
        test = board.separating_test()
        source, candidates = "separating", [(test["sensor_group"], int(test.get("cycles") or 1))] if test else []
    return source, candidates


def read_candidates(board, reads="gain"):
    """(source, [(sensor group, cycles)]) of the reads a policy proposes while nothing is committed and the ranking says
    read first, the reads that the ranking names (named_reads), and ("separating", []) otherwise."""
    named = None if board.committed else named_reads(board, reads)
    return named or ("separating", [])


def batch_reads(board, reads="gain"):
    """The reads the controller runs before a planning turn, the affordable prefix of read_candidates as (sensor group,
    cycles). Empty when nothing is affordable, which is when the rule declares insufficient evidence."""
    _, candidates = read_candidates(board, reads)
    reads = []
    for group, cycles in candidates:
        if not repair_budget.read_affordable(board, group, cycles):
            break
        reads.append((group, cycles))
    return reads


def advised_reads(board, reads="gain"):
    """The reads that the board advises without the schedule, as (sensor group, cycles), as the lines of advice print them: where
    the read follows the advice of the server, the read of that advice, or none when it advises no read; else batch_reads."""
    if follows_advice(board, "read"):
        advised = board.server_advice() or {}
        return [(advised["sensor_group"], int(advised.get("cycles") or 1))] if advised.get("action") == "read" else []
    return batch_reads(board, reads)


def server_reads(board, reads="gain"):
    """(source, [(sensor group, cycles)]) of the reads that the server names to the schedule, or None. Where the read follows the
    advice of the server, the advice names them: an advice to read names its read and then the other groups of the server's read
    batch, and any other advice names none. Otherwise they are the reads that the ranking names (named_reads)."""
    if not follows_advice(board, "read"):
        return named_reads(board, reads)
    advised = board.server_advice() or {}
    if advised.get("action") != "read":
        return None
    first = (advised["sensor_group"], int(advised.get("cycles") or 1))
    return "advice", [first] + [read for read in board.read_batch() if read[0] != first[0]]


def exploration(board, reads=None):
    """The schedule of the turn (candidate_plan_schedule.py), or None: with the switch off, before the profile, with the diagnostics server
    before a ranking, and after a commitment unless the last repair plan was refuted. Before a commitment the schedule packs
    reads while the server names reads (server_reads), and always without the server. After a refuted plan it packs reads only
    while no reading came since the refuting verdict and, with the server, the server names reads, and it lists what each
    remaining plan still needs. Every read leaves the reserve that the check of one read keeps: the completion of the repair of
    the first candidate (reserve_completion). The schedule differs from the board without it when
    its reads are not the reads the board advises, when a plan is left out or when it lists the remaining plans."""
    if not board.parallel or board.profile is None or (board.server and board.ranking is None):
        return None
    reads = board.read_advice if reads is None else reads
    refuted = bool(board.committed and board.refuted_last)
    if board.committed and not refuted:
        return None
    named = server_reads(board, reads) if board.server else None
    reads_open = named is not None if board.server else True
    if refuted:
        reads_open = reads_open and board.observations == board.refuted[-1]["observations"]
    tail = repair_budget.reserve_completion(board)
    reserve = Needs(tail.price, tail.calls, tail.cycles, tail.to_close)
    result = schedule(board, named[1] if named else [], reads_open, reserve, refuted)
    result.differs = refuted or bool(result.left_out) or [(r.group, r.cycles) for r in result.reads] != advised_reads(board, reads)
    return result


ARMS = {"batch": "the read batch by information gain per unit cost", "ec2": "the one step edge cutting read over repair plan classes",
        "separating": "the separating test of E10"}  # the advice of each arm of Ag7, by the source of read_candidates
