"""The rules of the check, one function per row of the table of the check of CausalDiagnosis/docs/general-agent-plan.md, in its
order, each rule(step, index, state) -> list[str] with state the WalkState of plan_check. A reason belongs to its step unless the
rule says it belongs to the whole plan. A rule whose key no row states, whose card the asset does not hold, or whose limit the
episode does not set gives no reason. Beside them, standing(step, index, state) says whether a step stands, which the check asks
for every step of its walk and the loop again before every call; a call with nothing new is one of its clauses, so no rule
function stands for it. Stubs of the freeze; the check builds them."""


def standing(step, index, state):
    """The Standing of a step whose effect already stands on the board, so that it makes no call and does not stop the plan, or
    None. A step stands under one of five clauses: episode_answer (a call of the facts tool or of a knowledge tool whose answer for
    the same arguments the board holds or an earlier step of the plan makes), surface_answer (a call of a tool that a binding
    needs first, the facts tool aside, answered, where no time passes), nothing_new and state_holds (as StepCost.stands of the
    walk states them), and same_commitment (a commit write equal to the commitment that stands). A step that known_tool,
    arguments, expectation or candidate_rank refuses does not stand, and a repair write, a wait and the write that ends the
    episode never stand, since no clause covers them."""
    raise NotImplementedError


def known_tool(step, index, state):
    """Reads the table; refuses a step that names a tool outside the table."""
    raise NotImplementedError


def arguments(step, index, state):
    """Reads the schema, the key values with the facts, the key minimum; refuses a step whose arguments its tool does not take."""
    raise NotImplementedError


def facts_first(step, index, state):
    """Reads the key facts, values and omitted where they need the facts, a need of the binding that is the facts tool, and the
    walk where the price or the time needs the facts; refuses a step, before the facts tool is answered, whose arguments or
    price need its answer, but for a knowledge read that stands after the facts tool in the same plan."""
    raise NotImplementedError


def order(step, index, state):
    """Reads the key after, the needs of a binding other than the facts tool, and an argument of a key each that a step leaves out;
    refuses a step without the earlier step that its tool needs."""
    raise NotImplementedError


def single_commitment(step, index, state):
    """Reads the key commits and the commitment on the board and earlier in the plan; refuses a commitment that differs from the
    one that stands."""
    raise NotImplementedError


def end_last(step, index, state):
    """Reads the key ends; refuses a step after the write that ends the episode."""
    raise NotImplementedError


def cites_observation(step, index, state):
    """Reads the key requires_observation; refuses a write that must cite an observation before any observation."""
    raise NotImplementedError


def candidate_rank(step, index, state):
    """Reads the candidates on the board and in the plan; refuses a follow or a candidate rank outside the candidates shown.
    Belongs to the whole plan for follow, the step for a rank."""
    raise NotImplementedError


def candidate_values(step, index, state):
    """Reads the key candidate; refuses a candidate with a value that the table or the facts tool does not list. Belongs to the
    whole plan."""
    raise NotImplementedError


def recall_rows(step, index, state):
    """Reads the collapsed rows of the board; refuses a recall of observations that the board did not collapse."""
    raise NotImplementedError


def expectation(step, index, state):
    """Reads the forms of the tool's answer and the modules; refuses an expectation that the tool's forms and the modules do not
    judge."""
    raise NotImplementedError


def open_repair(step, index, state):
    """Reads the write class repair with the key target, and the open targets of the ledger; refuses a repair write on a target
    whose repair is still open."""
    raise NotImplementedError


def precondition(step, index, state):
    """Reads the card precondition, the state writes with the key sets and the state at the start; refuses a call while the state
    has a value that its tool does not allow."""
    raise NotImplementedError


def end_guard(step, index, state):
    """Reads the keys ends and target, and Walk.end; refuses the write that ends the episode before the close of every repair and
    the observation that verifies the last one."""
    raise NotImplementedError


def budget(step, index, state):
    """Reads the prices of the walk and the budget left; refuses a plan that the budget left cannot pay. Belongs to the whole plan."""
    raise NotImplementedError


def time_limit(step, index, state):
    """Reads the times of the walk and the time limit; refuses a plan that reaches the time limit before the write that ends the
    episode can run. Belongs to the whole plan."""
    raise NotImplementedError


def step_limit(step, index, state):
    """Reads the calls of the walk and the steps left; refuses a plan that needs more counted calls than the steps left. Belongs to
    the whole plan."""
    raise NotImplementedError
