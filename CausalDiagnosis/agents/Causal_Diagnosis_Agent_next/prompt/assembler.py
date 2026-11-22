"""The assembler of the prompt (CausalDiagnosis/docs/general-agent-plan.md, section The prompt). A block is core (a constant with
slots), rendered (built from declared fields) or given by a module; no sentence is edited after it is written, and a block or
sentence of a part that is switched off is left out. Stubs of the freeze; the prompt builds them."""


def system_prompt(table, board, options, modules):
    """The system prompt of a turn, which reads the facts and the limits of the ledger from the board."""
    raise NotImplementedError


def orientation_turn(board, table, options, modules, retry=None):
    """The request for the differential: the system prompt, the task, the facts and the knowledge answers as the board prints
    them, then the ask; retry the reason why the last reply was not usable, or None."""
    raise NotImplementedError


def planning_turn(board, table, options, modules, retry=None, not_run=None, stood=None):
    """The planning request: the system prompt, the task, the board and the objective; retry the reasons of the last reply or
    plan that ran nothing, or None; not_run the cut of the last plan, {"step", "steps", "reasons"}, or None; stood the standing
    steps of the last plan, each (number, tool, reason), or None."""
    raise NotImplementedError


def earlier_turns_turn(board, table, options, modules, left_out=0):
    """The request for the summary of earlier turns at the compaction; left_out the count of entries left out of it."""
    raise NotImplementedError


def summary_turn(board, table, options, modules, note=None):
    """The request for the closing summary: every write of the episode with what it returned, every verdict and the limits."""
    raise NotImplementedError


def identifier_forms(table, board, modules):
    """The forms of the identifiers that a step may cite, each (form, what it names): those of the answer forms of the table,
    of the ledger of the kind, of the modules and of the knowledge tools."""
    raise NotImplementedError
