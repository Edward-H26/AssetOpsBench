"""The loop of the general agent (CausalDiagnosis/docs/general-agent-plan.md, section The loop). The controller calls a tool of
the asset only when a step names it; the calls it makes of its own are those of a module on an advice tool, which change nothing
on the asset and count no step. Stubs of the freeze; the loop builds them."""
from typing import Protocol


class BindingView(Protocol):
    """The read only view of the board and the ledger that a binding reads when it writes the arguments of a listed tool or reads
    its answer: the facts (None until the facts tool is answered), the status of the ledger, and paid(series), the units of a
    series that the episode has paid for."""

    facts: dict | None
    status: dict

    def paid(self, series):
        ...


class AgentRunner:
    """The general agent over the servers of its run: llm the model backend, server_paths {server name: launcher}, declaration
    the Declaration of the run, options the Options (their defaults where None). It builds the table of each episode from the
    listing of its sessions (Declaration.table, or Declaration.listed with the switch binding off)."""

    def __init__(self, llm, server_paths, declaration, options=None):
        raise NotImplementedError

    async def run(self, question):
        """One episode on the task message question: the EpisodeResult."""
        raise NotImplementedError


def request_settings(llm, temperature=None):
    """What a run folder records of the controller's requests on a backend: the blocks decoding and transport, at the
    controller's temperature unless a sampling run sends another."""
    raise NotImplementedError
