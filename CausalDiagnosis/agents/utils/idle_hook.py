"""The idle hook of every design: at the end of each outer iteration of an agent (a planning
turn, a ReAct round, a supervisor step) the design calls iteration_end(acted), acted being whether the iteration made a counted
call of the environment. An iteration that made none takes one no_call step: a counted call of the episode that the thinking
clock charges at its entry, with no cycles handed over by the harness, that counts one step of the step cap, settles no visit
and logs the event no_call. The driver binds the episode's no_call entry around the runner (bound); outside an episode the hook
does nothing. The binding is module-wide, so a port that plays its steps in a thread reaches it, as CausalDiagnosis.utils.model_request_retry's deadline."""
import asyncio
from contextlib import contextmanager

_NO_CALL = None


@contextmanager
def bound(no_call):
    """Bind the episode's no_call entry, a zero-argument callable that sends one no_call step and returns True when the
    environment ended the episode with it."""
    global _NO_CALL
    before, _NO_CALL = _NO_CALL, no_call
    try:
        yield
    finally:
        _NO_CALL = before


def no_call():
    """One no_call step of the bound episode: True when the environment ended the episode with it, False outside an episode."""
    return bool(_NO_CALL()) if _NO_CALL is not None else False


def iteration_end(acted):
    """The end of an outer iteration: a no_call step where it made no counted call. Returns True when the episode ended."""
    return False if acted else no_call()


async def aiteration_end(acted):
    """iteration_end for an asynchronous design; the step runs in a thread so the event loop of the runner keeps serving."""
    return False if acted else await asyncio.to_thread(no_call)
