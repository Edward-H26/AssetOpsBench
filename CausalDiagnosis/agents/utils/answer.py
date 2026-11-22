"""The one shape in which an answer reaches the controller of an agent, whoever read it: {"result" or "refused", "status",
"notifications", "observations"} (CausalDiagnosis/docs/general-agent-plan.md, section Reading answers). The module imports the
standard library alone and is shared code of the agents and the bindings."""


def answer(result=None, refused=None, status=None, notifications=(), observations=()):
    """An answer in the one shape: {"refused": refused} where refused is a text, the reason of a refusal, else {"result": result},
    so that a result of None is still a result; then "status", the public status that the answer showed ({} by default),
    "notifications", the events it notified, and "observations", the rows that the form of its tool read from it, each a list."""
    head = {"refused": refused} if isinstance(refused, str) else {"result": result}
    return {**head, "status": dict(status or {}), "notifications": list(notifications), "observations": list(observations)}
