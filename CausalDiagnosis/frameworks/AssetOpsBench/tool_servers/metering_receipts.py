"""The receipt line that the metering layer of AssetOpsBench's servers appends to an answer, and the end of an episode as
an answer shows it, read by every agent that plays these servers (the surface of the CausalDiagnosis agent and the
Agent-As-Tool port).

From the priced rows on, the layer appends to every answer one line holding a JSON object with the steps left, the budget left
and, on the rows with a clock, the cycle and the notifications. An answer shows that the environment ended the episode in three
ways: the refusal every server answers after the end, an accepted finish, and a receipt with no step left or with the end event
among its notifications.
"""
from CausalDiagnosis.agents.utils.tool_answers import EPISODE_OVER, json_object


def split_receipt(text):
    """(body, receipt) of an answer: the receipt is the JSON object of the last line when it holds steps_left, and None for an
    answer without one, whose body is then the whole text."""
    parts = (text or "").rstrip("\n").rsplit("\n", 1)
    if len(parts) == 2:
        receipt = json_object(parts[1])
        if receipt is not None and "steps_left" in receipt:
            return parts[0], receipt
    return text, None


def ended(tool, body, receipt, is_error):
    """Whether an answer shows that the environment ended the episode: a receipt with no step left or with the end event among
    its notifications, the refusal after the end, or an accepted finish."""
    if receipt is not None and (receipt.get("steps_left") == 0
                                or any(isinstance(n, dict) and n.get("event") == "end" for n in receipt.get("notifications") or [])):
        return True
    reply = json_object(body)
    if reply is not None and set(reply) == {"error"}:
        return str(reply["error"]).startswith(EPISODE_OVER)
    return tool == "finish" and not is_error and reply is not None
