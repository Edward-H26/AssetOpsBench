"""How every agent of the fork reads the answer of a tool: the JSON object an answer holds, and the start of the refusal
every server answers after the end of an episode. The module imports nothing of the fork, so the controller of the
CausalDiagnosis agent and the agents of a framework read the same answers by the same rules.
"""
import copy
import json
import math

EPISODE_OVER = "the episode is over"  # the start of the refusal every server answers after the end (CausalDiagnosis/mcp_servers/Environment_Actions_MCP/episode_state.py)


def json_object(text):
    """The JSON object a text holds, or None, also for a text that nests deeper than the decoder follows."""
    try:
        value = json.loads(text) if text else None
    except (TypeError, ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def split_receipt(text):
    """Return the body and a final-line receipt carrying steps_left, if present."""
    parts = (text or "").rstrip("\n").rsplit("\n", 1)
    if len(parts) == 2:
        receipt = json_object(parts[1])
        if receipt is not None and "steps_left" in receipt:
            return parts[0], receipt
    return text, None


def _pairs(items):
    value = {}
    for key, item in items:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _nonfinite(value):
    raise ValueError(f"nonfinite JSON value {value}")


def _float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JSON number exceeds the finite range")
    return number


def read(text, is_error, marks=()):
    """Normalize an envelope, a metered text answer or a refusal without raising.

    Legacy json_object and split_receipt keep their existing wire behavior. The
    new boundary returns named malformed-answer refusals instead of an exception.
    """
    try:
        if not isinstance(text, str):
            return {"refused": "tool answer must be text", "status": {}, "notifications": [], "observations": []}
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:
            return {"refused": "tool answer must contain complete UTF-8 characters",
                    "status": {}, "notifications": [], "observations": []}
        body, receipt = split_receipt(text)
        if receipt is not None:
            receipt = json.loads(text.rstrip("\n").rsplit("\n", 1)[1], object_pairs_hook=_pairs,
                                 parse_constant=_nonfinite, parse_float=_float)
        try:
            value = json.loads(body, object_pairs_hook=_pairs, parse_constant=_nonfinite, parse_float=_float)
        except json.JSONDecodeError:
            value = body
        except ValueError as error:
            return {"refused": f"malformed tool answer: {error}", "status": {}, "notifications": [], "observations": []}
        if not isinstance(value, dict):
            value = body
        envelope = isinstance(value, dict) and ("result" in value or "refused" in value)
        status = {}
        notifications, observations = [], []
        if envelope:
            nested = value.get("status", {})
            if not isinstance(nested, dict):
                raise ValueError("status must be an object")
            status.update(nested)
            status.update({key: item for key, item in value.items()
                           if key not in {"result", "refused", "status", "notifications", "observations"}})
            notifications = value.get("notifications", [])
            observations = value.get("observations", [])
            if not isinstance(notifications, list) or not all(isinstance(item, dict) for item in notifications):
                raise ValueError("notifications must be a list of objects")
            if not isinstance(observations, list) or not all(isinstance(item, dict) for item in observations):
                raise ValueError("observations must be a list of objects")
        if receipt is not None:
            events = receipt.get("notifications", [])
            if not isinstance(events, list) or not all(isinstance(item, dict) for item in events):
                raise ValueError("receipt notifications must be a list of objects")
            notifications = [*notifications, *events]
            status.update({key: item for key, item in receipt.items() if key not in {"notifications", "cycle"}})
            if "cycle" in receipt:
                status["time"] = receipt["cycle"]
        payload = {"result": value["result"] if envelope and "result" in value else value}
        if envelope and "refused" in value:
            if "result" in value or not isinstance(value["refused"], str) or not value["refused"].strip():
                raise ValueError("an envelope needs one result or nonempty refusal")
            payload = {"refused": value["refused"]}
        elif isinstance(value, dict) and set(value) == {"error"}:
            payload = {"refused": str(value["error"])}
        elif is_error or not envelope and (body.startswith(EPISODE_OVER)
                or any(mark in body for mark in marks if isinstance(mark, str) and mark)):
            payload = {"refused": body or "tool call returned an error"}
        return copy.deepcopy({**payload, "status": status, "notifications": notifications, "observations": observations})
    except (ValueError, TypeError, RecursionError, OverflowError) as error:
        return {"refused": f"malformed tool answer ({type(error).__name__}): {error}",
                "status": {}, "notifications": [], "observations": []}
    except Exception as error:
        return {"refused": "the answer could not be read", "status": {}, "notifications": [],
                "observations": [], "guard": type(error).__name__}
