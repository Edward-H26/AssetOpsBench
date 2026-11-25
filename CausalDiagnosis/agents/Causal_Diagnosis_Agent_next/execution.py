"""Core expectation judgments over new answers or retained observations."""

import copy
import json


def held_answer(board, step, row):
    """Use the latest retained observation of the same tool and series."""
    for call in reversed(board.calls):
        same = (call["args"].get(row.series) == step.args.get(row.series) if row.series
                else call["args"] == step.args)
        if call["tool"] != step.tool or not call["answered"] or not same:
            continue
        rows = call["answer"].get("observations", [])
        if rows:
            latest = max(item["n"] for item in rows)
            return {"observations": copy.deepcopy([item for item in rows if item["n"] == latest]), "notifications": []}
    return {"observations": [], "notifications": []}


def expectation_failures(step, answer, *, standing=False):
    failures = [answer["refused"]] if "refused" in answer else []
    rows = answer.get("observations", [])
    for key, expected in step.expect.items():
        if key == "no_event" and not standing:
            if any(item.get("event", item.get("kind")) in expected for item in answer.get("notifications", [])):
                failures.append("an excluded event occurred")
        elif key == "word" and not any(expected in item.get("words", []) for item in rows):
            failures.append("the expected word was absent")
        elif key == "outside":
            for field in expected:
                served = [item for item in rows if item.get("field") == field or
                          isinstance(item.get("field"), str) and item["field"].startswith(field + "_")]
                if any(item.get("inside_band") is True for item in served):
                    failures.append(f"{field} was served inside its normal range")
    return failures


def filled_arguments(step, row, board, summary=None):
    """Fill controller-owned arguments without changing the checked step."""
    args = copy.deepcopy(step.args)
    for name, source in row.fills.items():
        if source == "evidence":
            known = board.known_ids()
            args[name] = list(dict.fromkeys(identifier for identifier in step.evidence if identifier in known))
            if row.requires_observation:
                args[name] = list(dict.fromkeys([*args[name], *(item["id"] for item in board.latest_observations())]))
        elif source == "closing_summary":
            args[name] = summary
        elif source.startswith("fixed:"):
            value = source[len("fixed:"):]
            try:
                args[name] = json.loads(value)
            except ValueError:
                args[name] = value
        else:
            raise NotImplementedError(f"no provider for the {source} fill of {row.name}.{name}")
    return args
