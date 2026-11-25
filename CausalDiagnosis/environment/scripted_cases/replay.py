"""Exact scripted replay independent of dataset names and agent choices."""

import ast
import copy
import json
import math

from jsonschema import Draft202012Validator


def condition(expression, state):
    """Interpret a state comparison without evaluating Python source."""
    def value(node):
        if isinstance(node, ast.Name):
            if node.id not in state:
                raise KeyError(node.id)
            return state[node.id]
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.List, ast.Tuple)):
            return [value(item) for item in node.elts]
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            items = [bool(value(item)) for item in node.values]
            return all(items) if isinstance(node.op, ast.And) else any(items)
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, right = value(node.left), value(node.comparators[0])
            op = node.ops[0]
            if isinstance(op, ast.Eq):
                return left == right
            if isinstance(op, ast.NotEq):
                return left != right
            if isinstance(op, ast.In):
                return left in right
            if isinstance(op, ast.NotIn):
                return left not in right
        raise ValueError("unsupported state condition")
    if not isinstance(expression, str):
        raise ValueError("state condition must be text")
    try:
        return bool(value(ast.parse(expression, mode="eval").body))
    except KeyError:
        return False
    except (SyntaxError, TypeError, RecursionError) as error:
        raise ValueError("invalid state condition") from error


def _price(value):
    try:
        valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("a call price must be a finite nonnegative number")
    return value


class ScriptedEpisode:
    """Private scripts serve only the observation of the call actually made.

    Refusals count a call and cost zero. Public precondition failures end the
    episode; private precondition failures refuse the call without ending it.
    Repeated state-setting calls refuse without an unsafe transition. Only
    caller-supplied call and price limits apply.
    """

    def __init__(self, state, tools, actions, *, max_calls=None, budget=None, transform_observation=None):
        self.state = copy.deepcopy(state)
        self.tools = {item["name"]: copy.deepcopy(item) for item in tools}
        if len(self.tools) != len(tools):
            raise ValueError("duplicate tool name")
        if max_calls is not None and (isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 1):
            raise ValueError("max_calls must be a positive integer")
        self.max_calls, self.budget = max_calls, None if budget is None else _price(budget)
        self.actions = copy.deepcopy(actions)
        self.transform_observation = transform_observation
        self.calls, self.spent, self.over = 0, 0, False
        self.end_reason = None
        self.observed = False
        self.submission = None
        self.trace = []
        ids = set()
        for tool in self.tools.values():
            Draft202012Validator.check_schema(tool["schema"])
            _price(tool.get("cost", 0))
            for expression in tool.get("requires", []):
                condition(expression, self.state)
        for item in self.actions:
            if item["id"] in ids or item["tool"] not in self.tools:
                raise ValueError("scripted action identifiers and tools must be valid")
            ids.add(item["id"])
            if not isinstance(item["args"], dict) or not isinstance(item.get("updates", {}), dict):
                raise ValueError("scripted arguments and state updates must be objects")
            _price(item.get("cost", self.tools[item["tool"]].get("cost", 0)))
            for expression in item.get("requires", []):
                condition(expression, self.state)

    def status(self):
        return {"over": self.over, "end_reason": self.end_reason, "state": copy.deepcopy(self.state),
                "calls": self.calls, "spent": self.spent,
                "steps_left": None if self.max_calls is None else max(0, self.max_calls - self.calls),
                "budget_left": None if self.budget is None else self.budget - self.spent}

    def _answer(self, *, result=None, refused=None, observations=()):
        if not self.over and self.max_calls is not None and self.calls >= self.max_calls:
            self.over, self.end_reason = True, "call_limit"
        return {**({"refused": refused} if refused else {"result": result}), "status": self.status(),
                "notifications": [], "observations": list(observations)}

    def call(self, name, args):
        if self.over:
            return self._answer(refused="the episode is over")
        self.calls += 1
        trace = {"tool": name, "args": copy.deepcopy(args), "accepted": False, "action": None, "cost": 0}
        self.trace.append(trace)
        tool = self.tools.get(name)
        if tool is None:
            return self._answer(refused="unknown tool")
        errors = list(Draft202012Validator(tool["schema"]).iter_errors(args))
        if errors:
            return self._answer(refused="arguments do not satisfy the public tool schema")
        updates = tool.get("updates", {})
        if updates and all(self.state.get(key) == value for key, value in updates.items()):
            return self._answer(refused="the machine is already in the requested state")
        for expression in tool.get("requires", []):
            if not condition(expression, self.state):
                self.over, self.end_reason = True, "public_precondition"
                return self._answer(refused=f"the public tool requires {expression}")
        if tool.get("ends"):
            if not self.observed:
                return self._answer(refused="read an observation before submitting")
            self.submission = copy.deepcopy(args)
            self.over, self.end_reason = True, "submitted"
            trace["accepted"] = True
            return self._answer(result={"accepted": True})
        if "static_result" in tool:
            trace["accepted"] = True
            return self._answer(result=copy.deepcopy(tool["static_result"]))
        written = json.dumps(args, sort_keys=True, ensure_ascii=False, allow_nan=False)
        matched = [item for item in self.actions if item["tool"] == name and
                   json.dumps(item["args"], sort_keys=True, ensure_ascii=False, allow_nan=False) == written]
        if len(matched) != 1 and not (not matched and updates):
            return self._answer(refused="no unique scripted observation matches these exact arguments")
        action = matched[0] if matched else None
        for expression in action.get("requires", []) if action else ():
            if not condition(expression, self.state):
                return self._answer(refused=f"the call requires {expression}")
        price = action.get("cost", tool.get("cost", 0)) if action else tool.get("cost", 0)
        if self.budget is not None and self.spent + price > self.budget:
            self.over, self.end_reason = True, "budget"
            return self._answer(refused="the budget left cannot pay this call")
        observation = copy.deepcopy(action["observation"]) if action else {"state": {**self.state, **updates}}
        text = observation if isinstance(observation, str) else json.dumps(observation, ensure_ascii=False, allow_nan=False)
        try:
            materialized = self.transform_observation(observation) if self.transform_observation else observation
        except (OSError, ValueError):
            return self._answer(refused="the observation attachment could not be read")
        self.spent += price
        self.state.update(copy.deepcopy(updates))
        if action:
            self.state.update(copy.deepcopy(action.get("updates", {})))
        trace.update(accepted=True, action=action["id"] if action else None, cost=price)
        self.observed = True
        rows = [{"id": f"obs/{self.calls}", "n": self.calls, "text": text}]
        return self._answer(result=materialized, observations=rows)
