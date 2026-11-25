"""Interpret state comparisons using only the supplied expression and state."""

import ast


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
