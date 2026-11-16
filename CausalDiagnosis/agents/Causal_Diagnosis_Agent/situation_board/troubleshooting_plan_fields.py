def repair_steps(branch):
    """The work orders of a branch's repair plan. The server returns the plan as an object with steps, its price, its downtime
    cost and the cycles to the restored state; a plain list of steps is read too."""
    plan = branch.get("repair_plan") or {}
    return list(plan.get("steps") or []) if isinstance(plan, dict) else list(plan)


def cycles_to_restored(branch):
    plan = branch.get("repair_plan") or {}
    return branch.get("cycles_to_restored", plan.get("cycles_to_restored") if isinstance(plan, dict) else None)


def plan_prediction_id(branch):
    """The identifier of the prediction after every replacement of a branch's repair plan, or None when the server sends none."""
    plan = branch.get("repair_plan") or {}
    return plan.get("plan_prediction_id") if isinstance(plan, dict) else None


def branch_prediction_id(branch, component):
    """The prediction identifier a work order on the component cites along a branch. The choice lives here alone. The rule of
    the server side: every order of a repair plan cites the plan's prediction, the state after all of its
    replacements, which one reading after the last order closed verifies; the steps' own predictions describe one
    replacement executed alone and stay in the branch uncited. Without a plan identifier the step's own stands in."""
    plan_id = plan_prediction_id(branch)
    if plan_id:
        return plan_id
    for step in repair_steps(branch):
        if step.get("component") == component and step.get("prediction_id"):
            return step["prediction_id"]
    return None


def two_step_test(branch):
    """The two step pair of a branch, reported beside the confirm test when a pair separates better than any single group."""
    two = branch.get("two_step")
    if two is None:
        two = (branch.get("confirm_test") or {}).get("two_step")
    return two or None
