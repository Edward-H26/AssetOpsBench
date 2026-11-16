def port_counts(result, runner):
    """The counts of a run of the paper-era Agent-As-Tool: its steps are the supervisor's ReAct counter, the count the
    paper-era runner logged and the paper's Table 3 prints (CausalDiagnosis/frameworks/AssetOpsBench/agents/Agent_As_Tool_Paper/sub_agent_runtime.py and CausalDiagnosis/frameworks/AssetOpsBench/agents/Agent_As_Tool_Paper/supervisor_runner.py), and tool_calls every MCP call
    its sub-agents attempted, which the runner keeps also when the run raised. A run that returned no result, cut by the
    time limit or failed after its workflow, has the count its runner kept (steps_reached), which is None only when no
    supervisor was built."""
    steps = result.react_steps if result is not None else runner.steps_reached()
    return {"steps": steps, "tool_calls": len(runner.trajectory), "plan_steps": None, "react_steps": steps}


def pe_port_counts(result, runner):
    """The counts of a run of the paper-era Plan-Execute: its steps are the plan steps it executed, the step in which the run
    ended included and the summarization step not, which is how this reproduction reads the steps the paper prints for
    this design. plan_steps is the length of the plan that ran and tool_calls every MCP call its step agents attempted.
    A run that returned no result has the counts its runner kept, which are None only when no workflow was built
    (CausalDiagnosis/frameworks/AssetOpsBench/agents/Plan_Execute_Paper/runner.py)."""
    if result is None:
        return {"steps": runner.steps_reached(), "tool_calls": len(runner.trajectory),
                "plan_steps": (runner.record() or {}).get("plan_steps"), "react_steps": None}
    return {"steps": result.executed_steps, "tool_calls": len(runner.trajectory), "plan_steps": result.plan_steps, "react_steps": None}
