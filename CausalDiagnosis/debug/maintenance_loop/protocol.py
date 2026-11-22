"""Debug planner adaptation. No environment transitions or verdicts are replaced."""
from dataclasses import replace
import json


def progress(board):
    """Only public situation-board fields, never hidden state or oracle output."""
    orders = [{key: value for key, value in order.items()
               if key in ("id", "component", "created_t", "expected_close_t", "closed_t", "status")}
              for order in board.orders.values()]
    return {
        "cycle": board.time,
        "diagnosis_already_submitted": board.diagnosis is not None,
        "declaration_already_submitted": board.declaration is not None,
        "work_orders": orders,
        "wait_available": "wait" not in board.unserved and board.clock_runs,
        "forward_read_cycles_per_cycle": board.forward_read if board.profile is not None else None,
        "record_grows_with_thinking": board.record_grows,
        "recorded_until": board.status.get("recorded_until"),
        "unread_recorded_cycles": {g: len(rows) if rows is not None else None
                                   for g in (board.profile or {}).get("sensor_groups", {})
                                   for rows in [board.unpaid_cycles(g)]},
        "latest_read": getattr(board, "_debug_last_read", None),
    }


def instructions(board):
    state = progress(board)
    return (
        "\n\n## Debug execution contract (maintenance-loop-v4)\n"
        "Propose a short plan. The next action is checked and executed on the real environment. "
        "Unexecuted steps remain pending; they are checked again before execution. "
        "New observations, time changes or a surprise require replanning.\n"
        + json.dumps(state, ensure_ascii=False) + "\n"
        "An acknowledged work order is not a completed repair. Never repeat a submitted diagnosis/declaration, "
        "or order a component with an open order. Use its expected close and the current cycle. "
        "If wait is available, propose wait to advance toward completion. If wait is unavailable but forward "
        "reads advance time, select a sensor group that verifies the repair and a window that reaches its close. "
        "A history request beyond recorded_until can be empty; do not repeat an empty window expecting time to advance. "
        "If neither path exists and thinking does not advance time, report that limitation in why rather than inventing a wait. "
        "After completion, read actual post-repair measurements before finish whenever the configuration supports it. "
        "Do not diagnose again just to verify. A closed order alone does not establish recovery. "
        "Diagnosis covers all components, not only the component named in the question. "
        "If healthy, do not create an unnecessary order.\n"
        "With diagnostics: use its ranking, uncertainty, suggested tests and predicted repair effects as advice. "
        "Do not treat a high-ranked candidate as observed truth. On an order, prediction_id=estimate requests "
        "the controller's intervention estimate; use actual evidence IDs from the board. "
        "Without diagnostics: reason from the same observed readings and public failure modes. "
        "The same repair and evidence standards apply to both modes.\n"
    )


def unavailable_read(step, board):
    if step.kind != "read" or board.profile is None:
        return None
    group = step.args.get("sensor_group")
    if group not in board.profile["sensor_groups"]:
        return None  # let the original checker explain unknown groups
    if (board.unpaid_cycles(group) == [] and not board.reads_take_time
            and not board.forward_read and not board.record_grows):
        return (f"No unread recorded cycles exist for {group}. This history call cannot acquire a new sample or advance "
                "the declared clock. Use evidence already on the board for your decision, or explicitly wait for new data "
                "if wait is served. Do not repeat this empty read.")
    return None


def last_read(trajectory):
    for call in reversed(trajectory):
        if call.tool not in ("history", "read_sensors"):
            continue
        try:
            data, _ = json.JSONDecoder().raw_decode(call.response.lstrip())
            rows = data.get("observations", data.get("served", data.get("rows")))
            return {"tool": call.tool, "returned_rows": len(rows) if isinstance(rows, list) else data.get("returned"),
                    "start": data.get("start"), "end": data.get("end")}
        except (ValueError, TypeError, AttributeError):
            return {"tool": call.tool, "returned_rows": None}
    return None


def first_step(plan):
    """Execute the first model-chosen action, never substitute a controller-chosen action."""
    return replace(plan, steps=plan.steps[:1])


def install(prompts):
    original_prompt = prompts.planning_turn

    def planning_turn(board, plan_length, retry=None, surface=None):
        if board.profile is None or any(k not in board.opening_done and k not in board.unserved
                                       for k in ("failure_modes", "fix_history", *board.required_opening)):
            return original_prompt(board, plan_length, retry=retry, surface=surface) + instructions(board)
        # Avoid the old multi-step objective contradicting the current execution phase.
        kinds = [k for k in ("read", "diagnose", "declare", "order", "wait", "finish") if k not in board.unserved]
        if board.committed:
            kinds = [k for k in kinds if k not in ("diagnose", "declare")]
        text = (f"You control maintenance using actual tool feedback. Return only one JSON plan.\n"
                f"## Planning turn\nQuestion of the episode: {board.question}\n\n{board.render()}\n\n"
                f"Currently offered step kinds: {', '.join(kinds)}.\n"
                'Format: {"branch": null, "mode": "test_first", "why": "your evidence-based reason", '
                '"steps": [{"kind": "read", "args": {"sensor_group": "a listed group", "cycles": 1}, '
                '"expect": {}, "evidence": []}]}\n'
                "Use the actual listed group/component keys. branch is null or a 1-based candidate rank. "
                "diagnose args={rank: candidate rank}; without MCP list your own candidates as "
                "candidates=[{levels: component-to-level mapping, failure_mode_ids: actual IDs}]. "
                "order args={component: component key, prediction_id: actual prediction ID or estimate}; "
                "wait args={cycles: positive integer}; finish args={}. "
                "declare args={sensor_group: group, cycles: positive integer}, with evidence IDs of actual readings. "
                "expect may be empty; do not invent predictions or demand a rounded probability as an exact threshold. "
                "Use only real evidence IDs. Before any diagnosis/declaration, read actual telemetry. "
                "After evidence is available, decide whether to diagnose, gather additional available evidence, or declare "
                "insufficiency. Repeatedly requesting unavailable history adds no evidence. "
                "A diagnosis/declaration must precede orders. Never repeat a submitted diagnosis.\n")
        if retry:
            text += f"\nPrevious action could not run: {retry}\n"
        return text + instructions(board)

    prompts.planning_turn = planning_turn
    # Import after replacing the prompt: controller imports its builder by name.
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent import controller
    controller.planning_turn = planning_turn

    def model_plan(self, record):
        self.board._debug_last_read = last_read(getattr(self, "trajectory", []))
        pending = getattr(self, "_debug_pending", None)
        fingerprint = (self.board.observations, self.board.time)
        surprise = bool(getattr(self, "plans", [])) and self.plans[-1].surprise is not None
        if pending is not None and pending.steps:
            if not surprise and fingerprint == getattr(self, "_debug_checkpoint", None):
                step_plan = first_step(pending)
                problems = controller.check_plan(step_plan, self.board)
                blocked = unavailable_read(step_plan.steps[0], self.board)
                if blocked:
                    problems.append(blocked)
                if not problems:
                    self._debug_pending = replace(pending, steps=pending.steps[1:])
                    return step_plan, "debug_pending_model_step"
                self._retry = "Pending model action requires revision: " + "; ".join(problems)
            else:
                self._retry = "New evidence, elapsed time or a surprise requires updating the pending plan: " + json.dumps(pending.to_json())
            self._debug_pending = None
        retry, self._retry = self._retry, None
        proposed = self._parse(self._generate(planning_turn(
            self.board, self.options.plan_length, retry=retry, surface=self.surface)), record)
        if proposed is None:
            return self._no_plan(record, "no_plan_after_parse_failure", "could not be parsed",
                                 f"The reply could not be parsed: {record.parse_errors[-1]}")
        plan = first_step(proposed)
        if len(proposed.steps) > 1:
            self.board.notes.append("DEBUG maintenance-loop-v4: pending proposed tail (not yet executed): " +
                                    json.dumps([s.to_json() for s in proposed.steps[1:]]))
        problems = controller.check_plan(plan, self.board)
        if plan.steps:
            blocked = unavailable_read(plan.steps[0], self.board)
            if blocked:
                problems.append(blocked)
        if problems:
            record.check_problems.append(problems)
            return self._no_plan(record, "no_plan_after_check_failure", "; ".join(problems),
                                 "First proposed action was refused:\n- " + "\n- ".join(problems))
        self._debug_pending = replace(proposed, steps=proposed.steps[1:])
        self._debug_checkpoint = fingerprint
        return plan, "debug_model_first_step"

    controller.CausalLoopRunner._model_plan = model_plan
