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
        "forward_read_cycles_per_cycle": board.forward_read,
        "record_grows_with_thinking": board.record_grows,
        "recorded_until": board.status.get("recorded_until"),
    }


def instructions(board):
    state = progress(board)
    return (
        "\n\n## Debug execution contract (maintenance-loop-v3)\n"
        "Return exactly ONE proposed step per turn, including opening queries. "
        "The controller executes only the first proposed step, subject to the original checks; "
        "every tail step is discarded and recorded, not executed. Replan on actual feedback.\n"
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


def first_step(plan):
    """Execute the first model-chosen action, never substitute a controller-chosen action."""
    return replace(plan, steps=plan.steps[:1])


def install(prompts):
    original_prompt = prompts.planning_turn

    def planning_turn(board, plan_length, retry=None, surface=None):
        return original_prompt(board, plan_length, retry=retry, surface=surface) + instructions(board)

    prompts.planning_turn = planning_turn
    # Import after replacing the prompt: controller imports its builder by name.
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent import controller
    controller.planning_turn = planning_turn

    def model_plan(self, record):
        retry, self._retry = self._retry, None
        proposed = self._parse(self._generate(planning_turn(
            self.board, self.options.plan_length, retry=retry, surface=self.surface)), record)
        if proposed is None:
            return self._no_plan(record, "no_plan_after_parse_failure", "could not be parsed",
                                 f"The reply could not be parsed: {record.parse_errors[-1]}")
        plan = first_step(proposed)
        if len(proposed.steps) > 1:
            self.board.notes.append("DEBUG maintenance-loop-v3: discarded unexecuted plan tail: " +
                                    json.dumps([s.to_json() for s in proposed.steps[1:]]))
        problems = controller.check_plan(plan, self.board)
        if problems:
            record.check_problems.append(problems)
            return self._no_plan(record, "no_plan_after_check_failure", "; ".join(problems),
                                 "First proposed action was refused:\n- " + "\n- ".join(problems))
        return plan, "debug_model_first_step"

    controller.CausalLoopRunner._model_plan = model_plan
