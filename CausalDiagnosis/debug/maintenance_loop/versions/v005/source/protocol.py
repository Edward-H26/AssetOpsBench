"""Prompt and live-observation changes over frozen v4; no new action selection or world rules."""
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path

spec = importlib.util.spec_from_file_location("frozen_v4_protocol", Path(__file__).with_name("v4_protocol.py"))
v4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v4)


def public_progress(board):
    state = v4.progress(board)
    state["post_repair_verifications"] = {
        key: {k: verdict.get(k) for k in ("status", "consistent", "predicted_combination", "message") if k in verdict}
        for key, verdict in board.verifications.items()
    }
    state["budget_left"] = board.status.get("budget_left")
    state["steps_left"] = board.status.get("steps_left")
    state["read_prices_per_cycle"] = ((board.profile or {}).get("prices") or {}).get("read", {})
    return state


def offered_kinds(board):
    """Public planning affordances; actual action acceptance still uses v4 and native checks."""
    kinds = [kind for kind in ("read", "diagnose", "declare", "order", "wait", "time", "finish")
             if kind not in board.unserved]
    if board.committed:
        kinds = [kind for kind in kinds if kind not in ("diagnose", "declare")]
    else:
        kinds = [kind for kind in kinds if kind != "order"]
    if not board.clock_runs:
        kinds = [kind for kind in kinds if kind != "wait"]
    groups = (board.profile or {}).get("sensor_groups", {})
    if groups and all(v4.unavailable_read(_read_step(g), board) for g in groups):
        kinds = [kind for kind in kinds if kind != "read"]
    return kinds


def _read_step(group):
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.plan import Step
    return Step("read", {"sensor_group": group, "cycles": 1})


def contract(board):
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.plan.plan_schema import MODES
    groups = list((board.profile or {}).get("sensor_groups", {}))
    return {
        "required_root_fields": {"branch": "null or a 1-based candidate integer", "mode": list(MODES),
                                 "why": "string", "steps": "non-empty array of step objects"},
        "optional_root_fields": {"candidates": "array of {levels: component-to-level object, failure_mode_ids: string array}; required for your own diagnosis if no server candidate exists"},
        "offered_next_step_kinds": offered_kinds(board),
        "argument_types": {
            "read": {"sensor_group": {"type": "one string, never an array of channels", "enum": groups}, "cycles": "positive integer"},
            "diagnose": {"rank": "1-based candidate integer; candidates belong at ROOT, never inside diagnose.args"},
            "declare": {"sensor_group": "one listed group string", "cycles": "positive integer"},
            "order": {"component": list((board.profile or {}).get("components", [])), "prediction_id": "existing ID or estimate"},
            "wait": {"cycles": "positive integer"}, "time": {}, "finish": {},
        },
        "step_fields": {"kind": "controller step kind, never an MCP function name", "args": "object",
                        "expect": "object; may be empty", "evidence": "array of actual evidence ID strings; may be empty"},
    }


def planning_prompt(board, retry=None):
    state = public_progress(board)
    groups = (board.profile or {}).get("sensor_groups", {})
    ready_groups = [g for g in groups if not v4.unavailable_read(_read_step(g), board)]
    notes = [
        "Use exact group KEYS, not sensor/channel names. Do not add an answer or simulated tool results after the JSON.",
        "Before a final diagnosis or insufficient-evidence declaration, retrieve actual telemetry and cite observed evidence. The alarm text or a prior ranking alone is not a measurement.",
        "Choose read cycles and repairs whose prices fit budget_left and whose durations fit the remaining horizon. A large reading window is not required merely because a lot of history exists.",
        "A read of already exhausted history cannot create data. Read is currently offered only if a listed group can return unread history, play forward, or receive new data with the thinking clock.",
        "You may decide from existing evidence, ask for available additional evidence, or explicitly wait when served. Do not keep a confirmatory read at the head of every new plan when that read cannot return new data.",
        "Submit diagnosis once. Re-evaluate hypotheses after changes without submitting a second final diagnosis.",
        "An accepted work order is pending until it closes. Query time or wait for updates when served; do not reopen a component with an open order.",
        "After repair, read measurements and check the result. When your available evidence supports whole-device recovery, no order remains open and verification is positive, choose finish. Waiting again is not required merely to end the episode.",
        "A positive check of a partially repaired combination does not establish whole-device health. If faults remain, plan their repair within budget.",
        "MCP rankings and repair plans are advice, not ground truth or mandatory choices. Without MCP, construct your own candidates from the same public readings and failure modes.",
    ]
    return ("Return a proposed maintenance plan as ONE JSON object, using the contract below.\n"
            "## Planning turn\nQuestion of the episode: " + board.question + "\n\n" + board.render() +
            "\n\n## v5 public progress\n" + json.dumps(state, ensure_ascii=False) +
            "\n## v5 exact output contract\n" + json.dumps(contract(board), ensure_ascii=False) +
            "\nGroups currently offering data: " + json.dumps(ready_groups) +
            "\n" + "\n".join(notes) +
            ("\nPrevious reply/action was refused for this exact reason:\n" + retry if retry else "") +
            "\nReturn the plan only. Do not copy this contract as the answer. Choose your own actions and diagnosis.")


def trace(event, **fields):
    name = os.environ.get("HPML_DEBUG_TRACE")
    if not name:
        return
    path = Path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(), "event": event, **fields},
                              ensure_ascii=False, default=str) + "\n")


def install(prompts):
    v4.install(prompts)
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent import controller
    cls = controller.CausalLoopRunner
    native_generate = getattr(cls, "_v5_native_generate", cls._generate)
    cls._v5_native_generate = native_generate
    base_model_plan = cls._model_plan
    base_execute = getattr(cls, "_v5_native_execute", cls._execute)
    cls._v5_native_execute = base_execute

    def generate(self, prompt):
        board = self.board
        opening = board.profile is None or any(k not in board.opening_done and k not in board.unserved
                                              for k in ("failure_modes", "fix_history", *board.required_opening))
        if "## Planning turn" in prompt and not opening:
            retry = getattr(self, "_v5_retry", None)
            marker = "\nPrevious action could not run: "
            if marker in prompt:
                retry = prompt.split(marker, 1)[1].split("\n\n## Debug execution contract", 1)[0].strip()
            prompt = planning_prompt(board, retry)
        trace("model_request", prompt=prompt, progress=public_progress(board))
        try:
            reply = native_generate(self, prompt)
        except Exception as error:
            trace("model_error", error=f"{type(error).__name__}: {error}")
            raise
        trace("model_reply", reply=reply)
        return reply

    def model_plan(self, record):
        # v4 consumes _retry before calling generate. Preserve it only for the prompt, not for action selection.
        self._v5_retry = self._retry
        plan, source = base_model_plan(self, record)
        trace("plan_decision", index=record.index, source=source, plan=plan.to_json(),
              parse_errors=record.parse_errors, check_problems=record.check_problems)
        return plan, source

    async def execute(self, record):
        try:
            return await base_execute(self, record)
        finally:
            trace("execution_result", index=record.index, steps=record.steps,
                  progress=public_progress(self.board), end_reason=self.board.end_reason)

    cls._generate, cls._model_plan, cls._execute = generate, model_plan, execute
