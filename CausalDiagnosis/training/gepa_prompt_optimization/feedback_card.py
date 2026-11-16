"""The feedback card of a scored episode, the only record of it that reaches the reflection model of gepa_prompt_optimization/, the
online loop of GEPA over the tunable block of the three agents of the training table. The card keeps the flags and the
outcome of the verdict, the public trajectory and the messages of the diagnostics tools, and leaves out the details of the
checks, the hidden state, the oracle's quantities and the seed."""
import json
from pathlib import Path
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import CAUSAL_SERVER
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import HIDDEN
from CausalDiagnosis.utils.json_files import read_json


def diagnostics_messages(trajectory):
    """The message field of every answer of the diagnostics server in a trajectory, the evaluation trace GEPA reads."""
    messages = []
    for step in trajectory or []:
        if step.get("server") != CAUSAL_SERVER or step.get("error"):
            continue
        try:
            answer = json.loads(step.get("response") or "")
        except (TypeError, ValueError):
            continue
        if isinstance(answer, dict) and answer.get("message"):
            messages.append(f"{step.get('tool')}: {answer['message']}")
    return messages


STEP_FIELDS = ("server", "tool", "tool_args", "error")


def feedback_card(verdict, reward, trajectory):
    """The only record of a scored episode that reaches the reflection model. Per asset it keeps the flag of every check,
    the pass, the success, the failure class, the abstention and the count of unnecessary orders, and for the episode the
    reward, the realized cost, the end reason, the steps, the refusals, the public trajectory and the diagnostics messages.
    The details of the checks, the hidden start, the truth at the write, the oracle's cost, the normalized cost and the
    seed stay out, since they tell which components of a past episode were faulty. A single asset episode has one entry."""
    verdict = verdict or {}
    asset = {"checks": {name: isinstance(check, dict) and bool(check.get("ok")) for name, check in (verdict.get("checks") or {}).items()},
             "pass": verdict.get("pass"), "success": verdict.get("success"), "failure_class": verdict.get("failure_class"),
             "abstained": verdict.get("abstained"), "unnecessary_orders": verdict.get("unnecessary")}
    return {"assets": [asset], "reward": reward, "cost": verdict.get("cost"), "end_reason": verdict.get("end_reason"),
            "steps": verdict.get("steps"), "refusals": verdict.get("refusals"),
            "trajectory": [_public_step(step) for step in trajectory or []], "diagnostics_messages": diagnostics_messages(trajectory)}


def _public_step(step):
    """One tool call as the agent saw it, its response whole: the reflection request's context is the only bound."""
    return {**{k: step.get(k) for k in STEP_FIELDS}, "response": str(step.get("response") or "")}


def episode_card(ep_dir, reward):
    """The feedback card of an episode folder, from its verdict, the number the loop scores it by and its trajectory."""
    return feedback_card(read_json(Path(ep_dir) / HIDDEN / "verdict.json", {}), reward, read_json(Path(ep_dir) / "trajectory.json", []))


def feedback_string(card):
    """The card as the text the reflection model reads, built from the card alone."""
    lines = []
    for asset in card["assets"]:
        lines += [f"check {name}: {'ok' if ok else 'failed'}" for name, ok in asset["checks"].items()]
        lines.append(f"pass {asset['pass']}, success {asset['success']}, failure class {asset['failure_class']}, "
                     f"abstained {asset['abstained']}, unnecessary orders {asset['unnecessary_orders']}")
    lines.append(f"reward {card['reward']}, cost {card['cost']}, end reason {card['end_reason']}, steps {card['steps']}, "
                 f"refusals {card['refusals']}")
    if card["trajectory"]:
        lines.append("tool calls:")
        lines += [f"- {s['tool']} {json.dumps(s['tool_args'], default=str)}" + (f" error {s['error']}" if s["error"] else f" -> {s['response']}")
                  for s in card["trajectory"]]
    if card["diagnostics_messages"]:
        lines.append("diagnostics server messages:")
        lines += [f"- {m}" for m in card["diagnostics_messages"]]
    return "\n".join(lines)
