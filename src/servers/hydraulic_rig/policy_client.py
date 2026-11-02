"""Run a policy of envs.hydraulic_rig.policies through the tool interface, so that it plays on exactly what a language
model agent gets: the envelope that opens the episode, then only tool answers (docs/hydraulic_rig/action-server.md).

tool_call maps an env.step action to the tool and arguments that carry it, the inverse of the tools in main.py; play
runs one episode by sending each action and handing the answer back to the policy.
"""
import json


def tool_call(action):
    """The tool and arguments that carry an env.step action."""
    kind = action["type"]
    if kind == "get_asset_profile":
        return kind, {}
    if kind == "read_sensors":
        return kind, {"sensor_group": action["sensor_group"], "cycles": action["cycles"]}
    if kind == "advance_time":
        return kind, {"cycles": action["cycles"]}
    if kind == "summarize_changes_since":
        return kind, {"cycle": action["cycle"]}
    if kind == "get_failure_modes":
        return kind, {"component": action["component"]}
    if kind == "submit_diagnosis":
        return kind, {**action["diagnosis"], "evidence_ids": ",".join(action["evidence_ids"])}
    if kind == "declare_insufficient_evidence":
        test = action["separating_test"]
        return kind, {"sensor_group": test["sensor_group"], "cycles": test["cycles"], "evidence_ids": ",".join(action["evidence_ids"])}
    if kind == "create_work_order":
        return kind, {"component": action["component"], "prediction_id": action["prediction_id"], "evidence_ids": ",".join(action["evidence_ids"])}
    if kind == "finish":
        return kind, {"summary": action["summary"]}
    raise ValueError(f"no tool carries the action {action!r}")


def play(policy, opening, send):
    """Run one episode. send(action) delivers the action as a tool call and returns the answer text. The policy starts
    from the opening envelope, whose result is the asset profile, and after that sees only the answers; the episode is
    over when an answer says so."""
    policy.start(opening)
    obs = opening
    while True:
        obs = json.loads(send(policy.act(obs)))
        if obs["episode_over"]:
            return
