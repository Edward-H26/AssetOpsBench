"""Run an observation-to-action policy, such as the greedy reference policy, through the tool interface, so that it
plays on exactly what a language-model agent gets: the facts the question states at reset, then only tool answers.

An answer carries the newest history entry, with the measurement renamed and rounded (names.public_result), the cost,
the counters and the done flag (episode.call); play() rebuilds the policy's observation from these. The greedy policy
looks its held-out predictions up by the measured values, so PublicClassifiers keys that table the way the answers show
a measurement. Renaming and rounding must keep every stable cycle distinct at every tier, so that the policy decides
as it does in process; the constructor refuses a table in which two measurements would merge.
"""
import json

from envs.hydraulic_rig.classifiers import signature

from .episode import STATUS_KEYS
from .names import public_result


def tool_call(action):
    """The tool and arguments that carry an env.step action: the inverse of the tools in main.py."""
    kind = action["type"]
    if kind == "measure":
        return "measure", {"tier": action["tier"]}
    if kind == "submit" and action.get("insufficient_evidence"):
        return "submit_insufficient_evidence", {"next_action": action["next_action"]}
    if kind == "submit":
        return "submit", dict(action["diagnosis"])
    if kind == "replace":
        return "replace", {"component": action["component"]}
    if kind == "finish":
        return "finish", {}
    if kind == "advance_time":
        return "advance_time", {"cycles": action["cycles"]}
    raise ValueError(f"no tool carries the action {action!r}")


class PublicClassifiers:
    """A TierClassifiers whose held-out prediction table is keyed by measurements as the tool answers show them."""

    def __init__(self, classifiers):
        self.base, self.alpha = classifiers.base, classifiers.alpha
        self.lookup = {}
        for tier, table in classifiers.lookup.items():
            self.lookup[tier] = {signature(public_result(dict(key))): pred for key, pred in table.items()}
            if len(self.lookup[tier]) != len(table):
                raise ValueError(f"renaming and rounding make two tier-{tier} measurements identical")

    def predict(self, tier, result):
        return self.lookup[tier][signature(result)]


def play(policy, first_observation, send):
    """Run one episode. send(action) delivers the action as a tool call and returns the answer text. The policy sees the
    first observation, which holds the facts the question states, and after that only what the answers carry; the
    episode is over when an answer says so."""
    obs = {**first_observation, "history": []}
    while True:
        answer = json.loads(send(policy.act(obs)))
        if answer.get("event") is not None:
            obs["history"] = [*obs["history"], answer["event"]]
        obs.update((key, answer[key]) for key in STATUS_KEYS)
        if answer["episode_over"]:
            return
