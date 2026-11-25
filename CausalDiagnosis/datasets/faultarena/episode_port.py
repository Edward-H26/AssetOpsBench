"""Public tool declaration and private exact replay for a pinned case release."""

import copy
import base64
import json
import mimetypes
from graphlib import CycleError, TopologicalSorter
from pathlib import Path

from CausalDiagnosis.agents.utils.declaration import Declaration, ToolRow, Vocabulary
from CausalDiagnosis.debug.general_dataset.versions.v001.faultarena.data import FaultArenaRelease
from CausalDiagnosis.debug.general_dataset.versions.v001.faultarena.audit import audit_release
from CausalDiagnosis.debug.general_dataset.versions.v001.faultarena.download import relative_path
from CausalDiagnosis.debug.general_dataset.versions.v001.industryasseteqa.runner import digest
from CausalDiagnosis.environment.scripted_cases.replay import ScriptedEpisode
from CausalDiagnosis.utils.paths import ROOT
from CausalDiagnosis.utils.public_rules.scripted_cases import ScriptedRules


def tool_specs(release):
    """Derive roles and safety rules only from the published tool files."""
    output = [{"name": "get_task", "description": "Read the question, initial public state and run limits. Refusals count one tool call and cost zero.",
               "schema": {"type": "object", "properties": {}, "additionalProperties": False}, "role": "knowledge", "cost": 0}]
    for public in release.tools():
        ends = public.get("category") == "diagnosis"
        category = public.get("category")
        write = "answer" if ends else "repair" if category == "repair" else "state" if public.get("state_change") else "probe" if category in ("operation", "diagnostic", "configuration") else None
        requires = [f"power_status == {public['power_required']!r}"] if public.get("power_required") in ("ON", "OFF") else []
        description = public["description"]
        if requires:
            description += " Public requirement: " + "; ".join(requires) + ". Breaking it ends the episode."
        if public.get("state_change"):
            description += " State updates: " + json.dumps(public["state_change"]) + ". A repeated state-setting call is refused."
        output.append({"name": public["name"], "description": description, "schema": public["parameters"],
                       "role": "task" if ends else "act" if write else "observe", "write": write,
                       "ends": ends, "requires": requires, "updates": public.get("state_change", {}),
                       "cost": public.get("cost", 0), "destructive": public.get("is_destructive", False)})
    return output


def make_episode(release, case_id, limits):
    private = release.replay_case(case_id)
    actions = [{"id": item["action_id"], "tool": item["match_tool"], "args": item["match_args"],
                "requires": item.get("pre_conditions", []), "updates": item.get("state_updates", {}),
                "cost": item.get("action_cost", 0), "observation": item["observation"]}
               for item in private["hidden_environment"]]
    task = release.task(case_id)
    task["limits"] = limits
    tools = tool_specs(release)
    tools[0]["static_result"] = task

    def attachment(observation):
        if not isinstance(observation, dict) or not observation.get("image_path"):
            return observation
        relative = relative_path(observation["image_path"])
        if relative.parts[0] != "images":
            raise ValueError("image must be inside the release image directory")
        path = release.folder.joinpath(*relative.parts).resolve()
        path.relative_to(release.folder.resolve())
        mime, _ = mimetypes.guess_type(path.name)
        if mime not in ("image/png", "image/jpeg", "image/gif", "image/webp"):
            raise ValueError("unsupported image media type")
        return {**observation, "attachments": [{"type": "image", "mime_type": mime,
                 "data": base64.b64encode(path.read_bytes()).decode("ascii")}]}

    return ScriptedEpisode(private["initial_states"], tools, actions,
                           max_calls=limits.get("max_tool_calls"), budget=limits.get("budget"), transform_observation=attachment)


def score_trace(private, replay):
    """Trajectory diagnostics never substitute for a semantic diagnosis judge."""
    trace = replay.get("trace", [])
    accepted = [item for item in trace if item["accepted"]]
    order = {}
    for index, item in enumerate(accepted):
        if item.get("action"):
            order.setdefault(item["action"], index)
    terminal = [item["action_id"] for item in private["hidden_environment"] if item["match_tool"] == "submit_diagnosis"]
    if replay.get("submission") and len(terminal) == 1:
        order.setdefault(terminal[0], len(accepted))
    dag = private.get("golden_dag", {})
    dependencies = dag.get("dependencies", {})
    valid = True
    try:
        tuple(TopologicalSorter(dependencies).static_order())
    except CycleError:
        valid = False
    known = {item["action_id"] for item in private["hidden_environment"]}
    required = set(dag.get("mandatory_nodes", []))
    valid = valid and required <= known and set(dependencies) <= known and all(set(items) <= known for items in dependencies.values())
    return {"submitted": replay.get("submission") is not None,
            "refusals": sum(not item["accepted"] for item in trace),
            "cost": sum(item["cost"] for item in trace), "matched_actions": len(order),
            "reference_dag_valid": valid,
            "mandatory_coverage": len(required & order.keys()) / len(required) if valid and required else None,
            "precedence_satisfied": all(node in order and all(prior in order and order[prior] < order[node] for prior in priors)
                                         for node, priors in dependencies.items()) if valid else None,
            "diagnosis_quality": None, "diagnosis_quality_reason": "no semantic judge configured",
            "matching_policy": "exact_arguments", "official_matcher_parity": False}


class FaultArenaPort:
    server = "scripted_case"
    publication = "aggregate_only"

    def __init__(self, inputs):
        self.inputs = inputs
        self.audit = audit_release(inputs["release"])
        if any(item["problem"] != "cyclic DAG" for item in self.audit["problems"]):
            raise ValueError("release audit found structural problems")
        self.release = FaultArenaRelease(inputs["release"])
        self.specs = tool_specs(self.release)

    def case_ids(self, subset):
        if subset not in ("train", "test"):
            raise ValueError("scripted release partitions are train and test")
        return list(self.release.case_ids(subset))

    def task(self, case_id, limits):
        return {**self.release.task(case_id), "limits": copy.deepcopy(limits),
                "replay_policy": "exact tool arguments; refused calls cost zero and count one call; submission is not a semantic quality score"}

    def unsupported(self, case_id, *, supports_images=False):
        images = self.release.image_paths(case_id)
        if images and any(not self.release.folder.joinpath(*relative_path(path).parts).is_file() for path in images):
            return "source image files are missing"
        if not supports_images and "image" in self.release.modalities(case_id):
            return "image-backed observations require a multimodal transport; text-only replay is unsupported"
        return None

    def validate_case(self, case_id, limits):
        make_episode(self.release, case_id, limits)

    def declaration(self, case_id):
        rows = {}
        for item in self.specs:
            keys = dict(write=item.get("write"), ends=item.get("ends", False), facts=item["name"] == "get_task")
            if keys["ends"]:
                keys.update(commits=True, requires_observation=True)
            if keys["write"] == "state":
                keys["sets"] = json.dumps(item["updates"], sort_keys=True)
            rows[item["name"]] = ToolRow(item["name"], self.server, item["role"], **keys)
        return Declaration("faultarena", "scripted_cases", None,
                           Vocabulary("maintenance case", "machine", terms={"state": "machine state", "repair": "replacement", "target": "component"}),
                           rows, {}, {}, (), {}, (), ScriptedRules(self.specs), None)

    def launch(self, case_id, folder, limits):
        import mcp
        paths = [str(ROOT), str(Path(mcp.__file__).resolve().parents[1])]
        return (f"import sys\nsys.path[:0] = {paths!r}\n"
                "from CausalDiagnosis.debug.general_dataset.versions.v001.faultarena.data import FaultArenaRelease\n"
                "from CausalDiagnosis.datasets.faultarena.episode_port import make_episode\n"
                "from CausalDiagnosis.environment.scripted_cases.server import run_server\n"
                f"run_server(make_episode(FaultArenaRelease({str(self.inputs['release'])!r}), {case_id!r}, {limits!r}), {str(folder)!r})\n")

    def evaluate(self, case_id, episode, folder):
        path = folder / "hidden/replay.json"
        if not path.exists():
            return {"submitted": False, "diagnosis_quality": None, "reason": "no simulator record"}
        return score_trace(self.release.replay_case(case_id), json.loads(path.read_text(encoding="utf-8")))

    def identity(self):
        audit_release(self.inputs["release"])
        return {"release_manifest": digest(self.inputs["release"] / "source.json")}


def load(inputs):
    return FaultArenaPort(inputs)
