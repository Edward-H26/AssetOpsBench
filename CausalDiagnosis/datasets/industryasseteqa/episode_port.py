"""Register the versioned QA implementation on the declared episode driver."""

import json

from CausalDiagnosis.debug.general_dataset.versions.v001.industryasseteqa.data import QADataset
from CausalDiagnosis.debug.general_dataset.versions.v001.industryasseteqa.contract import declaration
from CausalDiagnosis.debug.general_dataset.versions.v001.industryasseteqa.evaluation import check_answer, direct_numeric_match
from CausalDiagnosis.debug.general_dataset.versions.v001.industryasseteqa.runner import launch_text, digest
from CausalDiagnosis.debug.general_dataset.versions.v001.industryasseteqa.splits import select_split


class QAPort:
    server = "maintenance_qa"
    publication = "aggregate_only"

    def __init__(self, inputs):
        self.inputs = inputs
        self.dataset = QADataset.load(inputs["facts"], inputs["qa"])
        self.split = json.loads(inputs["split"].read_text(encoding="utf-8"))

    def case_ids(self, subset):
        return select_split(self.dataset, self.inputs["facts"], self.inputs["qa"], self.split, subset)

    def task(self, case_id, limits):
        if any(value is not None for value in limits.values()):
            raise ValueError("QA declares no tool-call or price budget; use the model request budget")
        return self.dataset.task(case_id)

    def unsupported(self, case_id, *, supports_images=False):
        return None

    def validate_case(self, case_id, limits):
        self.task(case_id, limits)

    def declaration(self, case_id):
        return declaration()

    def launch(self, case_id, folder, limits):
        return launch_text(self.inputs["facts"], self.inputs["qa"], case_id, folder, False, normalized=True)

    def evaluate(self, case_id, episode, folder):
        held = episode.board.get("commitment")
        answer = held["args"].get("answer") if held and episode.end == "finish" else None
        return {**check_answer(answer, self.dataset.public_fact(case_id)),
                "direct_numeric_match": direct_numeric_match(answer, self.dataset.reference(case_id)),
                "semantic_quality": None, "semantic_quality_reason": "no semantic judge configured"}

    def identity(self):
        return {key: digest(path) for key, path in self.inputs.items()}


def load(inputs):
    return QAPort(inputs)
