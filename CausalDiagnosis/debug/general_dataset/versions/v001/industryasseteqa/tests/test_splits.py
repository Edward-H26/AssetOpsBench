"""Machine independence and input identity of development and held-out QA."""

import copy
import json

import pytest

from CausalDiagnosis.debug.general_dataset.versions.v001.industryasseteqa.data import QADataset
from CausalDiagnosis.debug.general_dataset.versions.v001.industryasseteqa.splits import make_split, select_split


def test_split_keeps_each_machine_together_and_rejects_leakage_or_changed_inputs(tmp_path):
    facts = [{"fact_id": f"f{index}", "asset_id": f"machine_{index // 2}",
              "features": [{"name": "value", "value": index}]} for index in range(8)]
    questions = [{"qa_id": f"q{index}", "fact_id": f"f{index}", "question": "What was measured?",
                  "task_type": "descriptive"} for index in range(8)]
    data = QADataset(facts, questions)
    fp, qp = tmp_path / "facts.jsonl", tmp_path / "qa.jsonl"
    fp.write_text("\n".join(map(json.dumps, facts)))
    qp.write_text("\n".join(map(json.dumps, questions)))
    manifest = make_split(data, fp, qp, 0.5)
    assert manifest == make_split(data, fp, qp, 0.5)
    dev = select_split(data, fp, qp, manifest, "dev")
    test = select_split(data, fp, qp, manifest, "test")
    assert not set(dev).intersection(test)
    assert set(dev + test) == set(data.question_ids())
    assert not {data.public_fact(qa)["asset_id"] for qa in dev}.intersection(
        data.public_fact(qa)["asset_id"] for qa in test)
    bad = copy.deepcopy(manifest)
    bad["partitions"]["dev"]["qa_ids"].append(test[0])
    with pytest.raises(ValueError, match="questions"):
        select_split(data, fp, qp, bad, "dev")
    fp.write_text(fp.read_text() + "\n")
    with pytest.raises(ValueError, match="hashes"):
        select_split(data, fp, qp, manifest, "test")
