"""Read a generic plan against the QA server's actual listed tool schemas."""

import asyncio
import json

from CausalDiagnosis.agents.utils.declaration import Declaration, ToolRow, Vocabulary
from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.plan.reply_parser import parse_usable_plan
from CausalDiagnosis.utils.public_rules.plan_rules import KindRules
from CausalDiagnosis.debug.general_dataset.versions.v001.industryasseteqa.data import QADataset
from CausalDiagnosis.debug.general_dataset.versions.v001.industryasseteqa.episode import QAEpisode
from CausalDiagnosis.debug.general_dataset.versions.v001.industryasseteqa.server import build_server


def test_a_qa_plan_reads_the_actual_listing_without_calls_or_private_references(tmp_path):
    fact = {"fact_id": "fact_a", "features": [{"name": "reading_mean", "value": 12.5}], "label": "evaluator_only"}
    question = {"qa_id": "question_a", "fact_id": "fact_a", "task_type": "descriptive", "question": "What was the mean?",
                "direct_answer": "evaluator_only", "reasoning_answer": "evaluator_only"}
    episode = QAEpisode(QADataset([fact], [question]), "question_a")
    listed = asyncio.run(build_server(episode, tmp_path).list_tools())
    rows = {"get_task": ToolRow("get_task", "maintenance_qa", "task", facts=True),
            "read_fact": ToolRow("read_fact", "maintenance_qa", "observe", answer="object"),
            "submit_answer": ToolRow("submit_answer", "maintenance_qa", "task", write="answer", commits=True,
                                     ends=True, requires_observation=True, after=("read_fact",))}
    declaration = Declaration(asset_id="qa_fixture", kind="qa_fixture", surface=None,
                              vocabulary=Vocabulary("maintenance question", "question"), rows=rows, listed_rows={},
                              server_roles={}, cards=(), advice={}, events=(), rules=KindRules(), binding=None)
    table = declaration.table({"maintenance_qa": listed})
    assert table.tools["submit_answer"].candidate == ("answer",)
    assert "evaluator_only" not in json.dumps([tool.model_dump() for tool in listed])
    proposed = {"direct_answer": "12.5", "reasoning_answer": "The released mean is 12.5.", "confidence": 0.9,
                "provenance": {"fact_id": "fact_a", "features": ["reading_mean"]}}
    body = {"why": "Read the released evidence before answering.", "steps": [
        {"tool": "get_task"}, {"tool": "read_fact"}, {"tool": "submit_answer", "args": {"answer": proposed}}]}
    parsed = parse_usable_plan(json.dumps(body), table, {}, None)
    assert parsed.refusal is None and parsed.unusable is None
    assert [step.tool for step in parsed.plan.steps] == list(rows)
    assert parsed.plan.steps[-1].args == {"answer": proposed} and parsed.plan.steps[-1].candidate is None
    assert episode.calls == [] and not episode.observed and not episode.over
    body["steps"][-1]["args"]["answer"] = "wrong argument type"
    rejected = parse_usable_plan(json.dumps(body), table, {}, None)
    assert len(rejected.plan.steps) == 2 and rejected.unusable.kinds == ("submit_answer",)
