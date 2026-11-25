"""Registered ports share one driver, including an independently declared alias."""

import copy
import base64
import json

import pytest

from CausalDiagnosis.datasets.declared_episodes import discover
from CausalDiagnosis.datasets.faultarena.episode_port import make_episode, score_trace
from CausalDiagnosis.debug.general_dataset.versions.v001.industryasseteqa.data import QADataset
from CausalDiagnosis.debug.general_dataset.versions.v001.industryasseteqa.splits import make_split
from CausalDiagnosis.debug.general_dataset.versions.v001.industryasseteqa.tests.test_qa import records
from CausalDiagnosis.debug.general_dataset.versions.v001.faultarena.download import download_release
from CausalDiagnosis.debug.general_dataset.versions.v001.faultarena.data import FaultArenaRelease
from CausalDiagnosis.debug.general_dataset.versions.v001.faultarena.tests.test_release import payloads, fetcher
from CausalDiagnosis.experiments.episode_runners.run_declared_episodes import main, run_dataset, validate_dataset

PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aD3kAAAAASUVORK5CYII="


@pytest.fixture
def anyio_backend():
    return "asyncio"


def qa_inputs(root):
    fact, question, _ = records()
    second = copy.deepcopy(fact)
    second.update(fact_id="fact_b", asset_id="asset_b")
    second_question = {**question, "qa_id": "question_b", "fact_id": "fact_b"}
    facts, qa, split = root / "facts.jsonl", root / "qa.jsonl", root / "split.json"
    facts.write_text("\n".join(map(json.dumps, (fact, second))), encoding="utf-8")
    qa.write_text("\n".join(map(json.dumps, (question, second_question))), encoding="utf-8")
    dataset = QADataset.load(facts, qa)
    manifest = make_split(dataset, facts, qa)
    split.write_text(json.dumps(manifest), encoding="utf-8")
    return {"facts": facts, "qa": qa, "split": split}


def release_inputs(root, *, images=False, cyclic=False):
    content = payloads(cycle=cyclic)
    all_cases = [json.loads(line) for line in content["data/all.jsonl"].decode().splitlines()]
    for record in all_cases:
        record["is_multimodal"], record["image_paths"] = images, ["images/example.png"] if images else []
        for action in record["case"]["hidden_environment"]:
            if action["match_tool"] == "report":
                action["match_tool"] = "submit_diagnosis"
            action["action_cost"] = 2 if action["match_tool"] == "inspect" else 0
            if images and action["match_tool"] == "inspect":
                action["observation"]["image_path"] = "images/example.png"
    for key, values in (("all", all_cases), ("train", all_cases[:1]), ("test", all_cases[1:])):
        content[f"data/{key}.jsonl"] = "\n".join(map(json.dumps, values)).encode()
    empty = {"type": "object", "properties": {}, "additionalProperties": False}
    tools = [{"name": "inspect", "description": "Read the public sample observation.", "parameters": empty, "category": "inspection", "cost": 2},
             {"name": "submit_diagnosis", "description": "Submit a diagnosis and finish.", "category": "diagnosis", "cost": 0,
              "parameters": {"type": "object", "properties": {"root_cause": {"type": "string"}, "solution": {"type": "string"}},
                             "required": ["root_cause", "solution"]}}]
    content["tools/tools_schema.json"] = json.dumps({"tools": tools}).encode()
    for key in [key for key in content if key.startswith("tools/toolslist/")]:
        del content[key]
    for tool in tools:
        content[f"tools/toolslist/{tool['name']}.json"] = json.dumps(tool).encode()
    if images:
        content["images/example.png"] = base64.b64decode(PNG)
    release = root / "release"
    download_release(release, fetch=fetcher(content), include_images=images)
    return {"release": release}


def reply(*steps):
    return {"text": json.dumps({"why": "Use the released measurements.", "steps": list(steps)})}


def test_discovery_lists_ports_without_loading_data_and_rejects_unknown_inputs(tmp_path, capsys):
    assert {"industryasseteqa_pdm", "faultarena"} <= discover().keys()
    assert main(["--list"]) == 0
    assert "industryasseteqa_pdm" in capsys.readouterr().out
    with pytest.raises(ValueError, match="requires exactly"):
        discover()["industryasseteqa_pdm"].load({"release": tmp_path})


def test_offline_validation_needs_no_endpoint_and_reports_selected_denominators(tmp_path):
    inputs = qa_inputs(tmp_path)
    report = validate_dataset(discover()["industryasseteqa_pdm"], inputs, "dev")
    assert report["valid"] and report["selected"] == report["validated"] == 1
    assert report["model_calls"] == 0 and report["semantic_quality"] is None
    args = ["--validate", "--dataset", "industryasseteqa_pdm", "--split", "dev"]
    for name, path in inputs.items():
        args.extend(["--input", f"{name}={path}"])
    assert main(args) == 0


@pytest.mark.anyio
@pytest.mark.parametrize("independent", [False, True])
async def test_registered_qa_and_an_independent_port_run_without_core_changes(tmp_path, monkeypatch, independent):
    inputs = qa_inputs(tmp_path)
    descriptor = discover()["industryasseteqa_pdm"]
    if independent:
        folder = tmp_path / "ports" / "another_qa"
        folder.mkdir(parents=True)
        (folder / "episode.json").write_text(json.dumps({"id": "another_qa", "kind": "maintenance_qa",
            "adapter": "independent_qa_port:load", "inputs": ["facts", "qa", "split"]}), encoding="utf-8")
        (tmp_path / "independent_qa_port.py").write_text(
            "from CausalDiagnosis.datasets.industryasseteqa.episode_port import QAPort\ndef load(inputs):\n    return QAPort(inputs)\n", encoding="utf-8")
        monkeypatch.syspath_prepend(str(tmp_path))
        descriptor = discover([tmp_path / "ports"])["another_qa"]
    port = descriptor.load(inputs)
    identifier = port.case_ids("dev")[0]
    fact = port.dataset.public_fact(identifier)
    answer = {"direct_answer": "12.5", "reasoning_answer": "The released mean is 12.5.", "confidence": 0.8,
              "provenance": {"fact_id": fact["fact_id"], "features": ["reading_mean"]}}
    prompts = []

    async def ask(messages, **kwargs):
        prompts.extend(messages)
        return reply({"tool": "get_task"}, {"tool": "read_fact"}, {"tool": "submit_answer", "args": {"answer": answer}})

    result = await run_dataset(descriptor, inputs, "dev", tmp_path / "run", ask=ask, max_turns=1, settings={"backend": "invented"}, limit=1)
    assert result["status"] == "completed"
    assert result["episodes"][0]["scores"]["provenance_ok"]
    assert "private_reference" not in json.dumps(prompts) and "private_label" not in json.dumps(prompts)
    assert result["publication"] == "aggregate_only" and result["adapter_code"]
    assert result["episodes"][0]["scores"]["semantic_quality"] is None
    if independent:
        assert result["dataset"] == "another_qa"


@pytest.mark.anyio
async def test_scripted_port_uses_the_same_driver_and_never_releases_reference_fields(tmp_path):
    inputs = release_inputs(tmp_path)
    prompts = []

    async def ask(messages, **kwargs):
        prompts.extend(messages)
        return reply({"tool": "get_task"}, {"tool": "inspect"},
                     {"tool": "submit_diagnosis", "args": {"root_cause": "A hypothesis", "solution": "A proposed repair"}})

    result = await run_dataset(discover()["faultarena"], inputs, "train", tmp_path / "run", ask=ask,
                               max_turns=1, settings={"backend": "invented"}, max_tool_calls=3)
    assert result["status"] == "completed", result
    scores = result["episodes"][0]["scores"]
    assert scores["submitted"] and scores["cost"] == 2
    assert scores["diagnosis_quality"] is None and not scores["official_matcher_parity"]
    assert "private invented answer" not in json.dumps(prompts)
    episode = json.loads((tmp_path / "run/episode_00000/episode.json").read_text())
    assert [item["tool"] for item in episode["trajectory"]] == ["get_task", "inspect", "submit_diagnosis"]


@pytest.mark.anyio
async def test_image_cases_stay_in_the_denominator_as_unsupported(tmp_path):
    inputs = release_inputs(tmp_path, images=True)

    async def never(*args, **kwargs):
        pytest.fail("an unsupported image case must not reach the text model")

    result = await run_dataset(discover()["faultarena"], inputs, "train", tmp_path / "run", ask=never,
                               max_turns=1, settings={"backend": "invented"})
    assert result["status"] == "incomplete" and len(result["case_ids"]) == 1
    assert result["episodes"][0]["end"] == "unsupported"
    summary = json.loads((tmp_path / "run/summary.json").read_text())
    assert summary["selected"] == summary["recorded"] == 1 and summary["ends"] == {"unsupported": 1}


@pytest.mark.anyio
async def test_images_enter_the_model_only_after_the_corresponding_tool_answers(tmp_path):
    inputs = release_inputs(tmp_path, images=True)
    prompts = []

    async def ask(messages, **kwargs):
        prompts.append(messages)
        if len(prompts) == 1:
            assert isinstance(messages[0]["content"], str) and PNG not in messages[0]["content"]
            return reply({"tool": "get_task"}, {"tool": "inspect"})
        content = messages[0]["content"]
        assert isinstance(content, list)
        images = [item for item in content if item["type"] == "image_url"]
        assert images == [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + PNG}}]
        assert PNG not in content[0]["text"] and "obs/2" in str(content)
        return reply({"tool": "submit_diagnosis", "args": {"root_cause": "A hypothesis", "solution": "A proposed repair"}})

    ask.supports_images = True
    result = await run_dataset(discover()["faultarena"], inputs, "train", tmp_path / "run", ask=ask,
                               max_turns=2, settings={"backend": "invented_multimodal"})
    assert result["status"] == "completed", result
    assert result["supports_images"] and "private invented answer" not in json.dumps(prompts)
    calls = json.loads((tmp_path / "run/episode_00000/llm_calls.json").read_text())
    assert "images" not in calls[0]["settings"]
    assert calls[1]["settings"]["images"][0]["observation_id"] == "obs/2"
    assert "data" not in calls[1]["settings"]["images"][0]


def test_cyclic_reference_is_not_repaired_or_given_a_trajectory_score(tmp_path):
    inputs = release_inputs(tmp_path, cyclic=True)
    release = FaultArenaRelease(inputs["release"])
    env = make_episode(release, "example_a", {})
    assert env.call("get_task", {})["result"]
    env.call("inspect", {})
    env.call("submit_diagnosis", {"root_cause": "guess", "solution": "repair"})
    score = score_trace(release.replay_case("example_a"), {"trace": env.trace, "submission": env.submission})
    assert not score["reference_dag_valid"] and score["mandatory_coverage"] is None
    assert score["precedence_satisfied"] is None


@pytest.mark.anyio
async def test_wrong_partition_or_input_identity_runs_no_model(tmp_path):
    inputs = qa_inputs(tmp_path)
    descriptor = discover()["industryasseteqa_pdm"]
    port = descriptor.load(inputs)

    async def never(*args, **kwargs):
        pytest.fail("invalid selection must not reach the model")

    with pytest.raises(ValueError, match="selected partition"):
        await run_dataset(descriptor, inputs, "dev", tmp_path / "wrong", ask=never, max_turns=1,
                          settings={}, case_ids=port.case_ids("test"))
    inputs["facts"].write_text(inputs["facts"].read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hashes"):
        await run_dataset(descriptor, inputs, "dev", tmp_path / "drift", ask=never, max_turns=1, settings={})
