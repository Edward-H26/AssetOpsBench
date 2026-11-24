"""Release controls use invented cases and never require published data."""

import json

import pytest

from CausalDiagnosis.debug.general_dataset.versions.v001.faultarena.audit import audit_release
from CausalDiagnosis.debug.general_dataset.versions.v001.faultarena.download import REQUIRED, REVISION, download_release
from CausalDiagnosis.debug.general_dataset.versions.v001.faultarena.data import FaultArenaRelease


def payloads(*, cycle=False, overlap=False, traversal=False):
    records = []
    for key in ("example_a", "example_b"):
        case = {"case_id": key, "environment_setup": {"initial_query": "Inspect an invented package."},
                "initial_states": {"power": "ON"}, "hidden_environment": [
                    {"action_id": "read", "match_tool": "inspect", "match_args": {}, "pre_conditions": [],
                     "state_updates": {}, "observation": {"text": "An invented observation.", "image_path": ""}},
                    {"action_id": "submit", "match_tool": "report", "match_args": {}, "pre_conditions": [],
                     "state_updates": {}, "observation": {"text": "", "image_path": ""}}],
                "golden_dag": {"mandatory_nodes": ["read", "submit"],
                               "dependencies": {"submit": ["read"], **({"read": ["submit"]} if cycle else {})}},
                "final_answer": {"root_cause": "private invented answer"}}
        records.append({"case_id": key, "initial_query": case["environment_setup"]["initial_query"], "case": case,
                        "is_multimodal": True, "image_paths": ["../outside.png" if traversal else "images/example.png"]})
    content = {name: b"{}" for name in REQUIRED}
    content["README.md"] = b"Invented release"
    content["LICENSE"] = b"Invented license"
    content["data/all.jsonl"] = "\n".join(map(json.dumps, records)).encode()
    content["data/train.jsonl"] = json.dumps(records[0]).encode()
    content["data/test.jsonl"] = json.dumps(records[0 if overlap else 1]).encode()
    tools = [{"name": name, "parameters": {"type": "object", "properties": {}}} for name in ("inspect", "report")]
    content["tools/tools_schema.json"] = json.dumps({"tools": tools}).encode()
    for tool in tools:
        content[f"tools/toolslist/{tool['name']}.json"] = json.dumps(tool).encode()
    return content


def fetcher(content):
    def fetch(url):
        if "/api/" in url:
            return json.dumps({"sha": REVISION, "siblings": [{"rfilename": name} for name in content]}).encode()
        return content[url.split(REVISION + "/", 1)[1]]
    return fetch


def test_download_identity_and_audit_without_releasing_answers(tmp_path):
    destination = tmp_path / "release"
    source = download_release(destination, fetch=fetcher(payloads()))
    assert source["status"] == "downloaded"
    report = audit_release(destination)
    assert (report["cases"], report["train"], report["test"]) == (2, 1, 1)
    assert report["missing_images"] == 1
    assert not report["problems"]
    assert "private invented answer" not in json.dumps(report)
    with pytest.raises(ValueError, match="must be new"):
        download_release(destination, fetch=fetcher(payloads()))
    (destination / "data/all.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="bytes differ"):
        audit_release(destination)


def test_cycles_are_reported_without_changing_or_excluding_source_cases(tmp_path):
    destination = tmp_path / "release"
    download_release(destination, fetch=fetcher(payloads(cycle=True)))
    before = (destination / "data/all.jsonl").read_bytes()
    report = audit_release(destination)
    assert report["cases"] == 2
    assert len(report["problems"]) == 2
    assert all(item["problem"] == "cyclic DAG" for item in report["problems"])
    assert (destination / "data/all.jsonl").read_bytes() == before


@pytest.mark.parametrize("name", ["tools/../escape", "tools/../../absolute", "tools/E:/outside", "tools/images\\..\\escape"])
def test_download_refuses_unsafe_release_filenames_before_writing(tmp_path, name):
    content = payloads()
    content[name] = b"{}"
    with pytest.raises(ValueError, match="safe relative paths"):
        download_release(tmp_path / "release", fetch=fetcher(content))
    assert not (tmp_path / "release").exists()


def test_audit_refuses_split_overlap_and_unsafe_image_references(tmp_path):
    for name, options, message in (("overlap", {"overlap": True}, "disjoint"),
                                   ("traversal", {"traversal": True}, "safe relative paths")):
        destination = tmp_path / name
        download_release(destination, fetch=fetcher(payloads(**options)))
        with pytest.raises(ValueError, match=message):
            audit_release(destination)


def test_public_projection_keeps_hidden_actions_and_reference_answers_private(tmp_path):
    destination = tmp_path / "release"
    download_release(destination, fetch=fetcher(payloads()))
    release = FaultArenaRelease(destination)
    assert release.case_ids("train") == ("example_a",)
    public = {"task": release.task("example_a"), "tools": release.tools()}
    assert "private invented answer" not in json.dumps(public)
    assert "hidden_environment" not in json.dumps(public)
    assert "golden_dag" not in json.dumps(public)
    hidden = release.replay_case("example_a")
    hidden["initial_states"]["power"] = "OFF"
    assert release.task("example_a")["initial_states"]["power"] == "ON"
