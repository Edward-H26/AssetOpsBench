"""engine_record.py: the record of the serving engine keeps the allow-list of /server_info and the launch flags in public form,
and no other value of the answer, the key above all, reaches a line, a message or a published file."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from CausalDiagnosis.experiments.cluster_runs import engine_record
from CausalDiagnosis.experiments.run_verification import publish_check
from CausalDiagnosis.experiments.run_verification import public_record
from CausalDiagnosis.datasets.dataset_loader.recordings import REPO

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "server_info_v0519.json"
ANSWER = FIXTURE.read_text(encoding="utf-8")
FIELDS = json.loads(ANSWER)
PRIVATE = (FIELDS["api_key"], FIELDS["admin_api_key"], FIELDS["host"], "someuser", "/home/someuser")
MODEL, COMMIT = "ibm-granite/granite-3.3-8b-instruct", "b" * 40
LAUNCH = ["--model-path", FIELDS["model_path"], "--served-model-name", "granite-3.3-8b-instruct", "--host", FIELDS["host"],
          "--port", "30123", "--context-length", "131072", "--dtype", "bfloat16", "--api-key=" + FIELDS["api_key"],
          "--chat-template", "/home/someuser/run/agent_chat_template.jinja", "--enable-lora",
          "--lora-paths", "v1=/home/someuser/adapters/v1", "/home/someuser/adapters/v2", "--mem-fraction-static", "0.85"]


def _no_private(text):
    return [value for value in PRIVATE if value in text]


def test_the_record_holds_the_reported_fields_and_the_public_flags_and_nothing_else_of_the_answer():
    lines = engine_record.engine_lines(ANSWER, LAUNCH, MODEL, COMMIT)
    assert lines[0] == "engine_version 0.5.19"
    assert lines[1] == "dtype bfloat16 dtype_launch bfloat16 quantization none kv_cache_dtype auto"
    args = json.loads(lines[2].split(" ", 1)[1])
    assert args == ["--model-path", f"{MODEL}@{COMMIT}", "--served-model-name", "granite-3.3-8b-instruct", "--context-length", "131072",
                    "--dtype", "bfloat16", "--chat-template", "agent_chat_template.jinja", "--enable-lora", "--lora-paths", "v1", "v2",
                    "--mem-fraction-static", "0.85"]
    assert lines[3].split()[1] == __import__("hashlib").sha256(lines[2].split(" ", 1)[1].encode()).hexdigest()
    assert all(len(line.split()) % 2 == 0 for line in lines)  # pairs, as ledger.endpoint_file reads them
    assert _no_private("\n".join(lines)) == [] and "30123" not in lines[2].replace("131072", "")


def test_a_server_started_without_a_dtype_keeps_both_the_reported_value_and_auto():
    without = [arg for arg in LAUNCH if arg not in ("--dtype", "bfloat16")]
    auto = json.dumps({**FIELDS, "dtype": "auto"})
    assert engine_record.engine_lines(auto, without, MODEL, COMMIT)[1].startswith("dtype auto dtype_launch auto ")


@pytest.mark.parametrize("field", engine_record.REPORTED)
def test_a_missing_field_stops_the_record_with_a_message_that_names_the_field_and_no_value(field, capsys):
    answer = json.dumps({key: value for key, value in FIELDS.items() if key != field})
    with pytest.raises(engine_record.EngineRecordError) as raised:
        engine_record.engine_lines(answer, LAUNCH, MODEL, COMMIT)
    assert field in str(raised.value) and _no_private(str(raised.value)) == [] and raised.value.__context__ is None
    done = subprocess.run([sys.executable, str(Path(engine_record.__file__)), "--model-id", MODEL, "--revision", COMMIT, "--", *LAUNCH],
                          input=answer, capture_output=True, text=True)
    assert done.returncode == 3 and done.stdout == "" and field in done.stderr and _no_private(done.stderr + done.stdout) == []


def test_an_answer_that_is_no_json_object_names_no_byte_of_it():
    for answer in ("not json " + FIELDS["api_key"], json.dumps([FIELDS["api_key"]])):
        with pytest.raises(engine_record.EngineRecordError) as raised:
            engine_record.reported(answer)
        assert _no_private(str(raised.value)) == [] and (raised.value.__context__ is None or raised.value.__suppress_context__)


def test_the_public_form_drops_the_host_the_port_and_the_key_in_both_forms_and_rewrites_every_path():
    argv = ["--host=10.0.0.1", "--port=8000", "--api-key", "secret-a", "--admin-api-key=secret-b", "--model-path=/home/u/m",
            "--tokenizer-path", "/scratch/u/tok", "--endpoint", "http://10.0.0.1:9/v1"]
    out = engine_record.public_launch_args(argv, MODEL, COMMIT)
    assert out == [f"--model-path={MODEL}@{COMMIT}", "--tokenizer-path", "tok", "--endpoint", ""]
    assert not any(value in " ".join(out) for value in ("10.0.0.1", "8000", "secret-a", "secret-b", "/home/u", "/scratch/u"))


def test_the_path_rule_is_the_rule_of_public_record():
    root = REPO
    for path in (root / "hpc" / "env_example.sh", root / "benchmarks" / "hydraulic_rig" / "ledger.py", Path("/home/someuser/models/m"),
                 Path("/scratch/x/agent_chat_template.jinja")):
        assert engine_record.public_path(str(path), root) == public_record.public_path(path)


def test_the_record_passes_the_scan_of_the_published_set_and_a_record_with_any_private_value_fails_it(tmp_path):
    record = tmp_path / "endpoint_lines.txt"
    record.write_text("\n".join(engine_record.engine_lines(ANSWER, LAUNCH, MODEL, COMMIT)) + "\n", encoding="utf-8")
    assert publish_check.text_problems(record, ["someuser"]) == []
    for value in (FIELDS["host"], FIELDS["model_path"], "http://10.9.8.7:30123/v1"):
        leaked = tmp_path / "leaked.txt"
        leaked.write_text(record.read_text(encoding="utf-8") + f"note {value}\n", encoding="utf-8")
        assert publish_check.text_problems(leaked, ["someuser"]), value
