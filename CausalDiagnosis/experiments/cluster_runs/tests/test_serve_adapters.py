"""SERVE_ADAPTERS of CausalDiagnosis/cluster_jobs/2_serve_model/serve_model.sbatch (CausalDiagnosis/cluster_jobs/README.md section 17): the agent model's launch receives every
LoRA adapter of the training folder, so a fine tuned cell of the training table cannot play on a server without its adapter. The test runs
the script's own lines from the adapter block to the agent model's launch in bash, with a stand-in launcher that prints the
flags it was given. It also holds after the Agent-As-Tool port, which launches the agent model with a copy of the flags taken
before the launch line, so it fails when the adapters join the flags after that copy. A job whose one model is the judge serves
it with the judge's flags, and an agent model never receives them, with or without a co-served judge."""
from CausalDiagnosis.utils.paths import CLUSTER_JOBS
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = CLUSTER_JOBS / "2_serve_model" / "serve_model.sbatch"
START = "# With SERVE_ADAPTERS=1"
LAUNCH = 'launch_server "$MODEL_DIR" "$SERVED_NAME" "$PORT"'
CO_SERVE = 'if [ -n "$CO_SERVE_JUDGE" ]; then'
JUDGE_FLAGS = "--mem-fraction-static 0.45 --reasoning-parser qwen3"
JUDGE_ALONE_FLAGS = "--cuda-graph-max-bs 16 --reasoning-parser qwen3"
AGENT_EXTRA = "--mem-fraction-static 0.85"
# the flag that the script adds to every launch of the agent model, as the agent block of serving_of_record.json declares
CACHE_REPORT = "--enable-cache-report"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _launch_section(start=START):
    lines = SCRIPT.read_text(encoding="utf-8").splitlines()
    first = next(i for i, line in enumerate(lines) if line.startswith(start))
    last = next(i for i, line in enumerate(lines) if line.startswith(LAUNCH))
    assert first < last
    return "\n".join(lines[first:last + 1])


def _served_flags(tmp_path, adapters, serve="1"):
    for name in adapters:
        (tmp_path / "adapters" / name).mkdir(parents=True)
        (tmp_path / "adapters" / name / "adapter_config.json").write_text("{}", encoding="utf-8")
    prelude = "\n".join([
        'launch_server() { echo "FLAGS $4"; }', "free_port() { echo 30000; }",
        f'SERVE_ADAPTERS="{serve}" HYD_TRAINING="{tmp_path}" MODEL_ADAPTER="" SGLANG_EXTRA="--context-length 131072"',
        'MODEL_DIR=/models/granite SERVED_NAME=granite LOGS=/tmp L=/tmp FORK=/tmp SLURM_JOB_ID=0'])
    out = subprocess.run(["bash", "-c", f"{prelude}\n{_launch_section()}"], capture_output=True, text=True, check=True).stdout
    return [line[len("FLAGS "):] for line in out.splitlines() if line.startswith("FLAGS ")]


def test_the_agent_model_is_launched_with_every_adapter_of_the_training_folder(tmp_path):
    names = ["rft_causal_f1", "rft_causal_unfiltered_f1", "rft_plan_execute_f2"]
    [flags] = _served_flags(tmp_path, names)
    assert flags.startswith("--context-length 131072 --enable-lora --max-loras-per-batch 4 --lora-paths ")
    for name in names:
        assert f"{name}={tmp_path / 'adapters' / name}" in flags


def test_without_serve_adapters_the_agent_model_is_launched_without_lora(tmp_path):
    [flags] = _served_flags(tmp_path, ["rft_causal_f1"], serve="")
    assert flags == f"--context-length 131072 {CACHE_REPORT}"


def _launches(model, judge, co_serve):
    """{served name: flags} of every server the script launches from the co-served judge to the agent model's launch."""
    prelude = "\n".join([
        'launch_server() { echo "LAUNCH $2 FLAGS $4"; }', "free_port() { echo 30000; }", "server_up() { return 0; }",
        "write_endpoint() { :; }", "engine_record() { :; }",  # the endpoint file of the co-served judge and its engine lines
        'model_dir() { echo /tmp; }',  # an existing folder, which the co-served judge checks
        f'MODEL_DIR=/models/{model} SERVED_NAME={model} JUDGE_MODEL="{judge}" JUDGE_SERVED_NAME="{judge}" CO_SERVE_JUDGE="{co_serve}"',
        f'JUDGE_SGLANG_EXTRA="{JUDGE_FLAGS}" JUDGE_ALONE_SGLANG_EXTRA="{JUDGE_ALONE_FLAGS}"',
        f'SGLANG_EXTRA="{AGENT_EXTRA if co_serve else ""}" SERVE_ADAPTERS="" MODEL_ADAPTER=""',
        "LOGS=/tmp L=/tmp FORK=/tmp SLURM_JOB_ID=0 JUDGE_UP_TRIES=1 JUDGE_ENDPOINT_FILE=/tmp/judge.txt"])
    out = subprocess.run(["bash", "-c", f"{prelude}\n{_launch_section(CO_SERVE)}"], capture_output=True, text=True, check=True).stdout
    return {line.split()[1]: line.split(" FLAGS ", 1)[1].strip() for line in out.splitlines() if line.startswith("LAUNCH ")}


def test_a_judge_served_alone_receives_the_judges_flags():
    """A judge job or the reflection model of the GEPA loops: MODEL and JUDGE_MODEL name the judge and nothing is co-served, so
    the judge takes the flags of a judge served alone, without the memory share of two models on one card."""
    assert _launches("qwen3-judge", "qwen3-judge", co_serve="") == {"qwen3-judge": JUDGE_ALONE_FLAGS}


@pytest.mark.parametrize("co_serve", ["", "1"])
def test_an_agent_model_never_receives_the_judges_flags(co_serve):
    """The agent model keeps SGLANG_EXTRA and the cache report alone, and a co-served judge keeps the judge's flags as before."""
    launches = _launches("granite", "qwen3-judge", co_serve)
    assert launches["granite"] == (f"{AGENT_EXTRA} {CACHE_REPORT}" if co_serve else CACHE_REPORT)
    assert launches.get("qwen3-judge") == (JUDGE_FLAGS if co_serve else None)
