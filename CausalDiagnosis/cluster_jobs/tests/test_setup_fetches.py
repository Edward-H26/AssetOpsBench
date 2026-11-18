"""The fetches of the setup chain on a link that breaks off and stalls, and the order and the ends of its jobs: one rule for the
shell (retry_fetch of functions.sh) and for Python (CausalDiagnosis/utils/download_retry.py), the caches that a job leaves on
scratch for the next one, the downloads after the environment job whose uv they fetch with, a failed job whose dependents leave
the queue, the image of a framework pulled in the setup and the data check before the metering check. The scripts run under
the stand-ins of test_cluster_jobs.py, and every fetch of Python against a local server that breaks off, stalls or fails; no
network, no Slurm and no image take part."""
from CausalDiagnosis.utils.paths import CLUSTER_JOBS, PACKAGE, ROOT
import hashlib
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import urllib.request

import pytest

from CausalDiagnosis.cluster_jobs.tests.test_cluster_jobs import (APPTAINER_PULL, DOWNLOAD, HUB_LISTING, PINS, README, _in, _jobs, _module_file,
                                                                _record, _run, _services, _sourced, _stand_in, _tree)
from CausalDiagnosis.cluster_jobs.tests.test_setup_and_guard import _call_of, _setup, _setup_tree, _submitted, _write_logs
from CausalDiagnosis.utils import download_retry
from CausalDiagnosis.utils.tests.test_download_retry import BODY, PlannedServer

FUNCTIONS = (CLUSTER_JOBS / "functions.sh").read_text(encoding="utf-8")
AGENT = "ibm-granite/granite-3.3-8b-instruct"
AOB_JOBS = PACKAGE / "frameworks" / "AssetOpsBench" / "cluster_jobs"


@pytest.fixture
def served():
    """served(body, plan) starts a planned server of test_download_retry.py; every server stops at the end of the test."""
    servers = []

    def start(body=BODY, plan=()):
        servers.append(PlannedServer(body, plan))
        return servers[-1]

    yield start
    for server in servers:
        server.close()


def _job(tree, name, env, *arguments):
    return subprocess.run(["bash", str(_in(tree, name)), *arguments], capture_output=True, text=True, cwd=tree.parent, env=env, timeout=120)


# ----- one rule of the fetches, in the shell and in Python -----

def test_the_shell_and_the_python_of_a_job_fetch_under_one_rule():
    """functions.sh and download_retry.py read FETCH_TIMEOUT, FETCH_RETRIES and FETCH_PAUSE with the same defaults, and the
    jobs export them, so the Python of a job fetches under the rule of its shell."""
    shell = dict(re.findall(r"^(FETCH_TIMEOUT|FETCH_RETRIES|FETCH_PAUSE)=\$\{\1:-(\d+)\}$", FUNCTIONS, re.M))
    assert shell == {"FETCH_TIMEOUT": str(download_retry.TIMEOUT_S), "FETCH_RETRIES": str(download_retry.RETRIES),
                     "FETCH_PAUSE": str(download_retry.PAUSE_S)}
    assert "export FETCH_TIMEOUT FETCH_RETRIES FETCH_PAUSE" in FUNCTIONS and re.search(r"^CACHE=\$\{CACHE:-\$ROOT/cache\}$", FUNCTIONS, re.M)


def test_retry_fetch_takes_a_fetch_up_while_it_brings_bytes_and_gives_up_after_the_retries_that_bring_nothing(tmp_path):
    """A fetch that breaks off four times, each time with more bytes under its folder, ends; one that brings nothing ends with
    the status of its last attempt after FETCH_RETRIES retries, and no retry line shows the arguments of the command."""
    tree, _root, _fake, env = _tree(tmp_path)
    moving = 'moving() { local n; n=$(ls "$1" | wc -l | tr -d " "); [ "$n" -ge 4 ] && return 0; head -c 9000 /dev/zero > "$1/part$n"; return 9; }'
    still = 'still() { echo try >> "$1"; return 7; }'
    done = _run(tmp_path, {**env, "FETCH_RETRIES": "2", "FETCH_PAUSE": "0"}, *_sourced(
        tree, moving, still, f'mkdir -p "{tmp_path}/m"', f'retry_fetch "{tmp_path}/m" moving "{tmp_path}/m"; echo "MOVING $?"',
        f'retry_fetch "{tmp_path}/nothing" still "{tmp_path}/tries"; echo "STILL $?"'))
    assert "MOVING 0" in done.stdout and "STILL 7" in done.stdout, done.stderr
    assert (tmp_path / "tries").read_text(encoding="utf-8").splitlines() == ["try"] * 3
    assert done.stderr.count("FETCH_RETRY ") == 6 and "FETCH_FAILED still with status 7 after 2 retries that brought nothing" in done.stderr
    assert str(tmp_path / "tries") not in done.stderr


# ----- the environment jobs: a sync or an install that breaks off is taken up from the cache under ROOT -----

UV_FLAKY = f"""echo "$*" >> "$FAKE_LOG/uv_calls"
env | grep -E '^(UV_CACHE_DIR|UV_HTTP_TIMEOUT)=' >> "$FAKE_LOG/uv_env"
fetch() {{  # a fetch of uv: a wheel more in the cache unless FAKE_NO_BYTES, and a failure until FAKE_BREAKS wheels are there
  local n; n=$(ls "$UV_CACHE_DIR" | wc -l | tr -d ' ')
  [ -n "${{FAKE_NO_BYTES:-}}" ] || head -c 9000 /dev/zero > "$UV_CACHE_DIR/wheel$n"
  [ "$n" -ge "${{FAKE_BREAKS:-0}}" ] || {{ echo "error: Request failed after 3 retries: operation timed out" >&2; exit 2; }}
}}
case "$1 ${{2:-}}" in
  "--version "*) echo "uv 0.11.7" ;;
  "python install") ;;
  "sync "*) mkdir -p "$UV_PROJECT_ENVIRONMENT/bin"; printf '#!/bin/bash\\nexec "{sys.executable}" "$@"\\n' > "$UV_PROJECT_ENVIRONMENT/bin/python"
            chmod +x "$UV_PROJECT_ENVIRONMENT/bin/python"; fetch ;;
  "pip install") fetch ;;
  "pip freeze") echo "package==1" ;;
esac"""


def _uv_tree(tmp_path, script):
    tree, root, _fake, env = _tree(tmp_path, (script,))
    log = tmp_path / "log"
    log.mkdir()
    (root / "tools" / "uv-0.11.7").mkdir(parents=True)
    _stand_in(root / "tools" / "uv-0.11.7", "uv", UV_FLAKY)
    return tree, root, {**env, "FAKE_LOG": str(log), "FETCH_PAUSE": "0"}


def test_the_environment_job_takes_up_a_sync_that_broke_off_from_the_cache_that_it_keeps_under_root(tmp_path):
    """The sync of the lock breaks off once after a wheel arrived: the job syncs again and ends with CD_SETUP_DONE, uv keeps its
    cache under ROOT, where the wheels stay after the job for the next one, and it fetches with the timeout of the fetches. The
    control: a sync that never brings a byte ends the job with status 3 after the retries of the rule."""
    tree, root, env = _uv_tree(tmp_path, "setup_env.sbatch")
    shutil.rmtree(root / "envs")
    done = _job(tree, "setup_env.sbatch", {**env, "FAKE_BREAKS": "1"})
    assert done.returncode == 0 and done.stdout.splitlines()[-1] == "CD_SETUP_DONE", done.stderr
    calls = (tmp_path / "log" / "uv_calls").read_text(encoding="utf-8").splitlines()
    assert len([call for call in calls if call.startswith("sync ")]) == 2 and "FETCH_RETRY 1 of 5 in 0 s, uv ended with status 2" in done.stderr
    assert sorted(p.name for p in (root / "cache" / "uv").iterdir()) == ["wheel0", "wheel1"]
    assert {f"UV_CACHE_DIR={root}/cache/uv", "UV_HTTP_TIMEOUT=300"} <= set((tmp_path / "log" / "uv_env").read_text(encoding="utf-8").splitlines())
    shutil.rmtree(root / "envs")
    done = _job(tree, "setup_env.sbatch", {**env, "FAKE_BREAKS": "99", "FAKE_NO_BYTES": "1", "FETCH_RETRIES": "1"})
    assert done.returncode == 3 and "CD_SETUP_DONE" not in done.stdout and "FETCH_FAILED uv with status 2 after 1 retries" in done.stderr


def test_the_training_environment_takes_up_an_install_that_timed_out(tmp_path):
    """The first install of the training environment times out at the package index, as on the cluster: the job installs again
    from the cache under ROOT and ends with CD_TRAIN_SETUP_DONE."""
    tree, root, env = _uv_tree(tmp_path, "setup_train_env.sbatch")
    shutil.copy(CLUSTER_JOBS / "0_setup" / "requirements-train.txt", _jobs(tree) / "0_setup" / "requirements-train.txt")
    venv = root / "train_venv" / "bin"
    venv.mkdir(parents=True)
    _stand_in(venv, "python", 'echo "torch 2.14.1 cuda 12.6"')
    done = _job(tree, "setup_train_env.sbatch", {**env, "FAKE_BREAKS": "1"})
    assert done.returncode == 0 and done.stdout.splitlines()[-1] == "CD_TRAIN_SETUP_DONE", done.stderr
    installs = [call for call in (tmp_path / "log" / "uv_calls").read_text(encoding="utf-8").splitlines() if call.startswith("pip install")]
    assert len(installs) == 3 and installs[0] == installs[1] and "torch==2.14.1" in installs[0] and "operation timed out" in done.stderr
    assert f"UV_CACHE_DIR={root}/cache/uv" in (tmp_path / "log" / "uv_env").read_text(encoding="utf-8").splitlines()


# ----- the model downloads: after the environment job, a folder made before its free space is read, a broken download -----

def test_the_model_downloads_follow_the_environment_job_whose_uv_they_fetch_with(tmp_path):
    """setup.sh submits both downloads and the data job after the environment job, and the image and the environment at once.
    The control: without uvx on its path a download job stops by name before it fetches, and does not fail later in the fetch."""
    tree, _root, _fake, env, calls = _setup_tree(tmp_path)
    done = _setup(tmp_path, tree, env)
    assert done.returncode == 0, done.stderr
    environment = _submitted(done.stdout)["setup_env.sbatch"][0]
    after = [*_call_of(calls, "download_model.sbatch"), *_call_of(calls, "fetch_data.sbatch")]
    assert len(after) == 3 and all(f"--dependency=afterok:{environment} --kill-on-invalid-dep=yes" in line for line in after)
    assert all("--dependency" not in line for script in ("pull_image.sbatch", "setup_env.sbatch") for line in _call_of(calls, script))
    tree, _root, fake, env = _tree(tmp_path / "alone", ("download_model.sbatch",))
    _stand_in(fake, "python3", f'exec "{sys.executable}" "$@"')
    bare = os.pathsep.join(entry for entry in env["PATH"].split(os.pathsep) if not os.access(os.path.join(entry, "uvx"), os.X_OK))
    done = _job(tree, "download_model.sbatch", {**env, "PATH": bare, "MODELS": str(tmp_path / "models"), "SHARED": "", "MODEL": AGENT})
    assert done.returncode == 2 and "run setup_env.sbatch first" in done.stderr and not (tmp_path / "models" / AGENT.split("/")[1]).exists(), done.stderr


DOWNLOAD_FLAKY = DOWNLOAD.replace('[ -z "${FAKE_FAIL:-}" ] || exit 1', """env | grep -E '^HF_HUB_(DOWNLOAD|ETAG)_TIMEOUT=' >> "$FAKE_LOG/hf_env"
n=$(cat "$FAKE_LOG/attempts" 2>/dev/null || echo 0); echo $((n + 1)) > "$FAKE_LOG/attempts"
if [ "$n" -lt "${FAKE_BREAKS:-0}" ]; then head -c 9000 /dev/zero > "$target/shard$n.incomplete"; exit 1; fi""")


def test_a_model_download_makes_its_folder_before_its_free_space_and_takes_up_a_download_that_broke_off(tmp_path):
    """A download into a models folder that no job has made yet reads the free space of a folder that exists, so its log holds
    no error of df, and a download that broke off after bytes arrived is made again, with the timeouts of the fetches handed to
    huggingface_hub, and ends with CD_DOWNLOAD_DONE."""
    tree, _root, fake, env = _tree(tmp_path, ("download_model.sbatch",))
    log = tmp_path / "log"
    log.mkdir()
    _stand_in(fake, "uvx", DOWNLOAD_FLAKY)
    _stand_in(fake, "curl", HUB_LISTING)
    models = tmp_path / "fresh" / "models"
    done = _job(tree, "download_model.sbatch", {**env, "FAKE_LOG": str(log), "MODELS": str(models), "SHARED": "", "MODEL": AGENT,
                                                 "FAKE_COMMIT": PINS[AGENT], "FAKE_BREAKS": "1", "FETCH_PAUSE": "0"})
    assert done.returncode == 0 and f"CD_DOWNLOAD_DONE {models / AGENT.split('/')[1]}" in done.stdout, done.stderr
    assert "No such file or directory" not in done.stderr and (log / "attempts").read_text(encoding="utf-8").strip() == "2"
    assert set((log / "hf_env").read_text(encoding="utf-8").splitlines()) == {"HF_HUB_DOWNLOAD_TIMEOUT=300", "HF_HUB_ETAG_TIMEOUT=300"}


# ----- a failed setup job takes the jobs after it out of the queue -----

SLURM_QUEUE = """import re, sys
from pathlib import Path
# the queue of Slurm for squeue -h -j JOB -o %T: a job that waits for a job that failed, or that the scheduler ended, stays
# PENDING for ever unless it was submitted with --kill-on-invalid-dep=yes, which ends it; every other job has left the queue
calls, failed = Path(sys.argv[1]).read_text().splitlines(), set(Path(sys.argv[2]).read_text().split())

def state(job):
    line = calls[int(job) - 101]
    waits = re.search(r"--dependency=afterok:(\\d+)", line)
    if not waits:
        return "ended"
    before = state(waits.group(1))
    if before == "PENDING":
        return "PENDING"
    if waits.group(1) in failed or before == "cancelled":
        return "cancelled" if "--kill-on-invalid-dep=yes" in line else "PENDING"
    return "ended"

found = state(sys.argv[sys.argv.index("-j") + 1])
print("PENDING" if found == "PENDING" else "", end="")
"""


def test_a_failed_setup_job_takes_the_jobs_after_it_out_of_the_queue_so_that_a_second_setup_submits_them_again(tmp_path):
    """The data job fails. Under the rule of Slurm, which leaves a job whose dependency can never be met pending for ever, the
    tables job and the digest job leave the queue only because setup.sh submitted them with --kill-on-invalid-dep=yes, so
    setup.sh status names them FAILED and a second setup.sh submits the data, the tables and the digests again."""
    tree, root, fake, env, calls = _setup_tree(tmp_path)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    assert _setup(tmp_path, tree, env).returncode == 0
    record = dict(row.split()[:2] for row in next((root / "setup").glob("*.jobs")).read_text(encoding="utf-8").splitlines())
    (tmp_path / "failed").write_text(record["data"], encoding="utf-8")
    _write_logs(root, {"image", "agent_model", "judge_model", "environment"}, failed={"data"})
    model = tmp_path / "slurm_queue.py"
    model.write_text(SLURM_QUEUE, encoding="utf-8")
    _stand_in(fake, "squeue", f'exec "{sys.executable}" "{model}" "{calls}" "{tmp_path / "failed"}" "$@"')
    status = _setup(tmp_path, tree, env, "status")
    assert "PENDING" not in status.stdout, status.stdout
    assert re.search(rf"^tables {record['tables']} FAILED", status.stdout, re.M) and re.search(rf"^digests {record['digests']} FAILED", status.stdout, re.M)
    again = _setup(tmp_path, tree, env)
    assert again.returncode == 0, again.stderr
    assert set(_submitted(again.stdout)) == {"fetch_data.sbatch", "precompute_tables.sbatch", "cpu_job.sbatch"}


# ----- the images: the image job pulls those of the frameworks, a pull that broke off goes on from its cache -----

def test_the_image_job_pulls_the_images_of_the_frameworks_even_beside_a_shared_sglang_image(tmp_path):
    """With the SGLang image readable in SHARED the job pulls no SGLang image and still calls every function of IMAGE_PULLS that
    the job_services.sh of a framework lists, with its node local folder, before CD_IMAGE_DONE."""
    tree, _root, fake, env = _tree(tmp_path, ("pull_image.sbatch",))
    _services(tree).write_text('pull_stand_in() { echo "PULLED into $1"; }\nIMAGE_PULLS+=(pull_stand_in)\n', encoding="utf-8")
    _stand_in(fake, "apptainer", APPTAINER_PULL)
    _stand_in(fake, "module", "exit 0")
    sif = tmp_path / "shared" / "images" / "sglang.sif"
    sif.parent.mkdir(parents=True)
    sif.write_text("image", encoding="utf-8")
    done = _job(tree, "pull_image.sbatch", {**env, "SHARED": str(tmp_path / "shared"), "SIF": str(sif)})
    assert done.returncode == 0 and done.stdout.splitlines()[-1] == f"CD_IMAGE_DONE {sif}", done.stderr
    assert any(line.startswith("PULLED into ") for line in done.stdout.splitlines()), done.stdout


APPTAINER_FLAKY = """echo "$1 $2 CACHE=${APPTAINER_CACHEDIR:-}" >> "$FAKE_LOG/apptainer_calls"
case "$1" in
  pull) n=$(ls "$APPTAINER_CACHEDIR" | wc -l | tr -d ' '); head -c 9000 /dev/zero > "$APPTAINER_CACHEDIR/layer$n"
        [ "$n" -ge "${FAKE_BREAKS:-0}" ] || { echo "FATAL: while making image: connection reset by peer" >&2; exit 255; }
        echo image > "$2" ;;
esac"""


def test_the_couchdb_image_is_pulled_in_the_setup_and_a_pull_that_broke_off_goes_on_from_its_layers(tmp_path):
    """The AssetOpsBench rows list couchdb_image in IMAGE_PULLS, and its pull breaks off once after a layer arrived: the next
    attempt goes on from the cache of the image under CACHE, the image takes its name only once it is whole, and the cache
    and the partial file are gone once it is in place."""
    tree, root, fake, env = _tree(tmp_path)
    framework = _jobs(tree).parent / "frameworks" / "AssetOpsBench" / "cluster_jobs"
    framework.mkdir(parents=True)
    for name in ("job_services.sh", "start_couchdb.sh"):
        shutil.copy(AOB_JOBS / name, framework / name)
    log = tmp_path / "log"
    log.mkdir()
    _stand_in(fake, "apptainer", APPTAINER_FLAKY)
    _stand_in(fake, "module", "exit 0")
    images = tmp_path / "images"
    done = _run(tmp_path, {**env, "FAKE_LOG": str(log), "IMAGES": str(images), "SHARED": "", "FAKE_BREAKS": "1", "FETCH_PAUSE": "0"},
                *_sourced(tree, 'echo "PULLS ${IMAGE_PULLS[*]}"', f'couchdb_image "{tmp_path}/local"'))
    assert done.returncode == 0 and "PULLS couchdb_image" in done.stdout, done.stderr
    assert (images / "couchdb-3.5.sif").read_text(encoding="utf-8") == "image\n" and sorted(p.name for p in images.iterdir()) == ["couchdb-3.5.sif"]
    pulls = (log / "apptainer_calls").read_text(encoding="utf-8").splitlines()
    assert len(pulls) == 2 and all(f"CACHE={root}/cache/apptainer/couchdb-3.5.sif" in line for line in pulls) and "FETCH_RETRY 1 of 5" in done.stderr
    assert not (root / "cache" / "apptainer" / "couchdb-3.5.sif").exists()


# ----- the metering check reads the data, so it checks the data first -----

def test_the_metering_check_stops_on_missing_data_before_it_starts_couchdb(tmp_path):
    """Without the data of record the conformance job stops with the line of the data check, which names the data job, before
    CouchDB starts; the control: with the data in place the same job goes on to CouchDB."""
    tree, _root, _fake, env = _tree(tmp_path, ("check_metering_layer.sh",))
    _services(tree).write_text('start_couchdb() { echo START_COUCHDB; COUCH_PID=; }\n', encoding="utf-8")
    (tree / "couchdb3.py").write_text("", encoding="utf-8")  # the package that the job asks for, as a stand-in
    done = _job(tree, "check_metering_layer.sh", {**env, "FAKE_DATA_CHECK": "1"})
    assert done.returncode == 2 and "run fetch_data.sbatch" in done.stderr and "START_COUCHDB" not in done.stdout, (done.stdout, done.stderr)
    done = _job(tree, "check_metering_layer.sh", env)
    assert "START_COUCHDB" in done.stdout.splitlines(), (done.stdout, done.stderr)


# ----- the Python fetches of the setup chain against a local server -----

def test_the_data_job_takes_up_a_download_of_the_archive_that_broke_off(served, tmp_path, monkeypatch):
    """The stream of the archive breaks off after its first bytes, as it did on a compute node: the download keeps them in the
    partial file and asks for the rest, and the archive takes its name once its SHA-256 is the pin."""
    from CausalDiagnosis.datasets.hyd_1 import download_and_build as fu
    server = served(plan=[("break", 5000)])
    monkeypatch.setattr(fu, "URL", server.url)
    monkeypatch.setenv("FETCH_PAUSE", "0")
    archive = fu.archive_for(tmp_path / "data", {fu.ARCHIVE_LINE: hashlib.sha256(BODY).hexdigest()})
    assert archive.read_bytes() == BODY and server.requests == [None, "bytes=5000-"]
    assert sorted(p.name for p in (tmp_path / "data").iterdir()) == [fu.ARCHIVE_NAME]


def test_the_data_job_takes_up_a_download_of_the_archive_that_stalled(served, tmp_path, monkeypatch):
    """The stream stalls after its first bytes: the attempt ends at FETCH_TIMEOUT, here one second, and the next one asks for
    the rest, long before the stall itself ends."""
    from CausalDiagnosis.datasets.hyd_1 import download_and_build as fu
    server = served(plan=[("stall", 4000, 30)])
    monkeypatch.setattr(fu, "URL", server.url)
    monkeypatch.setenv("FETCH_PAUSE", "0")
    monkeypatch.setenv("FETCH_TIMEOUT", "1")
    archive = fu.archive_for(tmp_path / "data", {fu.ARCHIVE_LINE: hashlib.sha256(BODY).hexdigest()})
    assert archive.read_bytes() == BODY and server.requests == [None, "bytes=4000-"]


def test_the_scenario_fetch_of_the_data_job_is_made_again_after_a_failure_of_the_server(served, monkeypatch):
    from CausalDiagnosis.frameworks.AssetOpsBench.scenarios import download_scenarios as fetch_scenarios
    server = served(body=b"the scenario file", plan=[("status", 503)])
    monkeypatch.setattr(fetch_scenarios, "source_url", lambda: server.url)
    monkeypatch.setenv("FETCH_PAUSE", "0")
    assert fetch_scenarios.fetch() == b"the scenario file" and len(server.requests) == 2


def test_the_reproduction_fetches_the_scenario_file_of_the_paper_against_its_pin(served, tmp_path, monkeypatch):
    """The job of the reproduction fetches the file through fetch_scenario_file: a download that broke off goes on, a file of
    another SHA-256 at the path is fetched again and replaced, a file at its pin is not fetched again, and a download of
    another SHA-256 is refused and leaves no file."""
    from CausalDiagnosis.frameworks.AssetOpsBench.reproduction import scenario_file
    script = (AOB_JOBS / "run_reproduction.sh").read_text(encoding="utf-8")
    assert "fetch_scenario_file" in script and "curl -sSfL" not in script
    server = served(plan=[("break", 5000)])
    monkeypatch.setattr(scenario_file, "SCENARIO_URL", server.url)
    monkeypatch.setattr(scenario_file, "SCENARIO_SHA256", hashlib.sha256(BODY).hexdigest())
    monkeypatch.setenv("FETCH_PAUSE", "0")
    path = tmp_path / "scenarios.jsonl"
    path.write_bytes(b"another file")
    assert scenario_file.fetch_scenario_file(path).read_bytes() == BODY and server.requests == [None, "bytes=5000-"]
    assert scenario_file.fetch_scenario_file(path) == path and len(server.requests) == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["scenarios.jsonl"]
    monkeypatch.setattr(scenario_file, "SCENARIO_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="not the pinned"):
        scenario_file.fetch_scenario_file(tmp_path / "other.jsonl")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["scenarios.jsonl"]


def test_a_tree_of_the_reproduction_is_fetched_again_after_a_failed_fetch(tmp_path, monkeypatch):
    """The shallow fetch of a pinned commit fails once: it is made again under the rule of the fetches, with the timeout of the
    fetches as the time git waits on a stalled transfer, and the paths of the commit arrive."""
    from CausalDiagnosis.frameworks.AssetOpsBench.reproduction import fetch_public_servers_and_data as layer
    if shutil.which("git") is None:
        pytest.skip("needs git")
    origin = tmp_path / "origin"
    (origin / "src" / "servers").mkdir(parents=True)
    (origin / "src" / "servers" / "probe.py").write_text("PROBE = 1\n", encoding="utf-8")
    git = ["git", "-C", str(origin), "-c", "user.name=t", "-c", "user.email=t@example.org"]
    alone = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    for argv in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "one"]):
        subprocess.run([*git, *argv], check=True, capture_output=True, env=alone)
    commit = subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True, env=alone).stdout.strip()
    run, fetches = subprocess.run, []

    def flaky(argv, *args, **kwargs):
        if "fetch" in argv:
            fetches.append(argv)
            if len(fetches) == 1:
                raise subprocess.CalledProcessError(128, argv)
        return run(argv, *args, **kwargs)

    monkeypatch.setattr(layer.subprocess, "run", flaky)
    monkeypatch.setenv("FETCH_PAUSE", "0")
    tree = layer.fetch_tree(commit, ("src/servers",), tmp_path / "out", repository=f"file://{origin}")
    assert (tmp_path / "out" / "src" / "servers" / "probe.py").read_text(encoding="utf-8") == "PROBE = 1\n" and len(fetches) == 2
    assert re.fullmatch(r"[0-9a-f]{40}", tree) and "http.lowSpeedTime=300" in fetches[0]


# ----- the optional data of the data job: the scenario file of the anchored preset -----

def test_no_step_of_the_setup_reads_the_scenario_file_of_the_anchored_preset(tmp_path, monkeypatch):
    """The fact that makes the scenario fetch optional: without the file, every digest family that covers the anchored preset
    (config, starts, logs and verdicts, texts, the episodes of the AssetOpsBench rows) runs on it, the scripted family of server
    processes leaves it out, and the tables job resolves a line of it, while nothing asks for the file; only the plan of a run
    of the preset reads it."""
    from CausalDiagnosis.experiments.run_verification import invariant_digests as digests
    from CausalDiagnosis.frameworks.AssetOpsBench.scenarios import scenario_anchors as anchors
    from CausalDiagnosis.frameworks.AssetOpsBench.tool_servers import digest_families as families
    from CausalDiagnosis.experiments.cluster_runs import precompute_grid_tables as precompute_tables
    from CausalDiagnosis.experiments.episode_runners.run_configuration import load_world
    from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path

    if not default_facts_path().exists():
        pytest.skip("needs the data file of record")
    monkeypatch.setenv(anchors.SCENARIOS_ENV, str(tmp_path / "absent.json"))
    monkeypatch.setenv("CD_TABLES_READ_ONLY", "1")
    asked = []
    real = anchors.load_scenarios
    monkeypatch.setattr(anchors, "load_scenarios", lambda *args, **kwargs: asked.append(args) or real(*args, **kwargs))
    assert "anchored" in digests.presets() and "anchored" not in digests.process_presets()
    for module in (digests, families):
        monkeypatch.setattr(module, "presets", lambda: ["anchored"])
    found, _detail = digests.compute(default_facts_path(), families=("config", "starts", "logs", "texts"))
    rows = families.rows_digests(default_facts_path())
    assert {key.split(".")[0] for key in found if ".anchored" in key} == {"config", "starts", "logs", "verdicts", "texts"}
    assert rows and all(".anchored." in key for key in rows)
    grid = tmp_path / "grid.txt"
    grid.write_text("--config anchored --agent scripted\n", encoding="utf-8")
    _facts, _cycles, blocks = load_world()
    assert [config.anchors is not None for _name, config in precompute_tables.configurations([str(grid)], [], blocks=blocks)] == [True]
    assert asked == []


def test_a_failed_fetch_of_optional_data_names_its_dataset_and_the_data_job_goes_on(tmp_path):
    """A fetch of OPTIONAL_DATA_FETCHES, the name of its dataset and then its command, runs after the data check; one that fails
    prints one warning that names its dataset, the job goes on and closes with CD_DATA_DONE and the optional datasets missing,
    and one that succeeds is not listed. The control: a fetch of DATA_FETCHES that fails stops the job with status 3 before the
    data check. The AssetOpsBench rows list the scenario fetch as optional, under the dataset that the fetch pins."""
    from CausalDiagnosis.frameworks.AssetOpsBench.scenarios import download_scenarios

    tree, _root, _fake, env = _tree(tmp_path, ("fetch_data.sbatch",))
    services = _services(tree)
    services.write_text('OPTIONAL_DATA_FETCHES+=("org/kept true" "org/lost false optional_fetch" "org/gone false")\n', encoding="utf-8")
    done = _job(tree, "fetch_data.sbatch", env)
    warnings = [line for line in done.stderr.splitlines() if line.startswith("WARNING ")]
    assert done.returncode == 0 and len(warnings) == 2 and "org/kept" not in done.stderr, done.stderr
    assert warnings[0].startswith("WARNING the optional dataset org/lost is missing: its fetch failed")
    assert done.stdout.splitlines()[-1] == "CD_DATA_DONE optional datasets missing: org/lost org/gone"
    services.write_text('DATA_FETCHES+=("false required_fetch")\n', encoding="utf-8")
    done = _job(tree, "fetch_data.sbatch", env)
    assert done.returncode == 3 and "CD_DATA_DONE" not in done.stdout and "WARNING" not in done.stderr
    text = (AOB_JOBS / "job_services.sh").read_text(encoding="utf-8")
    entry = f'OPTIONAL_DATA_FETCHES+=("{download_scenarios.DATASET} python -m {download_scenarios.__name__}")'
    assert entry in text.splitlines() and "DATA_FETCHES+=(" not in text.replace("OPTIONAL_DATA_FETCHES+=(", "")


def test_setup_status_reads_the_closing_line_with_the_optional_datasets_missing_as_done(tmp_path):
    """setup.sh status counts the data job done on the closing line that lists the optional datasets missing, prints that line
    and prints SETUP_OK once the seven jobs closed."""
    tree, root, _fake, env, _calls = _setup_tree(tmp_path)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    assert _setup(tmp_path, tree, env).returncode == 0
    _write_logs(root, {"image", "agent_model", "judge_model", "environment", "data", "tables", "digests"})
    closing = "CD_DATA_DONE optional datasets missing: ibm-research/AssetOpsBench"
    for row in next((root / "setup").glob("*.jobs")).read_text(encoding="utf-8").splitlines():
        step, _job_id, log = row.split()
        if step == "data":
            with open(log, "w", encoding="utf-8") as out:
                out.write(f"working\n{closing}\n")
    done = _setup(tmp_path, tree, env, "status")
    assert done.returncode == 0 and done.stdout.splitlines()[-1] == "SETUP_OK", (done.stdout, done.stderr)
    assert re.search(rf"^data \d+ {re.escape(closing)}$", done.stdout, re.M), done.stdout


def test_a_run_of_the_anchored_preset_without_its_scenario_file_stops_at_its_start_with_the_fetch_command(tmp_path, monkeypatch):
    """Where the optional fetch failed, a line of the anchored preset stops before its run folder exists, with the command that
    fetches the file in its message, through the driver of every grid line."""
    from CausalDiagnosis.experiments.episode_runners import run_episodes
    from CausalDiagnosis.frameworks.AssetOpsBench.scenarios import scenario_anchors as anchors
    from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path

    if not default_facts_path().exists():
        pytest.skip("needs the data file of record")
    monkeypatch.setenv(anchors.SCENARIOS_ENV, str(tmp_path / "absent.json"))
    out = tmp_path / "runs"
    with pytest.raises(FileNotFoundError, match=re.escape(anchors.FETCH_COMMAND)):
        run_episodes.main(["--config", "anchored", "--agent", "scripted", "--seeds", "1", "--out", str(out)])
    assert not out.exists()


# ----- the opening call reads the thinking rate where the asset file keeps it -----

# a stand-in of the asset loader for the opening call: three assets, one whose profile cannot load, one without a clock and hyd_1
LOADER_STAND_IN = {
    "CausalDiagnosis/__init__.py": "",
    "CausalDiagnosis/datasets/__init__.py": "",
    "CausalDiagnosis/datasets/dataset_loader/__init__.py": "",
    "CausalDiagnosis/datasets/dataset_loader/dataset_packages.py": "def asset_ids():\n    return ['broken', 'plain', 'hyd_1']\n",
    "CausalDiagnosis/datasets/dataset_loader/profile_registry.py": (
        "from types import SimpleNamespace\n\n\ndef get_profile(asset_id):\n"
        "    think = {'plain': None, 'hyd_1': 10}[asset_id]\n"
        "    return SimpleNamespace(default_config=lambda: SimpleNamespace(think_seconds_per_cycle=think))\n"),
}


def _opening_call(fork):
    """The opening call of run_lines.sh, run as the job runs it with the tree's Python and FORK, against a port where nothing
    listens: the rates of its OPENING_CALL line, {asset: rate}, and the seconds of the call."""
    script = (CLUSTER_JOBS / "3_run" / "run_lines.sh").read_text(encoding="utf-8")
    call = script.index('python - "$(endpoint_url')
    code = script[script.index("<<'PY'\n", call) + len("<<'PY'\n"):script.index("\nPY\n", call)]
    env = {"PATH": os.environ["PATH"], "PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "src")])}
    done = subprocess.run([sys.executable, "-", "http://127.0.0.1:1/v1", "m", str(fork)], input=code, capture_output=True, text=True,
                          cwd=ROOT, env=env, timeout=300)
    line = next((line for line in done.stdout.splitlines() if line.startswith("OPENING_CALL ")), done.stdout + done.stderr[-2000:])
    found = re.fullmatch(r"OPENING_CALL failed \w+ seconds ([0-9.]+) cycles (.*)", line)
    assert found, line
    return dict((*part.split(" ", 1), "")[:2] for part in found.group(2).split("; ")), float(found.group(1))


def test_the_opening_call_prints_the_cycles_at_the_thinking_rate_of_every_asset(tmp_path):
    """The opening call prints for hyd_1 its seconds over the thinking seconds of a cycle that the loader of its asset file
    gives, 10 under the defaults of the clock, and not the "cycles none" of a read of a key at the top of the file. Through a
    stand-in of the loader, an asset whose profile cannot load prints unknown for itself alone and one without a clock is left
    out, while hyd_1 keeps its rate."""
    from CausalDiagnosis.datasets.dataset_loader.profile_registry import get_profile

    rates, seconds = _opening_call(ROOT)
    think = get_profile("hyd_1").default_config().think_seconds_per_cycle
    assert think == 10 and "hyd_1" in rates and abs(float(rates["hyd_1"]) - seconds / think) < 0.002, rates
    for name, text in LOADER_STAND_IN.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text, encoding="utf-8")
    rates, seconds = _opening_call(tmp_path)
    assert set(rates) == {"broken", "hyd_1"} and rates["broken"] == "unknown, KeyError", rates
    assert abs(float(rates["hyd_1"]) - seconds / 10) < 0.002, rates


# ----- no documented command writes a tracked file of the clone it runs from -----

# a stand-in of the offline table that writes as the module does, through the shared atomic writer, which the test places in the
# tree, so that the documented line meets a runs root that no earlier job made
OFFLINE_STAND_IN = """import argparse
from pathlib import Path
from CausalDiagnosis.utils.atomic_writes_and_locks import replaced_atomically
ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, default=Path({default!r}))
out = ap.parse_args().out
with replaced_atomically(out) as handle:
    handle.write("prehistory,readings\\n0,1\\n")
print(f"wrote {{out}}")
"""
# sbatch that runs the job script on the spot, as a node would, its log under FAKE_LOG, and prints a job id as --parsable does
SBATCH_RUNS_THE_JOB = """args=("$@"); i=0
while [ "$i" -lt "${#args[@]}" ]; do case "${args[$i]}" in *.sbatch|*.sh) break ;; esac; i=$((i + 1)); done
bash "${args[@]:$i}" >> "$FAKE_LOG/job.out" 2>&1
echo "4242;cluster" """


def _documented_offline_lines():
    """The lines of the registered order of the page from the module of the offline table to its submit line."""
    order = README[README.index("\n### The registered order\n"):README.index("\n## 4. After a run\n")]
    lines = [line for block in re.findall(r"```bash\n(.*?)```", order, re.S) for line in block.splitlines()]
    first = next(i for i, line in enumerate(lines) if line.startswith("O=") and "offline_table" in line)
    last = next(i for i in range(first, len(lines)) if lines[i].startswith("bash $H/submit.sh cpu_job.sbatch"))
    return lines[first:last + 1]


def test_the_documented_offline_table_leaves_the_clone_equal_to_its_commit(tmp_path):
    """The lines of the page that compute the offline table, run in a clone through submit.sh and cpu_job.sbatch with an sbatch
    that runs the job at once, write the table under the runs root of the tree and leave git status of the clone empty. The
    control: the command without --out, as the page wrote it before, rewrites the tracked file, which git status names. The
    stand-in of the module writes to the default of the real module, the published copy of the results folder."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline import offline_table

    if shutil.which("git") is None:
        pytest.skip("needs git")
    published = offline_table.RESULTS_PATH.relative_to(ROOT).as_posix()
    assert published == "CausalDiagnosis/results/offline_mcp.csv"
    tree, root, fake, env = _tree(tmp_path, ("cpu_job.sbatch",))
    _module_file(tree, offline_table.__name__, OFFLINE_STAND_IN.format(default=str(tree / published)))
    writer = "CausalDiagnosis.utils.atomic_writes_and_locks"
    _module_file(tree, writer, (ROOT / (writer.replace(".", "/") + ".py")).read_text(encoding="utf-8"))
    (tree / published).parent.mkdir(parents=True, exist_ok=True)
    (tree / published).write_text("prehistory,readings\n", encoding="utf-8")
    (tree / ".gitignore").write_text("*.calls\n", encoding="utf-8")  # the records of the stand-ins of the data check
    _record(tree, "DEVELOPMENT")
    git = {**env, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.org", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.org",
           "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    for argv in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "the clone"]):
        subprocess.run(["git", "-C", str(tree), *argv], check=True, capture_output=True, env=git)
    log = tmp_path / "log"
    log.mkdir()
    _stand_in(fake, "sbatch", SBATCH_RUNS_THE_JOB)
    settings = {**git, "FAKE_LOG": str(log), "ACCOUNT": "acct", "PARTITION_CPU": "cpu"}

    def status_after(lines):
        done = _run(tmp_path, settings, f"H={_jobs(tree)}", *lines)
        assert done.returncode == 0 and "SUBMITTED cpu_job.sbatch 4242" in done.stdout, (done.stdout, done.stderr)
        return subprocess.run(["git", "-C", str(tree), "status", "--porcelain"], capture_output=True, text=True, env=git).stdout

    lines = _documented_offline_lines()
    assert status_after(lines) == "", (lines, (log / "job.out").read_text(encoding="utf-8"))
    tree_id = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True, env=git).stdout[:12]
    assert (root / "runs" / "development" / tree_id / "offline_mcp.csv").is_file()
    before = [lines[0], "bash $H/submit.sh cpu_job.sbatch -- python -m $O"]
    assert status_after(before) == f" M {published}\n"


# ----- two CouchDB servers on one node -----

# the lines of the vm.args of the couchdb:3.5 image that the fault involves, as the cluster read them: the Erlang distribution
# fixed at port 9100, with no interface named, so on every interface
IMAGE_VM_ARGS = "-kernel inet_dist_listen_min 9100\n-kernel inet_dist_listen_max 9100\n+Bd -noinput"
# a CouchDB that listens for the Erlang distribution where its args file puts it, the last value of a -kernel key winning as in
# erlexec and the first free port of the range taken as Erlang takes it, and answers HTTP where its ini file puts it; it binds
# the loopback address whatever interface the file names, so that the test opens no port to other hosts, and prints the one named
COUCHDB_STAND_IN = """import configparser, http.server, os, socket, sys
with open(os.environ["FAKE_PIDS"], "a", encoding="utf-8") as pids:
    pids.write(f"{os.getpid()}\\n")
kernel = {}
for line in open(os.environ["APPTAINERENV_COUCHDB_ARGS_FILE"], encoding="utf-8"):
    words = line.split()
    if len(words) == 3 and words[0] == "-kernel":
        kernel[words[1]] = words[2]
for port in range(int(kernel["inet_dist_listen_min"]), int(kernel["inet_dist_listen_max"]) + 1):
    dist = socket.socket()
    try:
        dist.bind(("127.0.0.1", port))
        break
    except OSError:
        dist.close()
else:
    print("Protocol 'inet_tcp': register/listen error: eaddrinuse", flush=True)
    sys.exit(1)
dist.listen()
print("DISTRIBUTION", port, kernel.get("inet_dist_use_interface", "every interface"), flush=True)
ini = configparser.ConfigParser()
ini.read(os.environ["APPTAINERENV_COUCHDB_INI_FILES"].split()[-1])


class Welcome(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'{"couchdb":"Welcome"}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


http.server.HTTPServer((ini["chttpd"]["bind_address"], int(ini["chttpd"]["port"])), Welcome).serve_forever()
"""
# apptainer that builds the sandbox as a folder, prints the vm.args of the image and runs the CouchDB above for bin/couchdb
APPTAINER_COUCHDB = """case "$1" in
  build) mkdir -p "$3" ;;
  exec) shift; [ "$1" != --bind ] || shift 2
        case "$2" in
          cat) printf '%s\\n' "$FAKE_VM_ARGS" ;;
          */bin/couchdb) exec "$FAKE_PYTHON" "$FAKE_COUCHDB" ;;
        esac ;;
esac"""


def _kernel_of(args_file):
    """The -kernel values of an args file, the last of a key winning as in erlexec."""
    words = [line.split() for line in args_file.read_text(encoding="utf-8").splitlines()]
    return {line[1]: line[2] for line in words if len(line) == 3 and line[0] == "-kernel"}


def test_two_couchdb_starts_on_one_node_each_take_a_distribution_port_of_their_own(tmp_path):
    """Two jobs on one node start CouchDB one after the other through start_couchdb, from the vm.args of the image, which fixes
    the Erlang distribution at port 9100: both answer, and each listens for the distribution at a free port of its own on the
    loopback address alone, apart from its HTTP port. The control: the CouchDB stand-in started twice on the args of the image
    stops the second with eaddrinuse, as the second job on the cluster did."""
    tree, _root, fake, env = _tree(tmp_path)
    framework = _jobs(tree).parent / "frameworks" / "AssetOpsBench" / "cluster_jobs"
    framework.mkdir(parents=True)
    for name in ("job_services.sh", "start_couchdb.sh"):
        shutil.copy(AOB_JOBS / name, framework / name)
    server = tmp_path / "couchdb_stand_in.py"
    server.write_text(COUCHDB_STAND_IN, encoding="utf-8")
    _stand_in(fake, "apptainer", APPTAINER_COUCHDB)
    _stand_in(fake, "module", "exit 0")
    _stand_in(fake, "python", f'exec "{sys.executable}" "$@"')
    image = tmp_path / "couchdb-3.5.sif"
    image.write_text("image\n", encoding="utf-8")
    pids = tmp_path / "pids"
    settings = {**env, "SHARED": "", "IMAGES": str(tmp_path / "images"), "COUCH_IMAGE": str(image), "FAKE_PYTHON": sys.executable,
                "FAKE_COUCHDB": str(server), "FAKE_VM_ARGS": IMAGE_VM_ARGS, "FAKE_PIDS": str(pids)}
    try:
        starts = {job: (tmp_path / f"local{job}") for job in ("101", "102")}
        done = {job: _run(tmp_path, {**settings, "SLURM_JOB_ID": job}, *_sourced(tree, f'start_couchdb "{local}"', 'echo "URL $COUCHDB_URL"'))
                for job, local in starts.items()}
        assert [done[job].returncode for job in starts] == [0, 0], [done[job].stderr[-2000:] for job in starts]
        ports = {}
        for job, local in starts.items():
            url = next(line.split()[1] for line in done[job].stdout.splitlines() if line.startswith("URL "))
            kernel = _kernel_of(local / "couchdb" / "vm.args")
            dist = kernel.get("inet_dist_listen_min")
            assert kernel.get("inet_dist_listen_max") == dist and kernel.get("inet_dist_use_interface") == "{127,0,0,1}", kernel
            log = (local / "couchdb" / "couchdb.log").read_text(encoding="utf-8")
            assert f"DISTRIBUTION {dist} {{127,0,0,1}}" in log and url != f"http://127.0.0.1:{dist}", (log, url)
            ports[job] = (url, int(dist))
        assert ports["101"][1] != ports["102"][1]
        for url, _dist in ports.values():
            with urllib.request.urlopen(url + "/", timeout=10) as answer:
                assert answer.status == 200 and b"Welcome" in answer.read()
        control = tmp_path / "control"
        control.mkdir()
        (control / "vm.args").write_text(IMAGE_VM_ARGS + "\n", encoding="utf-8")
        (control / "aob.ini").write_text(f"[chttpd]\nport = {_free_port()}\nbind_address = 127.0.0.1\n", encoding="utf-8")
        inside = {"PATH": os.environ["PATH"], "FAKE_PIDS": str(pids), "APPTAINERENV_COUCHDB_ARGS_FILE": str(control / "vm.args"),
                  "APPTAINERENV_COUCHDB_INI_FILES": f"/opt/couchdb/etc/default.ini {control / 'aob.ini'}"}
        first = subprocess.Popen([sys.executable, str(server)], stdout=subprocess.PIPE, text=True, env=inside)
        assert first.stdout.readline().split()[:2] == ["DISTRIBUTION", "9100"]
        (control / "aob.ini").write_text(f"[chttpd]\nport = {_free_port()}\nbind_address = 127.0.0.1\n", encoding="utf-8")
        second = subprocess.run([sys.executable, str(server)], capture_output=True, text=True, env=inside, timeout=60)
        assert second.returncode == 1 and "register/listen error: eaddrinuse" in second.stdout
        first.terminate()
        first.wait(timeout=30)
        first.stdout.close()
    finally:
        for pid in pids.read_text(encoding="utf-8").split() if pids.exists() else ():
            try:
                os.kill(int(pid), signal.SIGTERM)
            except ProcessLookupError:
                pass


def _free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]
