"""The fetches of the setup chain on a link that breaks off and stalls, and the order and the ends of its jobs: one rule for the
shell (retry_fetch of functions.sh) and for Python (CausalDiagnosis/utils/download_retry.py), the caches that a job leaves on
scratch for the next one, the downloads after the environment job whose uv they fetch with, a failed job whose dependents leave
the queue, the image of a framework pulled in the setup and the data check before the metering check. The scripts run under
the stand-ins of test_cluster_jobs.py, and every fetch of Python against a local server that breaks off, stalls or fails; no
network, no Slurm and no image take part."""
from CausalDiagnosis.utils.paths import CLUSTER_JOBS, PACKAGE
import hashlib
import os
import re
import shutil
import subprocess
import sys

import pytest

from CausalDiagnosis.cluster_jobs.tests.test_cluster_jobs import APPTAINER_PULL, DOWNLOAD, HUB_LISTING, PINS, _in, _jobs, _run, _services, _sourced, _stand_in, _tree
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
