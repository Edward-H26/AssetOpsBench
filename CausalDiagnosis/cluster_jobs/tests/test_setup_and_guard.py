"""The setup command of a clone (CausalDiagnosis/cluster_jobs/0_setup/setup.sh) and the guard of serve_model.sbatch against a model service that exits while
the script of the job runs. The scripts run under stand-ins for sbatch and squeue, as in test_cluster_jobs.py."""
from CausalDiagnosis.utils.paths import CLUSTER_JOBS, ROOT
import re
import subprocess

from CausalDiagnosis.cluster_jobs.tests.test_cluster_jobs import DIGESTS_MODULE, SCRIPTS, SERVING_FILE, _in, _run, _stand_in, _tree

CHAIN = ("setup.sh", "submit.sh", "pull_image.sbatch", "download_model.sbatch", "setup_env.sbatch", "fetch_data.sbatch",
         "precompute_tables.sbatch", "cpu_job.sbatch")
SERVING = '{"format": 1, "agent": {"hub_id": "org/agent"}, "judge": {"hub_id": "org/judge-model"}}'
SETTINGS = {"ACCOUNT": "acct", "PARTITION_CPU": "cpu", "PARTITION_GPU": "gpu"}


def _setup_tree(tmp_path, queue=""):
    """A tree with the scripts of the chain and the serving file; sbatch numbers its jobs from 101 and writes each call as one
    line of calls.txt; squeue prints QUEUE for any job."""
    tree, root, fake, env = _tree(tmp_path, CHAIN)
    (tree / SERVING_FILE.relative_to(ROOT)).write_text(SERVING, encoding="utf-8")
    calls, counter = tmp_path / "calls.txt", tmp_path / "counter"
    counter.write_text("100", encoding="utf-8")
    _stand_in(fake, "sbatch", f'n=$(( $(cat "{counter}") + 1 )); echo "$n" > "{counter}"; echo "$*" >> "{calls}"; echo "$n;cluster"')
    _stand_in(fake, "squeue", f'printf "%s" "{queue}"')
    return tree, root, fake, {**env, **SETTINGS}, calls


def _setup(tmp_path, tree, env, *arguments):
    return _run(tmp_path, env, f'bash "{_in(tree, "setup.sh")}" {" ".join(arguments)}')


def _submitted(stdout):
    """{script: job id} of the SUBMITTED lines, the two downloads under download_model.sbatch in their order."""
    found = {}
    for line in stdout.splitlines():
        if line.startswith("SUBMITTED "):
            _, script, job, _log = line.split()
            found.setdefault(script, []).append(job)
    return found


def _call_of(calls, script):
    return [line for line in calls.read_text(encoding="utf-8").splitlines() if f"/{script}" in line]


def test_setup_submits_seven_jobs_with_the_afterok_edges_of_the_chain(tmp_path):
    tree, root, _fake, env, calls = _setup_tree(tmp_path)
    done = _setup(tmp_path, tree, env)
    assert done.returncode == 0, done.stderr
    jobs = _submitted(done.stdout)
    assert sum(len(ids) for ids in jobs.values()) == 7 and len(jobs["download_model.sbatch"]) == 2
    assert f"--dependency=afterok:{jobs['setup_env.sbatch'][0]}" in _call_of(calls, "fetch_data.sbatch")[0]
    assert f"--dependency=afterok:{jobs['fetch_data.sbatch'][0]}" in _call_of(calls, "precompute_tables.sbatch")[0]
    digests = _call_of(calls, "cpu_job.sbatch")[0]
    assert f"--dependency=afterok:{jobs['precompute_tables.sbatch'][0]}" in digests and f"python -m {DIGESTS_MODULE} " in digests
    assert "--label clone" in digests and "CD_DIGESTS_DONE" in digests
    downloads = _call_of(calls, "download_model.sbatch")
    assert "--export=ALL,MODEL=org/judge-model" in downloads[1] and "MODEL=" not in downloads[0]
    # both downloads fetch with the uv that the environment job installs, so they follow it as the data job does
    assert all(f"--dependency=afterok:{jobs['setup_env.sbatch'][0]}" in line for line in downloads)
    for parallel in ("pull_image.sbatch", "setup_env.sbatch"):
        assert all("--dependency" not in line for line in _call_of(calls, parallel))
    chain = [line for line in done.stdout.splitlines() if line.startswith("SETUP_CHAIN ")]
    assert len(chain) == 1 and len(chain[0].split()) == 8
    ids = list((root / "setup").glob("*.ids"))
    assert len(ids) == 1 and ids[0].read_text(encoding="utf-8").strip() == chain[0]


def _write_logs(root, done_steps, failed=()):
    """Writes the log of every recorded job: its DONE line for a step of done_steps, a last error line for a failed one."""
    record = next((root / "setup").glob("*.jobs"))
    lines = {"image": "CD_IMAGE_DONE /sif", "agent_model": "CD_DOWNLOAD_DONE /m/a", "judge_model": "CD_DOWNLOAD_DONE /m/j",
             "environment": "CD_SETUP_DONE", "data": "CD_DATA_DONE", "tables": "CD_TABLES_DONE", "digests": "CD_DIGESTS_DONE"}
    for row in record.read_text(encoding="utf-8").splitlines():
        step, _job, log = row.split()
        if step in done_steps:
            text = f"working\n{lines[step]}\n"
        elif step in failed:
            text = "working\nthe data could not be fetched\n"
        else:
            continue
        with open(log, "w", encoding="utf-8") as out:
            out.write(text)


def test_a_second_setup_skips_each_job_whose_done_line_exists_and_drops_its_edge(tmp_path):
    tree, root, _fake, env, calls = _setup_tree(tmp_path)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    assert _setup(tmp_path, tree, env).returncode == 0
    _write_logs(root, {"image", "agent_model", "judge_model", "environment"}, failed={"data"})
    calls.write_text("", encoding="utf-8")
    done = _setup(tmp_path, tree, env)
    assert done.returncode == 0, done.stderr
    skipped = [line.split()[1] for line in done.stdout.splitlines() if line.startswith("SKIPPED ")]
    assert skipped == ["image", "environment", "agent_model", "judge_model"]
    assert set(_submitted(done.stdout)) == {"fetch_data.sbatch", "precompute_tables.sbatch", "cpu_job.sbatch"}
    assert "--dependency" not in _call_of(calls, "fetch_data.sbatch")[0]


def test_a_setup_refuses_while_a_job_of_the_last_chain_is_in_the_queue(tmp_path):
    tree, _root, _fake, env, _calls = _setup_tree(tmp_path)
    assert _setup(tmp_path, tree, env).returncode == 0
    _stand_in(tmp_path / "fakebin", "squeue", 'echo PENDING')
    done = _setup(tmp_path, tree, env)
    assert done.returncode == 4 and "still in the queue" in done.stderr and "SUBMITTED" not in done.stdout


def test_setup_status_prints_setup_ok_only_on_seven_done_lines(tmp_path):
    tree, root, _fake, env, _calls = _setup_tree(tmp_path)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    assert _setup(tmp_path, tree, env).returncode == 0
    steps = {"image", "agent_model", "judge_model", "environment", "data", "tables"}
    _write_logs(root, steps, failed={"digests"})
    done = _setup(tmp_path, tree, env, "status")
    assert done.returncode == 1 and "SETUP_OK" not in done.stdout
    assert re.search(r"^digests \d+ FAILED the data could not be fetched$", done.stdout, re.M)
    assert re.search(r"^data \d+ CD_DATA_DONE$", done.stdout, re.M)
    _write_logs(root, steps | {"digests"})
    done = _setup(tmp_path, tree, env, "status")
    assert done.returncode == 0 and done.stdout.splitlines()[-1] == "SETUP_OK" and len(done.stdout.splitlines()) == 8


def test_setup_status_names_a_job_still_in_the_queue(tmp_path):
    tree, _root, _fake, env, _calls = _setup_tree(tmp_path)
    assert _setup(tmp_path, tree, env).returncode == 0
    _stand_in(tmp_path / "fakebin", "squeue", 'echo PENDING')
    done = _setup(tmp_path, tree, env, "status")
    assert done.returncode == 1 and re.search(r"^image \d+ PENDING$", done.stdout, re.M)


def _function(text, name):
    """The lines of the shell function NAME of a script, from its first line to the closing brace."""
    lines = text.splitlines()
    start = lines.index(f"{name}() {{")
    return "\n".join(lines[start:lines.index("}", start) + 1])


SERVE = (CLUSTER_JOBS / "2_serve_model" / "serve_model.sbatch").read_text(encoding="utf-8")


def _guard(tmp_path, agent_body, script_body="sleep 30"):
    """watch_services of serve_model.sbatch on a stand-in script and a stand-in agent server; prints the variables after it."""
    return subprocess.run(["bash", "-c", "\n".join([
        "SGL_PID= JUDGE_PID= SCRIPT_PID= SERVICE_EXITED= SCRIPT_GRACE=2",
        _function(SERVE, "watch_services"),
        f"( {agent_body} ) & SGL_PID=$!",
        f"set -m; bash -c '{script_body}' & SCRIPT_PID=$!; set +m",
        "watch_services",
        'echo "AFTER service=$SERVICE_EXITED"',
        'kill -0 "$SCRIPT_PID" 2>/dev/null && echo SCRIPT_ALIVE || echo SCRIPT_ENDED',
    ])], capture_output=True, text=True, cwd=tmp_path, timeout=60)


def test_the_guard_ends_the_script_when_the_agent_server_exits(tmp_path):
    done = _guard(tmp_path, "sleep 1; exit 7")
    assert "SERVICE_EXITED agent 7" in done.stdout and "AFTER service=agent" in done.stdout and "SCRIPT_ENDED" in done.stdout


def test_the_guard_returns_quietly_when_the_script_ends_with_the_servers_alive(tmp_path):
    done = _guard(tmp_path, "sleep 30", script_body="sleep 1")
    assert "SERVICE_EXITED" not in done.stdout and "AFTER service=" in done.stdout


def test_the_job_ends_with_status_3_after_the_guard_names_a_dead_service():
    block = SERVE[SERVE.index('if [ -n "${RUN_AFTER_SERVE:-}" ]; then\n  # The script runs'):]
    assert "set -m" in block and "watch_services" in block
    assert re.search(r'if \[ -n "\$SERVICE_EXITED" \]; then stop_servers; exit 3; fi', block)


def test_no_launch_of_a_model_server_names_allow_auto_truncate():
    named = [p.name for p in SCRIPTS if "--allow-auto-truncate" in p.read_text(encoding="utf-8")]
    assert named == []
