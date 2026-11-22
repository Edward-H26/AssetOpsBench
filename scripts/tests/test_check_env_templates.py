"""The env template hook, scripts/check_env_templates.py, on a shell template such as env.example.sh: the tracked template
passes, a word shaped like a credential fails wherever it stands, a private env file still fails by its name, and .env.public
keeps its rules. Every fake credential is built at run time, so that this file holds none."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HOOK = ROOT / "scripts" / "check_env_templates.py"
TEMPLATE = ROOT / "CausalDiagnosis" / "cluster_jobs" / "env.example.sh"
TEMPLATE_PATH = "CausalDiagnosis/cluster_jobs/env.example.sh"
_spec = importlib.util.spec_from_file_location("check_env_templates", HOOK)
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)

# shorter than the long opaque token shape, so only its own shape can catch it
OPENAI_KEY = "sk" + "-" + "Zy9Xw8Vu7Ts6Rq5P"
GITHUB_TOKEN = "ghp" + "_" + "a1B2c3D4" * 5
AWS_KEY_ID = "AKIA" + "Q7RT2WX9" * 2
UUID = "-".join(["aaaa1111", "bbbb", "cccc", "dddd", "eeee2222ffff"])
JWT = ".".join(["ey" + "Jh" * 8, "ey" + "Jz" * 8, "s1" * 8])
OPAQUE = "Q9" * 14
# mixed case, as a base64 token is, so it cannot be a shell variable name
PADDED = "Zy9Xw8Vu7Ts6Rq5P" + "abcdEFgh"
KEY_NAME = "WATSONX" + "_APIKEY"
SECRET_NAMES = ["API" + "_KEY", "HF" + "_TOKEN", "COUCHDB" + "_PASSWORD"]


def problems_of(text):
    return hook.check_shell_template("env.example.sh", text)


def staged_run(repo, files):
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    for name, text in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "-f", *files], check=True)
    return subprocess.run([sys.executable, str(HOOK)], cwd=repo, capture_output=True, text=True)


def test_the_tracked_shell_template_passes():
    assert problems_of(TEMPLATE.read_text(encoding="utf-8")) == []


def test_variable_names_are_no_values():
    assert problems_of("WANDB_PROJECT_DEVELOPMENT=${WANDB_PROJECT_DEVELOPMENT:-dev}\n"
                       'export JUDGE_ALONE_SGLANG_EXTRA; [ -z "$JUDGE_ALONE_SGLANG_EXTRA" ]\n'
                       "export WANDB_PROJECT_DEVELOPMENT\nlong_lower_case_variable_name=1\n") == []


EMPTY_FORMS = ["{0}=${{{0}:-}}", "{0}=", '{0}=""', '{0}="${{{0}:-}}"', '[ -z "${0}" ] || export {0}', ': "${{{0}:=}}"',
               "OTHER=${{{0}:-}}", 'export "{0}="', 'x="$({0}= cmd)"']


@pytest.mark.parametrize("name", SECRET_NAMES)
def test_a_secret_key_that_ships_empty_passes(name):
    assert problems_of("".join(form.format(name) + "\n" for form in EMPTY_FORMS)) == []


@pytest.mark.parametrize("line", ["{0}=${{{0}:-hunter2}}", "{0}=hunter2hunter2", "{0}=${{{0}-local}}", "export {0}=local",
                                  "if true; then {0}='local'; fi", ': "${{{0}:=hunter2}}"', ": ${{{0}=hunter2}}", "{0}+=hunter2",
                                  'export "{0}=hunter2"', 'x="$({0}=hunter2 cmd)"', "x=`{0}=hunter2 cmd`", "{0}[0]=hunter2",
                                  "OTHER=${{{0}:-hunter2}}", 'echo "key: ${{{0}-hunter2}}"'])
@pytest.mark.parametrize("name", SECRET_NAMES)
def test_a_secret_key_with_a_value_fails(line, name):
    problems = problems_of(line.format(name) + "\n")
    assert len(problems) == 1 and "Secret keys must ship empty" in problems[0] and "local" not in problems[0] \
        and "hunter2" not in problems[0]

@pytest.mark.parametrize("line", [
    "MODEL=${MODEL:-%s}",
    "AGENT_DTYPE=${AGENT_DTYPE-%s}",
    "export WANDB_ENTITY=%s",
    'SGLANG_EXTRA=${SGLANG_EXTRA:-"--api-key %s"}',
    "SGLANG_EXTRA=--api-key=%s",
    "FORK_URL=${FORK_URL:-https://github.com/%s/fork.git}",
    "# the key of the remote route: %s.",
    'WANDB_ENTITY="%s="',
])
@pytest.mark.parametrize("credential", [OPENAI_KEY, GITHUB_TOKEN, AWS_KEY_ID, UUID, JWT, OPAQUE])
def test_a_credential_fails_wherever_it_stands(line, credential):
    problems = problems_of(line % credential + "\n")
    assert len(problems) == 1 and credential not in problems[0]


@pytest.mark.parametrize("line", ["# the key of the remote route: %s=", "WANDB_ENTITY=team %s==", "export %s="])
@pytest.mark.parametrize("credential", [PADDED, GITHUB_TOKEN, AWS_KEY_ID])
def test_a_credential_in_the_place_of_a_name_fails(line, credential):
    problems = problems_of(line % credential + "\n")
    assert len(problems) == 1 and credential not in problems[0]


def test_the_hook_checks_the_staged_shell_template(tmp_path):
    template = TEMPLATE.read_text(encoding="utf-8")
    assert staged_run(tmp_path / "clean", {TEMPLATE_PATH: template}).returncode == 0
    planted = staged_run(tmp_path / "planted", {TEMPLATE_PATH: template + f"WANDB_ENTITY=${{WANDB_ENTITY:-{GITHUB_TOKEN}}}\n"})
    assert planted.returncode == 1 and "a GitHub token" in planted.stderr and GITHUB_TOKEN not in planted.stderr


def test_a_python_module_named_env_is_code(tmp_path):
    assert staged_run(tmp_path, {"CausalDiagnosis/environment/env.py": "ROOT = 'not an env file'\n"}).returncode == 0


def test_a_private_env_file_still_fails(tmp_path):
    run = staged_run(tmp_path, {"CausalDiagnosis/cluster_jobs/env.sh": "ROOT=/scratch/someone\n"})
    assert run.returncode == 1 and "is a private env file" in run.stderr


@pytest.mark.parametrize("text, code", [(f"{KEY_NAME}=\n", 0), (f"{KEY_NAME}=local\n", 1), ("COUCHDB_USERNAME=admin\n", 0),
                                        ("COUCHDB_USERNAME=root\n", 1)])
def test_env_public_keeps_its_rules(tmp_path, text, code):
    assert staged_run(tmp_path, {".env.public": text}).returncode == code
