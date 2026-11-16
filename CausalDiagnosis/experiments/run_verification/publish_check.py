"""The check of run folders and results folders before they go to the branch of the runs (CausalDiagnosis/cluster_jobs/README.md section 9): it reads
exactly the published set of each folder (public_record.published_files), copied into a staging folder, and passes only when
1. no file holds an absolute path, the user name of this machine or a name of the names file, a URL of a host outside
   public_record.PUBLIC_HOSTS or an IPv4 address,
2. gitleaks finds no secret, once with the repository's .gitleaks.toml and once with its default rules alone, since that file
   leaves out every Markdown file and every path through tests, docs or examples,
3. no file is larger than GitHub takes (MAX_FILE_BYTES, a warning above WARN_FILE_BYTES) and the set stays under MAX_PUSH_BYTES,
4. the ledger check of every run folder on the staged copy reads ("checked", []).

usage, from the fork root:
  python -m CausalDiagnosis.experiments.run_verification.publish_check FOLDER ... [--names FILE] [--into TREE]
With --into the published set of every folder is copied into TREE/<folder name> and checked there, as the user then commits it.
"""
import argparse
import getpass
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

from CausalDiagnosis.experiments.run_verification import run_ledger as ledger  # noqa: E402
from CausalDiagnosis.experiments.run_verification import public_record  # noqa: E402
from CausalDiagnosis.experiments.run_verification import verify_results_folder as verify_results  # noqa: E402

# GitHub's limits: https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
# ("GitHub blocks files larger than 100 MiB", a warning above 50 MiB) and
# https://docs.github.com/en/get-started/using-git/troubleshooting-the-2-gb-push-limit ("a maximum 2 GiB limit for a single push")
MAX_FILE_BYTES = 100 * 2**20  # GitHub refuses a file above 100 MiB
WARN_FILE_BYTES = 50 * 2**20  # and warns above 50 MiB
MAX_PUSH_BYTES = 2 * 2**30  # and refuses a push above 2 GiB
ABSOLUTE_PATH = re.compile(r"(?:^|[\s\"'(=,\[])(?:~|/[A-Za-z0-9._@-]+)(?:/[A-Za-z0-9._@-]+)+")
URL = re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s\"'<>)\]]+", re.IGNORECASE)
IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
DEFAULT_RULES = "[extend]\nuseDefault = true\n"


def staged(folder, target):
    """Copies the published set of a folder into target byte for byte; returns the copied paths."""
    copied = []
    for rel in public_record.published_files(folder):
        (target / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(folder) / rel, target / rel)
        copied.append(target / rel)
    return copied


def text_problems(path, names):
    """Check 1 on one file, as "<file>: <what>" lines."""
    text = path.read_bytes().decode("utf-8", errors="replace")
    problems = [f"{path}: absolute path {m.group(0).strip()!r}" for m in ABSOLUTE_PATH.finditer(text)]
    problems += [f"{path}: URL {m.group(0)!r} of a host that no public file names" for m in URL.finditer(text)
                 if public_record.public_url(m.group(0)) is None]
    problems += [f"{path}: IPv4 address {m.group(0)!r}" for m in IPV4.finditer(text)]
    problems += [f"{path}: the name {name!r}" for name in names if name and name in text]
    return problems


def gitleaks_problems(folder, config):
    """Check 2 on a staged folder: gitleaks with the repository's rules and with its default rules alone."""
    tool = shutil.which("gitleaks")
    if tool is None:
        return ["gitleaks is not installed, so no secret scan ran"]
    problems = []
    with tempfile.TemporaryDirectory() as scratch:
        plain = Path(scratch) / "default.toml"
        plain.write_text(DEFAULT_RULES, encoding="utf-8")
        for rules in (config, plain):
            done = subprocess.run([tool, "dir", str(folder), "--config", str(rules), "--no-banner", "--redact", "--exit-code", "1"],
                                  capture_output=True, text=True)
            if done.returncode != 0:
                problems.append(f"gitleaks with {Path(rules).name} found a secret or failed: {(done.stdout + done.stderr).strip()[-400:]}")
    return problems


def size_problems(files):
    """Check 3: (problems, warnings)."""
    problems = [f"{p}: {p.stat().st_size} bytes, above GitHub's limit of {MAX_FILE_BYTES}" for p in files if p.stat().st_size > MAX_FILE_BYTES]
    warnings = [f"{p}: {p.stat().st_size} bytes, above GitHub's warning at {WARN_FILE_BYTES}" for p in files
                if WARN_FILE_BYTES < p.stat().st_size <= MAX_FILE_BYTES]
    total = sum(p.stat().st_size for p in files)
    if total > MAX_PUSH_BYTES:
        problems.append(f"the set holds {total} bytes, above the {MAX_PUSH_BYTES} of one push")
    return problems, warnings


def check(folders, into, names=(), config=None):
    """(problems, warnings) of the published set of the folders, staged under into."""
    config = config or ledger.run_data.ROOT / ".gitleaks.toml"
    names = [getpass.getuser(), *names]
    problems, files = [], []
    for folder in (Path(f) for f in folders):
        target = Path(into) / folder.name
        if target.exists():
            problems.append(f"{target} exists already")
            continue
        files += staged(folder, target)
        if (target / ledger.LEDGER).is_file():
            state, found = verify_results.check_ledger(target)
            if state != "checked" or found:
                problems.append(f"{folder.name}: the ledger check of the published copy reads {state}: {'; '.join(found[:5])}")
    for path in files:
        problems += text_problems(path, names)
    problems += gitleaks_problems(into, config)
    found, warnings = size_problems(files)
    return problems + found, warnings


def main(argv=None):
    ap = argparse.ArgumentParser(description="the check of run folders and results folders before a push to the branch of the runs")
    ap.add_argument("folders", nargs="+", help="run folders or results folders")
    ap.add_argument("--names", type=Path, help="a file of names that no public file may hold, one per line: account names, the scratch root")
    ap.add_argument("--into", type=Path, help="the folder of the branch of the runs, into which the published set is copied")
    a = ap.parse_args(argv)
    names = [line.strip() for line in a.names.read_text(encoding="utf-8").splitlines() if line.strip()] if a.names else []
    if a.into:
        problems, warnings = check(a.folders, a.into, names)
    else:
        with tempfile.TemporaryDirectory() as scratch:
            problems, warnings = check(a.folders, Path(scratch), names)
    for line in warnings:
        print(f"WARNING {line}")
    for line in problems:
        print(line)
    print(f"PUBLISH_CHECK_{'FAILED' if problems else 'OK'} {len(problems)} problems in {len(a.folders)} folders")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
