from pathlib import Path
from . import run_record_checks, run_replay


HERE = Path(__file__).resolve().parent
from CausalDiagnosis.experiments.run_verification import run_ledger as ledger  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import episode_dirs  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import ROOT  # noqa: E402
from CausalDiagnosis.utils.json_files import read_json, write_json  # noqa: E402


CERTIFICATE_FORMAT = 1
TOOL = "replay_check.py"
# the code that writes a certificate into a run folder, hashed into the event of the run's ledger
CODE = ("CausalDiagnosis/experiments/run_verification/replay_check/__init__.py", "CausalDiagnosis/experiments/run_verification/replay_check/run_record_checks.py", "CausalDiagnosis/experiments/run_verification/replay_check/episode_replay.py", "CausalDiagnosis/experiments/run_verification/replay_check/run_replay.py", "CausalDiagnosis/experiments/run_verification/replay_check/certificate.py", "CausalDiagnosis/experiments/run_verification/replay_check/command_line.py", "CausalDiagnosis/experiments/run_verification/replay_check/__main__.py", "CausalDiagnosis/experiments/run_verification/verify_results_folder.py")
ADVISORY = 3  # the exit status of a replay that passed and counts nowhere, since it ran on another store kind or platform
# the exit status of a replay that passed on a tree that is no clone of the run's commit, such as an upload, where no certificate
# can be written: no difference of the run, and apart from the 2 and 4 with which the CPU job stops before its command
NOT_HERE = 5


# ----- the certificate -----
def certificate_name(commit):
    """The certificate's file name, keyed by the first 12 hex digits of the commit it was written at and binds."""
    return f"replay_{commit[:12]}.json"


def certificate_for(run_dir, commit):
    """The certificate of a run folder written at a commit, or None."""
    return read_json(Path(run_dir) / certificate_name(commit)) if commit else None


def certify(run_dir, root=ROOT):
    """(exit status, the replay, the path of the certificate or None). The tree at root must be a repository at the run's own
    commit (run.json code.commit): the certificate binds that commit and the platform of the replay. A run that replays with no
    problem and no advisory reason gets the certificate named by the commit, written into its folder inside ledger.snapshot and
    ledger.record_changes, as rescore_under_verifier_scope.py records its copy, so the ledger check accepts the file. So does a run whose only
    problems are generated tables (table_problems), which the certificate names in data_differences: every value of such a table
    that an episode or the verifier reads shows in an answer, in a file the episode left or in a score cell, and the replay compared
    all of them. Every certificate names the GPU card of the run's model and judge endpoints as run.json records them
    (serving_card). The status is 0 with the certificate, 1 with a difference or a run that cannot be replayed, ADVISORY when the
    replay passed and counts nowhere, and NOT_HERE when the replay passed on a tree at another commit or with no repository of
    its own, where no certificate can be written."""
    run_dir = Path(run_dir)
    meta = read_json(run_dir / "run.json", {}) or {}
    result = run_replay.replay_run(run_dir, root=root)
    beyond_tables = [problem for problem in result["problems"] if problem not in result["table_problems"]]
    if result["state"] != "replayed" or beyond_tables:
        return 1, result, None
    if result["advisory"]:
        return ADVISORY, result, None
    if not (run_dir / ledger.LEDGER).exists():
        result["problems"].append(f"{run_dir.name}: the run has no ledger, which would record the certificate")
        return 1, result, None
    run_commit, commit = (meta.get("code") or {}).get("commit"), ledger.code_identity.git_state([], root)[0]
    if commit is None or commit != run_commit:
        here = "is no repository of its own" if commit is None else f"stands at commit {commit[:12]}"
        result["problems"].append(f"{run_dir.name}: cannot certify here, since the tree {here} and the run was played at commit "
                                  f"{(run_commit or 'none')[:12]}; certify on a clone of the run's commit")
        return NOT_HERE, result, None
    agent = meta.get("agent")
    certificate = {"format": CERTIFICATE_FORMAT, "run": run_dir.name, "run_identity": (meta.get("code") or {}).get("identity"),
                   "run_commit": run_commit, "checked_identity": ledger.identity_of(ledger.code_identity.tree_files(root), agent),
                   "checked_commit": commit, "agent": agent, "agent_files_equal": result["agent_files_equal"],
                   "changed": result["changed"], "loaded_changed": result["loaded_changed"], "store": result["store"],
                   "platform": result["platform"], "data_equal": result["data_equal"], "data_differences": result["data_differences"],
                   "listings_equal": result["listings_equal"], "scores_equal": not result["scoring_problems"],
                   "scoring_problems": result["scoring_problems"], "table_problems": result["table_problems"], "lines": result["lines"],
                   "serving_card": {role: ledger.card_of((meta.get("serving") or {}).get(role)) for role in ("agent", "judge")},
                   "episodes": sorted(result["transcripts"]), "transcripts": result["transcripts"], "date": ledger.now()}
    path = run_dir / certificate_name(commit)
    before = ledger.snapshot(run_dir)
    write_json(path, certificate)
    ledger.record_changes(run_dir, before, TOOL, f"replay certificate of every episode on a clone of the run's commit {commit[:12]}", CODE)
    return 0, result, path


def recorded_by_ledger(run_dir, name, digest):
    """Whether the last event of the run's ledger that names the file gives it this hash."""
    last = None
    for event in (ledger.load(run_dir) or {}).get("events", []):
        if name in event.get("files", {}):
            last = event["files"][name]["new"]
    return digest is not None and last == digest


def certified(run_dir, meta):
    """The reason a finished run does not stand as replayed by its certificate, or None. Each condition is read from the files, and
    the certificate is trusted for the replay alone: the certificate named by the run's commit, recorded by the run's ledger,
    written for the run's identity at the run's commit (checked_commit) on the platform that run.json recorded, naming every
    episode, equal listings, the recordings and the scenario records of the run (RECORDED_DATA) equal, equal scores and the store
    kind of the run on both sides. A generated table that the certificate names may differ: the replay compared every answer, every
    file and every score cell that a value of the table reaches. A clone of the run's own commit holds the run's agent and harness
    files by construction, so no file list is compared here."""
    run_dir = Path(run_dir)
    code = meta.get("code") or {}
    commit = code.get("commit")
    if not commit:
        return "its run.json names no commit, so no certificate can be bound to it"
    certificate = certificate_for(run_dir, commit)
    if certificate is None:
        return f"its folder holds no replay certificate of its commit {commit[:12]}"
    if not recorded_by_ledger(run_dir, certificate_name(commit), ledger.sha256_of(run_dir / certificate_name(commit))):
        return "its ledger does not record its replay certificate"
    if certificate.get("checked_commit") != commit:
        return f"its replay certificate was written at commit {(certificate.get('checked_commit') or 'none')[:12]}, not at its commit {commit[:12]}"
    if certificate.get("run_identity") != code.get("identity"):
        return "its replay certificate names another identity of the run"
    if certificate.get("platform") != meta.get("platform"):
        return "its replay certificate was written on another platform than its run.json records"
    if sorted(certificate.get("episodes") or []) != [ep_dir.name for ep_dir in episode_dirs(run_dir)]:
        return "its replay certificate does not name every episode of the run"
    if not certificate.get("listings_equal") or any(name in run_record_checks.RECORDED_DATA for name in certificate.get("data_differences") or ()):
        return "its replay certificate names other tool listings or other data"
    if certificate.get("scores_equal") is not True:
        return "its replay certificate shows other scores than the run holds"
    kind = run_record_checks.store_kind(meta)
    if (certificate.get("store") or {}).get("run") != kind or (certificate.get("store") or {}).get("replay") != kind:
        return "its replay certificate was written on another store kind than the run's"
    return None
