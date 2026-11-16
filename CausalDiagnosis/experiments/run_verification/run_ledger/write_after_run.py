from pathlib import Path
from CausalDiagnosis.utils.json_files import read_json
from CausalDiagnosis.utils import frameworks
from CausalDiagnosis.experiments.run_verification.public_record import add_private
from . import hashing, code_identity, ledger_file


def write_after(run_dir, scorer_listing=None):
    """Writes the ledger of a finished run of a declared kind whose ledger may be written after the fact (run_kind) and that
    has none: the fields of run.json with the job and the host it names, the code as
    the SHA-256 prefixes run.json recorded, since the files that ran are not at hand, and the data, the episodes and the
    run's files as they are found now. written_after_run says when and on which host, and the command and the environment
    of the run stay unknown. Judge records and scores already in the folder enter as one event, as its scorer's would, with
    the scorer code hashed in scorer_listing, lines of sha256sum taken on the machine that scored, when given."""
    run_dir = Path(run_dir)
    if (run_dir / ledger_file.LEDGER).exists():
        raise SystemExit(f"{run_dir} already has a ledger")
    meta = read_json(run_dir / "run.json", {}) or {}
    kind = ledger_file.run_kind(meta)
    if kind is None or not kind.after_the_fact:
        kinds = ", ".join(name for name in frameworks.run_kinds() if getattr(frameworks.code("RUN_KINDS", name), "after_the_fact", False))
        raise SystemExit(f"{run_dir} is no run of a kind whose ledger may be written after the fact ({kinds})")
    if not meta.get("ended"):
        raise SystemExit(f"{run_dir} holds no run.json with an end time, so the run has not finished")

    def as_recorded(name, endpoint, _):
        return {"name": name, "endpoint": endpoint, "endpoint_file": None, "endpoint_models": None} if name else None

    after = {"at": ledger_file.now(),
             "note": "written after the run from run.json and the files as found, with the code as the SHA-256 prefixes "
                     "run.json recorded and the command and the environment of the run unknown"}
    code = {"commit": None, "dirty": None, "upload_manifest_sha256": None, "differs_from_upload": None, "files": None,
            "recorded_prefixes": code_identity.recorded_code(meta)}
    # a run.json written before the published set was defined names its job and host, which go to the private file alone
    add_private(run_dir, "starts", {"at": after["at"], "written_after": True,
                                    "job": {"slurm_job_id": meta.get("slurm_job_id"), "host": meta.get("host")}})
    ledger_file._write(run_dir, ledger_file._record(run_dir, ledger_file.run_fields(meta, serve=as_recorded), meta.get("started"), None, None, code, after))
    ledger = ledger_file.finish(run_dir)
    found = {}
    for name, episode in ledger["episodes"].items():
        for path in [p for p in episode["files"] if kind.scorer_output(p)]:
            found[f"{name}/{path}"] = (None, episode["files"].pop(path)["sha256"])
    for path in [p for p in ledger["run_files"] if kind.scorer_output(p)]:
        found[path] = (None, ledger["run_files"].pop(path)["sha256"])
    if found:
        listing = hashing.read_manifest(scorer_listing) if scorer_listing else None
        ledger["events"].append(ledger_file._event(
            "ledger.write_after", kind.found_scores + ", found when the ledger "
            "was written" + (f", with the scorer code as {scorer_listing} hashed it" if listing else ", by scorer code the ledger cannot name"),
            found, {path: listing.get(path) for path in kind.scorer_code} if listing else None))
    ledger_file._write(run_dir, ledger)
    return ledger
