import json
from pathlib import Path
from CausalDiagnosis.experiments.run_verification.public_record import public_value  # noqa: E402
from CausalDiagnosis.utils.json_files import read_json
from CausalDiagnosis.experiments.table_definitions.training_variants import TRAINING_SPLIT_KEY  # noqa: E402


MODE = "gepa_online"
RESULT = "gepa_online.json"
UNFINISHED = ("running", "failed")  # the statuses of a result file whose loop --resume takes up


def loop_identity(meta):
    """What a resume compares: every field of run.json that says which loop a folder holds, the settings of the loop, the asset,
    the split, the model, the tool surface, the adapter and the code. The times, the label and the seeds of the stream, which follow
    from the rest, are left out, and so are the endpoints of the model and of the reflection model: a requeued job serves the same
    models at new addresses, so the names alone say which model a loop reads, as ledger.resume_refusal compares them."""
    code = meta.get("code") or {}
    names = ("config_name", "config", "agent", "servers", "model", "starts", "master_seed", "seed_pool",
             "training_variant", "training_fold", TRAINING_SPLIT_KEY, "adapter")
    gepa = {key: value for key, value in (meta.get("gepa") or {}).items() if key != "reflection_endpoint"}
    return {**{name: meta.get(name) for name in names}, "gepa": gepa, "store": (meta.get("backend") or {}).get("kind"),
            "code": code.get("identity") or code}


def as_stored(value):
    """A value as it reads back from a JSON file in its public form, so that an identity built now compares equal to the one a
    file holds."""
    return json.loads(json.dumps(public_value(value), default=str))


def loops_of(out, identity, statuses):
    """The run folders under out that hold a loop of this identity whose result file has one of the statuses, in name order."""
    found = []
    for path in sorted(Path(out).glob(f"*/{RESULT}")):
        saved = read_json(path, {}) or {}
        if saved.get("status") in statuses and saved.get("identity") == identity:
            found.append(path.parent)
    return found


def already_finished(out, identity):
    """The run folder of a finished loop of this identity under out when no unfinished loop of it lies there, else None, so that a
    requeued job, which starts every loop again with --resume, plays a finished loop never again."""
    if loops_of(out, identity, UNFINISHED):
        return None
    done = loops_of(out, identity, ("finished",))
    return done[0] if done else None


def folder_to_resume(out, identity):
    """The run folder of the unfinished loop of this identity under out, None when there is none. Several stop the command,
    since a resume would have to choose."""
    found = loops_of(out, identity, UNFINISHED)
    if len(found) > 1:
        raise SystemExit(f"--resume found {len(found)} unfinished loops of these settings under {out}: "
                         f"{', '.join(path.name for path in found)}. Move all but one aside")
    return found[0] if found else None
