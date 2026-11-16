import contextlib
import io
import types
from pathlib import Path
from CausalDiagnosis.experiments.run_verification import run_ledger as ledger  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import ROOT  # noqa: E402
from CausalDiagnosis.experiments.run_verification.run_identity_file_lists import AGENT_LIST  # noqa: E402
from CausalDiagnosis.experiments.table_definitions.training_variants import asset_of  # noqa: E402  the asset of a run as the tables read it


# the entries of the data record that a run reads as they were recorded, the recordings and the fetched scenario records. Every
# other entry is a generated table, which a fix computes again although no value that an episode reads changes, and a value that an
# episode or the verifier reads shows in an answer, in a file the episode left or in a score cell
RECORDED_DATA = ledger.RECORDED_DATA


# ----- the record of a run -----
def store_kind(meta):
    """The kind of store a run played its metered servers on, None for a run on the rig's action server."""
    return (meta.get("backend") or {}).get("kind")


def store_of(meta):
    """(store settings of a replay, the reason it is advisory or None). A run on the rig's action server needs no store. A run on
    CouchDB replays on the CouchDB that the environment names, as the driver reaches it, and else on the memory store, which is
    advisory for it."""
    kind = store_kind(meta)
    if kind is None:
        return None, None
    if kind != "memory":
        from CausalDiagnosis.experiments.episode_runners import run_episodes

        try:
            from CausalDiagnosis.utils import frameworks
            surface = next(name for name, entry in frameworks.surfaces().items() if kind in entry["stores"])
            return run_episodes.store_settings(types.SimpleNamespace(servers=surface, store=kind)), None
        except SystemExit:
            pass
    return {"kind": "memory"}, (None if kind == "memory" else f"the run played on the store {kind} and the replay on the memory store")


def platform_reason(meta, root=ROOT):
    """Why the platform here is not the platform the run recorded, or None when the two records are equal."""
    recorded = meta.get("platform")
    if not recorded:
        return "the run recorded no platform"
    here = ledger.platform_of(root)
    differing = sorted(key for key in set(recorded) | set(here) if recorded.get(key) != here.get(key))
    return f"the platform here differs from the run's in {', '.join(differing)}" if differing else None


def data_differences(meta, facts_path):
    """(names of the data files and generated tables whose hash in the run's record differs from the file here, a note or None).
    A run without the data record is compared on the prefix of its facts file, and the note says so."""
    recorded = meta.get("data")
    if recorded is None:
        prefix, digest = meta.get("facts_sha256") or "", ledger.sha256_of(facts_path) or ""
        return ([] if prefix and digest.startswith(prefix) else ["facts"]), "the run recorded no data record, so only the prefix of its facts file was compared"
    here = ledger.run_data.data_record(facts_path, asset_of(meta))
    return sorted(name for name in set(recorded) | set(here) if recorded.get(name) != here.get(name)), None


def changed_files(meta, root=ROOT):
    """{list: [paths]} of the files that differ between the code record of a run and the tree at root, a file gone or added included.
    A run before the lists recorded SHA-256 prefixes of its code, which compare under the empty name, and the AppleDouble files
    (._name) that a tar stream from macOS left beside the code are no code."""
    code = meta.get("code") or {}
    if code.get("files"):
        now, changed = ledger.code_identity.tree_files(root), {}
        for name in sorted(set(code["files"]) | set(now)):
            was, found = code["files"].get(name) or {}, now.get(name) or {}
            paths = sorted(path for path in set(was) | set(found) if was.get(path) != found.get(path))
            if paths:
                changed[name] = paths
        return changed
    paths = sorted(path for path, prefix in ledger.recorded_code(meta).items()
                   if not Path(path).name.startswith("._") and (not (Path(root) / path).is_file() or not ledger.sha256_of(Path(root) / path).startswith(prefix)))
    return {"": paths} if paths else {}


def agent_list(meta):
    return f"{AGENT_LIST}{meta.get('agent')}"


def loaded_changed(meta, changed):
    """{list: [paths]} of the changed files that the driver process of the run had loaded, as run.json names them."""
    loaded = set(meta.get("loaded_files") or [])
    return {name: hits for name, paths in changed.items() if (hits := [path for path in paths if path in loaded])}


def _driver_config(argv, blocks):
    """The configuration that the driver composes from its command line and plays (run_episodes.main): the preset with its overrides
    and its price table, widened by run_episodes.played_config for a run of the development starts or of the development split."""
    from CausalDiagnosis.experiments.episode_runners import run_episodes

    a = run_episodes.build_parser().parse_args(argv)
    return run_episodes.played_config(a, run_episodes.configure(a)[1], blocks)


def _loop_config(argv, blocks):
    """The configuration of the online loop of GEPA, which composes the command line of the driver from its own and plays what
    run_episodes.played_config gives for it, the development starts in a loop of the development split (gepa_online.main)."""
    from CausalDiagnosis.training import gepa_prompt_optimization as gepa_online
    from CausalDiagnosis.experiments.episode_runners import run_episodes

    settings = gepa_online.driver_settings(gepa_online.build_parser().parse_args(argv))
    return run_episodes.played_config(settings, run_episodes.configure(settings)[1], blocks)


def rescored_scope(meta):
    """The verifier scope under which rescore_under_verifier_scope.py scored a copy of a run again (run.json rescored.scope), or None for a run
    as it played. The copy holds the command of its run's ledger and the state file of every episode as the run played them, while
    its run.json, its verdicts and its score cells follow the scope of the rescoring."""
    return (meta.get("rescored") or {}).get("scope")


def config_problem(run_dir, meta, blocks):
    """The problem of a run whose recorded command composes in this tree another configuration than the one its run.json records, or
    None. The command is the argv of the run's ledger, composed as the program that starts a run of its mode composes it and, for a
    copy that rescore_under_verifier_scope.py scored again, under the scope of the rescoring (rescored_scope), so a fix of
    a preset, of the layer composition, of a price table or of the development starts is named, although every episode replays from
    the configuration of its own state file. The problem names the keys that differ and never a value."""
    from CausalDiagnosis.training import gepa_prompt_optimization as gepa_online

    name = Path(run_dir).name
    argv = ((ledger.load(run_dir) or {}).get("command") or {}).get("argv")
    if argv is None:
        return f"{name}: the ledger of the run records no command, so the replay cannot compose its configuration"
    compose = _loop_config if meta.get("mode") == gepa_online.MODE else _driver_config
    errors = io.StringIO()
    try:
        with contextlib.redirect_stderr(errors):
            composed = compose(argv, blocks).to_json()
    except (SystemExit, Exception) as exc:  # noqa: BLE001 - a command that composes nothing is named, and the replay goes on
        why = (errors.getvalue().strip().splitlines() or [str(exc)])[-1].split(": error: ")[-1].replace(f"{ROOT}/", "")
        return f"{name}: the command of the run composes no configuration in this tree ({why})"
    if (scope := rescored_scope(meta)) is not None:  # the one key of the configuration that the rescoring changes
        composed = {**composed, "verifier_scope": scope}
    recorded = meta.get("config") or {}
    differing = sorted(key for key in set(composed) | set(recorded) if composed.get(key) != recorded.get(key))
    if differing:
        return f"{name}: the configuration that the command of the run composes in this tree differs from run.json in {', '.join(differing)}"
    return None
