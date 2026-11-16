"""Digests of the hydraulic rig invariants that the asset layer refactor must keep (CausalDiagnosis/docs/adding-a-dataset.md).

Every digest is the SHA-256 of a canonical JSON text whose keys keep the order the code wrote them, so a reordered key
counts as a change. The digests are keyed by the SHA-256 of the facts file, since most of the values they cover depend on
it, and they hold no value computed from the data, which is why the digest file may enter the repository. The digest
presets are every preset file of configs/, composed as the drivers compose it and read from the folder when the digests
are computed, so a new preset file is covered without an edit. The families are config (EpisodeConfig() field by field
and every preset), starts (the start, schedule, hazard table and fix history of seeds 0 to 199 per preset), logs and
verdicts (the reference policies on the development seeds of every preset), scripted (the files of the scripted wiring
episode through both servers on every preset that replays byte for byte through server processes, and one episode of the
AssetOpsBench rows per preset and seed played in process through the metering layer on the memory store), texts (the
advertised tools of both servers, the profile and the briefing at every level of every preset on both surfaces, the tools
of the four metered servers per preset, the agent's prompts on a fixed board on the action server and on the surface of the
AssetOpsBench rows with the tool arguments that surface writes), and models (a fresh fit of models.json
without its date, and the rules key of the default grade table). An episode played in process reads every clock from the
tool: the start at a fixed origin and call k at the origin plus k times think_seconds_per_cycle of the preset, and on a
preset of the clock of generated tokens the tokens of one think_seconds_per_cycle before every call, so it replays byte for byte
on a preset that charges thinking time. A label of record names the commit its digests were
computed on: --write into the digest file of the checkout needs a commit hash, from --commit or from a checkout with
no change against its commit, and the golden tests compare only with such a label.

usage:
  python -m CausalDiagnosis.experiments.run_verification.invariant_digests --facts FACTS --label NAME --write CausalDiagnosis/experiments/run_verification/invariant_digests.json
  python -m CausalDiagnosis.experiments.run_verification.invariant_digests --facts FACTS --label NAME --check CausalDiagnosis/experiments/run_verification/invariant_digests.json
      [--expect-changed PREFIX ...] [--families config,starts,logs,scripted,texts,models] [--detail DIR]
--detail writes the per seed digests to DIR, a folder outside the repository, so a changed family can be traced to seeds.
"""
import argparse
import asyncio
import contextlib
import dataclasses
import datetime
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

from CausalDiagnosis.utils.paths import preset_files  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.servers_of_a_run import ROOT
from CausalDiagnosis.experiments.episode_runners.run_configuration import load_config, load_world  # noqa: E402  (common puts this checkout's src first on sys.path)
from CausalDiagnosis.datasets.dataset_loader import recordings as records  # noqa: E402
from CausalDiagnosis.datasets.dataset_loader.recordings import file_sha256  # noqa: E402  the SHA-256 of a file, read by test_digests.py through this module
from CausalDiagnosis.utils.json_digest import digest  # noqa: E402

FORMAT = 1
FAMILIES = ("config", "starts", "logs", "scripted", "texts", "models")
RESERVED_PRESET_NAMES = ("default", "rows")  # the keys config.default, scripted.rows and texts.rows hold these beside the presets
START_SEEDS = tuple(range(200))
SCRIPTED_SEEDS = 2
SCRIPTED_FILES = ("hidden/log.json", "released.json", "predictions.json", "episode.json")
DIGESTS_FILE = Path(__file__).resolve().parent / "invariant_digests.json"  # the digest file of this checkout, which holds the labels of record
# the clocks of the calls, on the wall, the monotonic and the thread clock, with the clock of the thinking time that they feed,
# the fields that the driver writes outside any call (run_episodes.DRIVER_STATE_FIELDS), the end of the episode and the flush of the
# model server's cache with its times, and the checkout's path
VOLATILE_KEYS = {"episode.json": ("last_call_wall_time", "episode_started_at", "called_at", "think_seconds_monotonic", "last_call_monotonic_time",
                                  "think_seconds_wall", "handover_wall", "handover_mono", "charged_until_mono", "transcript_cursor",
                                  "failures_cursor", "credited_segments", "started_servers", "open_service", "refused_processes",
                                  "credited_seconds", "uncredited_seconds", "failed_model_seconds", "failed_model_attempts", "facts_path",
                                  "episode_ended_at", "cache_flush")}
PLAN_LENGTH = None  # the agent's default, no count of steps per plan


def presets():
    """The digest presets: every preset file of configs/, the folder the drivers read, by its file name. The folder is read
    when the digests are computed, so a preset file that a later change adds is covered without an edit. A name that a key
    of a family holds beside the presets is refused."""
    from CausalDiagnosis.experiments.episode_runners import run_configuration

    names = sorted(path.stem for path in preset_files(run_configuration.CONFIGS).values())
    reserved = [name for name in names if name in RESERVED_PRESET_NAMES]
    if reserved:
        raise SystemExit(f"rename the preset files {reserved} of {run_configuration.CONFIGS}: the digest keys hold these names beside the presets")
    return names


def preset(name):
    """The EpisodeConfig of a preset of configs/, composed as the drivers compose it."""
    return load_config(name)[1]


def process_presets():
    """The presets whose scripted wiring episode replays byte for byte through the server processes: their clock charges no
    wall seconds between two calls, which the state file would keep, and they anchor no recorded scenario, since the driver
    plays every scenario of an anchored preset under every seed and its scenario file is fetched rather than tracked."""
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.clock import wall_clock

    configs = {name: preset(name) for name in presets()}
    return [name for name, config in configs.items() if not wall_clock.charges_wall_seconds(config) and config.anchors is None]


def config_digests(world, detail):
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig

    default = EpisodeConfig().to_json()
    out = {"config.default": digest(default)}
    out.update({f"config.default.{field}": digest(value) for field, value in default.items()})
    out.update({f"config.{name}": digest(preset(name).to_json()) for name in presets()})
    return out


def start_digests(world, detail):
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv

    cycles, blocks = world
    out = {}
    for name in presets():
        env = HydraulicRigEnv(cycles, blocks, preset(name))
        parts = {"start": [], "schedule": [], "hazard": [], "fix_history": []}
        for seed in START_SEEDS:
            env.reset(seed)
            parts["start"].append([list(env.initial), env.window_start, env.entry_offset])
            parts["schedule"].append(env.clock.schedule)
            parts["hazard"].append(digest(env.clock.hazard))
            parts["fix_history"].append(env.fix_history)
        for part, values in parts.items():
            out[f"starts.{name}.{part}"] = digest(values)
            detail[f"starts.{name}.{part}"] = [digest(v) for v in values]
    return out


def log_digests(world, detail):
    """The logs and the verdicts of the reference policies on every digest preset and its development seeds, the smallest
    seed of every start combination, played in process as run_reference_policies.py plays them."""
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment import start_state_seeds as start_states, verifier
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.gymnasium_wrapper import oracle_cost
    from CausalDiagnosis.environment.reference_policies.policies import run_episode
    from CausalDiagnosis.experiments.episode_runners.run_reference_policies import policy_table

    cycles, blocks = world
    table = policy_table(cycles, blocks)
    out = {}
    for name in presets():
        config = preset(name)
        seeds = start_states.combination_seeds(HydraulicRigEnv(cycles, blocks, config), 1)
        out[f"logs.{name}.seeds"] = digest(seeds)
        for policy, make in table.items():
            logs, verdicts = [], []
            for seed in seeds:
                env = HydraulicRigEnv(cycles, blocks, config)
                log = run_episode(env, make(env, seed), seed)
                # the verifier's own cost of the oracle episode, the normalizer of the drivers and the gym
                cost_oracle = oracle_cost(HydraulicRigEnv(cycles, blocks, config), seed)
                verdict = verifier.verify(log, verifier.released_from_env(env), env.oracle(), cost_oracle, grade=None,
                                          scope=config.verifier_scope)
                logs.append(digest(log))
                verdicts.append(digest(verdict))
            out[f"logs.{name}.{policy}"] = digest(logs)
            out[f"verdicts.{name}.{policy}"] = digest(verdicts)
            detail[f"logs.{name}.{policy}"] = dict(zip(map(str, seeds), logs))
            detail[f"verdicts.{name}.{policy}"] = dict(zip(map(str, seeds), verdicts))
    return out


def scripted_digests(world, detail, facts):
    """The files of the scripted wiring episode through both server processes on every preset of process_presets, without
    the wall clock and the path of the checkout, and one episode of the AssetOpsBench rows per digest preset and seed
    (a declared digest family). predictions.json is the diagnostics server's own file, so it changes with the server and the models."""
    from CausalDiagnosis.experiments.episode_runners import run_episodes
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.fit_models import MODELS_PATH, ensure_models

    ensure_models(facts_path=facts)
    out = {"scripted.models_used": digest(undated(json.loads(MODELS_PATH.read_text(encoding="utf-8"))))}
    with tempfile.TemporaryDirectory() as tmp:
        for name in process_presets():
            target = Path(tmp) / f"scripted_{name}"
            with contextlib.redirect_stdout(io.StringIO()):
                run_episodes.main(["--config", name, "--agent", "scripted", "--seeds", str(SCRIPTED_SEEDS), "--no-grade",
                                   "--facts", str(facts), "--out", str(target)])
            run_dir = next(p for p in sorted(target.iterdir()) if p.is_dir())
            for episode in sorted(p for p in run_dir.iterdir() if p.is_dir()):
                for rel in SCRIPTED_FILES:
                    path = episode / rel
                    if path.exists():
                        data = json.loads(path.read_text(encoding="utf-8"))
                        for key in VOLATILE_KEYS.get(path.name, ()):
                            data.pop(key, None)
                        out[f"scripted.{name}.{episode.name}.{rel}"] = digest(data)
    for family in digest_families():
        out.update(family.scripted(facts))
    return out


def text_digests(world, detail, facts):
    from CausalDiagnosis.experiments.episode_runners import run_episodes  # builds the agent's surface
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.situation_board import Board
    from CausalDiagnosis.agents.Causal_Diagnosis_Agent.prompts import orientation_turn, planning_turn, summary_turn, system_prompt
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.main import build_server as diagnostics_server
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.briefing import briefing
    from CausalDiagnosis.mcp_servers.Environment_Actions_MCP.main import build_server as action_server

    cycles, blocks = world
    with tempfile.TemporaryDirectory() as tmp:
        action = asyncio.run(action_server(Path(tmp) / "episode.json").list_tools())
    diagnostics = asyncio.run(diagnostics_server(None).list_tools())
    out = {"texts.tools.hydraulic_rig": digest([t.model_dump(mode="json") for t in action]),
           "texts.tools.causal_diagnosis": digest([t.model_dump(mode="json") for t in diagnostics])}
    for name in presets():
        config = preset(name)
        profile = HydraulicRigEnv(cycles, blocks, config).reset(0)["result"]
        out[f"texts.profile.{name}"] = digest(profile)
        for level in (1, 2, 3):
            out[f"texts.briefing.{name}.{level}"] = digest(briefing(profile, dataclasses.replace(config, guidance_level=level)))
        for family in digest_families():
            out.update(family.texts(world, name, config, facts))
    config = preset("static")
    env = HydraulicRigEnv(cycles, blocks, config)
    profile = env.reset(0)["result"]
    envelope = env.step({"type": "get_failure_modes", "component": ""})[0]
    question = briefing(profile, config)
    # the agent's texts as the driver builds them on the action server of the asset: its surface and the question of the episode
    surface = run_episodes.causal_surface(config.asset_id)
    board = Board(evidence_form=surface.evidence_form, reads_take_time=surface.reads_take_time, noun=surface.noun, question=question)
    board.set_profile(profile)
    board.set_failure_modes(envelope["result"])
    board.update_status(envelope)
    out["texts.agent.system"] = digest(system_prompt(PLAN_LENGTH, surface))
    out["texts.agent.orientation"] = digest(orientation_turn(question, profile, board.failure_modes, PLAN_LENGTH, surface=surface))
    out["texts.agent.planning"] = digest(planning_turn(board, PLAN_LENGTH, surface=surface))
    out["texts.agent.summary"] = digest(summary_turn(board, PLAN_LENGTH, surface=surface))
    for family in digest_families():
        out.update(family.agent_texts(question, profile, envelope, facts, detail))
    return out


def digest_families():
    """The digest families that the frameworks declare (utils/frameworks.py, DIGEST_FAMILIES), each imported on first use."""
    from CausalDiagnosis.utils import frameworks

    return [frameworks.resolve(reference) for _framework, _name, reference in frameworks.references("DIGEST_FAMILIES")]


def model_digests(world, detail, facts):
    """A fresh fit, never the cached models.json, so a stale file cannot hide a change of the fit."""
    from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline import fit_models as fit, diagnosability_grade as grade

    rules = grade.default_rules()
    return {"models.fit": digest(undated(fit.fit(str(facts)))), "models.grades_rules_key": grade.rules_key(rules, rules["budget"])}


def undated(models):
    """models.json without its generated date, so that a fit of another day digests as the same models."""
    models.pop("generated", None)
    return models


def compute(facts, families=FAMILIES):
    """{digest name: SHA-256} of the chosen families and the per seed detail, on this checkout's code."""
    _facts, cycles, blocks = load_world(str(facts))
    world, detail, out = (cycles, blocks), {}, {}
    for family in families:
        if family == "config":
            out.update(config_digests(world, detail))
        elif family == "starts":
            out.update(start_digests(world, detail))
        elif family == "logs":
            out.update(log_digests(world, detail))
        elif family == "scripted":
            out.update(scripted_digests(world, detail, facts))
        elif family == "texts":
            out.update(text_digests(world, detail, facts))
        elif family == "models":
            out.update(model_digests(world, detail, facts))
        else:
            raise SystemExit(f"unknown digest family {family!r}, use {', '.join(FAMILIES)}")
    return out, detail


def compare(stored, computed, expected=()):
    """(same, changed, missing, new, unexpected) between two digest maps. A change is expected when its name starts with
    one of the expected prefixes."""
    same = sorted(k for k in stored if k in computed and stored[k] == computed[k])
    changed = sorted(k for k in stored if k in computed and stored[k] != computed[k])
    missing = sorted(k for k in stored if k not in computed)
    new = sorted(k for k in computed if k not in stored)
    unexpected = [k for k in changed + missing + new if not any(k.startswith(p) for p in expected)]
    return same, changed, missing, new, unexpected


def load_baseline(path, facts_hash, label):
    """The stored digests of a label under the hash of a data file, refused by name when the file holds none."""
    baselines = json.loads(Path(path).read_text(encoding="utf-8"))["baselines"]
    if label not in baselines.get(facts_hash, {}):
        raise records.DataFileError(f"{path} holds no baseline {label} for the data file with SHA-256 {facts_hash}. Write it with "
                                    f"`python -m CausalDiagnosis.experiments.run_verification.invariant_digests --facts FILE --label {label} --write {path}`.")
    return baselines[facts_hash][label]["digests"]


def is_commit_hash(value):
    """Whether value names a commit as git prints it: 7 to 40 lowercase hexadecimal digits."""
    return isinstance(value, str) and 7 <= len(value) <= 40 and set(value) <= set("0123456789abcdef")


def record_digests(path, facts_hash, label):
    """(digests, None) when the digest file at path holds a label of record named label for the data file whose SHA-256
    is facts_hash, else (None, the reason). A label of record names the commit its digests were computed on, as the label
    that is written after the push in a clone of the pushed commit; a label without a commit hash is a working label."""
    entry = json.loads(Path(path).read_text(encoding="utf-8"))["baselines"].get(facts_hash, {}).get(label)
    if entry is not None and is_commit_hash(entry.get("commit")):
        return entry["digests"], None
    held = (f"holds no label {label}" if entry is None
            else f"holds the label {label} with the commit {entry.get('commit')!r}, which is no commit hash")
    return None, (f"{path} {held} for the data file with SHA-256 {facts_hash}, so it holds no label of record. That label is "
                  f"written after the push on the platform of record, in a clone of the pushed commit, by python "
                  f"CausalDiagnosis/experiments/run_verification/invariant_digests.py --facts FILE --label {label} --write {path} --commit HASH")


def of_families(digests, families):
    """The digests of the chosen families, the verdicts with the logs family that computes them."""
    return {k: v for k, v in digests.items() if k.split(".", 1)[0] in families or k.startswith("verdicts.") and "logs" in families}


def checkout_commit():
    """The short hash of the checkout's commit when no file of the checkout differs from it, else None, so that a label
    names only the commit its digests were computed on."""
    try:
        changed = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        if changed.strip():
            return None
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="digests of the hydraulic rig invariants, keyed by the facts file hash")
    ap.add_argument("--facts", required=True)
    ap.add_argument("--label", required=True, help="the name of the baseline, such as current")
    ap.add_argument("--commit", default=None, help="the commit the code is at, when the checkout is an export without git")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--write", type=Path, default=None, help="add the baseline to this digest file")
    ap.add_argument("--check", type=Path, default=None, help="compare with the baseline of this digest file")
    ap.add_argument("--expect-changed", nargs="*", default=(), help="digest name prefixes whose change the check accepts")
    ap.add_argument("--detail", type=Path, default=None, help="a folder outside the repository for the per seed digests")
    a = ap.parse_args(argv)
    commit = a.commit
    if commit is None and a.write is not None:
        commit = checkout_commit()
    if a.write is not None and a.write.resolve() == DIGESTS_FILE.resolve() and not is_commit_hash(commit):
        ap.error(f"{a.write} holds the labels of record, and a label of record names the commit its digests were computed on: "
                 f"give --commit HASH in a clone of the pushed commit, or write from a checkout with no change against its "
                 f"commit (the commit is {commit!r})")
    families = tuple(f for f in a.families.split(",") if f)
    facts_hash = file_sha256(a.facts)
    digests, detail = compute(Path(a.facts), families)
    if a.detail is not None:
        a.detail.mkdir(parents=True, exist_ok=True)
        (a.detail / f"{a.label}.json").write_text(json.dumps(detail, indent=1) + "\n", encoding="utf-8")
    status = 0
    if a.check is not None:
        stored = of_families(load_baseline(a.check, facts_hash, a.label), families)
        same, changed, missing, new, unexpected = compare(stored, digests, a.expect_changed)
        print(f"{len(same)} same, {len(changed)} changed, {len(missing)} missing, {len(new)} new against {a.label}")
        for name in changed + missing + new:
            state = "changed " if name in changed else "missing " if name in missing else "new "
            print(("expected " if name not in unexpected else "") + state + name)
        status = 1 if unexpected else 0
    if a.write is not None:
        data = json.loads(a.write.read_text(encoding="utf-8")) if a.write.exists() else {"format": FORMAT, "baselines": {}}
        data["baselines"].setdefault(facts_hash, {})[a.label] = {
            "commit": commit, "date": datetime.date.today().isoformat(), "families": list(families),
            "digests": digests}
        a.write.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {len(digests)} digests as {a.label} under the facts hash {facts_hash[:12]} to {a.write}")
    return status


if __name__ == "__main__":
    sys.exit(main())
