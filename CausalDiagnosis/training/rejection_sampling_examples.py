"""The training data of rejection sampling fine tuning (CausalDiagnosis/docs/experiments.md, section Training in the
environment): the examples of one agent and one fold built from the model calls of its sampling runs, the filtered arm from the
episodes that the verifier passed and the control, fine-tuning on unfiltered trajectories, from as many episodes drawn whatever their verdict.

usage, from the fork root with the fork's virtual environment first on PATH and HYD_TRAINING set:
  python -m CausalDiagnosis.training.rejection_sampling_examples --runs <sampling run folder> [<sampling run folder> ...]
  python -m CausalDiagnosis.training.rejection_sampling_examples --search <folder of run folders> [...] --agent <agent> --training-fold <fold>
the second form takes every finished sampling run of that agent and fold under the folders, by run.json.

The runs are the samples of one agent and one fold, the fold its model holds out, those of the first round, 1 to
samples_per_state of configs/training/<asset_id>/rft.json, or of the first round and those the minimum adds, run.json
in the mode training, each playing the training start states of the fold once (run_episodes/ --starts training
--training-sample k --training-fold f). Every seed is checked against the training start states of its configuration and
fold, so no test start state, reserved seed or label rollout seed enters training. The filtered arm keeps the episodes whose
hidden verdict passed. The control draws as many episodes, uniformly without replacement under control_seed, from every sampled
episode that holds a usable call, and never reads a verdict. Every usable model call of a kept episode becomes one example, the
prompt as the agent sent it and the reply as the completion, every kind of call alike. The CausalDiagnosis agent sends its
prompt as one user message on the chat route, which the model's chat template wraps, and the two paper-era ports,
Agent-As-Tool and Plan-Execute, send raw text on the completions route, whose reply ends
with the stop string that ended it. A call without its prompt, such as a call of the released Agent-As-Tool, which plays the reproduction
alone, is refused, so no example is encoded as a request the server never received. An example keeps the day of its call, which
the chat template prints, and the prompt tokens the server counted, which the trainer checks its own encoding against. A call
that raised, an empty reply and a reply cut at the token cap become no example, in both arms alike, and the trainer leaves out
an example longer than the maximum length.

data/<agent>_f<fold> of the training folder gets examples_filtered.jsonl, examples_unfiltered.jsonl and manifest.json with the
runs, the agent's label in scores.csv, the fold, the counts the data decides (episodes sampled, kept by the filter, examples per
arm, calls dropped by reason) and the SHA-256 of every file. Each arm also counts its episodes whose hidden log played a
combination that holds a single fault of the fold its model holds out: the split is closed under repair, so only a drop of the
layer 5 hazard leads there, and the table of training records prints the count. A run of a fine tuned variant records these
counts with its adapter (training.adapter_of), which is how that table reads them.

The minimum of kept episodes: the manifest's gate counts the kept episodes, those the verifier passed that hold a usable call,
against kept_per_training_state of the asset's rft.json times the training start states of the fold's model, read from the asset's split, with the
distinct training start states among them. Below it after the first round the gate asks for one more round of the same size,
the samples of training.extension_samples, and the builder runs again on the runs of both rounds; below it after the last round
the gate says trained false, train_lora.py refuses the data and the fine tuned variants of that fold write marker runs that the
tables replace by the runs without training. The command prints the gate in one line that starts with GATE."""
import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent

from CausalDiagnosis.experiments.episode_runners.run_folder_layout import HIDDEN, episode_name, read_rows
from CausalDiagnosis.utils.json_files import read_json  # noqa: E402
from CausalDiagnosis.utils import frameworks  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import verifier  # noqa: E402  the one predicate of a good episode
from CausalDiagnosis.experiments.table_definitions.training_variants import (AS_RELEASED, DATA_MANIFEST, DEFAULT_ASSET, DEVELOPMENT, FILTERED, FOLD_CONVENTION, FOLD_CONVENTION_KEY, REGISTERED,  # noqa: E402
                      TRAINING_MODE, TRAINING_SPLIT_KEY,
                      UNFILTERED, VISITS, asset_of, asset_training, check_fold, data_name, gate_of, numbers, round_samples, sha256_file,
                      training_root, visits_test_combination)

FILTER = "good"  # the rule of the filter that rft.json registers, verifier.good
FORMAT = 4  # the fold convention, with the gate of the minimum of kept episodes and the folds and the visits of the cross-fitting
EXAMPLES = {FILTERED: "examples_filtered.jsonl", UNFILTERED: "examples_unfiltered.jsonl"}
DROPPED = ("error", "empty", "cut")
# the route every model call of an agent takes: the CausalDiagnosis agent on the chat route, a declared agent on the route it declares
ROUTES = {**{name: entry["route"] for name, entry in frameworks.agents().items() if "route" in entry}, "causal": "chat"}
# the fields of run.json on which every sampling run of one agent and fold agrees, since the data of an adapter comes from one
# condition: the asset, the split, the configuration, the code that played it, the model, the tool servers, the switch of the MCP,
# the store, the sampling temperature and the development stamp, so that stamped runs never mix with registered ones. The endpoint
# of the model is no part of it, since two serving jobs of the same model, a requeued job among them, serve it at two addresses
IDENTITY = {"asset": asset_of, "split": lambda meta: meta.get(TRAINING_SPLIT_KEY) or REGISTERED, "configuration": lambda meta: meta.get("config_name"),
            "code identity": lambda meta: (meta.get("code") or {}).get("identity"), "model": lambda meta: meta.get("model"),
            "tool servers": lambda meta: meta.get("servers"),
            "MCP switch": lambda meta: meta.get("diagnostics_server"), "store": lambda meta: (meta.get("backend") or {}).get("kind"),
            "sampling temperature": lambda meta: meta.get("sampling_temperature"), "stamp": lambda meta: meta.get("stamp")}


def check_identity(runs):
    """Refuses sampling runs that differ in a field of IDENTITY, and names the field and two runs that differ in it."""
    first_dir, first = runs[0]
    for field, value_of in IDENTITY.items():
        for run_dir, meta in runs[1:]:
            if value_of(meta) != value_of(first):
                raise SystemExit(f"the sampling runs differ in their {field}: {first_dir.name} has {value_of(first)!r} and {run_dir.name} has "
                                 f"{value_of(meta)!r}, and the data of one adapter comes from one condition")


def sampling_runs(run_dirs, samples=None):
    """(agent, fold, asset, split, [(run folder, run.json)] in sample order, rounds) of the sampling runs of one agent and one fold
    of one asset and split,
    refused unless every run is a finished sampling run of the base weights and the runs hold the samples of whole rounds, 1 to
    samples times the rounds, exactly once, at most the first round and the extension rounds of the asset's minimum. samples is
    samples_per_state of the asset unless the caller names it."""
    runs = []
    for run_dir in map(Path, run_dirs):
        meta = read_json(run_dir / "run.json")
        if meta is None or not meta.get("ended"):
            raise SystemExit(f"{run_dir} is no finished run")
        if meta.get("mode") != TRAINING_MODE or meta.get("training_variant", AS_RELEASED) != AS_RELEASED:
            raise SystemExit(f"{run_dir} is a run of mode {meta.get('mode')!r} and variant {meta.get('training_variant')!r}, and the "
                             f"data of one round comes from sampling runs (mode {TRAINING_MODE}) of the agent without training")
        runs.append((run_dir, meta))
    agents, folds = {m.get("agent") for _, m in runs}, {m.get("training_fold") for _, m in runs}
    if len(agents) != 1 or len(folds) != 1:
        raise SystemExit(f"the runs mix the agents {sorted(map(str, agents))} or the folds {sorted(map(str, folds))}, and one adapter "
                         "learns one agent on one fold")
    check_identity(runs)
    fold, asset_id, split = folds.pop(), asset_of(runs[0][1]), runs[0][1].get(TRAINING_SPLIT_KEY) or REGISTERED
    check_fold(fold, asset_id, split)
    registered = numbers(asset_id)
    samples = samples or registered.samples_per_state
    played = sorted(m.get("training_sample") for _, m in runs)
    rounds = next((r for r in range(1, 2 + registered.extension_rounds) if played == list(range(1, r * samples + 1))), None)
    if rounds is None:
        raise SystemExit(f"the runs hold the training samples {played}, and the data needs whole rounds of {samples} samples from 1, "
                         f"at most {1 + registered.extension_rounds} rounds, each sample once")
    return agents.pop(), fold, asset_id, split, sorted(runs, key=lambda run: run[1]["training_sample"]), rounds


def label_of(runs):
    """The agent column of scores.csv over the sampling runs, the label under which the agent's evaluation runs enter the tables."""
    labels = {row["agent"] for run_dir, _ in runs for row in read_rows(run_dir)}
    if len(labels) != 1:
        raise SystemExit(f"the sampling runs carry the agent labels {sorted(labels)} in scores.csv, and one agent has one label")
    return labels.pop()


def episodes_of(runs):
    """Every episode the runs planned, in run and seed order, with what the builder reads of it and nothing of its verdict."""
    out = []
    for run_dir, meta in runs:
        date = datetime.fromisoformat(meta["started"]).date().isoformat()
        for seed in meta["seeds"]:
            name = episode_name(seed)
            out.append({"run": run_dir.name, "run_dir": run_dir, "episode": name, "ep_dir": run_dir / name, "seed": seed,
                        "config": meta["config_name"], "sample": meta["training_sample"], "date": date, "fold": meta["training_fold"],
                        "asset_id": asset_of(meta), "split": meta.get(TRAINING_SPLIT_KEY) or REGISTERED,
                        "test_set": (meta.get("master_seed"), meta.get("seed_pool"))})
    return out


def ended_completion(call):
    """The completion of a raw text call with the stop string that ended it, which a reply kept untrimmed (no_stop_trim, as the
    port asks) holds already and a trimmed reply lacks. SGLang reports the end token by its id or not at all, and the trainer
    then adds the end token."""
    reply, stop = call["reply"], call.get("matched_stop")
    if not isinstance(stop, str):
        return {"completion": reply, "ends_at_stop": False}
    return {"completion": reply if reply.endswith(stop) else reply + stop, "ends_at_stop": True}


def call_examples(ep_dir, agent):
    """(examples, Counter of the dropped calls by reason) of one episode's model calls, each on the route of its agent."""
    route = ROUTES[agent]
    examples, dropped = [], Counter()
    for index, call in enumerate(read_json(Path(ep_dir) / "llm_calls.json", []) or []):
        if "prompt" not in call:
            raise SystemExit(f"{ep_dir}: call {index} holds no prompt, as a call of the released Agent-As-Tool, so the request the "
                             "server received cannot be rebuilt")
        if call.get("error"):
            dropped["error"] += 1
        elif not (call.get("reply") or "").strip():
            dropped["empty"] += 1
        elif call.get("finish_reason") == "length":  # a reply cut at the cap would teach the model to stop in mid sentence
            dropped["cut"] += 1
        else:
            # the day of the call overrides the day the run started, and the server's prompt tokens let the trainer check its encoding
            recorded = {key: call[key] for key in ("date", "input_tokens") if call.get(key)}
            reply = ended_completion(call) if route == "completion" else {"completion": call["reply"]}
            examples.append({"call": index, "route": route, "prompt": call["prompt"], **reply, **recorded})
    return examples, dropped


def good(episode):
    """Whether the verifier found the episode good (verifier.good, a pass whose cited evidence the released record holds), read by
    the filter alone."""
    return verifier.good(read_json(Path(episode["ep_dir"]) / HIDDEN / "verdict.json", {}) or {})


def visits(episode):
    """Whether the episode played a combination that holds a single fault of the fold its model holds out, read from its hidden log."""
    return visits_test_combination(read_json(Path(episode["ep_dir"]) / HIDDEN / "log.json", []) or [], episode["fold"], episode["asset_id"],
                                   episode["split"])


def control_draw(pool, count, seed):
    """count episode names drawn uniformly without replacement from the pool, the same draw for the same names in any order."""
    return sorted(random.Random(seed).sample(sorted(pool), count))


def check_training_starts(episodes):
    """Refuses an episode whose seed enters at no training start state of its configuration, test set (the master seed and the
    seed pool its run recorded) and fold, or is a reserved seed, so no test start state, reserved seed or label rollout seed
    reaches the data. An episode of the development split is checked on its configuration widened to the development starts,
    as its run played it. Returns the count of training start states of the fold's model and the start state of every episode, by
    (run, episode), which the gate of the minimum of kept episodes reads."""
    from CausalDiagnosis.experiments.episode_runners.run_configuration import load_config, load_world
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment import start_state_seeds as start_states
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv

    worlds, counts, state_of = {}, set(), {}
    # the asset of an episode decides its world and the composition of its preset, so a port's runs are checked on their own records
    condition = lambda e: (e.get("asset_id") or DEFAULT_ASSET, e["config"], e["test_set"], e["fold"], e.get("split", REGISTERED))  # noqa: E731
    for asset, config_name, (master_seed, pool), fold, training_split in sorted(set(map(condition, episodes))):
        if asset not in worlds:
            worlds[asset] = load_world(None, asset)[1:]
        cycles, blocks = worlds[asset]
        config = load_config(config_name, asset)[1]
        if training_split == DEVELOPMENT:
            from CausalDiagnosis.experiments.episode_runners.run_episodes import development_config

            config = development_config(config, blocks)
        env = HydraulicRigEnv(cycles, blocks, config)
        split = asset_training(env.config.asset_id, cycles.facts_sha256, split=training_split).split
        states = set(start_states.training_start_states(env, split, fold, master_seed, pool))
        counts.add(len(states))
        reserved = start_states.reserved_seeds(env)
        for episode in (e for e in episodes if condition(e) == (asset, config_name, (master_seed, pool), fold, training_split)):
            env.reset(episode["seed"])
            if episode["seed"] in reserved or start_states.start_state(env) not in states:
                raise SystemExit(f"seed {episode['seed']} of {episode.get('run', 'the runs')} enters at no training start state of fold "
                                 f"{fold} of {config_name} or is a reserved seed or a label rollout seed, so it cannot be training data")
            state_of[(episode.get("run"), episode.get("episode"))] = start_states.start_state(env)
    if len(counts) != 1:
        raise SystemExit(f"the runs give the fold's model {sorted(counts)} training start states, and one model has one count")
    return {"training_states": counts.pop(), "states": state_of}


def build(run_dirs, out, samples=None, check_starts=check_training_starts):
    """Writes the two example files and the manifest of one agent and one fold into out and returns the manifest, from the
    registered numbers of the asset of the sampling runs."""
    agent, fold, asset_id, split, runs, rounds = sampling_runs(run_dirs, samples)
    training = asset_training(asset_id, split=split)
    config, config_file = training.config, training.config_file
    if config["rounds"] != 1 or config["filter"] != FILTER:
        raise SystemExit(f"{config_file} registers {config['rounds']} rounds and the filter {config['filter']!r}, and the builder makes "
                         f"one round filtered by the rule {FILTER}, a good episode of the verifier")
    episodes = episodes_of(runs)
    starts = check_starts(episodes)
    usable = {}
    for episode in episodes:
        examples, dropped = call_examples(episode["ep_dir"], agent)
        usable[(episode["run"], episode["episode"])] = (episode, examples, dropped)
    # an episode that met a single fault of the fold its model holds out leaves before the filter and before the draw of the
    # control, whatever its outcome, since it plays a combination that model is judged on
    left_out = {key for key, (episode, _, _) in usable.items() if visits(episode)}
    # the filter keeps an episode the verifier passed that holds a usable call, the episodes the gate counts
    kept = [key for key, (episode, examples, _) in usable.items() if key not in left_out and good(episode) and examples]
    pool = [key for key, (_, examples, _) in usable.items() if key not in left_out and examples]  # every sampled round, the control's pool
    first_round = set(round_samples(1, asset_id))
    if rounds > 1 and gate_of(sum(1 for key in kept if usable[key][0]["sample"] in first_round), 0, starts["training_states"], 1,
                              asset_id)["trained"]:
        raise SystemExit("the first round already kept the registered minimum, so no second round was due and its runs cannot "
                         "enter the data")
    gate = gate_of(len(kept), len({starts["states"][key] for key in kept}), starts["training_states"], rounds, asset_id)
    drawn = control_draw([f"{run}/{name}" for run, name in pool], len(kept), config["control_seed"])
    arms = {FILTERED: kept, UNFILTERED: [tuple(item.split("/", 1)) for item in drawn]}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"format": FORMAT, "agent": agent, "fold": fold, FOLD_CONVENTION_KEY: FOLD_CONVENTION, "asset_id": asset_id, TRAINING_SPLIT_KEY: split,
                "label": label_of(runs), "config": runs[0][1]["config_name"],
                "model": runs[0][1].get("model"), "stamp": runs[0][1].get("stamp"),
                "code_identity": (runs[0][1].get("code") or {}).get("identity"), "built": datetime.now().isoformat(timespec="seconds"),
                "training_config": {"file": str(config_file.relative_to(HERE.parents[1]) if config_file.is_relative_to(HERE.parents[1])
                                                else config_file), "sha256": sha256_file(config_file)},
                "runs": [{"run": run_dir.name, "training_sample": meta["training_sample"], "started": meta["started"],
                          "run_json_sha256": sha256_file(run_dir / "run.json"), "episodes": len(meta["seeds"])} for run_dir, meta in runs],
                "episodes_sampled": len(episodes), "episodes_with_a_usable_call": len(pool), "control_seed": config["control_seed"],
                "gate": {**gate, "episodes_sampled": len(episodes)}, "arms": {}}
    for arm, keys in arms.items():
        path, count, dropped = out / EXAMPLES[arm], 0, Counter({reason: 0 for reason in DROPPED})
        with open(path, "w", encoding="utf-8") as f:
            for key in sorted(keys):
                episode, examples, episode_dropped = usable[key]
                dropped.update(episode_dropped)
                for example in examples:
                    f.write(json.dumps({"agent": agent, "arm": arm, "fold": fold, "run": episode["run"], "episode": episode["episode"],
                                        "seed": episode["seed"], "sample": episode["sample"], "date": episode["date"], **example}) + "\n")
                    count += 1
        manifest["arms"][arm] = {"episodes": len(keys), "examples": count, "dropped": dict(dropped),
                                 VISITS: len(left_out), "file": EXAMPLES[arm], "sha256": sha256_file(path)}
    (out / DATA_MANIFEST).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def found_runs(roots, agent, fold):
    """The finished sampling runs of one agent and fold under the folders roots, read from run.json, in folder order; the
    builder then checks that they hold whole rounds of samples exactly once."""
    out = []
    for root in roots:
        for meta_path in sorted(Path(root).glob("*/run.json")):
            meta = read_json(meta_path) or {}
            if (meta.get("mode"), meta.get("agent"), meta.get("training_fold")) == (TRAINING_MODE, agent, fold) and meta.get("ended"):
                out.append(meta_path.parent)
    if not out:
        raise SystemExit(f"no finished sampling run of {agent} on fold {fold} under {', '.join(map(str, roots))}")
    return out


def main(argv):
    ap = argparse.ArgumentParser(description="the training data of rejection sampling fine tuning of one agent and one fold")
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--runs", nargs="+", help="the sampling run folders of one agent and fold, whole rounds of samples")
    source.add_argument("--search", nargs="+", help="folders of run folders, for example the grid folders, searched with --agent and --training-fold")
    ap.add_argument("--agent", default=None, help="with --search, the agent whose sampling runs the data takes")
    ap.add_argument("--training-fold", type=int, default=None, help="with --search, the fold whose sampling runs the data takes")
    ap.add_argument("--training-root", default=None, help="the training folder, HYD_TRAINING by default, outside the fork")
    a = ap.parse_args(argv)
    if a.search and (a.agent is None or a.training_fold is None):
        raise SystemExit("--search needs --agent and --training-fold")
    runs = a.runs or found_runs(a.search, a.agent, a.training_fold)
    agent, fold, asset_id, _split, _found, _rounds = sampling_runs(runs)
    out = training_root(a.training_root) / "data" / data_name(agent, fold, asset_id)
    manifest = build(runs, out)
    print(f"wrote {out}: {manifest['episodes_sampled']} episodes sampled, {manifest['arms'][FILTERED]['episodes']} kept by the filter, "
          f"{manifest['arms'][FILTERED]['examples']} and {manifest['arms'][UNFILTERED]['examples']} examples")
    gate = manifest["gate"]
    decision = "trained" if gate["trained"] else "second round due" if gate["second_round_due"] else "no adapter"
    print(f"GATE {agent} f{fold} kept {gate['kept']} of minimum {gate['minimum']} after {gate['rounds_sampled']} round(s): {decision}")
    print(f"LEFT OUT {agent} f{fold} {manifest['arms'][FILTERED][VISITS]} of {manifest['episodes_sampled']} episodes, which met a single "
          f"fault of fold {fold}, before the filter and the control")
    return manifest


if __name__ == "__main__":
    main(sys.argv[1:])
