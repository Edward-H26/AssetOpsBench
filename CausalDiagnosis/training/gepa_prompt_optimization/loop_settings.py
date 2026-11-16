import argparse
from CausalDiagnosis.experiments.episode_runners import run_episodes as drivers  # noqa: E402
from CausalDiagnosis.utils import frameworks  # noqa: E402
from CausalDiagnosis.utils.llm_backends import NO_THINKING  # noqa: E402
from CausalDiagnosis.experiments.table_definitions.experiment_tables import TRAINING_PRESET  # noqa: E402
from CausalDiagnosis.experiments.table_definitions.training_variants import AS_RELEASED, FOLD_CONVENTION, FOLD_CONVENTION_KEY, REGISTERED, SPLITS, TRAINED_AGENTS, asset_training, sampled_per_state  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.start_state_seeds import training_stream  # noqa: E402
from . import run_folders, candidate_proposals, online_loop, run_loop


LOOP_VARIANTS = (AS_RELEASED, "ft")  # the weights a loop plays on, the base for gepa and the filtered adapter for ft_gepa


# ----- the command line -----
def build_parser():
    ap = argparse.ArgumentParser(description="the online loop of GEPA over the tunable block of one agent of the training table")
    ap.add_argument("--model", required=True, help="the agent's model name at its endpoint")
    ap.add_argument("--agent", choices=TRAINED_AGENTS, default="causal", help="the agent whose block the loop tunes")
    ap.add_argument("--training-variant", choices=LOOP_VARIANTS, default=AS_RELEASED,
                    help="as_released plays the base weights, ft the agent's filtered adapter, the loop of the ft_gepa variant")
    ap.add_argument("--training-fold", type=int, required=True,
                    help="the fold that the loop's model holds out, one of the folds of the split of --training-split: the stream plays "
                         "the training start states of that model, and ft plays its adapter")
    ap.add_argument("--training-split", choices=SPLITS, default=REGISTERED,
                    help="the split of the fold, the registered one or the development one of a trial, whose loop plays "
                         "the preset widened to the development starts of the asset")
    ap.add_argument("--base-url", default=drivers.ROUTER_BASE_URL, help="the agent's OpenAI compatible endpoint")
    ap.add_argument("--reflection-model", required=True, help="the reflection model, the judge model of the job")
    ap.add_argument("--reflection-base-url", required=True, help="the endpoint of the reflection model, the judge endpoint of the job")
    ap.add_argument("--card", choices=online_loop.FEEDBACK, default=None,
                    help="the feedback that the reflection reads, the verifier's feedback card or the score alone as its control, both "
                         "with the score of the stream, the reward without the declaration credit")
    ap.add_argument("--config", default=TRAINING_PRESET, help="a preset of CausalDiagnosis/configs/, the preset of training by default")
    ap.add_argument("--servers", choices=("rig", *frameworks.surfaces()), default=None,
                    help="rig, the rig's action server, or aob, AssetOpsBench's servers behind the metering layer, the main row, which a row "
                         "of the environment table plays without the option (run_episodes.default_servers)")
    ap.add_argument("--store", choices=("memory", "couchdb"), default="memory", help="the database of the AssetOpsBench servers")
    # the defaults below are the registered numbers of configs/training/<asset_id>/rft.json, the same for the loops of the three agents
    ap.add_argument("--block", type=int, default=None, help="the seeds of a round, GEPA's minibatch of three")
    ap.add_argument("--window", type=int, default=None, help="the rounds over which the front and the incumbent are scored")
    ap.add_argument("--children", type=int, default=None, help="the proposals of a round, one as in an iteration of GEPA")
    ap.add_argument("--max-episodes", type=int, default=None,
                    help="the budget of the loop in episodes")
    ap.add_argument("--first-seed", type=int, default=0, help="the first seed that the stream tries")
    ap.add_argument("--master-seed", type=int, default=42, help="the generator seed of the test set whose start states the stream leaves out, "
                    "as for --starts sampled")
    ap.add_argument("--seed-pool", type=int, default=8, help="the smallest seeds per combination that the test set draws from")
    ap.add_argument("--gepa-seed", type=int, default=None, help="the generator seed of the parent draws")
    ap.add_argument("--facts", default=None, help="the rig data file, by default HYD_FACTS_PATH or the package's data folder")
    ap.add_argument("--out", default="CausalDiagnosis/results/runs")
    ap.add_argument("--resume", action="store_true",
                    help="take up the unfinished loop of these settings, asset and code under --out, and start a new loop when there is none")
    return ap


GEPA_NUMBERS = ("card", "block", "window", "children", "max_episodes", "gepa_seed")  # the registered numbers of a loop, under "gepa"


def with_registered_numbers(a, asset_id):
    """The arguments of the loop, every number that the command line leaves out taken from the gepa block of the rft.json of the
    asset of the loop's preset."""
    registered = asset_training(asset_id).config["gepa"]
    for name in GEPA_NUMBERS:
        if getattr(a, name) is None:
            setattr(a, name, registered[name])
    return a


def driver_settings(a):
    """The settings of run_episodes/ for the loop, which are the agent of --agent with the run's model on the config, the
    servers of the command line or the surface a row plays without the option (run_episodes.default_servers), the store of
    the command line, the loop's fold and split, and --starts sampled with the master seed and the pool of the test set, whose
    start states the stream leaves out."""
    return drivers.build_parser().parse_args(
        ["--config", a.config, "--agent", a.agent, "--training-variant", a.training_variant, "--training-fold", str(a.training_fold),
         "--model", a.model, "--base-url", a.base_url,
         "--servers", a.servers or drivers.default_servers(a.agent, a.config), "--store", a.store,
         "--training-split", a.training_split, "--starts", "sampled", "--master-seed", str(a.master_seed), "--seed-pool", str(a.seed_pool)]
        + (["--facts", a.facts] if a.facts else []))


def gepa_meta(a, asset_id):
    """The gepa block of run.json."""
    return {"agent": a.agent, "training_variant": a.training_variant, "training_fold": a.training_fold, FOLD_CONVENTION_KEY: FOLD_CONVENTION,
            "unanswered_proposals": run_loop.numbers(asset_id).unanswered_proposals, "proposal_attempts": run_loop.numbers(asset_id).proposal_attempts,
            "card": a.card, "reflection_model": a.reflection_model, "reflection_endpoint": a.reflection_base_url,
            "reflection_request": NO_THINKING, "reflection_max_tokens": None, "temperature": online_loop.TEMPERATURE, "block": a.block, "window": a.window,
            "children": a.children, "max_episodes": a.max_episodes, "first_seed": a.first_seed, "gepa_seed": a.gepa_seed,
            "components": list(candidate_proposals.MUTABLE), "result": run_folders.RESULT}


def loop_stream(a, config, cycles, blocks):
    """The stream of a loop: the training seeds of the model of its fold, which no replicate's start state enters, past every seed
    per training start state that a round of the sampling of its fine tuning may play. The weights of the loop do not enter it, so
    the loop on the base weights and the loop on the adapter of a fold read the same stream."""
    split = asset_training(config.asset_id, cycles.facts_sha256, split=a.training_split).split
    return training_stream(HydraulicRigEnv(cycles, blocks, config), split, a.max_episodes, a.training_fold, a.first_seed, a.master_seed,
                           a.seed_pool, skip_per_state=sampled_per_state(config.asset_id))
