from CausalDiagnosis.utils import frameworks


TRAINING_VARIANT_KEY = "training_variant"
TRAINING_FOLD_KEY = "training_fold"  # the run.json key of the fold, null on a run without training
# what the fold number of a training record names: the fold its model holds out. Records of the earlier naming used the same
# names and numbers for the fold a model trained on, so every reader of a training folder refuses a record without this value
FOLD_CONVENTION_KEY, FOLD_CONVENTION = "fold_convention", "held_out"
AS_RELEASED = "as_released"
TRAINING_VARIANTS = (AS_RELEASED, "gepa", "gepa_score_only", "ft", "ft_gepa", "ft_unfiltered")
# the agents of the training table, which train alike: the declared agents that train, and the CausalDiagnosis agent
TRAINED_AGENTS = (*(name for name, entry in frameworks.agents().items() if entry.get("trained")), "causal")
TRAINING_ROOT_ENV = "HYD_TRAINING"
TRAINING_MODE = "training"  # the mode of run.json of a sampling run, whose episodes are training data and enter no result table
ADAPTER_MANIFEST = "training_manifest.json"  # written by train_lora.py into the adapter folder
DATA_MANIFEST = "manifest.json"  # written by rejection_sampling_examples.py into data/<agent>_f<fold>/ of the training folder
FILTERED, UNFILTERED = "filtered", "unfiltered"
ARM_OF = {"ft": FILTERED, "ft_gepa": FILTERED, "ft_unfiltered": UNFILTERED}
# the weights the loop of GEPA played on and the feedback card it read, for each variant that plays a block: the score-only
# control runs on the base weights alone, so that a gain of GEPA is traced to the verifier's card
CARDS = ("verifier", "score_only")
BLOCK_OF = {"gepa": (AS_RELEASED, "verifier"), "gepa_score_only": (AS_RELEASED, "score_only"), "ft_gepa": ("ft", "verifier")}
# the training episodes left out of the data and of the stream of a loop, since they played a combination that holds a single
# fault of the fold their model holds out, the combinations that model is judged on
VISITS = "episodes_visiting_test_combinations"
LOOP_COUNTS = ("candidates_played", "stream_episodes", VISITS)  # what the table of what each agent trained on counts of a loop


def with_task_block(question, block):
    """The task message an agent receives: the briefing alone without a block, else the briefing, an empty line and the block."""
    return f"{question}\n\n{block}" if block else question


def is_marker(adapter):
    """Whether an adapter record stands for no adapter, the agent and fold below the minimum of kept episodes."""
    return bool(adapter) and adapter.get("trained") is False


def loop_counts(result):
    """What the table of what each agent trained on counts of a finished loop of GEPA, from its result file gepa_online.json: the
    candidates that played on the stream, the stream episodes and those the loop left out (LOOP_COUNTS)."""
    episodes = result.get("episodes") or []
    return {"candidates_played": len({e["candidate"] for e in episodes}), "stream_episodes": result.get("episodes_played"),
            VISITS: sum(1 for e in episodes if e["left_out"])}
