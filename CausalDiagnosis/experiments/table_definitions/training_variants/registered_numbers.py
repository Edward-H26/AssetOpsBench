from typing import NamedTuple
from . import held_out_folds


class Numbers(NamedTuple):
    """The registered numbers of the training of one asset that a run reads, from configs/training/<asset_id>/rft.json."""
    samples_per_state: int  # the samples of one round of rejection sampling per training start state
    kept_per_training_state: int  # an adapter trains only on at least this many kept episodes per training start state of its model
    extension_rounds: int  # a round that keeps fewer is followed by at most this many more rounds of the same size
    sampling_temperature: float  # of every model call of a sampling run, which the agents make at 0 otherwise
    unanswered_proposals: int  # a loop fails after this many proposals in a row without an answer (UNANSWERED_PROPOSALS_KEY)
    proposal_attempts: int  # the requests of one proposal to the reflection model, the first and the retries that quote its problems


def numbers(asset_id):
    """The registered numbers of the training of one asset, the only numbers of the training that a run of that asset reads."""
    training = held_out_folds.asset_training(asset_id)
    config = training.config
    try:
        return Numbers(config["samples_per_state"], config["minimum"]["kept_per_training_state"], config["minimum"]["extension_rounds"],
                       config["sampling_temperature"], config["gepa"][UNANSWERED_PROPOSALS_KEY], config["gepa"]["proposal_attempts"])
    except KeyError as missing:
        raise SystemExit(f"{training.config_file} holds no {missing.args[0]!r}, a registered number of the training, see "
                         "CausalDiagnosis/docs/adding-a-dataset.md") from None


# the number of the gepa block of rft.json after which a loop fails: proposals in a row whose every attempt, proposal_attempts of the
# same block, ends on an error of the reflection endpoint. The rig registers 2, the smallest number that tells an endpoint that is
# down from one failed proposal, since each proposal of the rig already asks twice
UNANSWERED_PROPOSALS_KEY = "unanswered_proposals"


def round_samples(round_no, asset_id):
    """The training samples of one round of sampling, counted from 1: round 1 plays the samples 1 to samples_per_state of the
    asset, round 2 the next as many, and so on."""
    per_round = numbers(asset_id).samples_per_state
    return list(range((round_no - 1) * per_round + 1, round_no * per_round + 1))


def extension_samples(asset_id):
    """The samples of the rounds that the minimum adds for an agent and fold whose first round kept too few episodes."""
    return [sample for round_no in range(2, 2 + numbers(asset_id).extension_rounds) for sample in round_samples(round_no, asset_id)]


def sampled_per_state(asset_id):
    """The samples per training start state that the rounds of the sampling may play, the first round and every extension round:
    (1 + extension_rounds) times samples_per_state of the asset. The stream of a loop passes over as many seeds per state."""
    registered = numbers(asset_id)
    return (1 + registered.extension_rounds) * registered.samples_per_state


def gate_of(kept, distinct_states, training_states, rounds, asset_id):
    """The record of the minimum for one agent and fold: kept_per_training_state episodes of the asset per training start state
    of the model, read from the asset's own split, a round below it is followed by another while extension_rounds allow, and
    below it after the last round no adapter trains."""
    registered = numbers(asset_id)
    minimum = registered.kept_per_training_state * training_states
    trained = kept >= minimum
    return {"minimum": minimum, "kept_per_training_state": registered.kept_per_training_state, "training_states": training_states,
            "rounds_sampled": rounds, "kept": kept, "distinct_training_states": distinct_states,
            "second_round_due": not trained and rounds < 1 + registered.extension_rounds, "trained": trained}
