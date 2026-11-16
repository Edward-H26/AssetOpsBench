from pathlib import Path
from CausalDiagnosis.evaluation.result_tables.tables_from_runs import STRATA, add_derived_fields, in_stratum  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import HIDDEN, episode_reward
from CausalDiagnosis.utils.json_files import read_json


WHOLE_BLOCK = "all"  # the stratum of every episode, so the best mean over the window always leads a stratum of the front


def score_of(row, ep_dir, horizon):
    """The score of a stream episode: the verifier's reward of its verdict with the declaration credit set to
    the registered stream_declaration_credit (common.episode_reward), the pass alone in the outcome term, else the reward of its row.
    None when the verifier could not score the episode."""
    verdict = read_json(Path(ep_dir) / HIDDEN / "verdict.json", {})
    if verdict.get("verified"):
        return episode_reward(verdict, horizon, stream=True)
    reward = row.get("reward")
    return float(reward) if reward not in ("", None) else None


def scored(values):
    return [v for v in values if v is not None]


def mean(values):
    """The mean of the scored values, or None without one, rounded so that equal means compare equal."""
    values = scored(values)
    return round(sum(values) / len(values), 12) if values else None


def improves(child, parent):
    """Whether a child's mean on the block admits it, a scored mean above its parent's or beside an unscored parent."""
    return child is not None and (parent is None or child > parent)


def strata_of(row, ep_dir):
    """The strata of an episode, those of tables_from_runs/ that it belongs to and the whole block. The protocol computes them
    from the verdict and the episode files, and the reflection model never sees them."""
    episode = add_derived_fields([{**row, "episode_dir": ep_dir}])[0]
    return [s for s in STRATA if in_stratum(episode, s)] + [WHOLE_BLOCK]
