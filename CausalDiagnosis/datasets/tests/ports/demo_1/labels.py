"""The label of a start of the made-up port, in the form the verifier reads: the misaligned press is inseparable under both
verdicts, since one value per channel does not tell it, and the aligned press is separable. TABLE_PATH is the table that
code of the port would compute on first use."""
from pathlib import Path

TABLE_PATH = Path(__file__).resolve().parent / "data" / "labels.json"


def label_of(env, rules=None):
    separable = tuple(env.initial) == env.profile.healthy_state()
    return {"separable": separable, "source": "made-up rule", "e18": {"separable": separable}}
