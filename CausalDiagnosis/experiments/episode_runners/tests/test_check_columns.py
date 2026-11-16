"""scores.csv reports the write count, timing and ordering checks of the verifier beside the provenance check, one boolean
cell each, from the ok of the checks count_gate, timing and causality; a verdict without checks leaves the four cells empty."""
from CausalDiagnosis.experiments.run_verification.tests.test_run_ledger import _driver


def _verdict(**failed):
    names = ("count_gate", "hard", "timing", "causality", "outcome", "provenance")
    return {"checks": {name: {"ok": name not in failed} for name in names}}


def test_each_check_of_the_verifier_has_its_own_cell_beside_provenance():
    run_episodes = _driver()
    row = run_episodes.verdict_row(_verdict(), 0.5, 10.0, None)
    assert {k: row[k] for k in ("provenance_ok", "write_count_ok", "timing_ok", "ordering_ok")} == dict.fromkeys(
        ("provenance_ok", "write_count_ok", "timing_ok", "ordering_ok"), True)
    # the control: a failed timing check fails its cell alone, and the checks that pass keep theirs
    row = run_episodes.verdict_row(_verdict(timing=True), 0.5, 10.0, None)
    assert (row["timing_ok"], row["write_count_ok"], row["ordering_ok"], row["provenance_ok"]) == (False, True, True, True)
    row = run_episodes.verdict_row(_verdict(count_gate=True, causality=True), 0.5, 10.0, None)
    assert (row["write_count_ok"], row["ordering_ok"], row["timing_ok"]) == (False, False, True)
    row = run_episodes.verdict_row({}, None, None, None)
    assert all(row[k] is None for k in ("provenance_ok", "write_count_ok", "timing_ok", "ordering_ok"))
