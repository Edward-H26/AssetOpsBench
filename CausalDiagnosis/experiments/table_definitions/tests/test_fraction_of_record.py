"""The registered fraction of record is a whole fraction of the replicates of the test set, read through registered.value() and
listed by missing() when it is open."""
import json
from fractions import Fraction

import pytest

from CausalDiagnosis.experiments.table_definitions import registered_analysis as registered
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.start_state_seeds import REPLICATES


def _whole_fraction(value):
    """Whether a value is r/REPLICATES for a whole r from 1 to REPLICATES, by exact arithmetic on its decimal text."""
    replicates = Fraction(str(value)) * REPLICATES
    return replicates.denominator == 1 and 0 < replicates <= REPLICATES


@pytest.mark.registered_file
def test_the_fraction_of_record_is_a_whole_fraction_of_the_replicates():
    value = registered.value("counting.fraction_of_record")
    assert value == 1.0 and _whole_fraction(value)
    assert "counting.fraction_of_record" in registered.TABLE_VALUES


@pytest.mark.parametrize("value", [0.2, 0.4, 0.6, 0.8, 1, 1.0])
def test_every_whole_fraction_of_five_replicates_passes(value):
    assert _whole_fraction(value)


@pytest.mark.parametrize("value", [0.3, 0.05, 0.25, 0, 1.2, -0.2])
def test_a_fraction_inside_a_replicate_or_outside_the_test_set_fails(value):
    # the control 0.3 leads: it cuts the second replicate, so the start combinations would weigh unequally in every mean
    assert not _whole_fraction(value)


def test_an_open_fraction_of_record_is_listed_and_stops_its_reader(tmp_path):
    data = registered.load()
    data["counting"]["fraction_of_record"] = None
    path = tmp_path / registered.PATH.name
    path.write_text(json.dumps(data), encoding="utf-8")
    assert "counting.fraction_of_record" in registered.missing(path)
    with pytest.raises(registered.OpenValue):
        registered.value("counting.fraction_of_record", path)
