from pathlib import Path

import pytest

from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import build_blocks, load_cycles
from CausalDiagnosis.datasets.hyd_1.verify_data_facts import DEFAULT_FACTS, DEFAULT_HYD1, collect_checks, file_checks, load_facts, load_hyd1, stable_blocks

if not Path(DEFAULT_FACTS).exists():
    pytest.skip("no data file in this clone, as ASSETS_WITHOUT_DATA=1 states", allow_module_level=True)

FACTS = load_facts(DEFAULT_FACTS)
CHECKS = file_checks(DEFAULT_FACTS, DEFAULT_HYD1) + collect_checks(FACTS, load_hyd1(DEFAULT_HYD1))


@pytest.mark.parametrize("check", CHECKS, ids=[c.fact for c in CHECKS])
def test_data_fact_holds_in_the_local_files(check):
    assert check.observed == check.expected


def test_environment_blocks_equal_the_independently_recomputed_stable_blocks():
    assert build_blocks(load_cycles(DEFAULT_FACTS)) == stable_blocks(FACTS)
