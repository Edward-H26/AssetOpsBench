from pathlib import Path

import pytest

from envs.hydraulic_rig.data import build_blocks, load_cycles
from verify_data_facts import (DEFAULT_FACTS, DEFAULT_HYD1, FACTS_SHA256, collect_checks, file_checks, load_facts, load_hyd1, sha256,
                               stable_blocks)

if not Path(DEFAULT_FACTS).exists():
    pytest.skip("hydraulic rig facts file not present; set HYD_FACTS_PATH", allow_module_level=True)
if sha256(DEFAULT_FACTS) != FACTS_SHA256:
    pytest.skip("the facts file is not the IndustryAssetEQA file that these checks describe; build_blocks.py --check "
                "compares a rebuilt file with it", allow_module_level=True)

FACTS = load_facts(DEFAULT_FACTS)
CHECKS = file_checks(DEFAULT_FACTS, DEFAULT_HYD1) + collect_checks(FACTS, load_hyd1(DEFAULT_HYD1))


@pytest.mark.parametrize("check", CHECKS, ids=[c.fact for c in CHECKS])
def test_data_fact_holds_in_the_local_files(check):
    assert check.observed == check.expected


def test_environment_blocks_equal_the_independently_recomputed_stable_blocks():
    assert build_blocks(load_cycles(DEFAULT_FACTS)) == stable_blocks(FACTS)
