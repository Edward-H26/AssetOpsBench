"""The four digest invariants of the hydraulic rig (CausalDiagnosis/docs/adding-a-dataset.md): the configuration with the start
mapping, the scripted fixture with the logs of the reference policies, the advertised tools and texts, and the hyd_1 fit.
Each test recomputes its families with CausalDiagnosis/experiments/run_verification/invariant_digests.py and compares them with the label LABEL of
invariant_digests.json, so any change of a covered value fails it. They are golden tests that run only with ASSET_DIGESTS=1,
not in the default suite, and only against a label of record, one that names the commit its digests were computed on.
While the digest file holds none for the data file in use they skip and name the command that writes it, after the push
on the platform of record."""
import os
import sys
from pathlib import Path

import pytest

from CausalDiagnosis.datasets.hyd_1.records import default_facts_path

from CausalDiagnosis.datasets.tests.conftest import DIGESTS_FILE, ROOT, needs_data

LABEL = "current"
pytestmark = [pytest.mark.skipif(os.environ.get("ASSET_DIGESTS") != "1", reason="digest invariants run only with ASSET_DIGESTS=1"),
              needs_data]


def _digest_tool():
    """invariant_digests.py, imported only when a test runs, since the import puts the drivers' folder on the import path."""
    from CausalDiagnosis.experiments.run_verification import invariant_digests as asset_digests

    return asset_digests


def _assert_digests_kept(families):
    tool = _digest_tool()
    facts = Path(default_facts_path())
    recorded, reason = tool.record_digests(DIGESTS_FILE, tool.file_sha256(facts), LABEL)
    if reason:
        pytest.skip(reason)
    stored = tool.of_families(recorded, families)
    computed, _detail = tool.compute(facts, families)
    _same, changed, missing, new, unexpected = tool.compare(stored, computed)
    assert not unexpected, f"against the label {LABEL}: changed {changed}, missing {missing}, new {new}"


def test_the_start_mapping_keeps_its_digest():
    _assert_digests_kept(("config", "starts"))


def test_the_scripted_fixture_keeps_its_digests():
    _assert_digests_kept(("scripted", "logs"))


def test_advertised_tools_and_texts_keep_their_digests():
    _assert_digests_kept(("texts",))


def test_the_hyd_1_fit_keeps_its_digest():
    _assert_digests_kept(("models",))
