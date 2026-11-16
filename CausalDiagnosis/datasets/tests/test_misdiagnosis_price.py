"""The misdiagnosis charge reads the price of the asset that an episode plays, for any asset of the recorded levels kind:
the tiny synthetic asset, whose price table holds a misdiagnosis price of its own, is charged that price for a wrong
diagnosis and nothing at a price of zero, and a log whose rules hold no misdiagnosis price, as a synthetic log of the tests
does, is charged nothing."""
import dataclasses

from CausalDiagnosis.datasets.dataset_loader import recordings as records
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.verifier import misdiagnosis_charge, misdiagnosis_charge_of, released_from_env, verify


def _missed(env, components):
    """The hidden start with the first component at another of its levels."""
    levels = dict(zip(components, env.initial))
    first = components[0]
    levels[first] = next(level for level in env.config.levels[first] if level != levels[first])
    return levels


def test_the_charge_reads_the_price_of_the_asset(tiny, tiny_recordings):
    cycles = records.load_cycles(tiny_recordings, tiny)
    blocks = records.build_blocks(cycles)
    config = tiny.default_config()
    price = config.prices["misdiagnosis"]
    assert price > 0
    for prices, expected in ((config.prices, [price]), ({**config.prices, "misdiagnosis": 0}, [])):
        env = HydraulicRigEnv(cycles, blocks, dataclasses.replace(config, prices=prices))
        env.reset(0)
        env.step({"type": "submit_diagnosis", "diagnosis": _missed(env, tiny.components), "evidence_ids": []})
        env.step({"type": "finish", "summary": "done"})
        verdict = verify(env.log, released_from_env(env), env.oracle(), None)
        assert [e["cost"] for e in env.log if e["event"] == "misdiagnosis"] == expected
        assert misdiagnosis_charge_of(verdict) == sum(expected) and not verdict["diagnosis_all_correct"]
        assert ("misdiagnosis_charge" in verdict) == bool(expected)
    reset = {"event": "reset", "step": None, "t_start": 0, "t_end": 0, "class": "system", "cost": 0, "hidden": list(env.initial)}
    for synthetic in (reset, {**reset, "config": {"asset_id": tiny.asset_id}}, {**reset, "config": {"asset_id": tiny.asset_id, "prices": {}}}):
        assert misdiagnosis_charge([synthetic], _missed(env, tiny.components)) == 0
