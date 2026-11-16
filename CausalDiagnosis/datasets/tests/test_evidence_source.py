"""The recording source of the recorded levels kind (CausalDiagnosis/datasets/dataset_loader/evidence_source.py) on worlds of the tiny asset: a served
reading is formatted once per source and handed out as a copy, and the tables of a world, the normal ranges and the start
set, belong to the very objects of that world. No test needs the facts file."""
from CausalDiagnosis.datasets.splits import development_split
from CausalDiagnosis.datasets.dataset_loader import evidence_source as evidence, recordings as records
from CausalDiagnosis.datasets.dataset_loader.evidence_source import RecordingSource
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv

from CausalDiagnosis.datasets.tests.conftest import write_recordings


def _world(profile, path):
    cycles = records.load_cycles(path, profile)
    return cycles, records.build_blocks(cycles)


def test_a_served_reading_is_a_copy_and_a_second_source_serves_its_own(tiny, tmp_path):
    """Changing the dict that a read returned does not change the next read of that recording, and two sources over two
    worlds whose recordings differ under the same indices serve their own values."""
    worlds = [_world(tiny, write_recordings(tmp_path / f"{seed}.jsonl", tiny, seed=seed)) for seed in (0, 1)]
    sources = [RecordingSource(tiny, {c.index: c for c in cycles}, blocks) for cycles, blocks in worlds]
    channels = tuple(tiny.channels)
    index = worlds[0][1][tiny.healthy_state()][0]
    served = sources[0].served_values(index, channels)
    served[next(iter(served))] = -1.0
    for source, (cycles, _blocks) in zip(sources, worlds):
        expected = evidence.served_values(next(c for c in cycles if c.index == index), channels, tiny)
        assert source.served_values(index, channels) == expected
    assert sources[0].served_values(index, channels) != sources[1].served_values(index, channels)


def test_two_worlds_have_their_own_bands_and_start_sets(tiny, tmp_path):
    """Two worlds of the tiny asset, the second with other recordings and one test combination removed, give environments
    whose normal ranges and start candidates are those that the world's own recordings give, whichever was built first and
    when one is built again. A store kept by the profile alone, or by the blocks alone, fails."""
    one = _world(tiny, write_recordings(tmp_path / "one.jsonl", tiny, seed=0))
    cycles, blocks = _world(tiny, write_recordings(tmp_path / "two.jsonl", tiny, seed=1))
    config = tiny.default_config()
    removed = next(k for k in RecordingSource(tiny, {c.index: c for c in cycles}, blocks).start_candidates(config) if k != tiny.healthy_state())
    two = ([c for c in cycles if c.combo != removed], {k: v for k, v in blocks.items() if k != removed})

    def own(world):
        source = RecordingSource(tiny, {c.index: c for c in world[0]}, world[1])
        moved = development_split.moved_to_development(tiny, world[1])
        return source._normal_ranges(), sorted(k for k in world[1] if config.within_fault_rule(k) and k not in moved)

    expected = {name: own(world) for name, world in (("one", one), ("two", two))}
    assert expected["one"] != expected["two"] and removed not in expected["two"][1]
    for name, world in (("one", one), ("two", two), ("one", one), ("two", two)):
        env = HydraulicRigEnv(*world, config)
        assert (env.normal_ranges, env.candidates) == expected[name], name
