import contextlib
from pathlib import Path
from CausalDiagnosis.experiments.run_verification import run_ledger as ledger  # noqa: E402
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import episode_name
from . import candidate_proposals


@contextlib.contextmanager
def playing(runner, candidate):
    """The candidate's block in place for one episode, appended to the task message by the runner, restored afterward whatever
    happens in between. Nothing else of the agent changes."""
    saved = runner.task_block
    runner.task_block = candidate["texts"][candidate_proposals.BLOCK]
    try:
        yield
    finally:
        runner.task_block = saved


class EpisodePlayer:
    """One episode of a candidate through the run's EpisodeRunner into the run folder, with its ledger line. It returns the
    row of scores.csv with the GEPA call and the episode folder."""

    def __init__(self, runner, settings, run_dir, meta):
        self.runner, self.settings, self.run_dir, self.meta = runner, settings, Path(run_dir), meta

    def __call__(self, candidate, seed, call):
        name = episode_name(seed, call=call)
        ep_dir = self.run_dir / name
        with playing(self.runner, candidate):
            row = self.runner.play(ep_dir, name, seed, None, self.runner.config, self.meta)
        row = dict(row, gepa_call=call)
        ledger.ledger_file.add_episode(self.run_dir, name, row)
        return row, ep_dir
