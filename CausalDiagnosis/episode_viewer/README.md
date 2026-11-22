# episode_viewer

Pages that show one episode or one run to a person who was not there.

## What lives here

| Part | What it shows |
| --- | --- |
| walkthrough | how one episode started, what the agent did step by step, and how the verifier scored it |
| step-through page | one slide per tool call of an episode, for diagnosing what happened |
| run review pack | for one run folder: where it came from, its ledger, its replay check and the walkthrough pages; also the fixed starts that the walkthroughs use (`fixed_starts.json`) |
| explorer | every run folder under a folder as pages served on 127.0.0.1: the run list, the episode page on one axis of cycles, what the agent knew and did at every step, the graph, the sensor readings and the files; or the page of one episode as one file |

## How to use it

```bash
python -m CausalDiagnosis episode_viewer walkthrough --help
python -m CausalDiagnosis episode_viewer step-through-page --help
python -m CausalDiagnosis episode_viewer run-review-pack --help
python -m CausalDiagnosis episode_viewer explorer serve --runs CausalDiagnosis/results/runs
```

The pages quote sensor readings of single cycles, so they stay out of the repository: `CausalDiagnosis/docs/walkthroughs/` is ignored by git. The section The step-through viewer of `CausalDiagnosis/docs/experiments.md` describes the page of one episode, and `CausalDiagnosis/docs/episode-explorer.md` describes the explorer.

## How to add to it

A new view reads the files of a run folder, which the section What a run writes of `CausalDiagnosis/docs/experiments.md` lists. Files here are not part of the identity of an episode run, so a committed change here leaves the group of runs that the tables read as it is.
