# Hydraulic rig environment

This package turns the public UCI hydraulic rig recordings into episodes of finding the faults of an asset and restoring it on a clock. Each episode hides one combination of the condition levels of four components, the agent pays for every sensor reading in price and in time, work orders are executed by a technician after a lead time, and the verifier scores the episode from the log and the hidden state alone. Gymnasium and OpenEnv wrappers expose the same episodes for policy learning.

```bash
export HYD_FACTS_PATH=~/rig-data/hyd_facts.jsonl
PATH=$PWD/.venv/bin:$PATH python -m pytest src/envs/hydraulic_rig -q
```

The tests that need the rig data file are skipped without it, and the fork does not ship that file. The data file is the IndustryAssetEQA facts file of the public EmbodiedQA repository at a fixed commit, named by `HYD_FACTS_PATH`, or a file with the same labels and served statistics rebuilt from the UCI archive, as [docs/hydraulic_rig/environment.md](../../../docs/hydraulic_rig/environment.md) describes.

[docs/hydraulic_rig/README.md](../../../docs/hydraulic_rig/README.md) gives the overview and the quick start, and [docs/hydraulic_rig/environment.md](../../../docs/hydraulic_rig/environment.md) documents this package: the configuration, the clock, the actions, the log, the oracle, the reference policies, the verifier and the wrappers.
