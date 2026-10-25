# Hydraulic rig environment

This package turns the public UCI hydraulic rig recordings into episodes of finding a fault and repairing the asset. Each episode hides one combination of the condition levels of four components, the agent pays for every sensor reading, and the score comes from the episode log alone.

```bash
uv run pytest src/envs/hydraulic_rig -q
```

The tests that need the rig data file are skipped without it, and the fork does not ship that file. The data file is the IndustryAssetEQA facts file of the public EmbodiedQA repository at a fixed commit, named by `HYD_FACTS_PATH`, or a file with the same labels and served statistics rebuilt from the UCI archive, as [docs/hydraulic_rig/environment.md](../../../docs/hydraulic_rig/environment.md) describes.

[docs/hydraulic_rig/README.md](../../../docs/hydraulic_rig/README.md) gives the overview and the quick start, and [docs/hydraulic_rig/environment.md](../../../docs/hydraulic_rig/environment.md) documents this package.
