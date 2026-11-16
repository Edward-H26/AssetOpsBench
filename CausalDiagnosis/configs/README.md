# configs

The presets of episodes, the price tables and the training settings. A preset is a JSON file that names the layers an episode switches on and their options; `python -m CausalDiagnosis experiments run-episodes --config <name>` takes it by its file name.

## What lives here

| Folder | What it holds |
| --- | --- |
| `layer_presets/` | one preset per layer of the layer ladder, from `base` to `deterioration` |
| `token_clock/` | the same rows with the token clock, which charges the thinking time of an agent by the tokens its model generates |
| `ablation_variants/` | the presets of the ablations, each one option changed against the preset of training |
| `development/` | presets for development and tests that no table runs |
| `prices/` | the price tables, which are not presets |
| `training/` | the folds and the filter of each dataset for training, which are not presets |

`healthy_control.json` lies directly in this folder; it plays starts without a fault.

## How to use it

Every program finds a preset by its name through one lookup, `preset_path` in `utils/paths.py`, whichever folder holds it, and stops when two files share a name. The section Configuration presets and price tables of `CausalDiagnosis/docs/experiments.md` lists every preset with what it changes.

## How to add to it

A new preset goes into the folder of its kind with a name no other preset has. The presets are part of the identity of an episode run, so a preset added after runs of record gives the runs played afterwards another group, and the tables read one group at a time.
