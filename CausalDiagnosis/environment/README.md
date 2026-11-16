# environment

The environment turns the description of an asset into episodes. Each episode hides one combination of the condition levels of the asset's components, serves only recorded cycles as readings, runs on a clock, and is scored by a verifier from its log and the hidden state.

## What lives here

| Folder | What it holds |
| --- | --- |
| `Causal_Diagnosis_Environment/` | the episode engine `env.py` with the twelve actions, the settings of an episode in `episode_config.py`, the oracle, the verifier, the tool contract that gives every tool its class, price and duration (`tool_contract.json`), the opening question an agent receives, and the Gymnasium and OpenEnv wrappers |
| `Causal_Diagnosis_Environment/layers/` | the six layers, one module each, which a preset switches on one after another: base, priced, real time, advanced real time, repair rules, deterioration and failure |
| `Causal_Diagnosis_Environment/clock/` | the clock of an episode and the two time tools, `get_current_time` and `wait_for_notification`, adapted from the Agents Research Environments of Meta, whose license stands in `THIRD_PARTY_NOTICES.md` at the repository root |
| `reference_policies/` | policies that use no model, such as the oracle, the greedy policy and doing nothing, and the diagnosability label of a start that the greedy policy computes |

The greedy policy, its classifiers and the diagnosability label are written for hyd_1 today, although they sit in this general folder.

## How to use it

```bash
python -m CausalDiagnosis examples run-one-episode --seed 0
python -m CausalDiagnosis environment diagnosability-label --help
```

The section Programmatic use of `CausalDiagnosis/docs/environment.md` shows how to create an episode in Python, play it and verify it.
`CausalDiagnosis/docs/environment.md` documents the configuration, the layers, the clock, the actions, the log, the oracle, the reference policies, the verifier and the wrappers.

## How to add to it

A layer is one module of `Causal_Diagnosis_Environment/layers/` with its fields, its options and its rules. `LAYERS` in `layers/__init__.py` lists the layers in order, and `compose` builds the configuration of an episode from a preset file. The section Layers of `CausalDiagnosis/docs/environment.md` describes what each layer adds.
A new tool of the action server is declared in `tool_contract.json` with its class, price and duration before the server serves it.
An asset never adds code here: everything that belongs to one asset lives in its folder under `datasets/`.
