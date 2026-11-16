# training

Training in the environment: the data of rejection sampling, LoRA fine-tuning of the agent model, and the online loop of GEPA over one block of prompt text.

## What lives here

| Part | What it does |
| --- | --- |
| rejection sampling | selects the episodes that pass and writes them as fine-tuning examples with a list of what each agent trained on |
| LoRA | fine-tunes the agent model on those examples |
| GEPA | runs the online prompt optimization loop for the agents of the training table, with the feedback card that the verifier fills |

## How to use it

```bash
python -m CausalDiagnosis training rejection-sampling-examples --help
python -m CausalDiagnosis training train-lora --help
python -m CausalDiagnosis training gepa-prompt-optimization --help
```

Fine-tuning runs on the cluster in its own environment, which `cluster_jobs/0_setup/setup_train_env.sbatch` builds. The section Training in the environment of `CausalDiagnosis/docs/experiments.md` describes the training table and its pre-registered numbers.

## How to add to it

A training variant is named among the training variants of `experiments/table_definitions/`, and the folds of a dataset come from `python -m CausalDiagnosis experiments write-training-split`. Files here are not part of the identity of an episode run, so a committed change here leaves the group of runs that the tables read as it is.
