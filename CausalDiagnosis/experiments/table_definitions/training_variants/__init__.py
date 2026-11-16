"""The variants of the training table and the files they read (CausalDiagnosis/docs/experiments.md, section Training
in the environment): the run.json key every run records, the tunable block that GEPA edits for the three agents alike, the
folds of the asset's registered split and the names under which the fine tuned adapters and the GEPA blocks of an agent and a fold are kept.

A run plays one variant. as_released plays the agent without training, which every table prints as no training, gepa plays it
with the final block of the online loop of GEPA on it, gepa_score_only with the block of the same loop under a feedback card
that carries the score alone, the control on the base weights, ft with its LoRA adapter, fine tuned by rejection sampling on the episodes
the verifier passed, ft_gepa with that adapter and the block of the loop run on it, and ft_unfiltered with the adapter of the
control, fine-tuning on unfiltered trajectories, which trains on as many episodes drawn whatever their verdict. Every trained variant has one model per fold of the
asset's registered split, configs/training/<asset_id>/folds.json, which asset_training() reads (a trial reads
the development split beside it, the same rule dealt over the development starts, and its files carry the split): the model
of fold h holds out fold h, trains on every start whose faults all lie outside fold h and from which no repair reaches one, and
plays only the test episodes of fold h, so every test episode is judged by a model that never started at its combination and
cannot reach it by a repair. The adapters and the blocks lie under the folder that HYD_TRAINING names,
adapters/<name>/ with the manifest the training job writes and blocks/<name>.txt with the record the export of gepa_prompt_optimization/
writes, so a grid line names its variant and its fold alone and every run checks that the files it plays are the ones its
variant, its agent and its fold call for, a block of ft_gepa included, which plays only on the adapter weights its loop played
on. run.json of a run records the adapter and the block with the counts of their training, what each agent trained on (the
table of training), so the tables read them from the runs that the results index lists and verify_results_folder.py rebuilds them. This module needs
the standard library alone, since train_lora.py imports it in the training environment."""

from . import held_out_folds
from . import registered_numbers
from . import variant_names
from . import adapter_and_block_records

from .held_out_folds import (
    CONFIG, CONFIG_FILE, CONFIG_NAME, DEFAULT_ASSET, DEVELOPMENT, FAULT_FOLDS, FOLDS, FOLDS_FILE, FOLD_COMPONENTS,
    REGISTERED, SPLITS, TRAINING_DIR, TRAINING_FOLDS, TRAINING_SPLIT_KEY, asset_of, asset_training, check_fold,
    combinations_played, fold_ids, fold_of, visits_test_combination)
from .registered_numbers import extension_samples, gate_of, numbers, round_samples, sampled_per_state
from .variant_names import (
    ADAPTER_MANIFEST, ARM_OF, AS_RELEASED, BLOCK_OF, DATA_MANIFEST, FILTERED, FOLD_CONVENTION, FOLD_CONVENTION_KEY,
    TRAINED_AGENTS, TRAINING_FOLD_KEY, TRAINING_MODE, TRAINING_VARIANTS, TRAINING_VARIANT_KEY, UNFILTERED, VISITS,
    is_marker, loop_counts, with_task_block)
from .adapter_and_block_records import (
    adapter_name, adapter_name_of_arm, adapter_of, block_name, block_of, built_from, check_fold_convention, data_name,
    read_record, run_reference, sha256_file, training_root)
