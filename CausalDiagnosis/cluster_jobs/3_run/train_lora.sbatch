#!/bin/bash
#SBATCH -J cd-train
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h100:1
#SBATCH -t 12:00:00
#SBATCH --requeue
# Trains one LoRA adapter of the training experiment (CausalDiagnosis/cluster_jobs/README.md) on one H100 in the training environment
# of setup_train_env.sbatch. AGENT names the agent (plan_execute, agent_as_tool or causal), ARM the arm, filtered or
# unfiltered, and FOLD the fold that the model holds out, and the examples are those of that model, which rejection_sampling_examples.py wrote
# under $HYD_TRAINING/data/<agent>_f<fold> from its training starts. The adapter
# lands in $HYD_TRAINING/adapters/rft_<agent>_f<fold> or rft_<agent>_unfiltered_f<fold> with its training_manifest.json, and a job
# that a preemptible partition requeues resumes from its latest checkpoint. With SMOKE=1 the job checks the training environment
# and the card without data: train_lora.py --smoke makes one optimizer step on four made-up examples into a training folder of its
# own, ROOT/training_smoke, and AGENT, ARM and FOLD are not read. The job checks its card as a serving job does, so a
# registered job, and any job once env.sh names the card of record, stops with GPU_NOT_OF_RECORD and status 2 on another card
# before the trainer starts, and train_lora.py writes the card into the training manifest beside the adapter. Submit as
#   AGENT=causal ARM=filtered FOLD=1 bash $H/submit.sh train_lora.sbatch
set -uo pipefail
source "${HPC_DIR:?submit this script with CausalDiagnosis/cluster_jobs/submit.sh}/env.sh"
require_model
if [ "${SMOKE:-}" != 1 ]; then
  case "${ARM:-}" in filtered|unfiltered) ;; *) echo "ARM must be filtered or unfiltered" >&2; exit 2;; esac
  case "${AGENT:-}" in plan_execute|agent_as_tool|causal) ;; *) echo "AGENT must be plan_execute, agent_as_tool or causal" >&2; exit 2;; esac
  case "${FOLD:-}" in ''|*[!0-9]*) echo "FOLD must be a fold of the asset's registered split, which train_lora.py checks" >&2; exit 2;; esac
fi
[ -x "$TRAIN_VENV/bin/python" ] || { echo "no training environment at $TRAIN_VENV; run setup_train_env.sbatch first" >&2; exit 2; }
MODEL_DIR=$(model_dir "$MODEL")
if [ "${SMOKE:-}" = 1 ]; then
  TRAIN_ARGS=(--smoke --training-root "$ROOT/training_smoke")
  WHAT="the smoke run of the trainer"
  DONE=smoke
else
  EXAMPLES=$HYD_TRAINING/data/${AGENT}_f$FOLD/examples_$ARM.jsonl
  [ -s "$EXAMPLES" ] || { echo "no examples at $EXAMPLES; run rejection_sampling_examples.py on the sampling runs of $AGENT on fold $FOLD first" >&2; exit 2; }
  TRAIN_ARGS=(--examples "$EXAMPLES")
  WHAT="LoRA of $AGENT on the $ARM arm of fold $FOLD"
  DONE="$AGENT $ARM $FOLD"
fi
L=$(node_local_dir); trap 'rm -rf "$L"' EXIT
export HF_HOME=$L/hf HF_HUB_OFFLINE=1 TMPDIR=$L/tmp; mkdir -p "$TMPDIR"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
runs_root
check_card
# the card reaches the training manifest through the environment of the trainer, in the form of the card line of an endpoint file
export HYD_TRAINING_CARD
HYD_TRAINING_CARD=$(card_line)
cd "$FORK" || exit 2
job_header "$WHAT, model $MODEL_DIR, restart ${SLURM_RESTART_COUNT:-0}"
"$TRAIN_VENV/bin/python" -m CausalDiagnosis.training.train_lora "${TRAIN_ARGS[@]}" --base-model "$MODEL_DIR" || exit 3
echo "CD_TRAIN_DONE $DONE"
