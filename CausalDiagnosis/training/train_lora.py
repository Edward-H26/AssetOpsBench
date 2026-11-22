"""LoRA fine tuning of the agent model on the examples of rejection_sampling_examples.py (the training table), one adapter per agent, arm and fold
of the cross-fitting. It runs in the training environment of CausalDiagnosis/cluster_jobs/0_setup/requirements-train.txt, a virtual environment of its own on the
cluster, and imports nothing of the fork, since that environment holds the training packages and not the fork's (CausalDiagnosis/docs/experiments.md, section Training in
the environment), beyond training_variants/, which needs the standard library alone and names the adapters.

usage, on one GPU of the cluster through CausalDiagnosis/cluster_jobs/3_run/train_lora.sbatch, with HYD_TRAINING naming the training folder:
  python -m CausalDiagnosis.training.train_lora --examples <examples_filtered.jsonl> --base-model <model folder>

Every example is tokenised as SGLang v0.5.19 tokenised the request of the agent. On the chat route (serving_chat.py) the prompt is
the one user message of the request, which the model's own chat template renders with the generation prompt, and the encoding
leaves out special tokens when the tokenizer adds them by itself. On the completions route (serving_completions.py) the raw prompt
is encoded as it was sent. The reply follows, with the stop string that ended it or else the end token, and only those tokens
carry the loss. The default system message of granite's template prints today's date, so the clock of the template is
pinned to the day of the call, which makes the tokens the ones of the request and the same on any training day, and the job stops
when the encoded prompt of an example differs in length from the prompt tokens the server counted for its call. An example longer
than the maximum length is left out and counted, never cut, since a cut prompt is a prompt the model never saw.
SFTTrainer of TRL then trains the adapter of the LoRA settings of the asset's rft.json on the pre-tokenised examples, with the
completion mask as the only loss mask, and saves a checkpoint every save_steps steps under checkpoints/<adapter> of the training
folder. Before the first step it writes what the checkpoints are made from, SOURCE.json with the hashes of the examples, of the data
manifest, of rft.json and of this script and the commit of the base model, and after a requeue it resumes from the latest
checkpoint only when that record equals the present one, else it stops and names the field that differs, and it moves and deletes
nothing. adapters/<adapter> gets the adapter, which SGLang serves under that name, rft_<agent>_f<fold> for the filtered arm and
rft_<agent>_unfiltered_f<fold> for the control, and training_manifest.json with the agent, the arm, the fold, the counts, the
episodes left out of the data as the data manifest counts them, the hashes of the inputs, the source, the times of the job with
their offset from UTC, which a run compares with the start of its server, the card of the job, which CausalDiagnosis/cluster_jobs/3_run/train_lora.sbatch checks
against the card of record and hands over in HYD_TRAINING_CARD, and the versions of the packages.

--smoke checks the training environment without any data: one optimizer step on four made-up examples, two on each route, into
a training folder of its own that is not the one HYD_TRAINING names, so the adapter smoke that it leaves is one no variant plays. It
prints the arguments of the trainer as the pinned release resolved them, the warmup steps among them."""
import argparse
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent

from CausalDiagnosis.experiments.table_definitions.training_variants import (ADAPTER_MANIFEST, DATA_MANIFEST, DEFAULT_ASSET, FILTERED, FOLD_CONVENTION, FOLD_CONVENTION_KEY,  # noqa: E402
                      TRAINING_SPLIT_KEY, VISITS, adapter_name_of_arm, asset_training, check_fold_convention, read_record, sha256_file, training_root)

PACKAGES = ("torch", "transformers", "trl", "peft", "accelerate", "datasets")
SOURCE = "SOURCE.json"  # what the checkpoints of a folder are made from, written before the first step
SMOKE = "smoke"  # the adapter of a smoke run, a name that no variant plays
CARD_VARIABLE = "HYD_TRAINING_CARD"  # the card line of the training job, which CausalDiagnosis/cluster_jobs/3_run/train_lora.sbatch exports
PIN_PROBES = (date(2001, 1, 2), date(2002, 3, 4))  # two days that no sampling run started on, to test that the pin reaches the template


def pinned_clock(day):
    """strftime_now of the chat template for one day, so that the date the template prints is the day of the sampling run."""
    moment = datetime.combine(date.fromisoformat(day) if isinstance(day, str) else day, datetime.min.time())
    return lambda fmt: moment.strftime(fmt)


def render(tokenizer, example, day=None):
    """The prompt of a chat example as the chat template renders the one user message, on the day of its sampling run."""
    return tokenizer.apply_chat_template([{"role": "user", "content": example["prompt"]}], tokenize=False, add_generation_prompt=True,
                                         strftime_now=pinned_clock(day or example["date"]))


def check_pin(tokenizer, examples):
    """Stops the job when the template reads the clock and the pin does not reach it, so no example carries the day of training.
    The check renders the first chat example, whose prompt holds the template's default system message, on two probe days."""
    if "strftime_now" not in (getattr(tokenizer, "chat_template", None) or ""):
        return
    probe = next((e for e in examples if e["route"] == "chat"), None)
    if probe is not None and render(tokenizer, probe, PIN_PROBES[0]) == render(tokenizer, probe, PIN_PROBES[1]):
        raise SystemExit("the chat template prints the date and the pinned clock did not reach it, so the prompts would carry the "
                         "day of training; check the transformers version of CausalDiagnosis/cluster_jobs/0_setup/requirements-train.txt")


def encode(tokenizer, example):
    """(input ids, completion mask) of one example: the prompt as SGLang encoded the request, the chat template's rendering on the
    chat route and the raw text on the completions route, then the reply, and the end token unless a stop string ended the reply."""
    if example["route"] == "completion":
        prompt = tokenizer.encode(example["prompt"])
    else:
        special = {"add_special_tokens": False} if len(tokenizer.encode("")) > 0 else {}
        prompt = tokenizer.encode(render(tokenizer, example), **special)
    end = [] if example.get("ends_at_stop") else [tokenizer.eos_token_id]
    completion = tokenizer.encode(example["completion"], add_special_tokens=False) + end
    return prompt + completion, [0] * len(prompt) + [1] * len(completion)


def read_examples(path):
    """The examples of one file, refused unless they belong to one agent, one arm and one fold."""
    examples = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not examples:
        raise SystemExit(f"{path} holds no example, so there is nothing to train on")
    kinds = {(e["agent"], e["arm"], e.get("fold")) for e in examples}
    if len(kinds) != 1:
        raise SystemExit(f"{path} mixes the agents, arms and folds {sorted(kinds, key=str)}, and one adapter learns one agent, one "
                         "arm and one fold")
    return examples


def check_gate(examples_path):
    """Refuses data whose manifest, beside the examples, holds no gate or a gate below the minimum, so that no
    adapter trains on too few kept episodes and no fold gets an adapter its record calls missing."""
    manifest = Path(examples_path).parent / DATA_MANIFEST
    record = read_record(manifest) if manifest.exists() else {}
    gate = record.get("gate")
    if gate is None:
        raise SystemExit(f"{manifest} holds no gate of the registered minimum, so rejection_sampling_examples.py has to build the data again")
    check_fold_convention(record, manifest)
    if not record.get("asset_id"):
        raise SystemExit(f"{manifest} names no asset, whose registered numbers the trainer would read, so rejection_sampling_examples.py has to build "
                         "the data again")
    if not gate.get("trained"):
        state = "a second round of sampling is due" if gate.get("second_round_due") else "no adapter trains for this agent and fold"
        raise SystemExit(f"{manifest} kept {gate.get('kept')} episodes against the registered minimum of {gate.get('minimum')}, so {state}")
    return gate


def prepare(tokenizer, examples, max_length):
    """(rows of the dataset, counts) with every example over max_length tokens left out and counted. An example that records the
    prompt tokens of its call must encode to that many prompt tokens, so no example trains on a prompt the model never saw."""
    check_pin(tokenizer, examples)
    rows, dropped, tokens, checked = [], 0, 0, 0
    for example in examples:
        input_ids, mask = encode(tokenizer, example)
        if example.get("input_tokens"):
            prompt_tokens = mask.index(1)
            if prompt_tokens != example["input_tokens"]:
                raise SystemExit(f"call {example.get('call')} of {example.get('run')}/{example.get('episode')} encodes to {prompt_tokens} prompt "
                                 f"tokens and the server counted {example['input_tokens']}, so the training prompt is not the one the model saw")
            checked += 1
        if len(input_ids) > max_length:
            dropped += 1
            continue
        rows.append({"input_ids": input_ids, "completion_mask": mask})
        tokens += len(input_ids)
    return rows, {"examples": len(examples), "kept": len(rows), "dropped_over_max_length": dropped, "tokens": tokens,
                  "prompt_tokens_checked": checked}


def source_record(examples_path, data_manifest, config_file, base_model):
    """What the checkpoints of an adapter are made from: the hashes of the examples, of the data manifest, of rft.json and of this
    script, and the commit of the base model."""
    commit = Path(base_model) / "HUB_COMMIT.txt"
    return {"examples_sha256": sha256_file(examples_path), "data_manifest_sha256": sha256_file(data_manifest) if data_manifest else None,
            "config_sha256": sha256_file(config_file), "trainer_sha256": sha256_file(Path(__file__)),
            "base_model_commit": commit.read_text(encoding="utf-8").split()[0] if commit.exists() else None}


def checkpoint_to_resume(checkpoints, record, last_checkpoint):
    """The checkpoint to resume from: None for a folder without one, after the record is written there, and the last checkpoint
    when the record of the folder equals the present one. A folder whose record is missing or differs stops the job and names the
    first field that differs. Nothing is moved or deleted. last_checkpoint is get_last_checkpoint of transformers."""
    checkpoints = Path(checkpoints)
    last = last_checkpoint(str(checkpoints)) if checkpoints.exists() else None
    source = checkpoints / SOURCE
    if last is None:
        checkpoints.mkdir(parents=True, exist_ok=True)
        source.write_text(json.dumps(record, indent=1), encoding="utf-8")
        return None
    if not source.exists():
        raise SystemExit(f"{checkpoints} holds the checkpoint {last} and no {SOURCE}, so nothing says what it was trained on; move the "
                         "folder aside")
    saved = json.loads(source.read_text(encoding="utf-8"))
    differs = [key for key in record if saved.get(key) != record[key]]
    if differs:
        raise SystemExit(f"{checkpoints} holds checkpoints of another {differs[0]}, {saved.get(differs[0])} there and {record[differs[0]]} "
                         "now; move the folder aside")
    return last


def smoke_examples():
    """The four made-up examples of a smoke run, two on the chat route and two on the completions route, of one agent, arm and fold."""
    base = {"agent": SMOKE, "arm": FILTERED, "fold": 1, "run": SMOKE, "episode": SMOKE, "seed": 0, "sample": 1, "date": "2026-01-05"}
    return [{**base, "call": 0, "route": "chat", "prompt": "Which component is faulty?", "completion": "The cooler."},
            {**base, "call": 1, "route": "chat", "prompt": "Which reading comes next?", "completion": "The pressure of the pump."},
            {**base, "call": 2, "route": "completion", "prompt": "Question: which part is worn?\nThought:", "completion": " the valve",
             "ends_at_stop": False},
            {**base, "call": 3, "route": "completion", "prompt": "Question: what is repaired?\nThought:", "completion": " the accumulator",
             "ends_at_stop": False}]


def smoke_inputs(root):
    """The training folder of a smoke run and its examples file, written there. The folder must be named and must not be the one
    HYD_TRAINING names, so that no variant reads the adapter the run leaves."""
    named = os.environ.get("HYD_TRAINING")
    if not root or (named and Path(root).resolve() == Path(named).resolve()):
        raise SystemExit("--smoke trains into a training folder of its own: name one with --training-root, and not the folder that "
                         "HYD_TRAINING names")
    path = Path(root) / "data" / SMOKE / "examples_smoke.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(example) + "\n" for example in smoke_examples()), encoding="utf-8")
    return Path(root), path


def job_card(environ=None):
    """The card of the training job from the line that CausalDiagnosis/cluster_jobs/3_run/train_lora.sbatch exports in CARD_VARIABLE, the card line of an endpoint
    file (gpu_name, gpu_mib, gpu_uuid, driver_version and cuda_version, the spaces of the name written as underscores), as a
    record with the spaces restored, or None outside such a job."""
    words = (os.environ if environ is None else environ).get(CARD_VARIABLE, "").split()
    if not words:
        return None
    card = dict(zip(words[::2], words[1::2]))
    if "gpu_name" in card:
        card["gpu_name"] = card["gpu_name"].replace("_", " ")
    return card


def versions():
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for name in PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def main(argv):
    ap = argparse.ArgumentParser(description="LoRA fine tuning of the agent model on the examples of one agent, arm and fold")
    ap.add_argument("--examples", default=None, help="examples_filtered.jsonl or examples_unfiltered.jsonl of rejection_sampling_examples.py")
    ap.add_argument("--base-model", required=True, help="the folder of the agent model that SGLang serves")
    ap.add_argument("--training-root", default=None, help="the training folder, HYD_TRAINING by default")
    ap.add_argument("--smoke", action="store_true",
                    help="one optimizer step on four made-up examples into the training folder of --training-root, which must not be HYD_TRAINING")
    a = ap.parse_args(argv)
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    if a.smoke:
        root, examples_path = smoke_inputs(a.training_root)
        examples, data_manifest, data, asset_id = read_examples(examples_path), None, {}, DEFAULT_ASSET
        name = SMOKE
    else:
        if not a.examples:
            ap.error("--examples is required without --smoke")
        examples = read_examples(a.examples)
        check_gate(a.examples)
        examples_path = Path(a.examples)
        data_manifest = examples_path.parent / DATA_MANIFEST
        data = read_record(data_manifest)
        asset_id = data["asset_id"]  # check_gate refuses a data manifest that names none
        root = training_root(a.training_root)
        name = adapter_name_of_arm(examples[0]["agent"], examples[0]["arm"], examples[0].get("fold"), asset_id)
    training = asset_training(asset_id)
    lora, train = training.config["lora"], training.config["training"]
    agent, arm, fold = examples[0]["agent"], examples[0]["arm"], examples[0].get("fold")
    out, checkpoints = root / "adapters" / name, root / "checkpoints" / name
    source = source_record(examples_path, data_manifest, training.config_file, a.base_model)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from transformers.trainer_utils import get_last_checkpoint
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(a.base_model)
    rows, counts = prepare(tokenizer, examples, train["max_length"])
    if not rows:
        raise SystemExit(f"every example of {a.examples} is longer than {train['max_length']} tokens")
    args = SFTConfig(
        output_dir=str(checkpoints), num_train_epochs=train["epochs"], learning_rate=train["learning_rate"],
        lr_scheduler_type=train["lr_scheduler"], warmup_steps=train["warmup"], weight_decay=train["weight_decay"],
        max_grad_norm=train["max_grad_norm"], per_device_train_batch_size=train["per_device_batch"],
        gradient_accumulation_steps=train["gradient_accumulation"], bf16=train["precision"] == "bf16",
        gradient_checkpointing=train["gradient_checkpointing"], gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=train["seed"], data_seed=train["seed"], logging_steps=train["logging_steps"], save_strategy="steps",
        save_steps=train["save_steps"], save_total_limit=2, report_to="none", completion_only_loss=True,
        max_length=train["max_length"], model_init_kwargs={"dtype": torch.bfloat16, "attn_implementation": train["attention"]},
        max_steps=1 if a.smoke else -1)
    peft_config = LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"], target_modules=lora["target_modules"],
                             bias="none", task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=a.base_model, args=args, train_dataset=Dataset.from_list(rows), processing_class=tokenizer, peft_config=peft_config)
    if a.smoke:
        print(f"SMOKE the trainer's arguments as the pinned release resolved them, warmup steps {trainer.args.warmup_steps}:")
        print(trainer.args.to_json_string())
    resume = checkpoint_to_resume(checkpoints, source, get_last_checkpoint)
    result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(out))
    commit = Path(a.base_model) / "HUB_COMMIT.txt"
    manifest = {"agent": agent, "arm": arm, "fold": fold, FOLD_CONVENTION_KEY: FOLD_CONVENTION, "asset_id": asset_id, "adapter": name,
                "smoke": a.smoke, "started": started,
                "ended": datetime.now().astimezone().isoformat(timespec="seconds"),
                "resumed_from": resume, "source": source, "examples": {"file": str(examples_path), "sha256": sha256_file(examples_path)},
                "data_manifest_sha256": sha256_file(data_manifest) if data_manifest else None,
                "config": {"file": str(training.config_file.relative_to(HERE.parents[1])), "sha256": sha256_file(training.config_file),
                           "lora": lora, "training": train},
                "base_model": {"folder": a.base_model, "commit": commit.read_text(encoding="utf-8").split()[0] if commit.exists() else None},
                "counts": counts, VISITS: ((data.get("arms") or {}).get(arm) or {}).get(VISITS), "steps": result.global_step,
                "stamp": data.get("stamp"), TRAINING_SPLIT_KEY: data.get(TRAINING_SPLIT_KEY), "built_from": data.get("runs"),
                "train_loss": result.training_loss,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "card": job_card(), "versions": versions()}
    from CausalDiagnosis.experiments.run_verification.public_record import public_value  # the manifest beside the adapter names no folder of the machine (public_record)

    (out / ADAPTER_MANIFEST).write_text(json.dumps(public_value(manifest), indent=1), encoding="utf-8")
    print(f"wrote {out}: {counts['kept']} of {counts['examples']} examples, {result.global_step} steps")
    return manifest


if __name__ == "__main__":
    main(sys.argv[1:])
