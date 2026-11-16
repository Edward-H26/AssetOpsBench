"""The trainer of the LoRA adapters (train_lora.py) without its training packages: a fake tokenizer stands
in for the model's, so the tests check that an example is encoded as SGLang encoded the request on its route, the reply alone
carries the loss, the date of the template is pinned to the sampling run, a long example is left out, never cut, one file holds
the examples of one agent, arm and fold, a checkpoint of another source is refused, and the smoke run takes its made-up examples
into a training folder of its own. No model is loaded and nothing is trained."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from CausalDiagnosis.training import train_lora as t
from CausalDiagnosis.experiments.table_definitions.training_variants import ADAPTER_MANIFEST, CONFIG, CONFIG_FILE, sha256_file

TEMPLATE = "{{ strftime_now('%B %d, %Y') }} stands for the template, which reads the clock"


class FakeTokenizer:
    """A template that prints a default system message with the date when the first message is not a system message, then every
    message and the generation prompt; one token per character, the end token 0."""

    chat_template = TEMPLATE
    eos_token_id = 0

    def __init__(self, adds_specials=False, ignores_pin=False):
        self.adds_specials, self.ignores_pin, self.calls = adds_specials, ignores_pin, []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, strftime_now=None):
        assert tokenize is False
        self.calls.append({"messages": messages, "add_generation_prompt": add_generation_prompt})
        text = ""
        if messages[0]["role"] != "system":
            text += "[system Today " + ("NOW" if self.ignores_pin else strftime_now("%B %d, %Y")) + "]"
        text += "".join(f"[{m['role']} {m['content']}]" for m in messages)
        return text + ("[assistant " if add_generation_prompt else "")

    def encode(self, text, add_special_tokens=True):
        return ([7] if self.adds_specials and add_special_tokens else []) + [ord(c) for c in text]


def _example(**fields):
    return {"agent": "causal", "arm": "filtered", "fold": 1, "date": "2026-11-21", "route": "chat", "prompt": "plan", "completion": "ok", **fields}


def test_a_chat_prompt_is_the_one_user_message_and_the_reply_alone_carries_the_loss():
    tokenizer = FakeTokenizer()
    ids, mask = t.encode(tokenizer, _example())
    prompt = "[system Today November 21, 2026][user plan][assistant "
    assert ids == [ord(c) for c in prompt] + [ord("o"), ord("k"), 0]
    assert mask == [0] * len(prompt) + [1, 1, 1]
    assert tokenizer.calls[-1] == {"messages": [{"role": "user", "content": "plan"}], "add_generation_prompt": True}
    # a tokenizer that adds a special token by itself encodes the rendered prompt without it, as SGLang's encode does
    ids, mask = t.encode(FakeTokenizer(adds_specials=True), _example())
    assert ids[0] == ord("[") and ids[-3:] == [ord("o"), ord("k"), 0] and sum(mask) == 3


def test_the_template_clock_is_pinned_to_the_sampling_run_and_a_template_that_ignores_the_pin_stops_the_job():
    rows, counts = t.prepare(FakeTokenizer(), [_example(date="2026-11-21"), _example(date="2026-11-23")], 10_000)
    texts = ["".join(chr(i) for i in row["input_ids"]) for row in rows]
    assert "November 21, 2026" in texts[0] and "November 23, 2026" in texts[1] and counts["kept"] == 2
    with pytest.raises(SystemExit, match="pinned clock"):
        t.prepare(FakeTokenizer(ignores_pin=True), [_example()], 10_000)
    # raw completions never pass through the template, so a file of them needs no pin
    t.prepare(FakeTokenizer(ignores_pin=True), [_example(route="completion", prompt="Q", completion="A", ends_at_stop=False)], 10_000)


def test_a_raw_completion_is_encoded_as_sent_and_a_stop_string_takes_the_place_of_the_end_token():
    tokenizer = FakeTokenizer()
    ids, mask = t.encode(tokenizer, _example(route="completion", prompt="Thought:", completion=" read\nObservation:", ends_at_stop=True))
    assert ids == [ord(c) for c in "Thought: read\nObservation:"] and sum(mask) == len(" read\nObservation:") and not tokenizer.calls
    ids, _mask = t.encode(tokenizer, _example(route="completion", prompt="Q", completion="A", ends_at_stop=False))
    assert ids == [ord("Q"), ord("A"), 0]


def test_an_example_over_the_maximum_length_is_left_out_and_counted_never_cut():
    short, long = _example(), _example(completion="x" * 500)
    rows, counts = t.prepare(FakeTokenizer(), [short, long], 100)
    assert len(rows) == 1 and rows[0] == dict(zip(("input_ids", "completion_mask"), t.encode(FakeTokenizer(), short)))
    assert counts == {"examples": 2, "kept": 1, "dropped_over_max_length": 1, "tokens": len(rows[0]["input_ids"]), "prompt_tokens_checked": 0}


def test_an_example_whose_prompt_encodes_to_other_than_the_servers_count_stops_the_job():
    """The server's count of prompt tokens is the check that the training prompt is the one the model saw: the same template,
    the same day and the same special tokens."""
    prompt_tokens = len("[system Today November 21, 2026][user plan][assistant ")
    _rows, counts = t.prepare(FakeTokenizer(), [_example(input_tokens=prompt_tokens)], 10_000)
    assert counts["prompt_tokens_checked"] == 1
    with pytest.raises(SystemExit, match="the server counted"):
        t.prepare(FakeTokenizer(), [_example(input_tokens=prompt_tokens + 1)], 10_000)
    with pytest.raises(SystemExit, match="the server counted"):  # a day other than the call's changes the prompt
        t.prepare(FakeTokenizer(), [_example(date="2026-11-01", input_tokens=prompt_tokens + 1)], 10_000)


def test_one_file_holds_the_examples_of_one_agent_one_arm_and_one_fold(tmp_path):
    path = tmp_path / "examples.jsonl"
    for other in (_example(arm="unfiltered"), _example(fold=2)):
        path.write_text("\n".join(json.dumps(e) for e in (_example(), other)), encoding="utf-8")
        with pytest.raises(SystemExit, match="one agent, one arm and one fold"):
            t.read_examples(path)
    path.write_text(json.dumps(_example(fold=2)), encoding="utf-8")
    assert t.read_examples(path)[0]["fold"] == 2 and t.adapter_name_of_arm("causal", "unfiltered", 2, "hyd_1") == "rft_causal_unfiltered_f2"
    path.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="no example"):
        t.read_examples(path)


def test_the_trainer_reads_the_registered_config_and_imports_no_training_package():
    # the trainer reads the rft.json of the asset that the data manifest names, and binds no constant of the default asset
    assert not hasattr(t, "CONFIG") and not hasattr(t, "CONFIG_FILE")
    assert not {"trl", "peft"} & set(sys.modules)
    lora, train = CONFIG["lora"], CONFIG["training"]
    assert (lora["r"], lora["alpha"], lora["dropout"]) == (16, 32, 0.05) and len(lora["target_modules"]) == 7
    assert (train["epochs"], train["learning_rate"], train["max_length"], train["per_device_batch"] * train["gradient_accumulation"]) == (2, 1e-4, 32768, 16)
    assert (CONFIG["samples_per_state"], CONFIG["sampling_temperature"], CONFIG["rounds"]) == (17, 0.7, 1)
    # the number of folds and the training start states follow from the split
    assert CONFIG["folds"] == {"file": "folds.json", "development_file": "folds_development.json"}


def test_the_trainer_refuses_data_below_the_minimum_or_without_a_gate(tmp_path):
    """No adapter trains on fewer kept episodes than the minimum, whether a second round is due or not."""
    from CausalDiagnosis.training import train_lora
    from CausalDiagnosis.experiments.table_definitions.training_variants import DATA_MANIFEST, FOLD_CONVENTION, FOLD_CONVENTION_KEY, gate_of
    stated = {FOLD_CONVENTION_KEY: FOLD_CONVENTION, "asset_id": "hyd_1"}

    examples = tmp_path / "causal_f1" / "examples_filtered.jsonl"
    examples.parent.mkdir()
    examples.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="holds no gate"):
        train_lora.check_gate(examples)
    for gate, message in ((gate_of(2, 2, 5, 1, "hyd_1"), "second round of sampling is due"), (gate_of(2, 2, 5, 2, "hyd_1"), "no adapter trains")):
        (examples.parent / DATA_MANIFEST).write_text(json.dumps({"gate": gate, **stated}), encoding="utf-8")
        with pytest.raises(SystemExit, match=message):
            train_lora.check_gate(examples)
    # data that names no asset is refused, since the trainer reads the registered numbers of the asset the data names
    (examples.parent / DATA_MANIFEST).write_text(json.dumps({"gate": gate_of(5, 4, 5, 1, "hyd_1"), FOLD_CONVENTION_KEY: FOLD_CONVENTION}),
                                                 encoding="utf-8")
    with pytest.raises(SystemExit, match="names no asset"):
        train_lora.check_gate(examples)
    (examples.parent / DATA_MANIFEST).write_text(json.dumps({"gate": gate_of(5, 4, 5, 1, "hyd_1"), **stated}), encoding="utf-8")
    assert train_lora.check_gate(examples)["trained"]


def test_a_checkpoint_of_another_source_is_refused(tmp_path):
    """The record of what the checkpoints are made from holds the hashes of the examples, of the data manifest, of rft.json and of
    the trainer and the commit of the base model. checkpoint_to_resume writes it into a folder without a checkpoint, returns the
    last checkpoint when the record of the folder equals the present one, and stops with the name of the field when the examples or
    rft.json differ or the record is missing, moving and deleting nothing."""
    examples, manifest, model = tmp_path / "examples.jsonl", tmp_path / "data_manifest.json", tmp_path / "model"
    examples.write_text("{}\n", encoding="utf-8")
    manifest.write_text("{}", encoding="utf-8")
    model.mkdir()
    (model / "HUB_COMMIT.txt").write_text("abc123 main\n", encoding="utf-8")
    record = t.source_record(examples, manifest, CONFIG_FILE, model)
    assert record == {"examples_sha256": sha256_file(examples), "data_manifest_sha256": sha256_file(manifest), "config_sha256": sha256_file(CONFIG_FILE),
                      "trainer_sha256": sha256_file(Path(t.__file__)), "base_model_commit": "abc123"}
    folder = tmp_path / "checkpoints" / "rft_causal_f1"

    def last(path):
        found = sorted(Path(path).glob("checkpoint-*"))
        return str(found[-1]) if found else None

    assert t.checkpoint_to_resume(folder, record, last) is None and json.loads((folder / t.SOURCE).read_text(encoding="utf-8")) == record
    (folder / "checkpoint-40").mkdir()
    assert t.checkpoint_to_resume(folder, record, last) == str(folder / "checkpoint-40")
    for field in ("examples_sha256", "config_sha256"):
        with pytest.raises(SystemExit, match=f"checkpoints of another {field}"):
            t.checkpoint_to_resume(folder, {**record, field: "f" * 64}, last)
    (folder / t.SOURCE).unlink()
    with pytest.raises(SystemExit, match="no SOURCE.json"):
        t.checkpoint_to_resume(folder, record, last)
    assert (folder / "checkpoint-40").is_dir()


def test_the_smoke_examples_are_two_per_route_and_the_trainer_takes_them(tmp_path, monkeypatch):
    monkeypatch.delenv("HYD_TRAINING", raising=False)
    root, path = t.smoke_inputs(tmp_path / "smoke")
    examples = t.read_examples(path)
    assert root == tmp_path / "smoke" and [e["route"] for e in examples] == ["chat", "chat", "completion", "completion"]
    assert {(e["agent"], e["arm"], e["fold"]) for e in examples} == {(t.SMOKE, "filtered", 1)}
    rows, counts = t.prepare(FakeTokenizer(), examples, 10_000)
    assert counts["kept"] == 4 and all(sum(row["completion_mask"]) for row in rows)


def test_the_smoke_run_needs_a_training_folder_of_its_own(tmp_path, monkeypatch):
    """--smoke without --training-root, or with the folder that HYD_TRAINING names, stops before any import of a training package."""
    monkeypatch.setenv("HYD_TRAINING", str(tmp_path / "training"))
    for argv in (["--smoke", "--base-model", "m"], ["--smoke", "--base-model", "m", "--training-root", str(tmp_path / "training")]):
        with pytest.raises(SystemExit, match="a training folder of its own"):
            t.main(argv)
    assert "trl" not in sys.modules and "peft" not in sys.modules


class _TrainerArguments(dict):
    """The arguments of the trainer as a stand-in of SFTConfig holds them."""

    warmup_steps = 0

    def to_json_string(self):
        return json.dumps(self, default=str)


class _Trainer:
    """A stand-in of SFTTrainer: one step, a loss, and the adapter folder that save_model makes."""

    def __init__(self, model, args, train_dataset, processing_class, peft_config):
        self.args = args

    def train(self, resume_from_checkpoint=None):
        return SimpleNamespace(global_step=1, training_loss=0.5)

    def save_model(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)


def test_the_training_manifest_records_the_card_of_the_job(tmp_path, monkeypatch):
    """CausalDiagnosis/cluster_jobs/3_run/train_lora.sbatch hands the line of its card to the trainer in HYD_TRAINING_CARD, in the form of the card line of an
    endpoint file, and the training manifest beside the adapter holds it as a record with the spaces of the name restored, or null
    outside such a job, written without the uuid of the card, which no public record names, while the manifest that main returns
    keeps it. A smoke run with stand-ins for the training packages writes the manifest of a made-up card."""
    line = "gpu_name NVIDIA_TEST_NVL gpu_mib 1000 gpu_uuid GPU-1 driver_version 1.2.3 cuda_version 9.9"
    card = {"gpu_name": "NVIDIA TEST NVL", "gpu_mib": "1000", "gpu_uuid": "GPU-1", "driver_version": "1.2.3", "cuda_version": "9.9"}
    assert t.CARD_VARIABLE == "HYD_TRAINING_CARD" and t.job_card({t.CARD_VARIABLE: line}) == card and t.job_card({}) is None
    stand_ins = {
        "torch": SimpleNamespace(bfloat16="bfloat16", cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda index: "NVIDIA TEST NVL")),
        "datasets": SimpleNamespace(Dataset=SimpleNamespace(from_list=lambda rows: rows)),
        "peft": SimpleNamespace(LoraConfig=lambda **settings: settings),
        "transformers": SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda path: FakeTokenizer())),
        "transformers.trainer_utils": SimpleNamespace(get_last_checkpoint=lambda path: None),
        "trl": SimpleNamespace(SFTConfig=lambda **settings: _TrainerArguments(settings), SFTTrainer=_Trainer)}
    for name, module in stand_ins.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delenv("HYD_TRAINING", raising=False)
    model = tmp_path / "model"
    model.mkdir()
    for environ, expected in (({t.CARD_VARIABLE: line}, card), ({}, None)):
        monkeypatch.delenv(t.CARD_VARIABLE, raising=False)
        for name, value in environ.items():
            monkeypatch.setenv(name, value)
        root = tmp_path / f"smoke_{len(environ)}"
        manifest = t.main(["--smoke", "--base-model", str(model), "--training-root", str(root)])
        written = json.loads((root / "adapters" / t.SMOKE / ADAPTER_MANIFEST).read_text(encoding="utf-8"))
        public = None if expected is None else {key: value for key, value in expected.items() if key != "gpu_uuid"}
        assert manifest["card"] == expected and written["card"] == public and written["gpu"] == "NVIDIA TEST NVL"
