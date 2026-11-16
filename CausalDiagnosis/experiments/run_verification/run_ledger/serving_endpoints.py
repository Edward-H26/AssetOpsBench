from CausalDiagnosis.utils.paths import PACKAGE
import json
import os
import re
import urllib.request
from pathlib import Path
from . import hashing


def endpoint_file(path):
    """An endpoint file of CausalDiagnosis/cluster_jobs/2_serve_model/serve_model.sbatch as its lines and fields, or None without a path: the base URL on the
    first line, then lines of key and value pairs (job, node, model, served_name, started, weights_commit, sglang, image,
    image_sha256, and the GPU card of the job, CARD_KEYS and CARD_RECORD_KEYS). A line that is not a list of pairs stays in
    lines only, so no value holds a space, and the job writes the name of its GPU with every space as an underscore."""
    if not path:
        return None
    path = Path(path)
    if not path.is_file():
        return {"file": str(path), "lines": None}
    lines = path.read_text(encoding="utf-8").splitlines()
    record = {"file": str(path.resolve()), "lines": lines, "base_url": lines[0].strip() if lines else None}
    for line in lines[1:]:
        tokens = line.split()
        if tokens and len(tokens) % 2 == 0:
            record.update(zip(tokens[0::2], tokens[1::2]))
    return record


def endpoint_models(base_url, api_key=None):
    """The /models answer of an endpoint that a serving job of this project runs, where SGLang names the served model, its weights
    folder and its context length, or None when it does not answer. The key, API_KEY of the environment unless given, goes out
    as the drivers send it and is never kept."""
    key = os.environ.get("API_KEY", "") if api_key is None else api_key
    request = urllib.request.Request(base_url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8")).get("data")
    except (OSError, ValueError):
        return None


def served(name, endpoint, file_variable):
    """A model or the judge as the run reached it: the name and endpoint of run.json, and, when the grid job passed the
    endpoint file of the serving job, its fields, whether its URL is the run's endpoint (a file left by an earlier serving
    job is not) and the endpoint's model list. Policies and runs without a judge get None."""
    if not name:
        return None
    record = endpoint_file(os.environ.get(file_variable))
    if record:
        record["matches_endpoint"] = record.get("base_url") == endpoint
    return {"name": name, "endpoint": endpoint, "endpoint_file": record,
            "endpoint_models": endpoint_models(endpoint) if record and endpoint else None}


# ----- the file lists, the identity, the platform, the serving file of record and the record check -----
SERVING_KEYS = ("weights_commit", "weights_files_sha256", "image_sha256", "template_sha256", "engine_version", "dtype", "dtype_launch",
                "quantization", "kv_cache_dtype", "launch_args_sha256")  # the keys of an endpoint file that the serving identity reads
# the GPU card of the serving job, which binds once env.sh names the card of record (GPU_OF_RECORD, GPU_MIB_OF_RECORD): a run of
# record then plays and judges on that card alone (card_differences)
CARD_KEYS = ("gpu_name", "gpu_mib")
# what the endpoint file records of the card besides, which binds nothing, and the file that the GPU sampler of the job appends to
CARD_RECORD_KEYS = ("gpu_uuid", "driver_version", "cuda_version", "gpu_sampler")
# the registered setting of each served model that the job hands the platform record (CausalDiagnosis/cluster_jobs/1_checks/platform_record.sh, from env.sh): per
# endpoint, the variable of its endpoint file and, per key of the record, the variable of the setting and the line of the endpoint
# file that must equal it. The served name is the key that every comparison of a judge reads, as the judge records carry it, and
# the Hugging Face id and the pinned revision stand beside it, each compared with its own equal.
SETTING = {"agent": ("HYD_ENDPOINT_FILE", {"model": ("HYD_MODEL", "served_name"), "hub_id": ("HYD_HUB_ID", "model"),
                                           "revision": ("HYD_REVISION", "weights_commit")}),
           "judge": ("HYD_JUDGE_ENDPOINT_FILE", {"model": ("HYD_JUDGE_MODEL", "served_name"), "hub_id": ("HYD_JUDGE_HUB_ID", "model"),
                                                 "revision": ("HYD_JUDGE_REVISION", "weights_commit")})}


def serving_identity(record):
    """The serving identity of a model as served() met it: the weights commit, the image and the chat template hashes and the
    SGLang version as the engine, from the endpoint file, the served model and its context length from the entry of the
    endpoint's model list named as the run names the model, and the GPU card of the serving job (CARD_KEYS) from the endpoint
    file. A key that the file or the list lacks reads None."""
    if not record:
        return None
    fields = record.get("endpoint_file") or {}
    entry = next((m for m in (record.get("endpoint_models") or []) if isinstance(m, dict) and m.get("id") == record.get("name")), {})
    return {**{key: fields.get(key) for key in SERVING_KEYS}, "engine": fields.get("sglang"), "model": entry.get("id"),
            "context_length": entry.get("max_model_len"), **{key: fields.get(key) for key in CARD_KEYS}}


def card_of(identity):
    """The GPU card of a serving identity or of the fields of an endpoint file, {gpu_name, gpu_mib}, or None when it names
    neither, as for an endpoint without a file or the file of a serving job that wrote no card."""
    card = {key: (identity or {}).get(key) for key in CARD_KEYS}
    return card if any(card.values()) else None


def card_text(card):
    """A GPU card as a reason names it, the name with its underscores read back as spaces."""
    if not card:
        return "no GPU card"
    return f"{str(card.get('gpu_name')).replace('_', ' ')} with {card.get('gpu_mib')} MiB"


def card_differences(endpoint_record):
    """Why the endpoint that served() met as endpoint_record is not on the card of record, [] when it is or when env.sh names no
    card of record (GPU_OF_RECORD empty). The card of record is GPU_OF_RECORD, the name as nvidia-smi prints it, and GPU_MIB_OF_RECORD,
    its memory in MiB when set; the endpoint file of the serving job must exist, name the run's endpoint and hold its card, whose
    name it writes with every space as an underscore."""
    if not os.environ.get("GPU_OF_RECORD"):
        return []
    fields = (endpoint_record or {}).get("endpoint_file")
    if not fields:
        return ["the endpoint has no endpoint file of its serving job, which would name its GPU card"]
    if fields.get("lines") is None:
        return [f"the endpoint file {fields['file']} does not exist"]
    if not fields.get("matches_endpoint"):
        return [f"the endpoint file {fields['file']} names {fields.get('base_url')}, not the endpoint of the run"]
    reason = card_record_reason(card_of(fields))
    return [reason] if reason else []


def card_record_reason(card):
    """Why a card {gpu_name, gpu_mib} is not the card of record of env.sh, None when it is or when no card of record is named."""
    want_name, want_mib = os.environ.get("GPU_OF_RECORD", ""), os.environ.get("GPU_MIB_OF_RECORD", "")
    if not want_name:
        return None
    want = {"gpu_name": want_name.replace(" ", "_"), "gpu_mib": want_mib or (card or {}).get("gpu_mib")}
    return None if card == want else f"the endpoint serves on {card_text(card)}, and the card of record is {card_text(want)}"


def serving_record(record):
    """What run.json records of the endpoint that served() met as record, or None without one: its serving identity with the
    GPU card, the keys of CARD_RECORD_KEYS but the sampler, the node of the serving job, the endpoint file and whether it names
    the run's endpoint, and the sampler file, whose SHA-256 and length hash_samplers() adds at the end of the run."""
    if not record:
        return None
    fields = record.get("endpoint_file") or {}
    return {**serving_identity(record), **{key: fields.get(key) for key in CARD_RECORD_KEYS if key != "gpu_sampler"},
            "node": fields.get("node"), "endpoint_file": fields.get("file"), "matches_endpoint": fields.get("matches_endpoint"),
            "sampler": {"path": fields.get("gpu_sampler"), "sha256": None, "bytes": None}}


def hash_samplers(serving):
    """Adds to every endpoint of run.json serving the SHA-256 and the length of its sampler file as it stands at the end of
    the run, None for a file that cannot be read."""
    for entry in (serving or {}).values():
        sampler = (entry or {}).get("sampler")
        if sampler and sampler.get("path"):
            sampler["sha256"], sampler["bytes"] = hashing.prefix_hash(sampler["path"])


class OpenServingFile(ValueError):
    """The serving file of record is missing, lacks a field, holds an empty one, or a weights commit that is no pinned Hub commit."""


SERVING_FILE = PACKAGE / "experiments" / "table_definitions" / "serving_of_record.json"
SERVING_FORMAT = 1
SERVING_ROLES = ("agent", "judge")
SERVING_MODES = ("alone", "co_served")
# the flags of a server that loads the LoRA adapters of the training table (CausalDiagnosis/cluster_jobs/2_serve_model/serve_model.sbatch, SERVE_ADAPTERS=1): their
# count and names come from the training output, so no tracked file declares them; the comparison sets them aside and
# adapter_differences holds them to the lora_adapters line of the endpoint file
ADAPTER_FLAG, ADAPTER_VALUED = "--enable-lora", ("--max-loras-per-batch", "--lora-paths")
# what the tree itself fixes of each served model, with the line of the endpoint file it is compared with
# (CausalDiagnosis/cluster_jobs/2_serve_model/serve_model.sbatch, engine_record.py). What only a serving job can report (the dtype, quantization and kv cache dtype the
# server reports, the hash of the image file, the chat template, the card) is recorded in run.json and never declared here.
SERVING_FIELDS = {"hub_id": "model", "served_name": "served_name", "weights_commit": "weights_commit", "dtype": "dtype_launch",
                  "engine_version": "sglang", "launch_args": "launch_args"}
COMMIT = re.compile(r"[0-9a-f]{40}")


def serving_of_record(path=None):
    """The serving file of record as {"format", "agent", "judge"}. Raises OpenServingFile with the reason when the file is missing,
    a field of a role is absent or empty, a launch mode holds no list of flags, or a weights commit is not 40 hex digits, as a value
    of the form config:<sha256> is not."""
    path = Path(path or SERVING_FILE)
    if not path.is_file():
        raise OpenServingFile(f"no serving file {path.name}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("format") != SERVING_FORMAT:
        raise OpenServingFile(f"{path.name} has the format {record.get('format')}, not {SERVING_FORMAT}")
    for role in SERVING_ROLES:
        block = record.get(role) or {}
        for field in SERVING_FIELDS:
            if block.get(field) in (None, ""):
                raise OpenServingFile(f"{path.name}: {role}.{field} is missing")
        for mode in SERVING_MODES:
            if not isinstance(block["launch_args"].get(mode), list):
                raise OpenServingFile(f"{path.name}: {role}.launch_args.{mode} holds no list of flags")
        if not COMMIT.fullmatch(str(block["weights_commit"])):
            raise OpenServingFile(f"{path.name}: {role}.weights_commit {block['weights_commit']} is no pinned Hub commit of 40 hex digits")
    return record


def public_launch(block, mode):
    """The launch flags of a role and mode as engine_record.py writes them into the endpoint file: the model path as
    <hub id>@<weights commit>, the served name, then the flags the serving file declares for the mode."""
    return ["--model-path", f"{block['hub_id']}@{block['weights_commit']}", "--served-model-name", block["served_name"],
            *block["launch_args"][mode]]


def _endpoint_value(fields, line):
    """The value of a line of an endpoint file: launch_args decoded from its JSON, None for an absent line."""
    value = fields.get(line)
    if line == "launch_args" and value is not None:
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def without_adapters(value):
    """The launch flags of an endpoint file without the adapter flags, ADAPTER_FLAG and each flag of ADAPTER_VALUED with its
    values (every word up to the next flag); the value itself when it is no list."""
    if not isinstance(value, list):
        return value
    kept, skipping = [], False
    for word in value:
        if str(word).startswith("--"):
            skipping = word in ADAPTER_VALUED
            if skipping or word == ADAPTER_FLAG:
                continue
        elif skipping:
            continue
        kept.append(word)
    return kept


def adapter_differences(fields, value):
    """[(field, declared, endpoint)] when the adapter flags of a launch disagree with the lora_adapters line of its endpoint file:
    with ADAPTER_FLAG at least one adapter and --max-loras-per-batch one more than the adapters, without it no adapter flag and
    no adapter; [] when they agree or the launch is no list."""
    if not isinstance(value, list):
        return []
    count = int(fields.get("lora_adapters") or 0)
    if ADAPTER_FLAG not in value:
        stray = [flag for flag in ADAPTER_VALUED if flag in value]
        return [("lora_adapters", "no adapter flag and 0 adapters", f"{count} adapters, flags {stray}")] if stray or count else []
    batch = value[value.index("--max-loras-per-batch") + 1] if "--max-loras-per-batch" in value[:-1] else None
    if count < 1 or batch != str(count + 1):
        return [("lora_adapters", "at least 1 adapter and --max-loras-per-batch one more", f"{count} adapters, --max-loras-per-batch {batch}")]
    return []


def serving_differences(endpoint_record, role, mode=None):
    """[(field, declared, endpoint)] of every field of the serving file of record that the endpoint file of endpoint_record (a record
    of served()) does not equal, [] when it equals all. The launch flags are compared with those of mode, or with mode None with
    those of either mode, the adapter flags set aside (without_adapters) and held to the endpoint's adapters (adapter_differences).
    Raises OpenServingFile as serving_of_record()."""
    declared = serving_of_record()[role]
    fields = (endpoint_record or {}).get("endpoint_file") or {}
    found = []
    for field, line in SERVING_FIELDS.items():
        value = _endpoint_value(fields, line)
        if field == "launch_args":
            modes = [mode] if mode else list(SERVING_MODES)
            if all(public_launch(declared, m) != without_adapters(value) for m in modes):
                found.append((f"launch_args.{'|'.join(modes)}", " ".join(public_launch(declared, modes[0])), value))
            found += adapter_differences(fields, value)
        elif value != declared[field]:
            found.append((field, declared[field], value))
    return found


def serving_shell(role=None):
    """The serving file as bash assignments CD_<ROLE>_<FIELD>, the declared flags of each launch mode as one quoted word."""
    import shlex

    record, lines = serving_of_record(), []
    for name in [role] if role else SERVING_ROLES:
        block = record[name]
        for field in SERVING_FIELDS:
            if field == "launch_args":
                for mode in SERVING_MODES:
                    lines.append(f"CD_{name.upper()}_LAUNCH_ARGS_{mode.upper()}={shlex.quote(' '.join(block['launch_args'][mode]))}")
            else:
                lines.append(f"CD_{name.upper()}_{field.upper()}={shlex.quote(str(block[field]))}")
    return lines


def setting_refusal(role, record, setting):
    """Why the endpoint file of a role, as endpoint_file() read it, differs from the setting that the job hands for the role, or
    None: every value that the setting gives must equal its line of the file (SETTING), and a setting with no file to check it
    against is refused as well. A name that is unset or empty checks nothing."""
    file_variable, keys = SETTING[role]
    given = {key: value for key, value in setting.items() if value}
    if not given:
        return None
    if not record or record.get("lines") is None:
        names = ", ".join(keys[key][0] for key in given)
        return f"the setting of the {role} ({names}) has no endpoint file of {file_variable} to check it against"
    differing = [f"{keys[key][1]} {record.get(keys[key][1])} where {keys[key][0]} names {value}"
                 for key, value in given.items() if record.get(keys[key][1]) != value]
    return f"the endpoint file {record['file']} of the {role} has {', and '.join(differing)}" if differing else None


def platform_serving(role, name=None):
    """The serving identity of the endpoint of a role (agent, judge) in the platform record: serving_identity() of the endpoint
    file that the variable of the role in SETTING names, under name, else the served name of the setting or of the file, with the
    keys of the setting. The served name of the setting is the model of the record, so the record names a judge whose job has
    ended and whose endpoint no longer answers, and hub_id and revision are those of the setting, None without one. Stops by
    reason, before anything is written, when the file differs from the setting (setting_refusal). None for a role with neither a
    file nor a name."""
    file_variable, keys = SETTING[role]
    setting = {key: os.environ.get(variable) or None for key, (variable, _line) in keys.items()}
    record = endpoint_file(os.environ.get(file_variable))
    if (refusal := setting_refusal(role, record, setting)) is not None:
        raise SystemExit(f"no platform record is written: {refusal}")
    name = name or setting["model"] or (record or {}).get("served_name")
    if not name:
        return None
    identity = serving_identity(served(name, (record or {}).get("base_url"), file_variable))
    return {**identity, "model": setting["model"] or identity["model"], "hub_id": setting["hub_id"], "revision": setting["revision"]}
