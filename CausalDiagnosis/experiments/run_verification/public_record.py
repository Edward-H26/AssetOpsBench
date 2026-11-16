"""What of a run folder goes public, and in which form: the one place of the rule (CausalDiagnosis/docs/experiments.md, the branch of
the runs).

The run folders go public as written, since the ledger hashes their files and a copy changed afterwards fails the ledger check. So
every program that writes a run folder writes it in its public form. A value falls in one class: a path inside the tree that played
the run is written relative to the fork root, a path outside it as its basename (CausalDiagnosis.datasets.dataset_loader.recordings.public_path); an endpoint
URL, a store URL, a host, a Slurm job id, the uuid of a card, a user name, a password, a key or a token is not written. What a
reader on the cluster or on a laptop still needs of those, the job and host of every start, the raw command, the endpoint files as
read and the data paths, goes into the private file PRIVATE_FILE of the run folder, which no ledger hashes and which lies outside
the published set (published_files). publish_check.py checks exactly that set."""
import json
import re
from pathlib import Path

from CausalDiagnosis.datasets.dataset_loader.recordings import public_path, resolve_recorded_path  # noqa: F401  the path rule of src and of the drivers

PRIVATE_FILE = "job.json"
# the hosts whose URLs a public file may name: the sources of the data, of the models and of the code
PUBLIC_HOSTS = ("huggingface.co", "archive.ics.uci.edu", "github.com", "raw.githubusercontent.com")
URL = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)
# keys that are never written into a public record, wherever they stand: the host and the job of a process, the uuid of a card
# and credentials; what a reader needs of the host and the job lives in PRIVATE_FILE
DROPPED_KEYS = frozenset({"host", "node", "job", "slurm_job_id", "slurm_array_job_id", "slurm_array_task_id", "slurm_restart_count",
                          "gpu_uuid", "password", "username", "api_key", "token"})
# keys that hold a URL of a served model or of a store, written as null so that a reader keeps the key
ENDPOINT_KEYS = frozenset({"endpoint", "judge_endpoint", "base_url", "reflection_endpoint", "judge_base_url", "url"})
# the keys of an endpoint file as the ledger read it that name its place on the cluster: the file, its raw lines, the sampler
ENDPOINT_FILE_PLACES = ("file", "lines", "gpu_sampler")
# what a public record keeps of an entry of an endpoint's model list; SGLang's entry also names the weights folder
MODEL_ENTRY_KEYS = ("id", "max_model_len")


def public_url(value):
    """A URL as a public file may hold it: one of PUBLIC_HOSTS kept, any other not written."""
    host = URL.sub("", value).split("/", 1)[0].split(":", 1)[0].split("@")[-1].lower()
    return value if host in PUBLIC_HOSTS or any(host.endswith("." + h) for h in PUBLIC_HOSTS) else None


def public_value(value):
    """A record in its public form, recursively: DROPPED_KEYS gone, ENDPOINT_KEYS null, a URL by public_url, an absolute path by
    public_path, the endpoint files and model lists of the ledger and run.json reduced (ENDPOINT_FILE_PLACES, MODEL_ENTRY_KEYS).
    Applying it twice gives the same record."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key in DROPPED_KEYS:
                continue
            if key in ENDPOINT_KEYS and not isinstance(item, (dict, list)):
                out[key] = None
            elif key == "endpoint_file" and isinstance(item, dict):
                out[key] = public_value({k: v for k, v in item.items() if k not in ENDPOINT_FILE_PLACES})
            elif key == "endpoint_file" and isinstance(item, str):
                out[key] = None
            elif key == "sampler" and isinstance(item, dict):
                out[key] = {k: v for k, v in item.items() if k != "path"}
            elif key == "endpoint_models" and isinstance(item, list):
                out[key] = [{k: m[k] for k in MODEL_ENTRY_KEYS if k in m} if isinstance(m, dict) else m for m in item]
            else:
                out[key] = public_value(item)
        return out
    if isinstance(value, list):
        return [public_value(item) for item in value]
    if isinstance(value, str):
        if URL.match(value):
            return public_url(value)
        if value.startswith(("/", "~")):
            return public_path(value)
    return value


def public_argv(argv):
    """A command line as a public record holds it: an absolute path by public_path, a URL by public_url (an empty string in its
    place, so that the options still parse), every other argument as it is; --flag=value forms alike."""
    out = []
    for arg in (str(a) for a in argv):
        flag, equals, rest = arg.partition("=") if arg.startswith("--") and "=" in arg else ("", "", arg)
        value = rest
        if URL.match(value):
            value = public_url(value) or ""
        elif value.startswith(("/", "~")):
            value = public_path(value)
        out.append(f"{flag}{equals}{value}" if equals else value)
    return out


def read_private(run_dir):
    """The private record of a run folder, {} without one."""
    path = Path(run_dir) / PRIVATE_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def add_private(run_dir, key, entry):
    """Appends entry to the list key of the private record of a run folder (starts, events) and writes it with mode 600."""
    record = read_private(run_dir)
    record.setdefault(key, []).append(entry)
    path = Path(run_dir) / PRIVATE_FILE
    path.write_text(json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8")
    path.chmod(0o600)
    return record


def published_files(folder):
    """The published set of a run folder or of a results folder: every file below it that the ledger records (ledger.file_hashes
    leaves out what ledger.ignored names, PRIVATE_FILE among them) with the ledger and the local events file, as paths relative to
    the folder in path order."""
    from CausalDiagnosis.experiments.run_verification import run_ledger as ledger  # here: ledger imports this module

    folder = Path(folder)
    return sorted(set(ledger.file_hashes(folder)) | {name for name in (ledger.LEDGER, ledger.LOCAL_EVENTS) if (folder / name).is_file()})
