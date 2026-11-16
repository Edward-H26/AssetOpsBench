"""The record of the serving engine that an endpoint file of CausalDiagnosis/cluster_jobs/2_serve_model/serve_model.sbatch carries: the version, the precision and the
launch arguments of the SGLang server, read from its GET /server_info (SGLang v0.5.19, the handler of that route in
python/sglang/srt/entrypoints/http_server.py, which returns the resolved arguments of ServerArgs in server_args.py) and from
the flags the job started it with.

The answer of /server_info holds the server's own api_key, host, port and model path beside everything else, so it is parsed once
and only the fields of REPORTED leave the parse; nothing else of it is kept, written or put into a message. The launch flags are
written in public form by public_launch_args. The script stands alone (the serving job runs it from the tree of the job), so its
path rule is written out here; tests/test_engine_record.py holds it equal to public_record.public_path.

usage: curl .../server_info | python engine_record.py --model-id ID --revision COMMIT [--fork DIR] -- <launch flags>
Prints the lines of the endpoint file; exits 3 with a message that names the missing field and no
value when the answer lacks one of REPORTED or is no JSON object."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

REPORTED = ("version", "dtype", "quantization", "kv_cache_dtype")  # the allow-list: the only fields read from /server_info
DROPPED = ("--host", "--port", "--api-key", "--admin-api-key")  # flags whose value names the machine or a secret
LIST_FLAGS = ("--lora-paths",)  # flags that take several values, each a name=path pair or a path


class EngineRecordError(Exception):
    """A /server_info answer that the record cannot use. Its message names a field, never a value of the answer."""


def reported(text):
    """{field: value} of REPORTED from the text of a /server_info answer; the rest of the answer is dropped here."""
    try:
        answer = json.loads(text)
    except ValueError:
        raise EngineRecordError("the /server_info answer is not JSON") from None
    if not isinstance(answer, dict):
        raise EngineRecordError("the /server_info answer is not a JSON object")
    missing = [field for field in REPORTED if field not in answer]
    if missing:
        raise EngineRecordError(f"the /server_info answer names no {', '.join(missing)}")
    values = {field: answer[field] for field in REPORTED}
    del answer
    return {field: "none" if value is None else str(value) for field, value in values.items()}


def public_path(value, fork=None):
    """A path in the public form of the run folders (CausalDiagnosis.datasets.dataset_loader.recordings.public_path): a relative path as it is, an absolute path inside the
    fork relative to the fork root, any other absolute path as its last part."""
    if not value.startswith(("/", "~")):
        return value
    absolute = Path(value).expanduser().resolve()
    if fork is not None:
        try:
            return absolute.relative_to(Path(fork).resolve()).as_posix() or "."
        except ValueError:
            pass
    return absolute.name


def _public_value(value, fork):
    if "://" in value:
        return ""
    return public_path(value, fork)


def public_launch_args(argv, model_id, revision, fork=None):
    """The launch flags of a server in public form. --host, --port, --api-key and --admin-api-key go with their values;
    --model-path becomes <model id>@<revision>, a model id that is a folder of the machine by the path rule; each value of --lora-paths becomes its adapter name (name=path) or the last part
    of its path; any other absolute path takes the path rule of public_path, and a URL becomes empty. Both the "--flag value" and
    the "--flag=value" forms are read."""
    out, i, listing = [], 0, False
    model = f"{public_path(model_id, fork)}@{revision}"
    argv = list(argv)
    while i < len(argv):
        arg = argv[i]
        flag, has_inline, inline = arg.partition("=") if arg.startswith("--") else (arg, "", "")
        if arg.startswith("--"):
            listing = flag in LIST_FLAGS
            if flag in DROPPED:
                i += 1 if has_inline else 2
                continue
            if flag == "--model-path":
                out.append(f"--model-path={model}" if has_inline else "--model-path")
                if not has_inline:
                    out.append(model)
                    i += 1
                i += 1
                continue
            out.append(f"{flag}={_public_value(inline, fork)}" if has_inline else arg)
        elif listing:
            name, eq, path = arg.partition("=")
            out.append(name if eq else public_path(arg, fork))
        else:
            out.append(_public_value(arg, fork))
        i += 1
    return out


def launch_dtype(argv):
    """The --dtype the server was started with, "auto" (SGLang's default, the dtype field of ServerArgs in server_args.py) when none was given."""
    for i, arg in enumerate(argv):
        if arg == "--dtype" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--dtype="):
            return arg.split("=", 1)[1]
    return "auto"


def engine_lines(text, argv, model_id, revision, fork=None):
    """The lines of an endpoint file for the engine: its version, the precision it reports beside the dtype it was started
    with (the pair is compared, since a BF16 checkpoint started with "auto" may report "auto"), and the launch flags in public
    form, written as one compact JSON token since no value of an endpoint file holds a space, with its SHA-256."""
    fields = reported(text)
    args = json.dumps(public_launch_args(argv, model_id, revision, fork), separators=(",", ":")).replace(" ", "\\u0020")
    return [f"engine_version {fields['version']}",
            f"dtype {fields['dtype']} dtype_launch {launch_dtype(argv)} quantization {fields['quantization']} kv_cache_dtype {fields['kv_cache_dtype']}",
            f"launch_args {args}", f"launch_args_sha256 {hashlib.sha256(args.encode('utf-8')).hexdigest()}"]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    launch = argv[argv.index("--") + 1:] if "--" in argv else []
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    parser.add_argument("--fork")
    a = parser.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
    try:
        print("\n".join(engine_lines(sys.stdin.read(), launch, a.model_id, a.revision, a.fork)))
    except EngineRecordError as error:
        print(f"ENGINE_RECORD_FAILED {error}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
