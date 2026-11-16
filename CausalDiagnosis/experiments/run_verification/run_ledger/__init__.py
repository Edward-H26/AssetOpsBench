"""The run ledger: ledger.json in every run folder, which records where, when and with which command, job, code, model,
judge, data and configuration a run happened, and the SHA-256 of every file the run wrote (CausalDiagnosis/docs/experiments.md).

The driver calls start() before the first episode, add_episode() after every episode and finish() after run.json, so a
run cut short keeps a ledger with the status running and every episode it finished. After the run, a tool that changes a
file of the run folder (a rejudge, a sync, an archive move) calls append_event() with the old and the new hash of each
file, and nothing else rewrites the ledger. The ledger holds names, settings and hashes and never a reading. It keeps the
endpoints, so it stays in the run folder, which git ignores. The environment is copied for the names of ENV_NAMES only, so
the API key and any other secret of the job's environment never reach it.

A run of a kind that a framework declares (RUN_KINDS of utils/frameworks.py, such as the reproduction of AssetOpsBench) writes
the same ledger from its own run.json through the fields its kind gives (run_kind), and its scorer's records enter as events
through record_changes(). write_after() writes the ledger of a finished run of such a kind that has none, after the fact, and
says so in the ledger.
"""

from . import hashing
from . import serving_endpoints
from . import run_data
from . import code_identity
from . import ledger_file
from . import write_after_run
from . import command_line

from .hashing import (
    IGNORED_DIRS, IGNORED_NAMES, canonical_sha256, file_hashes, ignored, models_sha256, prefix_hash, read_manifest,
    sha256_of)
from .serving_endpoints import (
    OpenServingFile, SERVING_FILE, card_differences, card_of, card_record_reason, card_text, endpoint_file,
    endpoint_models, hash_samplers, served, serving_differences, serving_identity, serving_of_record, serving_record)
from .run_data import ROOT, data_files, data_record, installed_sha256, platform_command, platform_of
from .code_identity import (
    UPLOAD_MANIFEST, code_files, code_record, data_reasons, git_lines, git_state, identity_lists, identity_of,
    open_reasons, record_reasons, recorded_code, tree_files, tree_record_reasons, untracked_loaded)
from .ledger_file import (
    ENV_NAMES, INTERRUPTED, LEDGER, LOCAL_EVENTS, RECORDED_DATA, RESUME_TOOL, add_episode, append_event, finish,
    in_progress, kept_episodes, load, now, record_changes, record_resume, resume_refusal, run_fields, run_kind,
    snapshot, start)
from .write_after_run import write_after
from .command_line import main
