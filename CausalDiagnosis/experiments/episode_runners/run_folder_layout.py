from CausalDiagnosis.utils.paths import RUNS
import csv
import re
from pathlib import Path
from CausalDiagnosis.utils.paths import ROOT
from CausalDiagnosis.utils import json_files


DEFAULT_RUNS = RUNS
STATE_FILE, RELEASED_FILE, PREDICTIONS_FILE = "episode.json", "released.json", "predictions.json"
HIDDEN = "hidden"  # the subfolder of an episode that no agent facing tool reads: oracle, verdict, score and log
HIDDEN_FILES = ("oracle.json", "verdict.json", "score.json", "log.json")
PUBLIC_FILES = (STATE_FILE, RELEASED_FILE, "trajectory.json", "llm_calls.json", "plans.json", "report.md", "question.txt")
# scores.csv: the run and episode identity, the factors of the ablations, the verifier's scores and the bookkeeping
META_COLUMNS = ("run", "seed", "scenario_id", "date", "config", "agent", "model", "clock_mode", "guidance_level",
                "diagnostics_server", "prices", "verifier_scope", "ranking_model", "block", "entry_offset", "n_faulty", "wall_s", "error")
SCORE_COLUMNS = ("pass", "graded_pass", "graded_pass_n", "graded_top1", "graded_top1_n", "success", "separable", "commitment_on_inseparable",
                 "diagnosis_all_correct", "faulty_set_right", "recovered", "unnecessary", "false_alarms", "cost",
                 "cost_normalized", "downtime_total",
                 "detection_delay", "repair_delay", "turns_to_correct_diagnosis", "abstained", "abstention_cites_reading", "abstention_correct",
                 "prediction_verdict", "failure_class", "end_reason", "time_used", "steps", "provenance_ok", "write_count_ok", "timing_ok", "ordering_ok", "read_gate", "reward",
                 "cost_oracle", "grade", "judge_pass", "static_faulty_set_correct", "static_repairs_match_oracle")
BOOKKEEPING_COLUMNS = ("tool_calls", "refusals", "causal_calls", "llm_calls", "input_tokens", "output_tokens", "plan_steps",
                       "plans", "surprises", "revisions", "fallbacks", "first_plan_mode", "replay_matches", "gepa_call",
                       "react_steps",  # the ReAct counter of the Agent-As-Tool supervisor, empty for every other agent
                       # the harness rule of every agent: who finished the episode, how the runner ended, a cut at the wall limit, the class
                       # of a failure, the replays, and the counts of failed calls, cut replies and refused diagnoses by reason
                       "finish_by", "runner_end", "wall_limit", "error_class", "replays", "failed_calls", "cut_replies", "diagnosis_refusals",
                       # the thinking clock of the episode: credited and uncredited seconds of the starts and the calls, the failed
                       # model attempts and their seconds, the time after the last counted call that nothing charged, how a cut ended,
                       # and 1 where the wall and the monotonic lengths of the agent's part differ by more than a second
                       "credited_s", "uncredited_s", "failed_model_s", "failed_model_attempts", "uncharged_after_last_s", "cut_end",
                       "clock_skew",
                       # 1 when the model server's cache was flushed before an episode of a preset with the wall clock, 0 when the
                       # flush failed, empty on another clock
                       "server_cache_flushed")
COLUMNS = (*META_COLUMNS, *SCORE_COLUMNS, *BOOKKEEPING_COLUMNS)


def episode_reward(verdict, horizon, stream=False):
    """The reward of an episode with the registered numbers (registered.reward_numbers), the one call of verifier.reward in the
    drivers; stream gives the declaration credit of the online loop of GEPA. The verifier keeps the same numbers as defaults for the
    callers outside benchmarks, and a test holds the two equal."""
    from CausalDiagnosis.experiments.table_definitions import registered_analysis as registered
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment import verifier

    numbers = registered.reward_numbers()
    credit = numbers["stream_declaration_credit"] if stream else numbers["declaration_credit"]
    return verifier.reward(verdict, horizon, w_c=numbers["w_c"], w_t=numbers["w_t"], declaration_credit=credit, cost_cap=numbers["cost_cap"])


def read_rows(run_dir):
    """The rows of a run's scores.csv, every cell a string."""
    path = Path(run_dir) / "scores.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# an error cell that names a context overflow or an error of the model endpoint, which every row shows: the
# messages of vLLM and SGLang for a prompt past the context and the exception classes of the OpenAI client
CONTEXT_OR_API_ERROR = re.compile(r"context[ _]?(?:length|window)|maximum context|maximum allowed length|exceeds the maximum|too many tokens"
                                  r"|\bAPI\w*Error\b|\b(?:RateLimit|BadRequest|InternalServer|Authentication|PermissionDenied|NotFound|UnprocessableEntity|Conflict)Error\b",
                                  re.IGNORECASE)


def context_or_api_error(text):
    """Whether an error cell of scores.csv names a context overflow or an error of the model endpoint."""
    return bool(text) and CONTEXT_OR_API_ERROR.search(str(text)) is not None


def input_tokens_per_call(ep_dirs):
    """(mean input tokens per model call, calls) over the llm_calls.json of the episode folders, (None, 0) without a call."""
    tokens = [call.get("input_tokens") for ep_dir in ep_dirs for call in (json_files.read_json(Path(ep_dir) / "llm_calls.json") or [])
              if call.get("input_tokens") is not None]
    return (sum(tokens) / len(tokens) if tokens else None), len(tokens)


def episode_dirs(run_dir):
    """The episode folders of a run, in name order; a folder is an episode when it holds the state file."""
    return sorted(p for p in Path(run_dir).iterdir() if p.is_dir() and (p / STATE_FILE).exists())


def episode_name(seed, scenario_id=None, call=None):
    """The folder of an episode inside its run folder, the one naming rule of the driver, of gepa_prompt_optimization/ and of every
    reader of scores.csv: seed000042, scenario7_seed000042 for an anchored scenario, call00003_seed000042 for a GEPA metric call."""
    name = f"seed{int(seed):06d}"
    if call not in (None, ""):
        return f"call{int(call):05d}_{name}"
    if scenario_id not in (None, ""):
        return f"scenario{scenario_id}_{name}"
    return name


def episode_folder(run_dir, row):
    """The episode folder of a row of scores.csv, named by episode_name from its seed, scenario and GEPA call."""
    return Path(run_dir) / episode_name(row["seed"], row.get("scenario_id"), row.get("gepa_call"))


def refuse_page_output(out, episode=None):
    """Stop before a page for people is written inside the fork, since the page quotes readings of single cycles that stay out
    of it, or inside its episode folder, whose files the run ledger hashes. The one guard of every writer of such a page."""
    out = Path(out).resolve()
    if out.is_relative_to(ROOT.resolve()):
        raise SystemExit(f"{out} lies inside the fork, and the page quotes readings that stay out of it")
    if episode is not None and out.is_relative_to(Path(episode).resolve()):
        raise SystemExit(f"{out} lies inside the episode folder, whose files the run ledger hashes")


def run_dirs(runs_root):
    """The run folders under a root, in name order; a folder is a run when it holds run.json."""
    root = Path(runs_root)
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "run.json").exists())
