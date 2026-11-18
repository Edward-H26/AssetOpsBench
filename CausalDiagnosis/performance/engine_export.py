"""The engine's record of each generation request, from the export that SGLang writes when its launch carries the flags of
CausalDiagnosis/performance/README.md, joined to the model calls of llm_calls.json, which splits the seconds of the model calls
into the queue, the prefill and the decode of the engine, the rest of the calls outside those three stages, and the calls that
the join cannot read.

The engine writes one JSON line per generation request into the files sglang-request-metrics-<date and hour>.log of the folder
that its launch names. Every stamp of a line is in epoch seconds of the serving node's wall clock, the engine's performance counter
moved by the difference that it calibrates, and queue_time is a duration in seconds, so a stamp of the engine compares with the
stamps of the client on the same node. A line also holds the request itself under request_parameters, with its prompt; the
reader keeps only the keys of KEPT, so no prompt reaches an output.

No response id ties a line to a call, so the join is by time: every line received while a call is open belongs to that call,
since the client's retry policy attempts a failed request again inside one recorded call. The stage times of the attempts add,
the call counts its lines beyond the first, its input tokens are checked against every line and its output tokens against the
line received last, the attempt whose answer the client kept. A line received while no call is open belongs to no call of the
client, as the engine's warm-up at its launch, the opening call of a job and a probe of the harness at a wall cut do not, and it
is passed over and counted; a line received while two calls are open cannot be told to one of them and is counted apart, taken
by neither. A call without a line, or without the two stamps of its span, stays unjoined. A token count of the client leaves no
line, since the engine writes the export only from the output of a generation. The time ties a line to its call only when one
episode at a time reaches the server, so the calls of a job at more runs in flight than one are not joined.

Of a line the queue is queue_time, the prefill runs from forward_entry_time to prefill_finished_time and the decode from
prefill_finished_time to request_finished_ts; its decode tokens are completion_tokens less the first, which the prefill gives, so
that the rate of the decode holds no prefill. cached_tokens are the prompt tokens that the engine found in its cache. The seconds
of the generation calls of an episode split into parts that add up to them: of a call whose lines agree with it in their tokens,
the queue, the prefill and the decode of its lines and the rest of the call outside those three stages; of a call without a line,
its seconds as the part of the unjoined calls; of a call whose lines differ from it in their tokens, its seconds as the part of
the differing calls. Each part also stands as its share of those seconds, so that the shares add up to one.

The rest outside the three stages holds the client, the transport and the waits between attempts, and also two spans of the
engine's own time that no stage holds: from the receipt of a request, request_received_ts, to its entry into the queue,
forward_entry_time less queue_time, and from the end of its decode, request_finished_ts, to the sending of its answer,
response_sent_to_client_ts. It goes below zero when the engine ran two attempts of the call at once, and by up to 0.005 s when
the stages fill nearly all of the call, since the client rounds the seconds of a call to 0.01 s."""
import bisect
import json
from pathlib import Path

from .stage_profile import TOKEN_COUNT

FILE_PREFIX = "sglang-request-metrics-"  # the engine names each file of the export by the hour in which it writes it
KEPT = ("request_received_ts", "queue_time", "forward_entry_time", "prefill_finished_time", "request_finished_ts", "prompt_tokens",
        "completion_tokens", "cached_tokens")
STAGES = ("queue_s", "prefill_s", "decode_s")
PARTS = (*STAGES, "outside_stages_s", "unjoined_calls_s", "differing_calls_s")
CALL_SHARES = tuple(f"call_share_{name[:-2]}" for name in PARTS)
COUNTS = ("engine_calls_joined", "engine_calls_unjoined", "engine_input_mismatches", "engine_output_mismatches", "engine_extra_lines")
COLUMNS = (*COUNTS, *PARTS, *CALL_SHARES, "cached_prompt_tokens", "decode_tokens", "decode_tokens_per_s", "engine_note")
CALL_COLUMNS = ("run", "episode", "call", "started_at", "call_s", "engine_lines", *STAGES, "outside_stages_s", "cached_prompt_tokens",
                "decode_tokens", "decode_tokens_per_s", "input_match", "output_match")


def export_files(paths):
    """The files of an export: a file as it is given, and of a folder every file in it that the engine names."""
    return [file for path in map(Path, paths) for file in (sorted(path.glob(f"{FILE_PREFIX}*.log")) if path.is_dir() else [path])]


def read_lines(paths):
    """(lines, unread): the lines of the export files in the order of their receipt, each with the keys of KEPT alone, and the
    count of the lines that are no JSON object or lack a number under a key of KEPT."""
    lines, unread = [], 0
    for path in paths:
        for text in Path(path).read_text(encoding="utf-8").splitlines():
            if not text.strip():
                continue
            try:
                record = json.loads(text)
            except ValueError:
                record = None
            if not isinstance(record, dict) or not all(isinstance(record.get(key), (int, float)) for key in KEPT):
                unread += 1
                continue
            lines.append({key: record[key] for key in KEPT})
    return sorted(lines, key=lambda line: line["request_received_ts"]), unread


def generation_calls(calls):
    """The calls of llm_calls.json that reach the generation of the engine, a token count left out."""
    return [call for call in calls if call.get("kind") != TOKEN_COUNT]


def join(calls, lines):
    """(matched, passed, ambiguous): per call, in the order of calls, the lines received while it was open, in the order of their
    receipt; the count of the lines received while no call was open; and the count of those received while two calls were open."""
    timed = sorted((i for i, call in enumerate(calls) if call.get("started_at") is not None and call.get("ended_at") is not None),
                   key=lambda i: float(calls[i]["started_at"]))
    starts = [float(calls[i]["started_at"]) for i in timed]
    matched, passed, ambiguous = [[] for _ in calls], 0, 0
    for line in lines:
        received = line["request_received_ts"]
        before = timed[:bisect.bisect_right(starts, received)]  # the calls started by the receipt
        open_calls = [i for i in before if float(calls[i]["ended_at"]) >= received]
        if len(open_calls) == 1:
            matched[open_calls[0]].append(line)
        elif open_calls:
            ambiguous += 1
        else:
            passed += 1
    return matched, passed, ambiguous


def call_figures(call, lines):
    """The engine's figures of one call from the lines of its attempts: the count of its lines, the sums of their stages, of their
    cached and their decode tokens, the rest of the call's seconds outside those stages, the decode rate, and whether the call agrees
    on its input tokens with every line and on its output tokens with the line received last."""
    stages = {"queue_s": sum(float(line["queue_time"]) for line in lines),
              "prefill_s": sum(line["prefill_finished_time"] - line["forward_entry_time"] for line in lines),
              "decode_s": sum(line["request_finished_ts"] - line["prefill_finished_time"] for line in lines)}
    seconds = None if call.get("seconds") is None else float(call["seconds"])
    decode_tokens = sum(max(0, int(line["completion_tokens"]) - 1) for line in lines)
    return {"call_s": seconds, "engine_lines": len(lines), **stages,
            "outside_stages_s": None if seconds is None else seconds - sum(stages.values()),
            "cached_prompt_tokens": sum(int(line["cached_tokens"]) for line in lines), "decode_tokens": decode_tokens,
            "decode_tokens_per_s": decode_tokens / stages["decode_s"] if stages["decode_s"] > 0 else None,
            "input_match": int(all(call.get("input_tokens") == line["prompt_tokens"] for line in lines)),
            "output_match": int(call.get("output_tokens") == lines[-1]["completion_tokens"])}


def agrees(figure):
    return figure is not None and bool(figure["input_match"] and figure["output_match"])


def episode_columns(entries):
    """The engine columns of one episode from its generation calls, each beside its figures or None for a call without a line:
    the counts of the join, the parts of the seconds of the calls that report them with their shares of those seconds, the cached
    and the decode tokens and the decode rate of the agreeing calls, and the note of what the join could not read."""
    figures = [figure for _call, figure in entries if figure is not None]
    timed = [(float(call["seconds"]), figure) for call, figure in entries if call.get("seconds") is not None]
    good = [figure for _seconds, figure in timed if agrees(figure)]
    parts = {**{name: sum((figure[name] for figure in good), 0.0) for name in (*STAGES, "outside_stages_s")},
             "unjoined_calls_s": sum((seconds for seconds, figure in timed if figure is None), 0.0),
             "differing_calls_s": sum((seconds for seconds, figure in timed if figure is not None and not agrees(figure)), 0.0)}
    total = sum(seconds for seconds, _figure in timed)
    counts = {"engine_calls_joined": len(figures), "engine_calls_unjoined": len(entries) - len(figures),
              "engine_input_mismatches": sum(1 for figure in figures if not figure["input_match"]),
              "engine_output_mismatches": sum(1 for figure in figures if not figure["output_match"]),
              "engine_extra_lines": sum(figure["engine_lines"] - 1 for figure in figures)}
    decode = (sum(figure["decode_tokens"] for figure in good), parts["decode_s"])
    notes = [f"{count} {text}" for count, text in ((counts["engine_calls_unjoined"], "model calls have no line of the engine"),
                                                   (len(figures) - sum(1 for figure in figures if agrees(figure)), "joined calls differ in their token counts"),
                                                   (counts["engine_extra_lines"], "lines of retried attempts are added to their calls")) if count]
    return {**counts, **parts, **{share: parts[name] / total if total else None for name, share in zip(PARTS, CALL_SHARES)},
            "cached_prompt_tokens": sum(figure["cached_prompt_tokens"] for figure in good), "decode_tokens": decode[0],
            "decode_tokens_per_s": decode[0] / decode[1] if decode[1] > 0 else None, "engine_note": "; ".join(notes)}


def job_join(episode_calls, paths):
    """(entries, calls, summary) of the export of one job at one run in flight: episode_calls maps (run, episode) to the calls of
    llm_calls.json of every episode of the job's model runs, left-out and cut episodes among them, since each of their calls left
    lines; entries maps each to its generation calls beside their figures, None for a call without a line; calls holds the figures
    of every joined call with its place among the generation calls of its episode, a token count left out; and summary counts the
    files, the lines read, unread, passed over and taken by no call for two open calls, the calls, the joined calls and the lines
    of their retried attempts."""
    files = export_files(paths)
    lines, unread = read_lines(files)
    keyed = [(key, index, call) for key, calls in episode_calls.items() for index, call in enumerate(generation_calls(calls))]
    matched, passed, ambiguous = join([call for _key, _index, call in keyed], lines)
    entries, rows = {key: [] for key in episode_calls}, []
    for (key, index, call), found in zip(keyed, matched):
        figure = call_figures(call, found) if found else None
        entries[key].append((call, figure))
        if figure is not None:
            rows.append({"run": key[0], "episode": key[1], "call": index, "started_at": call.get("started_at"), **figure})
    return entries, rows, {"export_files": len(files), "lines": len(lines), "unread_lines": unread, "passed_lines": passed,
                           "ambiguous_lines": ambiguous, "calls": len(keyed), "joined_calls": len(rows),
                           "extra_lines": sum(row["engine_lines"] - 1 for row in rows)}
