"""The judge record of an episode, hidden/judge.json, as the core reads it: its file name, the rule of a reviewed call and
the summary of a record over its planned calls. A judge of a framework writes the record; the tables, the rescoring and the
checks of a run read it through this module alone, so the format is one contract of the core whatever judge wrote it.
"""

JUDGE_NAME = "judge.json"


def reviewed(call):
    """Whether a judge call holds a parsed review, the one rule of summary_of, aggregate.judge_fields and rejudge.unreviewed: a
    call with an error or with an empty review object is a failed call."""
    return call.get("error") is None and bool(call.get("details"))


def summary_of(calls, planned):
    """The pass rate, the mean score and the pass of a judge record over its planned calls: a call without a parsed
    review, a backend error or an unparseable reply, counts as false in the pass rate with the denominator kept, as the agents table of
    the paper counts it (aggregate.judge_fields), the pass is the majority of the planned calls, so a record whose calls all
    failed reads false everywhere, and the mean score is over the parsed reviews alone, None without one."""
    parsed = [c for c in calls if reviewed(c)]
    pass_rate = sum(1 for c in parsed if c["passed"]) / planned
    return {"pass_rate": pass_rate, "score_mean": sum(c["score"] for c in parsed) / len(parsed) if parsed else None, "pass": pass_rate > 0.5}
