"""The judge record format of the core (evaluation/metrics/judge_records.py) is the one the judge of AssetOpsBench writes and
reads: one home for the file name, the rule of a reviewed call and the summary."""
from CausalDiagnosis.evaluation.metrics import judge_records
from CausalDiagnosis.frameworks.AssetOpsBench.LLM_Judges import episode_judge


def test_the_judge_reads_the_record_format_of_the_core():
    assert (episode_judge.JUDGE_NAME, episode_judge.reviewed, episode_judge.summary_of) == (
        judge_records.JUDGE_NAME, judge_records.reviewed, judge_records.summary_of)


def test_a_failed_call_counts_false_with_the_denominator_kept():
    calls = [{"passed": True, "score": 1.0, "details": {"a": 1}, "error": None}, {"passed": True, "score": 0.5, "details": {}, "error": None},
             {"passed": False, "score": 0.0, "details": None, "error": "judge backend error"}]
    assert judge_records.summary_of(calls, 5) == {"pass_rate": 0.2, "score_mean": 1.0, "pass": False}
