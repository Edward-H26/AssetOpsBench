"""The rubric scorer of AssetOpsBench (src/evaluation/scorers/llm_judge.py) on the whole trajectory.

The released scorer cuts the trajectory at 8,000 characters. That cut keeps the head of the prompt and hides the later
steps of a long trajectory from the judge, on the rig the diagnosis, the work orders and the repairs, so the judge of the
benchmark reads the trajectory uncut: FullTrajectoryJudgeScorer is the released scorer with that one difference, and
install() registers it under the released name, so the released Evaluator picks it. The prompt, the rubric, the parse of
the review and the score are the released ones, read from the released module.
"""

from evaluation.models import Scenario, ScorerResult
from evaluation.scorers import llm_judge, register
from llm import LLMBackend

__all__ = ["FullTrajectoryJudgeScorer", "install"]


class FullTrajectoryJudgeScorer(llm_judge.LLMJudgeScorer):
    """The released LLM-As-Judge scorer with the trajectory passed to the prompt uncut."""

    def __call__(self, scenario: Scenario, answer: str, trajectory_text: str) -> ScorerResult:
        characteristic = scenario.characteristic_form or scenario.expected_answer or ""
        if not characteristic:
            return ScorerResult(
                scorer=self.name,
                passed=False,
                rationale="scenario has neither characteristic_form nor expected_answer",
            )

        prompt = llm_judge._PROMPT_TEMPLATE.format(
            question=scenario.text,
            characteristic=characteristic,
            answer=answer,
            trajectory=trajectory_text,
        )

        try:
            raw = self._llm.generate(prompt)
        except Exception as exc:  # judge call failure is a scoring failure, not a crash
            llm_judge._log.exception("llm_judge: backend error")
            return ScorerResult(
                scorer=self.name,
                passed=False,
                rationale=f"judge backend error: {exc}",
            )

        review = llm_judge._parse_review(raw)
        if review is None:
            return ScorerResult(
                scorer=self.name,
                passed=False,
                rationale="judge returned unparseable JSON",
                details={"raw": raw[:2000]},
            )

        passed = (
            review.get("task_completion") is True
            and review.get("data_retrieval_accuracy") is True
            and review.get("generalized_result_verification") is True
            and review.get("agent_sequence_correct") is True
            and review.get("clarity_and_justification") is True
            and review.get("hallucinations") is False
        )
        score = sum(1 for k in llm_judge._RUBRIC_KEYS[:5] if review.get(k) is True) / 5.0
        if review.get("hallucinations") is True:
            score = max(0.0, score - 0.2)

        rationale = str(
            review.get("suggestions") or review.get("reason") or ""
        )[:500]
        return ScorerResult(
            scorer=self.name,
            passed=passed,
            score=round(score, 3),
            rationale=rationale,
            details=review,
        )


def install(llm: LLMBackend, name: str = "llm_judge") -> None:
    """Register the full trajectory scorer bound to ``llm`` under ``name``, the released scorer's name by default."""
    register(name, FullTrajectoryJudgeScorer(llm, name=name))
