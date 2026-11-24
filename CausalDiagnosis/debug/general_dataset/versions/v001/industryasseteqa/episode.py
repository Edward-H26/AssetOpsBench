"""A read-only evidence episode with one terminal answer submission."""

import copy

from .evaluation import check_answer


class QAEpisode:
    """Restrict observations and citations to the question's released fact."""

    def __init__(self, dataset, qa_id):
        self.task = dataset.task(qa_id)
        self._fact = dataset.public_fact(qa_id)
        self.observed = False
        self.over = False
        self.answer = None
        self.calls = []

    def read_fact(self):
        """Read measurements and source metadata for this question."""
        if self.over:
            raise ValueError("the episode has ended")
        self.observed = True
        self.calls.append({"tool": "read_fact", "args": {}})
        return copy.deepcopy(self._fact)

    def submit_answer(self, answer):
        """Commit a well-formed, grounded answer and end the episode."""
        if self.over:
            raise ValueError("the episode has ended")
        if not self.observed:
            raise ValueError("read_fact must be answered before submission")
        report = check_answer(answer, self._fact)
        self.calls.append({"tool": "submit_answer", "args": {"answer": copy.deepcopy(answer)}, "checks": report})
        if report["structure_ok"] and report["provenance_ok"]:
            self.answer = copy.deepcopy(answer)
            self.over = True
        return {"accepted": self.over, "checks": report}
