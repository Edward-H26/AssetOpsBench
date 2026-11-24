"""Connect the development QA driver to the shared declared-plan controller."""

import json
from dataclasses import asdict

from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.controller import AgentRunner
from .contract import declaration


class PlanModel:
    takes_response_format = True

    def __init__(self, ask, max_calls):
        self.ask, self.max_calls = ask, max_calls

    async def generate_with_usage(self, prompt, temperature=0.0, response_format=None):
        return await self.ask([{"role": "user", "content": prompt}], response_format=response_format)


async def run_general(ask, launcher, task, max_calls, checkpoint):
    runner = AgentRunner(PlanModel(ask, max_calls), {"maintenance_qa": launcher}, declaration())

    def convert(episode):
        common = asdict(episode)
        commitment = episode.board.get("commitment")
        submitted = commitment["args"].get("answer") if commitment and episode.end == "finish" else None
        return {"answer": submitted, "end": "submitted" if episode.end == "finish" else episode.end,
                "error": episode.error, "model_calls": runner.model_history,
                "trajectory": common["trajectory"], "episode": common}

    runner.on_update = lambda episode: checkpoint(convert(episode))
    episode = await runner.run(json.dumps(task, allow_nan=False))
    return convert(episode)
