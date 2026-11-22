"""The ranking module, of the advice kind ranking (CausalDiagnosis/docs/general-agent-plan.md, section Advice modules). A stub of
the freeze; the module builds it."""


class RankingModule:
    kinds = ("ranking",)
    switches = ("branches", "single_hypothesis", "reads")

    def __init__(self, tools, table, options):
        raise NotImplementedError

    def blocks(self):
        raise NotImplementedError

    def forms(self, row):
        raise NotImplementedError

    def step_problems(self, step, index, state):
        raise NotImplementedError

    def candidates(self):
        raise NotImplementedError

    def sections(self, board):
        raise NotImplementedError

    def known_ids(self):
        raise NotImplementedError

    def evidence_for(self, step, row):
        raise NotImplementedError

    async def before_turn(self, ctx):
        raise NotImplementedError

    async def before_step(self, step, row, ctx):
        raise NotImplementedError

    async def after_batch(self, steps, answers, ctx):
        raise NotImplementedError

    async def on_pending(self, pending, observation, ctx):
        raise NotImplementedError

    def to_json(self):
        raise NotImplementedError
