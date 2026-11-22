from CausalDiagnosis.agents.Causal_Diagnosis_Agent.controller import GUARD_COMMIT_FIRST
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.plan import OPENING_KINDS


COMMIT_KINDS = ("diagnose", "declare")
FORCED_FINISH = (-1, 0)  # the label of a call after every plan, a finish outside every plan


# ----- where the model calls and the plan steps fall -----
class Walk:
    """A cursor over the controller's tool calls that labels each call with the step that made it, in the order of
    CausalDiagnosis/agents/Causal_Diagnosis_Agent/agent_options.py, CausalDiagnosis/agents/Causal_Diagnosis_Agent/controller.py and CausalDiagnosis/agents/Causal_Diagnosis_Agent/episode_result.py."""

    def __init__(self, calls, tools):
        self.calls, self.tools, self.i, self.labels = calls, tools, 0, [None] * len(calls)

    def peek(self, ahead=0):
        j = self.i + ahead
        return self.calls[j].tool if j < len(self.calls) else None

    def take(self, tool, label):
        if tool is None or self.peek() != tool:
            return None
        self.labels[self.i] = label
        self.i += 1
        return self.calls[self.i - 1]

    def read(self, label, step=None):
        """A reading and, when it served, the ranking the controller asks for right after it, then the verifications that reading
        allows and the ranking a refuted plan asks for, since only a read a plan names verifies a repair."""
        call = self.take(self.tools["read"], label)
        if call is not None and call.result is not None:
            self.take("rank_root_causes", label)
            self.verifications(label)
            if ((step or {}).get("observed") or {}).get("refuted"):
                self.take("rank_root_causes", label)

    def verifications(self, label):
        """verify_outcome for every prediction whose orders have closed and that a reading after the close lets the server verify."""
        while True:
            if self.peek() == "verify_outcome":
                self.take("verify_outcome", label)
            elif (self.peek(), self.peek(1)) == ("rank_root_causes", "verify_outcome"):
                self.take("rank_root_causes", label)  # the belief after the repair, when the ranking was not current
            else:
                return

    def batch(self, steps, plan, first, count):
        """A batch of consecutive read steps of one plan: its count readings one after the other, the read that ended it when
        the environment refused that read after its call, then the one ranking, the verifications those readings allow and
        the ranking a refuted plan asks for, labelled with the last reading made. Returns the number of steps it handled."""
        for k in range(first, first + count):
            self.take(self.tools["read"], (plan, k))
        handled, last = count, (plan, first + count - 1)
        ended = steps[first + count - 1] if first + count <= len(steps) else None
        if ended is not None and ended["kind"] == "read":  # a read step right after the readings of a batch is the one that ended it
            if (ended.get("observed") or {}).get("refused") is not None:
                self.take(self.tools["read"], (plan, first + count))
            handled += 1
        self.take("rank_root_causes", last)
        self.verifications(last)
        if (steps[first + count - 2].get("observed") or {}).get("refuted"):
            self.take("rank_root_causes", last)
        return handled

    def wait(self, label, step=None):
        """The waits event by event, the change summary, the verifications of the closed plans and, when a verdict refuted a
        plan, the ranking that backtracking asks for."""
        if self.take(self.tools["wait"], label) is None:
            return
        while self.take(self.tools["wait"], label) is not None:
            pass
        self.take(self.tools["changes"], label)
        self.verifications(label)
        if ((step or {}).get("observed") or {}).get("refuted"):
            self.take("rank_root_causes", label)

    def finish(self, label, step=None):
        """The verifications a reading allows and the ranking a refuted plan asks for, then finish unless the step stopped on a surprise,
        since the controller never waits or reads for a finish."""
        self.verifications(label)
        if ((step or {}).get("observed") or {}).get("refuted") or (self.peek(), self.peek(1)) == ("rank_root_causes", self.tools["finish"]):
            self.take("rank_root_causes", label)
        if step is None or step.get("status") == "done" or "refused" in (step.get("note") or ""):
            self.take(self.tools["finish"], label)

    def step(self, step, label):
        kind, status, note = step["kind"], step.get("status"), step.get("note") or ""
        if (step.get("observed") or {}).get("stands"):  # a step whose effect already stood made no call of its own
            if kind == "read" or kind in COMMIT_KINDS:
                self.take("rank_root_causes", label)  # the ranking brought up to date where it was stale, as before a commitment
            return
        if kind == "read":
            self.read(label, step)
        elif kind == "failure_modes" and not (step.get("args") or {}).get("component"):
            while self.take(self.tools["failure_modes"], label) is not None:  # one call per component of the profile on both surfaces
                pass
        elif kind in OPENING_KINDS:
            self.take(self.tools.get(kind), label)
        elif kind in COMMIT_KINDS:
            self.take("rank_root_causes", label)  # the guard ranks again when the ranking is not on the current evidence
            if status == "done" or "refused" in note:
                self.take(self.tools[kind], label)
        elif kind == "order" and note != GUARD_COMMIT_FIRST:  # an order without a commitment before it calls nothing
            self.take("estimate_intervention_effect", label)
            if status == "done" or "refused" in note:
                self.take(self.tools["order"], label)
        elif kind == "wait":
            self.wait(label, step)
        elif kind == "finish":
            self.finish(label, step)


def causal_walk(calls, plans, tools, board=None):
    """(label per call, whether the walk placed every call consistently with what plans.json observed), with the tools of the
    surface the episode ran on. Every call belongs to a plan, since the model plans the reads of the opening too.
    With the server every turn opens with a fresh ranking when the belief moved since the last one, the first before any
    reading, labelled step 0 of the plan it was made for, then the executed steps of the plan follow, and a
    finish of the controller outside every plan closes the walk. With the schedule of candidate plans on, the
    batches of the board say which consecutive read steps ran as one batch with one ranking after it (Walk.batch)."""
    walk = Walk(calls, tools)
    board = board or {}
    batches = (board.get("batches") or []) if board.get("parallel") else []
    made = {(b["plan"], b["first_step"]): len(b.get("reads") or []) for b in batches if b.get("reads")}
    ranked_at = {(plan, first + i): (plan, first + count - 1) for (plan, first), count in made.items() for i in range(count)}
    for plan in plans:
        walk.take("rank_root_causes", (plan["index"], 0))
        steps = plan.get("steps") or []
        k = 1
        while k <= len(steps):
            count = made.get((plan["index"], k))
            if count:
                k += walk.batch(steps, plan["index"], k, count)
            else:
                walk.step(steps[k - 1], (plan["index"], k))
                k += 1
    if walk.i < len(calls):
        walk.finish(FORCED_FINISH)
    return walk.labels, walk.i == len(calls) and walk_matches_plans(calls, walk.labels, plans, tools, ranked_at)


def order_id(result):
    """The identifier of the order a call created, from the rig's envelope or from the answer of AssetOpsBench's wo."""
    return (result.get("work_order") or {}).get("id") or result.get("wonum")


def walk_matches_plans(calls, labels, plans, tools, ranked_at=None):
    """Every read step's ranking, the one after its batch for a read of a batch (ranked_at), holds the top mass plans.json
    observed, and every order step's call made the order it observed."""
    ranked_at = ranked_at or {}
    for plan in plans:
        for k, step in enumerate(plan.get("steps") or [], 1):
            mine = [c for c, label in zip(calls, labels) if label == (plan["index"], k)]
            observed = step.get("observed") or {}
            if step["kind"] == "read" and observed.get("top_mass") is not None:
                at = ranked_at.get((plan["index"], k), (plan["index"], k))
                ranks = [c for c, label in zip(calls, labels) if label == at and c.tool == "rank_root_causes" and (c.answer or {}).get("candidates")]
                if not ranks or ranks[0].answer["candidates"][0].get("probability") != observed["top_mass"]:
                    return False
            if step["kind"] == "order" and observed.get("work_order"):
                orders = [c for c in mine if c.tool == tools["order"] and c.result]
                if not orders or order_id(orders[0].result) != observed["work_order"]:
                    return False
    return True
