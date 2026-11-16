from CausalDiagnosis.experiments.episode_runners import run_episodes
from CausalDiagnosis.agents.Causal_Diagnosis_Agent.prompts import ORIENTATION_HEADING, PLANNING_HEADING, RETRY_HEADING, SUMMARY_HEADING
from CausalDiagnosis.utils import frameworks
from . import formatting, controller_walk


NO_MODEL_SOURCES = ("rule_only",)  # plans written without a model call, by the rule only test tool


def plan_column(ep):
    """The plan step behind every call, with a note when the calls cannot be placed. A Plan-Execute call shows the task
    of its step, and a call of the CausalDiagnosis controller shows its plan, its step, the expectation and the outcome."""
    if ep.kind == "plan_execute":
        return [f"task {formatting.num(c.plan_step)}, {formatting.quoted(c.task, 120)}" if c.task else "" for c in ep.calls], None
    if ep.kind != "causal" or not ep.plans:
        return [""] * len(ep.calls), None
    plans = ep.plans.get("plans") or []
    labels, placed = ep.walk
    if not placed:
        return [""] * len(ep.calls), ("The walk over the controller's plans could not account for every tool call, so the plan column "
                                      "is left out.")
    texts = []
    for n, (call, label) in enumerate(zip(ep.calls, labels)):
        if label == controller_walk.FORCED_FINISH:
            texts.append("controller finish outside every plan")
        elif label[1] == 0:
            texts.append(f"before plan {label[0]}, the fresh ranking the plan was made from")
        else:
            texts.append(f"plan {label[0]} step {label[1]}, {call_role(call, ep.calls[n - 1] if n else None, ep.tools)}")
    silent = []
    for plan in plans:
        indices = [i for i, label in enumerate(labels) if label[0] == plan["index"] and label[1] > 0]
        for k, step in enumerate(plan.get("steps") or [], 1):
            mine = [i for i in indices if labels[i][1] == k]
            if not mine:
                silent.append(f"Plan {plan['index']} step {k} made no tool call, {step_text(plan, k, ep.components)}.")
                continue
            anchor = next((i for i in mine if ep.calls[i].tool == ep.tools.get(step["kind"])), mine[0])
            texts[anchor] = step_text(plan, k, ep.components)
        if plan.get("after_surprise") and indices:
            texts[indices[0]] = "**re-plan** " + texts[indices[0]]
    return texts, " ".join(silent) or None


def call_role(call, previous, tools):
    """What a call does inside a step whose row is carried by another call."""
    if call.tool == "rank_root_causes":
        if previous is not None and previous.tool == tools["read"]:
            return "the ranking after the reading"
        if previous is not None and previous.tool == "verify_outcome":
            return "the ranking after a refuted plan, which backtracking reads"
        return "a fresh ranking before the commitment"
    if call.tool == "verify_outcome":
        return f"the verification of {(call.args or {}).get('prediction_id') if isinstance(call.args, dict) else call.args}"
    return {tools["changes"]: "the change summary after the wait", tools["failure_modes"]: "the failure modes of the next component",
            "estimate_intervention_effect": "the estimate the order cites", tools["wait"]: "a wait for the next event"}.get(
        call.tool, f"{call.tool} within the step")


def step_text(plan, k, components):
    step = plan["steps"][k - 1]
    return f"plan {plan['index']} step {k}, {step_args_text(step)}, {expectation_text(step, plan, components)}, {step_outcome(step)}"


def step_outcome(step):
    status, note = step.get("status"), step.get("note") or ""
    if status == "done":
        return "held" if step.get("expect") else "done"
    if status == "surprise":
        return f"surprise, {formatting.quoted(note, 220)}"
    return f"{status}" + (f", {formatting.quoted(note, 220)}" if note else "")


def step_args_text(step):
    args = step.get("args") or {}
    return {"read": lambda: f"read {args.get('sensor_group')} for {formatting.plural(args.get('cycles'), 'cycle')}",
            "diagnose": lambda: f"diagnose rank {args.get('rank')}",
            "declare": lambda: f"declare with {args.get('sensor_group')} for {formatting.plural(args.get('cycles'), 'cycle')}",
            "order": lambda: f"order the {args.get('component')} citing {args.get('prediction_id')}",
            "wait": lambda: f"wait {formatting.plural(args.get('cycles'), 'cycle')}",
            "finish": lambda: "finish"}.get(step["kind"], lambda: step["kind"])()


def expectation_text(step, plan, components):
    expect, candidates = step.get("expect") or {}, plan.get("candidates") or []

    def ranked(rank):
        if isinstance(rank, int) and 1 <= rank <= len(candidates):
            return f"rank {rank} {formatting.short_levels(candidates[rank - 1]['combination'], components)}"
        return f"rank {formatting.num(rank)}"

    kind = step["kind"]
    if kind == "read" and expect.get("effective_at_most") is not None:
        return f"expects at most {formatting.effective_count(expect['effective_at_most'])} after the reading"
    if kind == "read" and "top_rank" in expect:
        return f"expects {ranked(expect['top_rank'])} on top" + (f" at mass at least {formatting.prob(expect['top_mass_at_least'])}"
                                                                if expect.get("top_mass_at_least") is not None else "")
    if kind == "diagnose" and "top_rank" in expect:
        return f"expects {ranked(expect['top_rank'])} on top"
    if kind == "order" and expect:
        return f"expects prediction {expect.get('prediction_id')}" + (f" with predicted recovery at least {formatting.prob(expect['recovery_at_least'])}"
                                                                     if expect.get("recovery_at_least") is not None else "")
    if kind == "wait":
        wants = [w for w, key in (("no new alarm", "no_new_alarm"),) if expect.get(key, True)]
        return "expects " + formatting.joined(wants) if wants else "no expectation"
    return "no expectation"


def model_call_counts(ep):
    """The model calls made before each tool call returned, or None with a note when they cannot be placed."""
    if not ep.llm_calls:
        return [0] * len(ep.calls), None
    if ep.kind == "plan_execute":
        return plan_execute_counts(ep)
    if ep.kind == "causal" and ep.plans and ep.walk[1]:
        return causal_counts(ep)
    return None, (f"The files do not say where the {formatting.plural(len(ep.llm_calls), 'model call')} of this runner fall between its tool "
                  "calls, so the column shows n/a.")


def argument_markers():
    """The first line of every argument call of the released Plan-Execute executor, from the entry of each declared check without
    a model (the scripted wiring check plays that executor)."""
    return tuple(frameworks.code("AGENTS", name).argument_marker for name in run_episodes.SCRIPTED_AGENTS)


def plan_execute_counts(ep):
    """The planning calls come first, then one argument call per tool step in the order of the steps."""
    markers = argument_markers()
    arguments = [i for i, c in enumerate(ep.llm_calls) if (c.get("prompt") or "").startswith(markers)]
    first = arguments[0] if arguments else len(ep.llm_calls)
    if len(arguments) != len(ep.calls) or arguments != list(range(first, first + len(arguments))):
        return None, (f"The {formatting.plural(len(arguments), 'argument call')} of `llm_calls.json` do not pair with the "
                      f"{formatting.plural(len(ep.calls), 'tool call')} one to one, so the column shows n/a.")
    return [first + n for n in range(1, len(ep.calls) + 1)], None


def prompt_kind(prompt):
    if ORIENTATION_HEADING in prompt:
        return "orientation"
    if SUMMARY_HEADING in prompt:
        return "summary"
    if PLANNING_HEADING in prompt:
        return "retry" if RETRY_HEADING in prompt else "plan"
    return None


def causal_counts(ep):
    """The orientation falls after the plan that read the profile, before the first call of the next turn, every plan's calls (a retry
    or a repair among them) before the plan's first step call, which follows the fresh ranking of its turn, or before the first call
    of a later turn when it made none, and every closing summary before a finish call."""
    unplaced = None, "The model calls of `llm_calls.json` do not match the controller's plans, so the column shows n/a."
    plans = ep.plans.get("plans") or []
    labels, _ = ep.walk
    kinds = [prompt_kind(c.get("prompt") or "") for c in ep.llm_calls]
    groups = []
    for kind in kinds:
        if kind == "plan":
            groups.append(1)
        elif kind == "retry" and groups:
            groups[-1] += 1
        elif kind not in ("orientation", "summary"):
            return unplaced
    modelled = [p for p in plans if p.get("source") not in NO_MODEL_SOURCES]
    if len(groups) != len(modelled):
        return unplaced
    n = len(ep.calls)
    profiled = next((p["index"] for p in plans if any(s.get("kind") == "profile" and s.get("status") == "done" for s in p.get("steps") or [])), None)
    oriented = next((i for i, label in enumerate(labels) if profiled is not None and label[0] > profiled), n)
    starts = plan_starts(labels, plans)
    finishes = [i for i, c in enumerate(ep.calls) if c.tool == "finish"]
    summaries = kinds.count("summary")
    placed = ([oriented] * kinds.count("orientation")
              + [starts[p["index"]] for p, size in zip(modelled, groups) for _ in range(size)]
              + (finishes[:summaries] + [n] * max(0, summaries - len(finishes))))
    return [sum(1 for at in placed if at <= i) for i in range(n)], None


def plan_starts(labels, plans):
    """The index of every plan's first step call, which follows the fresh ranking of its turn, or of the first call of a later
    turn when the plan made none, or the number of calls when no call followed."""
    starts, following = {}, next((i for i, label in enumerate(labels) if label == controller_walk.FORCED_FINISH), len(labels))
    for plan in reversed(plans):
        own = [i for i, label in enumerate(labels) if label[0] == plan["index"]]
        starts[plan["index"]] = next((i for i in own if labels[i][1] > 0), following)
        following = own[0] if own else following
    return starts
