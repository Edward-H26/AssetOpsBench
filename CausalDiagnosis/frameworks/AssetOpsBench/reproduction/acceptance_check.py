"""The acceptance check of the two paper-era ports on AssetOpsBench's own scenarios: the Agent-As-Tool port
(agent_as_tool_paper) and the Plan-Execute port (plan_execute_paper).

usage, from the fork root with the fork's virtual environment first on PATH, after score_runs.py judged the runs:
  python -m CausalDiagnosis.frameworks.AssetOpsBench.reproduction.acceptance_check RUN_DIR [RUN_DIR ...] [--port agent_as_tool_paper|plan_execute_paper] [--out DIR]

A port is checked against the paper's values of its own design on runs that played it on the paper's scenarios,
granite-3-3-8b-instruct on SGLang with the paper's decoding. The rule has three parts:

- Decidable. The runs decide only when all four hold. Every run folder given is finished: run.json has its end and as many
  episodes done as planned. Every run that holds an episode of the port is a run of a registered set, runnable or
  recovered, with one run per scenario, the paper's own count. The episodes of the port hold every scenario of those runs
  exactly once, the 120 runnable ones among them. Every episode of the port has a step count: an episode without one never
  built its agent, so it is no observation of the port. For Agent-As-Tool every episode holds the judge verdict of the
  paper protocol, since score_runs.py judged the run. The pilot and a list of ids decide nothing.
- Steps (Table 3), both ports. For the single-agent and the multi-agent tasks apart, the 95 percent interval of the mean steps, from
  the bootstrap over scenarios of reproduction_tables.py, lies inside the paper's value plus or minus STEP_TOLERANCE: 5.3 and 6.6 for
  Agent-As-Tool, whose steps are the supervisor's ReAct counter, and 5.2 and 5.1 for Plan-Execute, whose steps are the
  executed plan steps, the step in which a run ends included and the summarization step not. This is an equivalence: a
  wide interval misses, where an overlap would let it pass. steps_meet_paper is the one place where this rule is written.
- Task completion (Figure 5), Agent-As-Tool only. The bracket of task completion over the paper's 141 scenarios, under the paper-era
  rule of that design, the majority of five judge calls at temperature 0.3, with the scenarios not run counted first as
  failed and then as accomplished, holds the paper's 35 percent. For Plan-Execute no judge value decides: the paper prints
  its granite values as counts on a denominator it does not state, so the report prints the port's rates beside them, under
  the paper-era rule of that design, one judge call at temperature 0.

The verdict is PASS when the runs decide and every check that decides passes (exit code 0), MISS when the runs decide and
one fails (1), and NOT DECIDABLE otherwise (3), with every condition that failed named. NOT DECIDABLE is no outcome of the
check: its cause lies outside the port, so the job is run once more from the same commit, with no file of a port changed.
A second NOT DECIDABLE in a row starts no third job: the conditions it names are read and their cause is fixed outside the
ports first.
Printed beside these with no rule: per task kind the half-width of the interval and the mean minus the paper's value, data
retrieval accuracy and result verification, the episodes that an error ended, and for Plan-Execute the plan length beside
the paper's planned steps, the reviewer's rounds and the history entries per episode.

A PASS says that the port met the step counts of the paper, and for Agent-As-Tool its task completion, under a rule fixed
before the run. It does not say that a port reproduces the paper's agent: the tools are the MCP servers of main-0.x and
the agents of a port are rebuilt from the public branches.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from CausalDiagnosis.frameworks.AssetOpsBench.reproduction.scenario_sets import ACCEPTANCE_REPEATS, ACCEPTANCE_SETS, SETS
from CausalDiagnosis.frameworks.AssetOpsBench.reproduction.run_folder_names import PE_PORT, PORT

from CausalDiagnosis.frameworks.AssetOpsBench.reproduction import reproduction_tables as tables
from CausalDiagnosis.frameworks.AssetOpsBench.agents.Plan_Execute_Paper.workflow import PLANS  # noqa: E402  the plans the port's planner tries before the last one runs

# The tolerance of the steps check, in steps: the interval of a port's mean steps must lie within this distance of the
# paper's value. Its ground: the paper prints one mean step count per design and task kind, to one decimal and without a
# spread, so no width can be read off it. The width is a choice of the rule, a design parameter, and steps_meet_paper is
# the one function that reads it.
STEP_TOLERANCE = 1.0
STEPS_RULE = (f"the 95 percent interval of the port's mean steps lies within {STEP_TOLERANCE:g} step{'' if STEP_TOLERANCE == 1 else 's'} "
              "of the paper's value on either side, an equivalence, so a wide interval misses where an overlap would let it pass")
KINDS = ("single", "multi")
NOT_DECIDABLE = 3  # the exit code of runs that cannot decide, beside 0 for a pass and 1 for a miss
VERDICTS = {True: "PASS", False: "MISS", None: "NOT DECIDABLE"}
# per port: the design whose paper values apply, its printed name, and whether the task completion check decides beside the steps
RULES = {PORT: {"paradigm": "agent_as_tool", "name": "Agent-As-Tool", "judge_decides": True},
         PE_PORT: {"paradigm": "plan_execute", "name": "Plan-Execute", "judge_decides": False}}
JUDGED = tuple(field for field, _ in tables.JUDGED)
FLOW_FILE = "plan_execute_trajectory.json"
NAMED = 12  # the most ids or episodes a reason lists before it gives the count of the rest


def paper_steps(port):
    return {kind: float(tables.TABLE3[(RULES[port]["paradigm"], kind)][0]) for kind in KINDS}


def paper_judge(port):
    return dict(zip(JUDGED, tables.FIGURE5[RULES[port]["paradigm"]]))


def listed(items):
    items = [str(item) for item in items]
    return ", ".join(items[:NAMED]) + (f" and {len(items) - NAMED} more" if len(items) > NAMED else "")


def undecidable(rows, metas, port):
    """The conditions of a decidable set that the runs fail, one sentence each, and none when the runs can decide. Read
    from run.json of every run (its end, its episodes done and planned, its set, its repeats and its scenario ids) and from
    the rows of the port (run, scenario, steps)."""
    rows = [r for r in rows if r["agent"] == port]
    reasons = []
    for meta in metas:
        if not meta.get("ended") or meta.get("episodes_done") != meta.get("episodes_planned"):
            reasons.append(f"run {meta['run']} is not finished: {meta.get('episodes_done')} of {meta.get('episodes_planned')} episodes done"
                           + ("" if meta.get("ended") else ", and run.json has no end"))
    if not rows:
        return reasons + [f"the runs hold no episode of {port}"]
    played = [meta for meta in metas if meta["run"] in {r.get("run") for r in rows}]
    for meta in played:
        if meta.get("set") not in ACCEPTANCE_SETS or int(meta.get("repeats") or 0) != ACCEPTANCE_REPEATS:
            reasons.append(f"run {meta['run']} is no run of a registered set: set {meta.get('set')}, {meta.get('repeats')} run(s) per scenario, "
                           f"where the rule asks for {' or '.join(ACCEPTANCE_SETS)} with {ACCEPTANCE_REPEATS}")
    held = Counter(int(r["scenario_id"]) for r in rows)
    planned = {int(i) for meta in played for i in meta.get("scenario_ids") or ()} | set(SETS["runnable"])
    missing, repeated = sorted(planned - set(held)), sorted(i for i, n in held.items() if n > 1)
    if missing:
        reasons.append(f"{len(missing)} scenario(s) of the registered set have no episode of the port: {listed(missing)}")
    if repeated:
        reasons.append(f"{len(repeated)} scenario(s) have more than one episode of the port: {listed(repeated)}")
    uncounted = sorted(r.get("episode") or f"scenario {r['scenario_id']}" for r in rows if tables.number(r, "steps") is None)
    if uncounted:
        reasons.append(f"{len(uncounted)} episode(s) have no step count, since their agent was never built: {listed(uncounted)}")
    if RULES[port]["judge_decides"]:  # an episode that score_runs.py did not judge is no observation of the task completion
        field = f"paper_{JUDGED[0]}_majority"
        unjudged = sorted(r.get("episode") or f"scenario {r['scenario_id']}" for r in rows if tables.number(r, field) is None)
        if unjudged:
            reasons.append(f"{len(unjudged)} episode(s) of the port hold no judge verdict under the paper protocol: {listed(unjudged)}")
    return reasons


def steps_meet_paper(found, paper):
    """Whether the steps of one task kind meet the paper's value: the interval found, a mean_interval of reproduction_tables.py or None
    before any step was counted, lies inside the paper's value plus or minus STEP_TOLERANCE (STEPS_RULE). The one function
    that decides the steps check."""
    return found is not None and found["low"] >= paper - STEP_TOLERANCE and found["high"] <= paper + STEP_TOLERANCE


def steps_check(rows, port):
    """Per task kind the paper's value, the port's mean with its interval, the half-width of the interval, the mean minus
    the paper's value, and the verdict of steps_meet_paper."""
    steps = {}
    for kind, paper in paper_steps(port).items():
        found = tables.mean_interval(tables.select(rows, kind=kind), "steps")
        steps[kind] = {"paper": paper, "mean": None, "low": None, "high": None, "half_width": None, "difference": None,
                       "pass": steps_meet_paper(found, paper)}
        if found is not None:
            steps[kind].update(mean=found["estimate"], low=found["low"], high=found["high"],
                               half_width=(found["high"] - found["low"]) / 2, difference=found["estimate"] - paper)
    return steps


def bracket(rows, field):
    """(floor, ceiling) in percent of the paper-era rule over the paper's 141 scenarios, or None before score_runs.py."""
    found = tables.mean_interval(rows, f"paper_{field}_majority")
    if found is None:
        return None
    run = tables.scenario_count(rows)
    floor = found["estimate"] * run / tables.PAPER_SCENARIOS
    return 100 * floor, 100 * (floor + (tables.PAPER_SCENARIOS - run) / tables.PAPER_SCENARIOS)


def judge_check(rows, port):
    """Per judge criterion the paper's value beside the port's: the bracket over 141 with whether it holds the paper's value
    for a port whose task completion decides, and else the rate over the port's scenarios under the paper-era rule of its
    design, with pass None, since nothing is checked."""
    rule, judge = RULES[port], {}
    for field, paper in paper_judge(port).items():
        if rule["judge_decides"]:
            found = bracket(rows, field)
            judge[field] = {"paper": paper, "low": found and found[0], "high": found and found[1],
                            "pass": found is not None and found[0] <= paper <= found[1]}
        else:
            found = tables.mean_interval(rows, tables.paper_era_field(rule["paradigm"], field))
            judge[field] = {"paper": paper, "rate": found and 100 * found["estimate"], "pass": None}
    return judge


def check(rows, metas, port=PORT):
    """The verdict of a port on the scores.csv rows and the run.json records of its runs. ``pass`` is True or False when the
    runs decide and None when they do not, with the conditions that failed in ``reasons``."""
    reasons = undecidable(rows, metas, port)
    rows = [r for r in rows if r["agent"] == port]
    steps, judge = steps_check(rows, port), judge_check(rows, port)
    deciding = [s["pass"] for s in steps.values()] + ([judge["task_completion"]["pass"]] if RULES[port]["judge_decides"] else [])
    planned = tables.mean_interval(rows, "plan_steps") if port == PE_PORT else None
    return {"port": port, "decidable": not reasons, "reasons": reasons, "steps": steps, **judge,
            "pass": None if reasons else all(deciding), "scenarios": tables.scenario_count(rows), "episodes": len(rows),
            "errors": sum(1 for r in rows if r.get("error")),
            "plan_steps": planned and {"mean": planned["estimate"], "low": planned["low"], "high": planned["high"]}}


def flow_records(run_dirs, rows):
    """(plans, history entries) of every Plan-Execute port episode whose folder holds its record of the run."""
    folders = {Path(d).name: Path(d) for d in run_dirs}
    flows = []
    for r in rows:
        if r["agent"] != PE_PORT or r.get("run") not in folders or not r.get("episode"):
            continue
        record = folders[r["run"]] / r["episode"] / FLOW_FILE
        if record.is_file():
            flow = json.loads(record.read_text(encoding="utf-8"))
            flows.append((len(flow.get("plans") or []), len(flow.get("history") or [])))
    return flows


def report(verdict, metas, flows=None):
    def number(value, digits=1, sign=""):
        return "not scored" if value is None else f"{value:{sign}.{digits}f}"

    port = verdict["port"]
    rule = RULES[port]
    out = [f"# Acceptance of the paper-era {rule['name']}, {metas[0]['model']}", "",
           f"Verdict: {VERDICTS[verdict['pass']]}. {verdict['episodes']} episodes over {verdict['scenarios']} scenarios, "
           f"{verdict['errors']} of them ended by an error.", ""]
    if not verdict["decidable"]:
        out += ["The runs cannot decide the check, and the values below are printed for the record alone:", ""]
        out += [f"- {reason}" for reason in verdict["reasons"]] + [""]
    out += ["| Task kind | Paper steps | Port mean [95% interval] | Half-width | Mean minus paper | Passes the steps rule |",
            "|---|---|---|---|---|---|"]
    for kind, s in verdict["steps"].items():
        out.append(f"| {kind}-agent | {s['paper']:.1f} | {number(s['mean'], 2)} [{number(s['low'], 2)}, {number(s['high'], 2)}] | "
                   f"{number(s['half_width'], 2)} | {number(s['difference'], 2, '+')} | {s['pass']} |")
    out += ["", f"The steps rule: {STEPS_RULE}."]
    if rule["judge_decides"]:
        out += ["", "| Criterion | Paper | Port floor to ceiling over 141 | Holds the paper |", "|---|---|---|---|"]
        for field, name in tables.JUDGED:
            j = verdict[field]
            out.append(f"| {name} | {j['paper']} | {number(j['low'])} to {number(j['high'])} | {j['pass']} |")
        out += ["", "The steps check and the task completion check decide the verdict. Data retrieval accuracy and result verification "
                "are printed beside their paper values."]
    else:
        out += ["", "| Criterion | Paper (count of Figure 5b) | Port rate over its scenarios, one judge call at temperature 0 |", "|---|---|---|"]
        for field, name in tables.JUDGED:
            out.append(f"| {name} | {verdict[field]['paper']} | {number(verdict[field]['rate'])} |")
    if port == PE_PORT:
        p = verdict["plan_steps"] or {"mean": None, "low": None, "high": None}
        out += ["", f"Plan length: mean {number(p['mean'], 2)} [{number(p['low'], 2)}, {number(p['high'], 2)}] steps, beside the paper's "
                    f"{tables.TABLE12_PLANNED[0]} planned steps over all paradigms."]
        if flows:
            rounds, entries = zip(*flows)
            out += [f"Reviewer rounds per episode: mean {sum(rounds) / len(rounds):.2f}, at most {max(rounds)}, "
                    f"{sum(1 for n in rounds if n >= PLANS)} of {len(rounds)} episodes ran the last plan after {PLANS} invalid reviews.",
                    f"History entries per episode, the executed plan steps that returned and the summarization step: mean "
                    f"{sum(entries) / len(entries):.2f} over {len(entries)} episodes."]
        else:
            out.append(f"Reviewer rounds and history entries per episode: no {FLOW_FILE} in these runs.")
        out += ["", "The steps check alone decides the verdict. The judge criteria and the plan figures are printed beside the paper's values."]
    return out


def main(argv):
    ap = argparse.ArgumentParser(description="the acceptance check of a paper-era port against the paper's steps and judge scores")
    ap.add_argument("runs", nargs="+", help="run folders of run_scenarios.py with --agents naming the port, scored by score_runs.py")
    ap.add_argument("--port", choices=tuple(RULES), default=PORT, help="the port to check, the Agent-As-Tool port by default")
    ap.add_argument("--out", default=None, help="a folder for the report, beside the printout")
    a = ap.parse_args(argv)
    metas, rows = tables.load_runs(a.runs)
    verdict = check(rows, metas, a.port)
    text = "\n".join(report(verdict, metas, flow_records(a.runs, rows) if a.port == PE_PORT else None))
    print(text, flush=True)
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        name = "acceptance.md" if a.port == PORT else f"acceptance_{a.port}.md"
        (Path(a.out) / name).write_text(text + "\n", encoding="utf-8")
    return {True: 0, False: 1, None: NOT_DECIDABLE}[verdict["pass"]]


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
