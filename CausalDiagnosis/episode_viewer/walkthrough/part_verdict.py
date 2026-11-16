import json
from . import formatting


VERDICT_WORDS = {True: "right", False: "wrong", None: "not submitted"}


# ----- part 4, the verdict -----
def part_verdict(ep):
    out = [f"## {formatting.PARTS[3]}", ""]
    verdict, score, oracle = ep.hidden("verdict.json"), ep.hidden("score.json") or {}, ep.hidden("oracle.json")
    if not verdict:
        return out + ["The verdict is not in this episode folder.", ""]
    if not verdict.get("verified"):
        return out + [f"The verifier could not run on this episode, {formatting.quoted(verdict.get('error'))}.", ""]
    components = ep.components
    out += ["The verifier reads the log and the hidden state, never the answer text, and runs its six checks in this order "
            "(`hidden/verdict.json`).", ""]
    out += formatting.table(["Check", "Result", "Evidence"], check_rows(verdict["checks"], components))
    grade = verdict.get("grade")
    out += ["", "The scores follow (`hidden/verdict.json` and `hidden/score.json`).", ""]
    out += formatting.table(["Score", "Value"], [
        ("Failure class", verdict["failure_class"]),
        ("Pass, a correct diagnosis, a recovered rig and no unnecessary order", formatting.num(verdict["pass"])),
        ("Success", formatting.num(verdict["success"])),
        ("Separable start, by the label and by E18", f"{formatting.num(verdict['separable'])} and {formatting.num(verdict.get('separable_e18'))}"
         + (f", E18 cost {formatting.num((grade.get('e18') or {}).get('cost'))}" if grade else ", grade unknown")
         + ((", grade from the table" if score["grade_source"] == "table" else f", grade source {formatting.quoted(score['grade_source'], 160)}")
            if score.get("grade_source") else "")),
        ("Abstained", formatting.num(verdict["abstained"])),
        ("Hidden start", formatting.long_levels(verdict["hidden_start"], components)),
        *truth_rows(verdict, components),
        ("Final combination", formatting.long_levels(verdict["final"], components)),
        ("Cost", f"{formatting.num(verdict['cost'])}, prices {formatting.num(verdict['cost_prices'])}, penalties {formatting.num(verdict['penalties'])}, downtime {formatting.num(verdict['downtime_total'])}"),
        ("Cost of the oracle policy", formatting.num(verdict["cost_oracle"])),
        ("Cost over the oracle's", formatting.prob(verdict["cost_normalized"])),
        ("Downtime", formatting.num(verdict["downtime_total"])),
        ("Detection delay", delays(verdict["detection_delay"])),
        ("Repair delay", delays(verdict["repair_delay"])),
        ("Turns to the correct diagnosis", formatting.num(verdict["turns_to_correct_diagnosis"])),
        ("Predictions cited by orders", formatting.joined(f"{r['work_order']} cites {r['prediction_id']}, found {formatting.num(r['found'])}, predicted recovery "
                                               f"{formatting.prob(r['predicted_recovery'])}" for r in verdict.get("prediction_verdict") or []) or "none"),
        ("Summary check, a heuristic", summary_check_text(verdict["summary_check"])),
        ("End", f"{verdict['end_reason']} at cycle {formatting.num(verdict['time_used'])}, {formatting.plural(verdict['steps'], 'environment step')}, "
                f"{formatting.num(verdict['refusals'])} refused, failures {formatting.joined(verdict['failures']) or 'none'}"),
        ("Reward", formatting.prob(score.get("reward"))),
        ("Judge", judge_text(ep.hidden("judge.json")))])
    if oracle:
        out += ["", f"The oracle's reference (`hidden/oracle.json`) diagnoses {formatting.long_levels(oracle['diagnosis'], components)} and orders "
                + (formatting.joined(f"the {o['component']} with window end {formatting.num(o.get('window_end'))}" for o in oracle["work_orders"]) or "nothing")
                + ", then finishes."]
    return out + [""]


def truth_rows(verdict, components):
    """The cycle of the diagnosis and the truth it is scored against, which the verifier records since version 7."""
    if verdict.get("truth_at_write") is None:
        return []
    return [("Diagnosis written at cycle", formatting.num(verdict.get("write_t"))), ("Truth at the write", formatting.long_levels(verdict["truth_at_write"], components))]


def check_rows(checks, components):
    describe = {
        "count_gate": lambda c: ("writes " + (", ".join(f"{k} {v}" for k, v in c["writes"].items()) or "none")
                                 + (", problems " + formatting.joined(formatting.quoted(p) for p in c["problems"]) if c["problems"] else ", no problem")),
        "hard": lambda c: ("diagnosis " + ", ".join(f"{k} {VERDICT_WORDS[c['diagnosis_per_component'][k]]}" for k in components)
                           + f", orders on {formatting.joined(c['work_orders_agent']) or 'nothing'} against the oracle's {formatting.joined(c['work_orders_oracle']) or 'nothing'}"),
        "timing": lambda c: (formatting.joined(f"{o['id']} on the {o['component']} closed at {formatting.num(o['closed_t'])} against the window end {formatting.num(o['window_end'])}"
                                    for o in c["orders"]) or "no agent order on a component with a scheduled or hazard drop"),
        "causality": lambda c: f"commitment at log entry {formatting.num(c['commit_index'])}, first order at log entry {formatting.num(c['first_order_index'])}",
        "outcome": lambda c: (f"recovered {formatting.num(c['recovered'])}, {formatting.plural(c['unnecessary'], 'unnecessary order')}"
                              + (" " + formatting.joined(c["unnecessary_orders"]) if c["unnecessary_orders"] else "")),
        "provenance": lambda c: (f"{formatting.plural(c['cited'], 'identifier')} cited, "
                                 + (f"unmatched {formatting.joined(c['unmatched'][:5])}" if c["unmatched"] else "every one issued by the episode")
                                 if c["cited"] else "no identifier cited"),
    }
    return [(f"`{name}`", "passed" if check["ok"] else "failed", describe[name](check) if name in describe else formatting.quoted(json.dumps(check), 160))
            for name, check in checks.items()]


def delays(by_component):
    return ", ".join(f"{c} {formatting.num(v)}" for c, v in by_component.items() if v is not None) or "none"


def summary_check_text(check):
    return (f"claims recovery {formatting.num(check['claims_recovery'])}, claims replacements of {formatting.joined(check['claimed_replacements']) or 'nothing'}, "
            f"unsupported {formatting.joined(check['unsupported_replacements']) or 'none'}, contradicts the log {formatting.num(check['contradicts_log'])}")


def judge_text(judge):
    if not judge:
        return "no judge record for this episode"
    calls = judge.get("calls") or []
    errors = [c["error"] for c in calls if c.get("error")]
    text = (f"protocol {judge.get('protocol')}, judge model {judge.get('judge_model')}, pass {formatting.num(judge.get('pass'))}, pass rate "
            f"{formatting.num(judge.get('pass_rate'))}, mean score {formatting.num(judge.get('score_mean'))}")
    return text + (f", {len(errors)} of {formatting.plural(len(calls), 'call')} failed with {formatting.quoted(errors[0], 120)}" if errors else "")
