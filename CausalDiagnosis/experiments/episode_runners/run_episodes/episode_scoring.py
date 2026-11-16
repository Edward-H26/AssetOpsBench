import json
from CausalDiagnosis.experiments.episode_runners.run_folder_layout import HIDDEN, PREDICTIONS_FILE, RELEASED_FILE, SCORE_COLUMNS, STATE_FILE, episode_reward  # noqa: E402
from CausalDiagnosis.utils.json_files import read_json, write_json  # noqa: E402
from CausalDiagnosis.datasets.dataset_loader import dataset_packages as ports  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment import verifier  # noqa: E402
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.env import HydraulicRigEnv  # noqa: E402
from CausalDiagnosis.utils.public_rules.visits import is_no_call  # noqa: E402
from CausalDiagnosis.utils import model_request_retry as retry  # noqa: E402
from CausalDiagnosis.mcp_servers.Environment_Actions_MCP import episode_state as episode  # noqa: E402
from . import reference_policy_episodes


def grade_of_start(env, released):
    """The label of the episode's start in the form the verifier reads: the label that the asset of the episode declares, the
    function that LABELS of its package names called on the released rules (CausalDiagnosis.datasets.dataset_loader.dataset_packages.start_label), or none for an asset
    that declares none, under which the verifier counts every start separable. The rig declares separability.start_grade: the
    independent label, the greedy reference's majority over ten rollouts, with the server's E18 grade nested under
    "e18", read from the grade table of the row's rules, so that one rule scores every row, the rows without the server included,
    whatever ranking a run serves. That grade is None only for a start outside the fault rule of the sampler, such as an anchored day, and a start of the
    fault rule that either table lacks raises, as a failed lookup does. Returns the grade and its source."""
    if ports.declared(env.config.asset_id, "LABELS") is None:
        return None, "no label declared by the asset"
    grade = ports.start_label(env, released["rules"])
    return grade, "not in table" if grade is None else "table"


# the score column of each check of the verifier that scores.csv reports beside provenance_ok
CHECK_COLUMNS = {"write_count_ok": "count_gate", "timing_ok": "timing", "ordering_ok": "causality"}


def _check_ok(checks, name):
    value = (checks or {}).get(name)
    if isinstance(value, dict):
        return value.get("ok", value.get("passed"))
    return value


def _delay(value):
    """The largest delay over components, from a per component dict or a number; None when nothing was measured."""
    if isinstance(value, dict):
        measured = [v for v in value.values() if isinstance(v, (int, float))]
        return max(measured) if measured else None
    return value


def verdict_row(verdict, reward, cost_oracle, judge):
    """The verifier's verdict as the score columns of scores.csv; a key the verdict lacks leaves its cell empty."""
    row = {k: verdict.get(k) for k in SCORE_COLUMNS}
    row["detection_delay"] = _delay(verdict.get("detection_delay"))
    row["repair_delay"] = _delay(verdict.get("repair_delay"))
    row["provenance_ok"] = _check_ok(verdict.get("checks"), "provenance")
    # the write count, timing and ordering checks, reported for every episode beside provenance and outside pass
    for column, check in CHECK_COLUMNS.items():
        row[column] = _check_ok(verdict.get("checks"), check)
    row["reward"] = reward
    row["cost_oracle"] = cost_oracle
    row["judge_pass"] = None if judge is None else judge.get("pass")
    return row


def verify_episode(ep_dir, config, cycles, blocks, seed, options, env_class=HydraulicRigEnv):
    """Replay the served episode, write the hidden files and return the score columns and the start combination. A log
    the verifier cannot read raises, so the caller records the error and leaves the scores empty. options holds the
    judge settings (or None), whether the grade is looked up, and the ranking model of the grade table. env_class is
    the engine the episode played, the one with the record in the AssetOpsBench rows."""
    env = episode.replay(episode.load_state(ep_dir / STATE_FILE), env_class=env_class)
    oracle = env.oracle()
    released = read_json(ep_dir / RELEASED_FILE, {})
    cost_oracle = reference_policy_episodes.oracle_reference(cycles, blocks, config, seed)
    if config.verifier_scope == "all":
        oracle["reads"] = reference_policy_episodes.greedy_reference(cycles, blocks, config, seed)
    grade, grade_source = None, "disabled"
    if options["grade"]:
        # a lookup that fails raises like a verdict that could not run, since a verdict under an unknown label would score
        # every declaration of the run as an abstention on a separable start with the reason in hidden/score.json alone
        grade, grade_source = grade_of_start(env, released)
    verdict = verifier.verify(env.log, released, oracle, cost_oracle, grade=grade, scope=config.verifier_scope,
                              predictions=read_json(ep_dir / PREDICTIONS_FILE))
    hidden = ep_dir / HIDDEN
    hidden.mkdir(exist_ok=True)
    write_json(hidden / "log.json", env.log)
    write_json(hidden / "oracle.json", oracle)
    write_json(hidden / "verdict.json", verdict)
    if not verdict.get("verified"):
        raise RuntimeError(f"the verifier could not read the episode: {verdict.get('error')}")
    reward = episode_reward(verdict, config.horizon)
    judge = run_judge(ep_dir, options["judge"]) if options["judge"] else None
    row = verdict_row(verdict, reward, cost_oracle, judge)
    write_json(hidden / "score.json", {**row, "scope": config.verifier_scope, "grade_source": grade_source, "judge": judge})
    return row, [oracle["diagnosis"][c] for c in config.components]


def run_judge(ep_dir, options, static=False):
    """The AssetOpsBench judge on the trajectory and the closing answer of the runner, or on the static answer when static is set,
    through the judge that options names (JUDGES of a framework's declaration), which writes hidden/judge.json itself; a judge that
    cannot run, an episode without a trajectory among them, leaves the pass empty rather than failing the episode. A judge whose
    context is unknown or whose endpoint no longer answers is no judge failure of one episode: it is raised, so the run stops with
    the reason."""
    from CausalDiagnosis.utils import frameworks

    judge = frameworks.code("JUDGES", options["judge"])

    try:
        return judge.judge_episode(ep_dir, options["backend"], options["protocol"], options["agent_model"], options["base_url"], static=static,
                                   serving=options.get("serving"), budget=options.get("budget"))
    except (judge.ContextUnknown, retry.ServiceDown):
        raise
    except Exception as exc:  # noqa: BLE001 - a judge failure is recorded, not raised
        return {"pass": None, "error": f"{type(exc).__name__}: {exc}"}


def tool_counts(state):
    calls = [call for call in state.get("tool_calls", []) if not is_no_call(call)]  # a no_call step is no tool call
    refused = 0
    for call in calls:
        if call.get("refused"):  # the metering layer names the refusal beside its receipt line
            refused += 1
            continue
        try:
            refused += "refused" in json.loads(call["response"])
        except (TypeError, ValueError, KeyError):
            pass
    return {"tool_calls": len(calls), "refusals": refused}
