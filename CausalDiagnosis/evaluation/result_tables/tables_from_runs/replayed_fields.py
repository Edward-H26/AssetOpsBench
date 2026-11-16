import json
from pathlib import Path
from CausalDiagnosis.experiments.episode_runners.episode_launchers import DEFAULT_RANKING_MODEL  # noqa: E402
from CausalDiagnosis.utils.csv_rows import as_bool  # noqa: E402
from CausalDiagnosis.utils.json_files import read_json  # noqa: E402
from CausalDiagnosis.evaluation.metrics import mcp_metrics  # noqa: E402
from . import episode_files, episode_selection


def elimination_fields(episode):
    """The realized elimination of the paid reads of an episode and its calibration against the server's forecasts
    (realized_read_elimination.py), from released.json and the rank_root_causes answers of trajectory.json: the paid reads, the
    realized elimination per read and per unit of the cost of E21 as charged, the reads that the server forecast on the
    belief they met with their mean expected and realized elimination and the gap, and the digest of the models.json the
    replay read. An episode without a released read loads no model, and neither does one without a start combination, the empty
    block cell of an episode whose verification failed, which has no block for the fits to leave out."""
    from CausalDiagnosis.evaluation.metrics import realized_read_elimination as read_elimination  # the server's filter loads only when a table reads the elimination

    folder = episode.get("episode_dir")
    released = read_json(Path(folder) / "released.json", {}) if folder is not None else {}
    if not episode.get("block") or not any(e.get("event") == "read" for e in released.get("events", [])):
        return {**read_elimination.episode_measures([]), "yardstick": ""}
    left_out = tuple(json.loads(episode["block"]))  # the start, a JSON list in scores.csv, whose block the fits leave out
    records = read_elimination.read_records(released, left_out=left_out)
    read_elimination.calibrate(records, released, episode_files.ranking_answers(episode_files._trajectory(episode)), left_out=left_out)
    return {**read_elimination.episode_measures(records), "yardstick": read_elimination.yardstick(left_out=left_out).digest if records else ""}


def add_elimination_fields(episodes):
    """The fields of elimination_fields on the episodes that ELIM and A13_A17 read, those of the five agents, of the
    switches of the causal agent and of the reference policies, so that no other episode replays the filter."""
    measured = {id(e) for e in episode_selection.elimination_episodes(episodes) + episode_selection.causal_episodes(episodes)}
    for e in episodes:
        if id(e) in measured:
            e.update(elimination_fields(e))
    return episodes


def add_mcp_fields(episodes):
    """The measures of mcp_metrics.py under the key mcp of every episode of the agent tables that has an episode folder, read
    once from the files that the servers and the verifier wrote, and the production lost while thinking as the cell
    think_loss, which the rows with the MCP pair against the rows without it. Only the rows with the MCP replay the post repair
    check."""
    for e in episode_selection.agent_episodes(episodes, every_clock_mode=True):
        if e.get("episode_dir") is None:
            continue
        verdict = read_json(Path(e["episode_dir"]) / "hidden" / "verdict.json", {})
        e["mcp"] = mcp_metrics.episode_measures(e["episode_dir"], mcp_metrics.start_of(e.get("block")), e.get("ranking_model") or DEFAULT_RANKING_MODEL,
                                                verdict, with_verdicts=as_bool(e.get("diagnostics_server")) is True)
        e["think_loss"] = e["mcp"].get("think_loss")
    return episodes
