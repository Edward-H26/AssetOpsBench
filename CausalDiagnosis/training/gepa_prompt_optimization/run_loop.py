from datetime import datetime
from pathlib import Path
from . import run_folders, online_loop, episode_player


HERE = Path(__file__).resolve().parent
from CausalDiagnosis.experiments.run_verification import run_ledger as ledger  # noqa: E402
from CausalDiagnosis.experiments.episode_runners import run_episodes as drivers  # noqa: E402
from CausalDiagnosis.utils.llm_backends import NO_THINKING, endpoint_backend  # noqa: E402
from CausalDiagnosis.utils.json_files import read_json
from CausalDiagnosis.experiments.table_definitions.training_variants import check_fold, numbers  # noqa: E402


def run_online(a, argv, settings, runner, config, price_table, stream, reflection):
    """One online loop into a run folder, with run.json in the mode gepa_online, the ledger written before the first episode and
    closed after run.json, the episodes, gepa_online.json, the candidates and the reflection calls. With --resume the folder is
    that of the unfinished loop of the same identity under --out, when there is one, and a finished loop of that identity there
    returns its folder and no loop. A loop that fails leaves its folder without run.json, as an unfinished run, and raises
    LoopFailed."""
    runner.label = drivers.run_record.run_label(runner.config_name, f"{run_folders.MODE}_{a.card}_{a.agent}_{a.training_variant}_f{a.training_fold}_{a.model}")
    settings.seeds_list = list(stream)
    # the reasons of the start check before run.json, so that a loop that is not registered carries the stamp, as a run of
    # run_episodes/ does: the runner alone asks the tree at its first episode, after run.json of the loop is made
    runner.unregistered = drivers.start_check.registration(settings, config, runner.config_name, runner.facts_path)
    meta = drivers.run_meta(settings, runner, config, price_table, [], None)
    meta.update({"mode": run_folders.MODE, "gepa": loop_settings.gepa_meta(a, config.asset_id)})
    identity = run_folders.as_stored(run_folders.loop_identity(meta))
    if a.resume and (done := run_folders.already_finished(a.out, identity)) is not None:
        return done, None
    taken_up = run_folders.folder_to_resume(a.out, identity) if a.resume else None
    run_dir = taken_up or drivers.run_folder(a.out, runner.label)
    registered = numbers(config.asset_id)
    loop = online_loop.OnlineLoop(episode_player.EpisodePlayer(runner, settings, run_dir, meta), reflection, run_dir, a.card, stream, block=a.block,
                      window=a.window, children=a.children, max_episodes=a.max_episodes, gepa_seed=a.gepa_seed,
                      reflection_model=a.reflection_model, reflection_endpoint=a.reflection_base_url, horizon=config.horizon,
                      unanswered_proposals=registered.unanswered_proposals, proposal_attempts=registered.proposal_attempts,
                      asset_id=config.asset_id, training_fold=a.training_fold, split=a.training_split, agent=a.agent,
                      training_variant=a.training_variant,
                      identity=identity)
    if taken_up:
        loop.restore(read_json(run_dir / run_folders.RESULT))
        last = loop.interrupted[-1]
        ledger.append_event(run_dir, "gepa_online.py", f"resumed after round {last['after_round']}, {len(last['episodes'])} episodes of "
                                                       f"the round cut short moved to {last['folder']}")
    else:
        ledger.start(run_dir, meta, argv)
    loop.run()
    meta["episodes_planned"] = loop.calls
    drivers.write_run(run_dir, meta, loop.rows)
    ledger.finish(run_dir)
    return run_dir, loop


def main(argv):
    if argv[:1] == ["export"]:
        return block_export.export(argv[1:])
    a = loop_settings.build_parser().parse_args(argv)
    settings = loop_settings.driver_settings(a)
    facts_path, cycles, blocks = drivers.load_world(a.facts)
    config_name, config, price_table = drivers.configure(settings)
    config = drivers.played_config(settings, config, blocks)  # widened to the development starts in a loop of a trial
    check_fold(a.training_fold, config.asset_id, a.training_split)  # the parser takes any integer, so a fold the split does not name stops here
    loop_settings.with_registered_numbers(a, config.asset_id)
    for name in ("block", "window", "children", "max_episodes"):
        if getattr(a, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be at least 1")
    if config.anchors is not None:
        raise SystemExit(f"--config {a.config} anchors its scenarios, and the online loop needs the seeded starts of a stream")
    stream = loop_settings.loop_stream(a, config, cycles, blocks)
    runner, _judge = drivers.build_runner(settings, config_name, config, price_table, cycles, blocks, facts_path)
    if drivers.is_marker(runner.adapter):
        # no adapter on this fold, so the loop of the ft_gepa variant does not run and its marker run reads no block
        print(f"SKIPPED {a.agent} f{a.training_fold} {a.training_variant}: no adapter, the fold kept fewer episodes than the registered minimum")
        return None
    reflection = endpoint_backend(a.reflection_model, a.reflection_base_url, extra_body=NO_THINKING)
    reflection.max_tokens = None  # no reply cap: the request carries none, whatever the backend's default
    try:
        run_dir, loop = run_online(a, argv, settings, runner, config, price_table, stream, reflection)
    except online_loop.LoopFailed as failure:
        raise SystemExit(f"FAILED {a.agent} f{a.training_fold} {a.training_variant}: {failure}. The folder keeps the loop for --resume") from None
    if loop is None:
        print(f"FINISHED {run_dir}: the loop of these settings has finished there, so --resume plays nothing")
        return run_dir
    print(f"wrote {run_dir}: {loop.played} stream episodes in {len(loop.rounds)} rounds, {len(loop.candidates)} candidates, final "
          f"{loop.incumbent}, {sum(1 for e in loop.episodes if e['left_out'])} episodes left out, ended {datetime.now():%H:%M:%S}")
    return run_dir


# Imported after the definitions: this module uses them only when a function runs, and they may load
# this module first (the order of the package is in __init__.py).
from . import loop_settings, block_export
