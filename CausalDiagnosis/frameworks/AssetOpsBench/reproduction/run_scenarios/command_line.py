import argparse
import os
import tempfile
from datetime import datetime
from pathlib import Path
from CausalDiagnosis.frameworks.AssetOpsBench.reproduction.scenario_sets import ACCEPTANCE_REPEATS, SETS
from CausalDiagnosis.frameworks.AssetOpsBench.reproduction.run_folder_names import PE_PORT, PORT, PORTS, RUN_FILE
from CausalDiagnosis.frameworks.AssetOpsBench.reproduction.scenario_file import SRC, load_scenarios
from CausalDiagnosis.frameworks.AssetOpsBench.reproduction import fetch_public_servers_and_data as layer_b
from CausalDiagnosis.experiments.episode_runners import run_episodes  # noqa: E402
from CausalDiagnosis.utils.json_files import write_json  # noqa: E402  the rig's atomic JSON writer
from CausalDiagnosis.experiments.run_verification.public_record import public_value  # noqa: E402  the public form of run.json
from CausalDiagnosis.experiments.run_verification import run_ledger as ledger  # noqa: E402  the rig's run ledger, which records these runs as well
from . import server_launchers, play_episode, run_record


def parse_args(argv):
    ap = argparse.ArgumentParser(description="play the paper-era Agent-As-Tool and Plan-Execute on AssetOpsBench scenarios")
    ap.add_argument("--scenarios", required=True, help="the scenarios.jsonl of the --layer-b folder (README.md)")
    ap.add_argument("--layer-b", default=os.environ.get("AOB_LAYER_B"), help="the folder fetch_public_servers_and_data.py builds (default AOB_LAYER_B)")
    ap.add_argument("--out", required=True, help="the folder that receives the run folder")
    ap.add_argument("--set", choices=sorted(SETS), default="runnable", help="the 120 runnable scenarios, the ten of the pilot, or with a rows file the runnable ones and those it unlocks")
    ap.add_argument("--ids", type=int, nargs="+", help="scenario ids instead of a set")
    ap.add_argument("--agents", nargs="+", choices=PORTS, default=list(PORTS),
                    help=f"{PORT} the paper-era Agent-As-Tool and {PE_PORT} the paper-era Plan-Execute, both by default")
    ap.add_argument("--repeats", type=int, default=ACCEPTANCE_REPEATS, help="runs per scenario, the paper's one by default")
    ap.add_argument("--model", default=os.environ.get("SERVED_NAME"), help="the served model name (default SERVED_NAME)")
    ap.add_argument("--base-url", default=os.environ.get("BASE_URL"), help="the OpenAI compatible endpoint (default BASE_URL)")
    ap.add_argument("--episode-timeout", type=float, default=1800.0, help="seconds after which a run counts as failed")
    ap.add_argument("--no-reload", action="store_true", help="skip the CouchDB reload before each episode")
    ap.add_argument("--server", action="append", default=[], metavar="NAME=SCRIPT",
                    help="run SCRIPT as the server NAME, for tests. When given, the servers are exactly those named")
    ap.add_argument("--allow-unpinned", action="store_true", help="accept a scenario file other than the pinned one")
    return ap.parse_args(argv)


def main(argv):
    a = parse_args(argv)
    if a.set == "recovered" and not a.ids:
        if not a.layer_b:
            raise SystemExit("--set recovered needs --layer-b, a folder built with --recovered-rows")
        ids = layer_b.recovered_ids(a.layer_b)
    else:
        ids = list(a.ids or SETS[a.set])
    set_name = "ids" if a.ids else a.set
    rows = load_scenarios(a.scenarios, ids, pinned=not a.allow_unpinned)
    api_key = os.environ.get("API_KEY")
    if not (a.model and a.base_url and api_key):
        raise SystemExit("a run needs --model (or SERVED_NAME), --base-url (or BASE_URL) and the key in API_KEY")
    if not a.layer_b and (not a.server or not a.no_reload):
        raise SystemExit("a run needs --layer-b (or AOB_LAYER_B), the folder fetch_public_servers_and_data.py builds")
    layer = Path(a.layer_b).resolve() if a.layer_b else None
    servers = server_launchers.parse_servers(a.server)
    shared, fmsr = server_launchers.server_env(a.model, a.base_url, api_key)
    with tempfile.TemporaryDirectory(prefix="aob_launchers_") as folder:
        if layer:
            os.environ.update(layer_b.server_settings(layer, Path(folder) / "tsfm_outputs"))  # the loaders read these too
            shared.update(layer_b.server_settings(layer, Path(folder) / "tsfm_outputs"))
        launchers = server_launchers.write_launchers(folder, servers, shared, fmsr, src=layer / "servers/src" if layer else SRC)
        found = server_launchers.check_servers(launchers)  # before the run folder exists, so a server that cannot start leaves none behind
        label = f"aob_{set_name}"
        run_dir = run_episodes.run_folder(a.out, run_episodes.run_record.run_label(label, a.model))
        meta = run_record.run_meta(a, run_dir, set_name, ids, servers, found, shared, api_key)
        write_json(run_dir / RUN_FILE, public_value(meta))
        ledger.start(run_dir, meta, argv)
        settings = play_episode.Settings(launchers, a.model, a.base_url, api_key, not a.no_reload, a.episode_timeout, layer)
        for repeat in range(1, a.repeats + 1):
            for sid in ids:
                for agent in a.agents:
                    record = play_episode.play(agent, rows[sid], repeat, run_dir, settings)
                    meta["episodes_done"] += 1
                    meta["agent_errors"] += record["error"] is not None
                    write_json(run_dir / RUN_FILE, public_value(meta))
                    ledger.ledger_file.add_episode(run_dir, record["episode"], record)
                    print(f"{record['episode']}: {record['runtime_s']:.1f} s, {record['steps']} steps"
                          + (f", error {record['error'][:200]}" if record["error"] else ""), flush=True)
    meta["ended"] = datetime.now().isoformat(timespec="seconds")
    write_json(run_dir / RUN_FILE, public_value(meta))
    ledger.finish(run_dir)
    print(f"AOB_REPRO_RUN {run_dir}", flush=True)
    return run_dir
