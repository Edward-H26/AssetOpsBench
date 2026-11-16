import argparse
import sys
from pathlib import Path
from CausalDiagnosis.utils.json_files import write_json
from . import serving_endpoints, run_data, code_identity, ledger_file, write_after_run


def main(argv):
    """python -m CausalDiagnosis.experiments.run_verification.run_ledger event RUN_DIR TOOL WHAT appends an event without files, for a tool that
    moves a run folder instead of changing its files,
    python -m CausalDiagnosis.experiments.run_verification.run_ledger after RUN_DIR [--scorer-sha256 FILE] writes the ledger of a finished
    AssetOpsBench reproduction run that has none,
    python -m CausalDiagnosis.experiments.run_verification.run_ledger platform --out FILE [--facts PATH] [--model NAME] [--judge-model NAME] writes the
    platform record of the node, with the registered setting of each model that the variables of SETTING hand it,
    python -m CausalDiagnosis.experiments.run_verification.run_ledger record-check [--facts PATH] [--asset ID] prints RECORD <commit> or DEVELOPMENT with one
    line REASON <text> per reason of rules 1, 2, 4, 5 and 6, exit 0 in both cases, and
    python -m CausalDiagnosis.experiments.run_verification.run_ledger serving --shell [--role ROLE] prints the serving file of record as bash assignments
    (exit 3 while it is open), while serving --check ENDPOINT_FILE --role ROLE [--mode MODE] compares an endpoint file with it and
    prints SERVING_OK or SERVING_DIFFERS with one line REASON per field, exit 1 on a difference."""
    ap = argparse.ArgumentParser(description="the ledger of a run folder, the platform record of a node, the record check of a tree and the serving file of record")
    commands = ap.add_subparsers(dest="command", required=True)
    event = commands.add_parser("event", help="append an event without files to the ledger of a run folder")
    event.add_argument("run_dir", type=Path)
    event.add_argument("tool")
    event.add_argument("what")
    after = commands.add_parser("after", help="write the ledger of a finished reproduction run that has none")
    after.add_argument("run_dir", type=Path)
    after.add_argument("--scorer-sha256", type=Path, help="sha256sum lines of the scorer code on the machine that scored the run")
    record = commands.add_parser("platform", help="write the platform record of this node")
    record.add_argument("--out", type=Path, required=True)
    record.add_argument("--asset", default=None, help="the asset of the record (default: the engine's default asset)")
    record.add_argument("--facts", default=None,
                        help="the data file of the asset (default: the file that the variable of its asset file names, else its release file)")
    record.add_argument("--model", default=None,
                        help="the served name of the agent model, whose endpoint file HYD_ENDPOINT_FILE names (default: HYD_MODEL, else served_name of that file)")
    record.add_argument("--judge-model", default=None,
                        help="the served name of the judge, whose endpoint file HYD_JUDGE_ENDPOINT_FILE names (default: HYD_JUDGE_MODEL, else served_name of that file)")
    check = commands.add_parser("record-check", help="whether this tree can play runs of record, with every reason it cannot")
    check.add_argument("--facts", default=None, help="the data file (default: the release file of the asset)")
    check.add_argument("--asset", default=None, help="the asset (default: the engine's default asset)")
    serving = commands.add_parser("serving", help="the serving file of record as shell assignments, or an endpoint file against it")
    shape = serving.add_mutually_exclusive_group(required=True)
    shape.add_argument("--shell", action="store_true")
    shape.add_argument("--check", type=Path, metavar="ENDPOINT_FILE")
    serving.add_argument("--role", choices=serving_endpoints.SERVING_ROLES, default=None)
    serving.add_argument("--mode", choices=serving_endpoints.SERVING_MODES, default=None)
    a = ap.parse_args(argv)
    if a.command == "after":
        written = write_after_run.write_after(a.run_dir, a.scorer_sha256)
        print(f"wrote {a.run_dir / ledger_file.LEDGER} after the run: {len(written['episodes'])} episodes, status {written['status']}")
    elif a.command == "event":
        if not ledger_file.append_event(a.run_dir, a.tool, a.what):
            print(f"{a.run_dir} has no ledger, nothing recorded")
    elif a.command == "platform":
        write_json(a.out, run_data.platform_command(a.facts, a.model, a.judge_model, a.asset))
        print(f"wrote the platform record {a.out}")
    elif a.command == "record-check":
        reasons = code_identity.record_reasons(a.facts, a.asset)
        print("DEVELOPMENT" if reasons else f"RECORD {code_identity.git_state([], run_data.ROOT)[0]}")
        for reason in reasons:
            print(f"REASON {reason}")
    elif a.shell:
        try:
            print("\n".join(serving_endpoints.serving_shell(a.role)))
        except serving_endpoints.OpenServingFile as e:
            print(e, file=sys.stderr)
            return 3
    else:
        if a.role is None:
            ap.error("serving --check needs --role")
        record = serving_endpoints.endpoint_file(a.check)
        record["matches_endpoint"] = True
        try:
            found = serving_endpoints.serving_differences({"endpoint_file": record}, a.role, a.mode)
        except serving_endpoints.OpenServingFile as e:
            print(e, file=sys.stderr)
            return 3
        print(f"{'SERVING_DIFFERS' if found else 'SERVING_OK'} {a.role}")
        for field, declared, value in found:
            print(f"REASON {field} declared {declared} endpoint {value}")
        return 1 if found else 0
    return 0
