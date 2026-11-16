import argparse
from pathlib import Path
from . import certificate


# ----- the command line -----
def main(argv=None):
    ap = argparse.ArgumentParser(description="the replay check of a finished run and its certificate")
    commands = ap.add_subparsers(dest="command", required=True)
    certify_parser = commands.add_parser("certify", help="replay a run folder on a clone of its commit and write its certificate "
                                                         "when the replay passes and counts")
    certify_parser.add_argument("run", type=Path)
    a = ap.parse_args(argv)
    status, result, path = certificate.certify(a.run)
    for line in result["problems"] + [f"advisory: {reason}" for reason in result["advisory"]]:
        print(line)
    print(f"{a.run.name}: {result['state']}, {result['episodes']} episodes, {result['lines']} transcript lines compared, "
          + (f"certificate {path.name}" if path else "no certificate"))
    return status
