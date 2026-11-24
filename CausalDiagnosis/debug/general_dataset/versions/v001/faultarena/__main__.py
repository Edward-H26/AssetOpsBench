"""Download and audit the FaultArena release before implementing its replay."""

import argparse
import json
from pathlib import Path

from .audit import audit_release
from .download import download_release


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download")
    download.add_argument("--out", type=Path, required=True)
    download.add_argument("--include-images", action="store_true")
    audit = commands.add_parser("audit")
    audit.add_argument("--release", type=Path, required=True)
    audit.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            manifest = download_release(args.out, include_images=args.include_images)
            print(json.dumps({"revision": manifest["revision"], "files": len(manifest["files"]), "out": str(args.out)}))
        else:
            report = audit_release(args.release)
            if args.out:
                with args.out.open("x", encoding="utf-8") as stream:
                    stream.write(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
            if report["problems"]:
                return 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
