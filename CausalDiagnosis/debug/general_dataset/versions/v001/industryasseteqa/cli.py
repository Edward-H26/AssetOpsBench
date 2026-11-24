"""Commands for downloading inputs, inspecting questions and checking answers."""

import argparse
import asyncio
import json
import math
import os
import subprocess
from pathlib import Path

from .data import DataError, QADataset, read_jsonl
from .evaluation import check_answer, direct_numeric_match


def evaluate(dataset, predictions):
    """Check every question, keeping missing and errored predictions in the total."""
    indexed = {}
    for prediction in predictions:
        qa_id = prediction.get("qa_id")
        if not isinstance(qa_id, str) or qa_id not in dataset.question_ids():
            raise DataError(f"prediction names an unknown qa_id: {qa_id!r}")
        if qa_id in indexed:
            raise DataError(f"duplicate prediction: {qa_id}")
        indexed[qa_id] = prediction
    checks = []
    for qa_id in dataset.question_ids():
        prediction = indexed.get(qa_id, {})
        report = check_answer(prediction.get("answer"), dataset.public_fact(qa_id))
        checks.append({"qa_id": qa_id, "missing": qa_id not in indexed,
                       "error": prediction.get("error"), **report,
                       "direct_numeric_match": direct_numeric_match(prediction.get("answer"), dataset.reference(qa_id))})
        if prediction.get("error"):
            checks[-1].update(structure_ok=False, provenance_ok=False)
            if checks[-1]["direct_numeric_match"] is not None:
                checks[-1]["direct_numeric_match"] = False
    numeric = [row["direct_numeric_match"] for row in checks if row["direct_numeric_match"] is not None]
    return {"check_protocol": "structure_provenance_and_single_numeric_answer", "total": len(checks),
            "structure_ok_rate": sum(row["structure_ok"] for row in checks) / len(checks),
            "provenance_ok_rate": sum(row["provenance_ok"] for row in checks) / len(checks),
            "numeric_scorable": len(numeric), "direct_numeric_match_rate": sum(numeric) / len(numeric) if numeric else None,
            "checks": checks}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="fetch the publisher's raw CSV files")
    download.add_argument("--out", type=Path, required=True)
    build = commands.add_parser("build", help="build descriptive facts and QA with upstream code")
    build.add_argument("--raw", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    for command in ("inspect", "check", "serve", "run", "split"):
        sub = commands.add_parser(command)
        source = sub.add_mutually_exclusive_group(required=True)
        source.add_argument("--facts", type=Path)
        source.add_argument("--db", type=Path)
        sub.add_argument("--qa", type=Path, required=True)
        if command == "check":
            sub.add_argument("--predictions", type=Path, required=True)
            sub.add_argument("--out", type=Path, required=True)
        elif command == "run":
            selection = sub.add_mutually_exclusive_group(required=True)
            selection.add_argument("--qa-id", action="append")
            selection.add_argument("--split", type=Path)
            sub.add_argument("--subset", choices=("dev", "test"), default="dev")
            sub.add_argument("--limit", type=int)
            sub.add_argument("--out", type=Path, required=True)
            sub.add_argument("--model", required=True)
            sub.add_argument("--base-url", required=True)
            sub.add_argument("--api-key-env", default="API_KEY")
            sub.add_argument("--temperature", type=float, default=0.0)
            sub.add_argument("--max-turns", type=int, required=True)
            sub.add_argument("--agent", choices=("json", "general"), default="json")
        elif command == "split":
            sub.add_argument("--out", type=Path, required=True)
            sub.add_argument("--dev-fraction", type=float, default=0.2)
            sub.add_argument("--seed", default="0")
        else:
            sub.add_argument("--qa-id", required=True)
        if command == "serve":
            sub.add_argument("--episode-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            from .download import download_raw

            manifest = download_raw(args.out)
            print(json.dumps({"revision": manifest["revision"], "destination": str(args.out)}))
            return
        if args.command == "build":
            from .build import build_descriptive

            print(build_descriptive(args.raw, args.out))
            return
        dataset = QADataset.load(args.facts or args.db, args.qa, sqlite=args.db is not None)
        if args.command == "split":
            from .splits import make_split
            manifest = make_split(dataset, args.facts or args.db, args.qa, args.dev_fraction, args.seed)
            with args.out.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(manifest, indent=2) + "\n")
            print(json.dumps({name: {"assets": len(part["assets"]), "questions": len(part["qa_ids"])}
                              for name, part in manifest["partitions"].items()}))
            return
        if args.command == "run":
            from .model import ChatModel
            from .runner import run_questions
            from urllib.parse import urlsplit

            selection = None
            qa_ids = args.qa_id
            if args.split:
                from .splits import select_split, input_hash
                manifest = json.loads(args.split.read_text(encoding="utf-8"))
                qa_ids = select_split(dataset, args.facts or args.db, args.qa, manifest, args.subset)
                selection = {"manifest_sha256": input_hash(args.split), "subset": args.subset}
            if args.limit is not None:
                if args.limit < 1:
                    raise ValueError("limit must be positive")
                qa_ids = qa_ids[:args.limit]

            url = urlsplit(args.base_url)
            if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.query or url.fragment:
                raise ValueError("base-url must be an HTTP endpoint without credentials, query or fragment")
            if not math.isfinite(args.temperature) or args.max_turns < 1:
                raise ValueError("temperature must be finite and max-turns must be positive")
            key = os.environ.get(args.api_key_env)
            if not key:
                raise ValueError(f"set the environment variable {args.api_key_env}")

            async def run():
                model = ChatModel(args.model, args.base_url, key, args.temperature)
                try:
                    return await run_questions(args.facts or args.db, args.qa, qa_ids, args.out,
                                               sqlite=args.db is not None, ask=model,
                                               settings={"model": args.model, "base_url": args.base_url,
                                                         "temperature": args.temperature, "route": "chat/completions",
                                                         "transport": "json_schema" if args.agent == "general" else "json_tool_steps", "api_key_env": args.api_key_env,
                                                         "selection": selection},
                                               max_turns=args.max_turns, agent=args.agent)
                finally:
                    await model.close()

            result = asyncio.run(run())
            print(json.dumps({"status": result["status"], "episodes": len(result["episodes"]), "out": str(args.out)}))
            if result["status"] != "completed":
                parser.exit(1, "the run did not submit an answer for every selected question\n")
        elif args.command == "inspect":
            print(json.dumps({"task": dataset.task(args.qa_id), "fact": dataset.public_fact(args.qa_id)}, indent=2))
        elif args.command == "check":
            result = evaluate(dataset, read_jsonl(args.predictions))
            with args.out.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(result, indent=2) + "\n")
        else:
            from .episode import QAEpisode
            from .server import build_server

            if args.episode_dir.exists():
                raise ValueError("episode-dir must be new to avoid mixing transcripts")
            args.episode_dir.mkdir(parents=True)
            build_server(QAEpisode(dataset, args.qa_id), args.episode_dir).run(transport="stdio")
    except (OSError, ValueError, KeyError, ImportError, subprocess.SubprocessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
