"""Run a registered declared episode dataset with the shared agent and records."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

from CausalDiagnosis.agents.Causal_Diagnosis_Agent_next.controller import AgentRunner
from CausalDiagnosis.datasets.declared_episodes import discover
from CausalDiagnosis.utils.paths import PACKAGE, ROOT


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def code_identity():
    roots = [PACKAGE / path for path in ("agents/Causal_Diagnosis_Agent_next", "agents/utils", "datasets",
             "environment/scripted_cases", "utils/public_rules", "debug/general_dataset")]
    paths = {Path(__file__), PACKAGE / "__main__.py", ROOT / "uv.lock",
             PACKAGE / "utils/paths.py", PACKAGE / "utils/atomic_writes_and_locks.py"}
    for root in roots:
        for folder, dirs, files in os.walk(root):
            dirs[:] = [name for name in dirs if name not in {"data", "tests", "__pycache__"}]
            paths.update(Path(folder) / name for name in files if name.endswith(".py") or name == "episode.json")
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


class DeclaredModel:
    takes_response_format = True

    def __init__(self, ask, max_calls):
        self.ask, self.max_calls = ask, max_calls
        self.supports_images = bool(getattr(ask, "supports_images", False))
        if callable(getattr(ask, "fits_context", None)):
            self.fits_context = ask.fits_context

    async def generate_with_usage(self, prompt, temperature=0.0, response_format=None, images=None):
        content = prompt
        if images:
            if not self.supports_images:
                raise ValueError("the model declares no image transport")
            content = [{"type": "text", "text": prompt}]
            for item in images:
                content.append({"type": "text", "text": f"Image evidence from {item['observation_id']}"})
                content.append({"type": "image_url", "image_url": {"url": f"data:{item['mime_type']};base64,{item['data']}"}})
        return await self.ask([{"role": "user", "content": content}], response_format=response_format)


def select_cases(port, subset, case_ids, limit):
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer")
    available = port.case_ids(subset)
    selected = list(available if case_ids is None else case_ids)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(available):
        raise ValueError("case_ids must be nonempty, unique and belong to the selected partition")
    return selected if limit is None else selected[:limit]


def validate_dataset(descriptor, inputs, subset, *, case_ids=None, limit=None, max_tool_calls=None, budget=None, supports_images=False):
    """Validate input identity and selected case structure without a model."""
    port = descriptor.load(inputs)
    selected = select_cases(port, subset, case_ids, limit)
    limits = {"max_tool_calls": max_tool_calls, "budget": budget}
    errors, unsupported, checked = {}, {}, 0
    for case_id in selected:
        try:
            port.validate_case(case_id, limits)
            checked += 1
        except (ValueError, KeyError, TypeError) as error:
            name = type(error).__name__
            errors[name] = errors.get(name, 0) + 1
        reason = port.unsupported(case_id, supports_images=supports_images)
        if reason:
            unsupported[reason] = unsupported.get(reason, 0) + 1
    return {"dataset": descriptor.id, "kind": descriptor.kind, "subset": subset, "selected": len(selected),
            "validated": checked, "errors": errors, "unsupported": unsupported, "valid": not errors,
            "inputs": port.identity(), "adapter_code": descriptor.source_identity(),
            "semantic_quality": None, "model_calls": 0, "supports_images": supports_images}


async def run_dataset(descriptor, inputs, subset, out, *, ask, max_turns, settings,
                      case_ids=None, limit=None, max_tool_calls=None, budget=None):
    if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
        raise ValueError("max_turns must be a positive integer")
    if max_tool_calls is not None and (isinstance(max_tool_calls, bool) or not isinstance(max_tool_calls, int) or max_tool_calls < 1):
        raise ValueError("max_tool_calls must be a positive integer")
    if budget is not None and (isinstance(budget, bool) or not math.isfinite(budget) or budget < 0):
        raise ValueError("budget must be finite and nonnegative")
    port = descriptor.load(inputs)
    selected = select_cases(port, subset, case_ids, limit)
    limits = {"max_tool_calls": max_tool_calls, "budget": budget}
    tasks = {case_id: port.task(case_id, limits) for case_id in selected}
    out = Path(out).resolve()
    if out.exists():
        raise ValueError("run output must be new")
    identities, code, adapter_code = port.identity(), code_identity(), descriptor.source_identity()
    out.mkdir(parents=True)
    meta = {"format": "declared_episode_run", "dataset": descriptor.id, "kind": descriptor.kind,
            "subset": subset, "case_ids": selected, "inputs": identities, "code": code, "adapter_code": adapter_code,
            "settings": settings, "supports_images": bool(getattr(ask, "supports_images", False)),
            "max_turns": max_turns, "limits": limits, "status": "running",
            "publication": port.publication, "runtime": {"python": sys.version, "platform": sys.platform,
                "packages": {name: importlib.metadata.version(name) for name in ("mcp", "pydantic", "jsonschema", "openai")}}, "episodes": []}
    write_json(out / "run.json", meta)
    try:
        for index, case_id in enumerate(selected):
            folder = out / f"episode_{index:05d}"
            folder.mkdir()
            write_json(folder / "task.json", tasks[case_id])
            unsupported = port.unsupported(case_id, supports_images=bool(getattr(ask, "supports_images", False)))
            if unsupported:
                row = {"case_id": case_id, "folder": folder.name, "end": "unsupported", "reason": unsupported,
                       "scores": None, "model_calls": 0, "tool_calls": 0}
                write_json(folder / "result.json", row)
                meta["episodes"].append(row)
                write_json(out / "run.json", meta)
                continue
            launcher = folder / "launcher.py"
            launcher.write_text(port.launch(case_id, folder, limits), encoding="utf-8")
            runner = AgentRunner(DeclaredModel(ask, max_turns), {port.server: launcher}, port.declaration(case_id))

            def checkpoint(episode):
                write_json(folder / "episode.json", asdict(episode))
                write_json(folder / "llm_calls.json", runner.model_history)

            runner.on_update = checkpoint
            episode = await runner.run(json.dumps(tasks[case_id], ensure_ascii=False, allow_nan=False))
            checkpoint(episode)
            scores = port.evaluate(case_id, episode, folder)
            row = {"case_id": case_id, "folder": folder.name, "end": episode.end, "error": episode.error,
                   "scores": scores, "model_calls": episode.model_calls, "tool_calls": len(episode.trajectory)}
            write_json(folder / "result.json", row)
            meta["episodes"].append(row)
            write_json(out / "run.json", meta)
        if port.identity() != identities or code_identity() != code or descriptor.source_identity() != adapter_code:
            raise ValueError("inputs or runtime source changed during the run")
        meta["status"] = "completed" if all(row["end"] == "finish" for row in meta["episodes"]) else "incomplete"
    except BaseException as error:
        meta.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                    error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        ends = {}
        for row in meta["episodes"]:
            ends[row["end"]] = ends.get(row["end"], 0) + 1
        write_json(out / "summary.json", {"selected": len(selected), "recorded": len(meta["episodes"]), "ends": ends,
                                          "semantic_quality": None, "publication": port.publication})
        write_json(out / "run.json", meta)
    return meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--multimodal", action="store_true")
    parser.add_argument("--dataset")
    parser.add_argument("--input", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--split")
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--max-tool-calls", type=int)
    parser.add_argument("--budget", type=float)
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env", default="API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args(argv)
    registry = discover()
    if args.list:
        for item in registry.values():
            print(f"{item.id}: {item.kind}; inputs: {', '.join(item.inputs)}")
        return 0
    try:
        if args.dataset not in registry:
            raise ValueError("name a registered dataset with --dataset; --list shows the choices")
        if not args.split:
            raise ValueError("split is required")
        inputs = {}
        for value in args.input:
            name, separator, path = value.partition("=")
            if not separator or not path or name in inputs:
                raise ValueError("each input must be a distinct NAME=PATH")
            inputs[name] = Path(path)
        if args.validate:
            report = validate_dataset(registry[args.dataset], inputs, args.split, case_ids=args.case_id, limit=args.limit,
                                      max_tool_calls=args.max_tool_calls, budget=args.budget, supports_images=args.multimodal)
            print(json.dumps(report, indent=2))
            return 0 if report["valid"] else 1
        if not all((args.out, args.max_turns, args.model, args.base_url)):
            raise ValueError("out, max-turns, model and base-url are required for a model run")
        url = urlsplit(args.base_url)
        if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("base-url must be an HTTP endpoint without embedded credentials, query or fragment")
        key = os.environ.get(args.api_key_env)
        if not key:
            raise ValueError(f"set the environment variable {args.api_key_env}")
        if not math.isfinite(args.temperature):
            raise ValueError("temperature must be finite")
        import asyncio
        from CausalDiagnosis.agents.utils.chat_model import ChatModel

        async def run():
            model = ChatModel(args.model, args.base_url, key, args.temperature, supports_images=args.multimodal)
            try:
                return await run_dataset(registry[args.dataset], inputs, args.split, args.out, ask=model,
                                         max_turns=args.max_turns, settings={"model": args.model, "base_url": args.base_url,
                                         "api_key_env": args.api_key_env, "temperature": args.temperature},
                                         case_ids=args.case_id, limit=args.limit, max_tool_calls=args.max_tool_calls, budget=args.budget)
            finally:
                await model.close()
        result = asyncio.run(run())
        print(json.dumps({"status": result["status"], "selected": len(result["case_ids"]), "recorded": len(result["episodes"])}))
        return 0 if result["status"] == "completed" else 1
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
