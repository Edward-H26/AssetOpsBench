"""A development QA driver using shared MCP sessions and a version-local loop."""

import csv
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

from CausalDiagnosis.agents.utils.mcp_sessions import open_sessions
from CausalDiagnosis.utils.paths import ROOT
from .tool_loop import run_tool_loop

from .data import QADataset
from .evaluation import check_answer, direct_numeric_match


INSTRUCTIONS = ("Answer the maintenance question using released evidence. Read the fact before submitting. "
                "The submission answer has direct_answer and reasoning_answer as nonempty strings, "
                "provenance with fact_id and a nonempty features list, and confidence from zero to one. "
                "Optional provenance file and row must match the released evidence. "
                "A structurally accepted answer ends the task; acceptance is not a quality score.")


def write_json(path, value):
    """Replace a checkpoint atomically inside the new run folder."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def launch_text(facts, questions, qa_id, folder, sqlite, *, normalized=False):
    """Make the shared isolated session process import this checkout and runtime."""
    import mcp

    paths = [str(ROOT), str(Path(mcp.__file__).resolve().parents[1])]
    return ("import sys\n" + f"sys.path[:0] = {paths!r}\n"
            "from pathlib import Path\n"
            "from CausalDiagnosis.debug.general_dataset.versions.v001.industryasseteqa.data import QADataset\n"
            "from CausalDiagnosis.debug.general_dataset.versions.v001.industryasseteqa.episode import QAEpisode\n"
            "from CausalDiagnosis.debug.general_dataset.versions.v001.industryasseteqa.server import build_server\n"
            f"dataset = QADataset.load({str(facts)!r}, {str(questions)!r}, sqlite={sqlite!r})\n"
            f"build_server(QAEpisode(dataset, {qa_id!r}), Path({str(folder)!r}), normalized={normalized!r}).run(transport='stdio')\n")


def finish(tool, arguments, text, is_error):
    if tool != "submit_answer" or is_error:
        return None
    try:
        response = json.loads(text)
    except (ValueError, TypeError):
        return None
    if isinstance(response, dict) and response.get("accepted") is True:
        return arguments["answer"]
    return None


def code_files():
    """Hash the driver and shared execution code used by a development run."""
    paths = [*Path(__file__).parent.glob("*.py"),
             ROOT / "CausalDiagnosis/agents/utils/mcp_sessions.py",
             ROOT / "CausalDiagnosis/agents/utils/chat_model.py",
             ROOT / "CausalDiagnosis/agents/utils/declaration.py",
             ROOT / "CausalDiagnosis/agents/utils/answer_forms.py",
             ROOT / "CausalDiagnosis/agents/utils/tool_answers.py",
             ROOT / "CausalDiagnosis/agents/utils/core_texts.py",
             ROOT / "CausalDiagnosis/utils/public_rules/plan_rules.py",
             *(path for path in (ROOT / "CausalDiagnosis/agents/Causal_Diagnosis_Agent_next").rglob("*.py")
               if "tests" not in path.parts),
             ROOT / "CausalDiagnosis/utils/model_request_retry.py",
             ROOT / "CausalDiagnosis/utils/atomic_writes_and_locks.py",
             ROOT / "CausalDiagnosis/utils/paths.py",
             ROOT / "uv.lock",
             * (ROOT / "CausalDiagnosis/mcp_servers/utils").glob("*.py")]
    return {path.relative_to(ROOT).as_posix(): digest(path) for path in sorted(paths)}


async def run_questions(facts, questions, qa_ids, out, *, sqlite, ask, settings, max_turns, agent="json"):
    """Run selected questions with partial records, input hashes and explicit status.

    This development format is separate from the hydraulic driver and its ledger.
    """
    facts, questions, out = Path(facts).resolve(), Path(questions).resolve(), Path(out).resolve()
    dataset = QADataset.load(facts, questions, sqlite=sqlite)
    if agent not in ("json", "general"):
        raise ValueError("agent must be json or general")
    if not qa_ids or len(set(qa_ids)) != len(qa_ids):
        raise ValueError("qa_ids must be nonempty and unique")
    for qa_id in qa_ids:
        dataset.task(qa_id)
    if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
        raise ValueError("max_turns must be a positive integer")
    if out.exists():
        raise ValueError("run output must be new")
    source_hashes = {"facts": digest(facts), "qa": digest(questions)}
    out.mkdir(parents=True)
    meta = {"format": "maintenance_qa_development", "status": "running", "agent": "general" if agent == "general" else "json_tool_loop",
            "settings": settings, "max_turns": max_turns, "qa_ids": qa_ids,
            "inputs": source_hashes, "code": code_files(), "episodes": [],
            "runtime": {"python": sys.version, "platform": sys.platform,
                        "packages": {name: importlib.metadata.version(name) for name in ("mcp", "pydantic")}}}
    write_json(out / "run.json", meta)
    rows = []
    try:
        for index, qa_id in enumerate(qa_ids):
            folder = out / f"episode_{index:05d}"
            folder.mkdir()
            write_json(folder / "task.json", dataset.task(qa_id))
            launcher = folder / "launcher.py"
            launcher.write_text(launch_text(facts, questions, qa_id, folder, sqlite, normalized=agent == "general"), encoding="utf-8")

            def checkpoint(result):
                write_json(folder / "result.json", result)

            try:
                if agent == "general":
                    from .general import run_general
                    result = await run_general(ask, launcher, dataset.task(qa_id), max_turns, checkpoint)
                else:
                    async with open_sessions({"maintenance_qa": launcher}) as sessions:
                        result = await run_tool_loop(ask, sessions, dataset.task(qa_id), instructions=INSTRUCTIONS,
                                                     max_turns=max_turns, finish=finish, checkpoint=checkpoint)
                    result["server_returncodes"] = sessions.returncodes
            except Exception as error:
                saved = folder / "result.json"
                result = json.loads(saved.read_text(encoding="utf-8")) if saved.exists() else {
                    "answer": None, "model_calls": [], "trajectory": []}
                result.update(end="error", error={"type": type(error).__name__, "message": str(error)})
            report = check_answer(result["answer"], dataset.public_fact(qa_id))
            result["checks"] = report
            checkpoint(result)
            write_json(folder / "prediction.json", {"qa_id": qa_id, "answer": result["answer"], "error": result["error"]})
            row = {"qa_id": qa_id, "end": result["end"], "model_calls": len(result["model_calls"]),
                   "tool_calls": len(result["trajectory"]), "structure_ok": report["structure_ok"],
                   "provenance_ok": report["provenance_ok"],
                   "direct_numeric_match": direct_numeric_match(result["answer"], dataset.reference(qa_id))}
            rows.append(row)
            meta["episodes"].append({"qa_id": qa_id, "folder": folder.name, "end": result["end"]})
            write_json(out / "run.json", meta)
        if source_hashes != {"facts": digest(facts), "qa": digest(questions)}:
            raise ValueError("input files changed during the run")
        meta["status"] = "completed" if all(row["end"] == "submitted" for row in rows) else "incomplete"
    except BaseException as error:
        meta.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                    error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        with (out / "scores.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("qa_id", "end", "model_calls", "tool_calls", "structure_ok", "provenance_ok", "direct_numeric_match"))
            writer.writeheader()
            writer.writerows(rows)
        write_json(out / "run.json", meta)
    return meta
