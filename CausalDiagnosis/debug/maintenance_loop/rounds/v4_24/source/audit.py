"""Versioned debug experiments: immutable snapshots, independent episodes and generated reports."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CausalDiagnosis/__main__.py").is_file())
VERSIONS = HERE / "versions"
SOURCES = ("run.py", "protocol.py", "prompts_v2.py", "test_protocol.py", "scenario1.json")


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def journal(title, text):
    """Append an event to runs/journal.md, a local file that the folder's .gitignore keeps out of the repository, so that the
    times, job ids and results of runs stay local; never regenerate or overwrite earlier analysis."""
    path = HERE / "runs" / "journal.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(f"\n## {now()}: {title}\n\n{text}\n")


def version_path(name):
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("version must contain only letters, numbers, underscore or hyphen")
    return VERSIONS / name


def prepare(name, design, config=None, first_seed=0, scenario_ids="2007"):
    folder = version_path(name)
    folder.mkdir(parents=True, exist_ok=False)  # never replace an earlier version
    source = folder / "source"
    source.mkdir()
    filenames = ["scenario.json" if config and name == "scenario1.json" else name for name in SOURCES]
    for name in filenames:
        shutil.copyfile(Path(config) if config and name == "scenario.json" else HERE / name, source / name)
    manifest = {"design": design, "version": folder.name,
                "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "sha256": {name: digest(source / name) for name in filenames},
                "config_file": "scenario.json" if config else "scenario1.json", "first_seed": first_seed,
                "scenario_ids": scenario_ids or None,
                "status": "prepared", "formal_evaluation": False}
    write(folder / "version.json", manifest)
    (folder / "README.md").write_text(
        f"# {folder.name}\n\n设计：{design}\n\n代码冻结于 source/，哈希见 version.json。"
        "每次运行在 runs/ 独立保存；同版本不得修改 source/，新修改必须建立新版本。\n"
        "\n原工具契约、环境及 verifier 不变；±MCP 的工具可用性差异仍是比较限制。\n", encoding="utf-8")
    (folder / "ANALYSIS.md").write_text(
        f"# {folder.name} 分析\n\n## 设计\n\n{design}\n\n"
        "## 实测证据\n\n尚未运行，不能推断收益。\n\n"
        "## 结果解释与下一版\n\n完成后引用运行目录和实际步骤追加分析。\n", encoding="utf-8")
    journal(f"建立 {folder.name}", f"设计：{design}\n\n冻结代码：[版本目录](../versions/{folder.name}/README.md)。")
    print(folder)


def audit(dest):
    scores = []
    for file in dest.rglob("scores.csv"):
        with file.open(encoding="utf-8-sig", newline="") as f:
            scores.extend(dict(row, score_file=str(file.relative_to(dest))) for row in csv.DictReader(f))
    tools, empty_reads, orders, problems = Counter(), 0, [], Counter()
    for file in dest.rglob("trajectory.json"):
        rows = json.loads(file.read_text(encoding="utf-8"))
        for row in rows:
            tools[f"{row.get('server')}.{row.get('tool')}"] += 1
            response = str(row.get("response", ""))
            if row.get("tool") in ("history", "read_sensors"):
                try:
                    parsed, _ = json.JSONDecoder().raw_decode(response.lstrip())
                    # Count only explicit empty arrays, not unrecognized schemas.
                    if parsed == [] or (isinstance(parsed, dict) and any(parsed.get(k) == [] for k in ("observations", "rows", "readings", "served", "data"))):
                        empty_reads += 1
                except (ValueError, TypeError):
                    pass
            if row.get("tool") in ("generate_work_order", "create_work_order"):
                orders.append({"step": row.get("step"), "arguments": row.get("tool_args"),
                               "error": row.get("error")})
    for file in dest.rglob("plans.json"):
        data = json.loads(file.read_text(encoding="utf-8"))
        for plan in data.get("plans", []):
            for item in plan.get("check_problems", []):
                for problem in (item if isinstance(item, list) else [item]):
                    problems[str(problem)] += 1
    return {"audit_revision": 2, "scores": scores, "tool_counts": dict(tools), "explicit_empty_reads": empty_reads,
            "order_attempts": orders, "check_problems": dict(problems.most_common(12))}


def report(run, summary):
    lines = [f"# {summary['version']} / {run.name}", "", f"状态：{summary['status']}。开发试跑，不计入正式主表。", "",
             "| 模式 | pass | 诊断全对 | 恢复 | 工具调用 | 诊断MCP调用 | 时间cycle | 结束原因 |", "|---|---|---|---|---|---|---|---|"]
    for result in summary["results"]:
        if not result["audit"]["scores"]:
            lines.append(f"| {result['mode']} | 无评分记录 | - | - | - | - | - | 进程返回 {result['returncode']} |")
        for row in result["audit"]["scores"]:
            keys = ("pass", "diagnosis_all_correct", "recovered", "tool_calls", "causal_calls", "time_used", "end_reason")
            lines.append("| " + " | ".join([result["mode"], *(row.get(k, "") for k in keys)]) + " |")
    lines += ["", "## 证据与限制", "", "完整原始记录在各模式的 episodes/；命令、进程日志、拒绝原因和计数单独保留。",
              "进程返回0不等于任务通过；回合超时/API失败与诊断失败分别记录。空读计数只识别明确的空数组，需结合原始返回核实。",
              "±MCP 沿用正式工具契约，两组的等待等工具可能不同，不能把差异全部归于诊断信息。", ""]
    for result in summary["results"]:
        lines += [f"## {result['mode']}", "", "工具计数：`" + json.dumps(result["audit"]["tool_counts"], ensure_ascii=False) + "`", ""]
        for error, count in result["audit"]["check_problems"].items():
            lines.append(f"- {count} 次：{error}")
        for row in result["audit"]["scores"]:
            if row.get("error"):
                lines += ["", "运行错误：" + row["error"]]
    lines += ["", "## 分析与下一版本", "", "自动报告只描述可观测结果。人工分析在 ANALYSIS.md 追加，并引用实际步骤；未经实测的设计不得写成收益。", ""]
    (run / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run_pair(name, base_url, model, modes=("mcp", "no_mcp")):
    folder = version_path(name)
    version = json.loads((folder / "version.json").read_text(encoding="utf-8"))
    for filename, expected in version["sha256"].items():
        if digest(folder / "source" / filename) != expected:
            raise ValueError(f"frozen source changed: {filename}; create a new version")
    # Check the existing endpoint, without recording its authorization header.
    request = urllib.request.Request(base_url.rstrip("/") + "/models")
    if os.environ.get("API_KEY"):
        request.add_header("Authorization", "Bearer " + os.environ["API_KEY"])
    with urllib.request.urlopen(request, timeout=30) as response:
        models = json.load(response)
    ids = [r.get("id") for r in models.get("data", [])]
    if model not in ids:
        raise ValueError(f"requested model is not listed: {model}; available: {ids}")
    run = folder / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run.mkdir(parents=True, exist_ok=False)
    summary = {"version": name, "status": "running", "started_at": now(), "results": [],
               "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "model": model, "models": models,
               "source_sha256": version["sha256"], "base_commit": version["base_commit"]}
    write(run / "summary.json", summary)
    journal(f"启动 {name}/{run.name}", f"作业：{summary['slurm_job_id']}；模式：{', '.join(modes)}；模型：{model}。")
    for mode in modes:
        extra = ["--no-diagnostics"] if mode == "no_mcp" else []
        dest = run / mode
        dest.mkdir()
        command = [sys.executable, str(folder / "source/run.py"), "experiments", "run-episodes",
                   "--config", str(folder / "source" / version.get("config_file", "scenario1.json")),
                   "--agent", "causal", "--servers", "aob", "--store", "memory", "--starts", "seeds",
                   "--seeds", "1", "--first-seed", str(version.get("first_seed", 0)), "--model", model, "--base-url", base_url,
                   "--judge", "none", "--out", str(dest / "episodes"), *extra]
        scenario_ids = version.get("scenario_ids", "2007")
        if scenario_ids:
            command += ["--scenario-ids", scenario_ids]
        write(dest / "command.json", {"command": command, "started_at": now()})
        write(run / "progress.json", {"mode": mode, "status": "running", "updated_at": now()})
        print("DEBUG_RUN_START", mode, run, flush=True)
        with (dest / "stdout.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        item = {"mode": mode, "returncode": result.returncode, "finished_at": now(), "audit": audit(dest / "episodes")}
        write(dest / "result.json", item)
        summary["results"].append(item)
        write(run / "summary.json", summary)
        report(run, summary)
        print("DEBUG_RUN_END", mode, result.returncode, flush=True)
    summary.update(status="completed", finished_at=now())
    write(run / "summary.json", summary)
    write(run / "progress.json", {"status": "completed", "updated_at": now()})
    report(run, summary)
    lines = []
    for result in summary["results"]:
        for row in result["audit"]["scores"]:
            lines.append(f"- {result['mode']}：pass={row.get('pass')}，诊断全对={row.get('diagnosis_all_correct')}，"
                         f"恢复={row.get('recovered')}，结束={row.get('end_reason')}，错误={row.get('error') or '无'}。")
    journal(f"完成 {name}/{run.name}", "\n".join(lines) + f"\n\n[本次结果](../versions/{name}/runs/{run.name}/README.md)。")
    print("DEBUG_PAIR_RECORDED", run, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--version", required=True)
    p.add_argument("--design", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--first-seed", type=int, default=0)
    p.add_argument("--scenario-ids", default="2007", help="empty string disables anchored scenario selection")
    p = sub.add_parser("run-pair")
    p.add_argument("--version", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", default="granite-3.3-8b-instruct")
    p.add_argument("--modes", nargs="+", choices=("mcp", "no_mcp"), default=["mcp", "no_mcp"])
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.version, args.design, args.config, args.first_seed, args.scenario_ids)
    else:
        run_pair(args.version, args.base_url, args.model, args.modes)
