"""Run one frozen scenario-major round; archive failures and release serving job after completion."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analysis(folder, item):
    lines = [f"# {item['scenario']} / {item['mode']}", "", "开发试跑，单seed，不作为正式显著性结论。", ""]
    for row in item["audit"]["scores"]:
        lines += ["| 指标 | 结果 |", "|---|---|"]
        for key in ("pass", "diagnosis_all_correct", "recovered", "clock_mode", "time_used", "cost", "downtime_total", "end_reason", "error"):
            lines.append(f"| {key} | {row.get(key, '')} |")
    lines += ["", "## 观测到的问题类型", ""]
    parses, examples = Counter(), []
    for file in (folder / "episodes").rglob("plans.json"):
        data = json.loads(file.read_text(encoding="utf-8"))
        for plan in data.get("plans", []):
            for error in plan.get("parse_errors", []):
                parses[str(error)] += 1
            problems = plan.get("check_problems")
            if (problems or plan.get("surprise")) and len(examples) < 4:
                examples.append({"plan_index": plan.get("index"), "proposed_or_executed": plan.get("plan"),
                                 "check_problems": problems, "surprise": plan.get("surprise")})
    write(folder / "evidence_examples.json", examples)
    if parses:
        lines.append("- 格式/协议：存在解析失败。模型输出与prompt/schema共同影响，不能仅凭解析失败断言模型能力不足。")
        lines += [f"  - {n}次：{e}" for e, n in parses.most_common(5)]
    if item["audit"]["check_problems"]:
        lines.append("- 控制器：存在执行前拒绝。具体原因见result.json和evidence_examples.json，不等同于环境物理规则。")
    if item["audit"]["explicit_empty_reads"]:
        lines.append(f"- 数据/时钟接口：{item['audit']['explicit_empty_reads']}次明确空读；需结合clock_mode和窗口判断，不能直接判为数据集覆盖不足。")
    for row in item["audit"]["scores"]:
        if row.get("error"):
            lines.append("- 运行异常：" + row["error"] + "。与普通任务失败分开，不直接计作推理能力结论。")
        if row.get("clock_mode") == "generation_time":
            lines.append("- 动态环境：启用思考计时，实际cycle=" + row.get("time_used", "") + "；仍须核对事件日志，不能只用最终pass判断环境是否演化。")
    lines += ["", "## 原始证据", "", "完整输入/输出、工具反馈、计划和原评分在episodes/。本报告只做证据分类，不自动宣称唯一根因。", ""]
    (folder / "ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")


def overview(root, state):
    lines = ["# v4 八场景 × 三配置", "", f"状态：{state['state']}；已完成 {len(state['completed'])}/24。", "",
             "三个配置使用同一个LLM与服务。我们的两配置使用冻结v004；Plan-and-Execute沿用原版。每场景依次跑三组。",
             "当前配置来自此前重建的8个debug场景，非正式测试集统计。±MCP仍有部分工具可用性差异。", "",
             "| 场景 | 模式 | pass | 诊断全对 | 恢复 | cycle | 空读 | 结束原因 | 异常 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for item in state["completed"]:
        scores = item["audit"]["scores"]
        if not scores:
            lines.append(f"| {item['scenario']} | {item['mode']} | 无评分 | — | — | — | — | — | 进程返回{item['returncode']} |")
        for row in scores:
            cells = [item["scenario"], item["mode"], *(row.get(k, "") for k in ("pass", "diagnosis_all_correct", "recovered", "time_used")),
                     str(item["audit"]["explicit_empty_reads"]), row.get("end_reason", ""), row.get("error", "")]
            lines.append("| " + " | ".join(str(c).replace("|", "/") for c in cells) + " |")
    lines += ["", "## 综合分析原则", "",
              "按静态诊断、动态劣化、维修依赖、失效截止、证据不足及健康对照分别查看，不混成一个pass数字。",
              "程序异常/服务异常、解析失败、控制器拒绝、环境按规则演化、真实诊断/维修错误分别记录。",
              "prompt和模型能力通常不能仅凭一条失败严格分离；保留模型原文和具体反馈再作人工归因。",
              "成功率、完整诊断、修复、生产损失、空读、调用数量共同查看。候选排名正确不等于完成维修。", ""]
    (root / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main(root, base_url):
    root = root.resolve()
    repo = next(p for p in root.parents if (p / "CausalDiagnosis/__main__.py").is_file())
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for filename, expected in manifest["sha256"].items():
        if sha(root / filename) != expected:
            raise ValueError("frozen round file changed: " + filename)
    spec = importlib.util.spec_from_file_location("round_audit", root / "source/audit.py")
    audit_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit_module)
    state_file = root / "status.json"
    state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {
        "state": "running", "started_at": stamp(), "completed": [], "current": None}
    state.update(state="running", gpu_service_job_id=os.environ.get("SLURM_JOB_ID"))
    request = urllib.request.Request(base_url.rstrip("/") + "/models", headers={"Authorization": "Bearer " + os.environ["API_KEY"]})
    with urllib.request.urlopen(request, timeout=30) as response:
        write(root / "models.json", json.load(response))
    endpoint = os.environ.get("ENDPOINT_FILE")
    if endpoint and Path(endpoint).exists():
        (root / "serving.txt").write_text(Path(endpoint).read_text(), encoding="utf-8")
    for task in manifest["tasks"]:
        if any(x["id"] == task["id"] for x in state["completed"]):
            continue
        parent = root / "runs" / task["id"]
        attempt = 1
        while (parent / f"attempt_{attempt:02d}").exists():
            attempt += 1
        folder = parent / f"attempt_{attempt:02d}"
        folder.mkdir(parents=True)
        state["current"] = dict(task, attempt=attempt, started_at=stamp())
        write(state_file, state)
        overview(root, state)
        args = [sys.executable]
        if task["agent"] == "causal":
            args += [str(root / "source/run.py")]
        else:
            args += ["-m", "CausalDiagnosis"]
        args += ["experiments", "run-episodes", "--config", str(root / task["config"]), "--agent", task["agent"],
                 "--servers", "aob", "--store", "memory", "--starts", "seeds", "--seeds", "1", "--first-seed", str(task["seed"]),
                 "--model", "granite-3.3-8b-instruct", "--base-url", base_url, "--judge", "none", "--out", str(folder / "episodes")]
        if task.get("scenario_id"):
            args += ["--scenario-ids", str(task["scenario_id"])]
        if task["mode"] == "causal_no_mcp":
            args.append("--no-diagnostics")
        write(folder / "command.json", {"command": args, "started_at": stamp(), "task": task})
        print("ROUND_START", task["id"], flush=True)
        with (folder / "stdout.log").open("w", encoding="utf-8") as log:
            proc = subprocess.run(args, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
        result = dict(task, returncode=proc.returncode, finished_at=stamp(), attempt=attempt,
                      audit=audit_module.audit(folder / "episodes"))
        write(folder / "result.json", result)
        analysis(folder, result)
        state["completed"].append(result)
        state["current"] = None
        write(state_file, state)
        overview(root, state)
        print("ROUND_END", task["id"], proc.returncode, flush=True)
    state.update(state="completed", finished_at=stamp(), gpu_shutdown="native serving script stops server after this command returns")
    write(state_file, state)
    overview(root, state)
    # Archive the reproducible and reviewable records, excluding private launch configurations and keys.
    allowed = {"SUMMARY.md", "ANALYSIS.md", "evidence_examples.json", "manifest.json", "status.json", "models.json", "serving.txt",
               "command.json", "result.json", "stdout.log", "trajectory.json", "plans.json", "llm_calls.json", "episode.json",
               "released.json", "transcript.jsonl", "report.md", "scores.csv", "predictions.json", "debug_variant.json"}
    target = root.parent / (root.name + "_completed.zip")
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in root.rglob("*"):
            if file.is_file() and (file.name in allowed or "source" in file.relative_to(root).parts or "configs" in file.relative_to(root).parts):
                archive.write(file, file.relative_to(root).as_posix())
    write(root / "archive.json", {"path": str(target), "bytes": target.stat().st_size, "sha256": sha(target), "created_at": stamp()})
    print("ROUND_COMPLETE_ARCHIVED_STOP_GPU", target, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", required=True, type=Path)
    parser.add_argument("--base-url", required=True)
    a = parser.parse_args()
    main(a.round, a.base_url)
