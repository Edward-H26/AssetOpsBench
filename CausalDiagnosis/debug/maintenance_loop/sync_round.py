"""Read-only local watcher: sync round status and verified archive, then exit after GPU job ends."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time
import zipfile


def remote_state(host, remote, job):
    script = f'''from pathlib import Path
import json,subprocess
r=Path({remote!r})
out={{}}
for name in ['status.json','archive.json','cluster.json']:
 p=r/name
 if p.exists():out[name]=json.loads(p.read_text())
p=r/'SUMMARY.md'
if p.exists():out['summary_markdown']=p.read_text()
s=subprocess.run(['bash','-lc','squeue -h -j {int(job)} -o %T'],capture_output=True,text=True)
out['gpu_job_active']=bool(s.stdout.strip())
out['gpu_job_state']=s.stdout.strip()
print(json.dumps(out))
'''
    encoded = base64.b64encode(script.encode()).decode()
    command = "python3 -c " + shlex.quote("import base64;exec(base64.b64decode(" + repr(encoded) + "))")
    result = subprocess.run(["ssh", "-4", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=12", host, command],
                            capture_output=True, text=True, encoding="utf-8", timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout)


def watch(host, remote, job, local, interval):
    local = local.resolve()
    local.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            data = remote_state(host, remote, job)
            for name in ["status.json", "cluster.json"]:
                if name in data:
                    (local / name).write_text(json.dumps(data[name], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if "summary_markdown" in data:
                (local / "SUMMARY.md").write_text(data["summary_markdown"], encoding="utf-8")
            sync = {"synced_at": datetime.now(timezone.utc).isoformat(), "gpu_job": job,
                    "gpu_job_active": data["gpu_job_active"], "gpu_job_state": data["gpu_job_state"],
                    "round_state": data.get("status.json", {}).get("state", "queued_or_starting")}
            archive = data.get("archive.json")
            if archive and not (local / "SYNC_COMPLETE.json").exists():
                dest = local.parent / (local.name + "_completed.zip")
                cmd = ["scp", "-4", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=12",
                       host + ":" + archive["path"], str(dest)]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                if result.returncode:
                    raise RuntimeError(result.stderr.strip())
                if hashlib.sha256(dest.read_bytes()).hexdigest() != archive["sha256"]:
                    raise ValueError("archive SHA256 mismatch")
                with zipfile.ZipFile(dest) as z:
                    for name in z.namelist():
                        if not (local / name).resolve().is_relative_to(local):
                            raise ValueError("archive path outside round")
                    z.extractall(local)
                (local / "archive.json").write_text(json.dumps(archive, indent=2) + "\n", encoding="utf-8")
                sync["archive_verified"] = True
                (local / "SYNC_COMPLETE.json").write_text(json.dumps(sync, indent=2) + "\n", encoding="utf-8")
            (local / "sync.json").write_text(json.dumps(sync, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(sync), flush=True)
            if (local / "SYNC_COMPLETE.json").exists() and not data["gpu_job_active"]:
                return
            if not data["gpu_job_active"] and sync["round_state"] != "completed":
                # Keep the incomplete results visible; do not silently call them a completed round.
                print("GPU job ended before full archive; inspect cluster log", flush=True)
                return
        except Exception as error:
            print(datetime.now(timezone.utc).isoformat(), type(error).__name__, str(error), flush=True)
        time.sleep(interval)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="insomnia")
    p.add_argument("--remote", required=True)
    p.add_argument("--job", required=True)
    p.add_argument("--local", type=Path, required=True)
    p.add_argument("--interval", type=int, default=120)
    a = p.parse_args()
    watch(a.host, a.remote, a.job, a.local, a.interval)
