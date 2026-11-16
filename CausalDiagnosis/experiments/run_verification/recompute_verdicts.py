"""Recompute the verdict of every episode folder under a runs folder with this checkout's verifier, so that a refactor
can show that the verifier gives the same verdict on the same stored inputs (CausalDiagnosis/docs/adding-a-dataset.md).

An episode folder holds hidden/log.json, hidden/oracle.json, hidden/verdict.json and released.json, and predictions.json
when the diagnostics server stored predictions. The stored verdict supplies the inputs that the run computed outside the
log, which are the cost of the oracle policy, the grade, and the verifier scope. The output holds, for every
folder, the digest of the recomputed verdict and the keys in which it differs from the stored one. Run folders stay
outside the repository, so the output belongs outside it too.

usage:
  python -m CausalDiagnosis.experiments.run_verification.recompute_verdicts --runs RUNS --out OUT.json [--label NAME]
  python -m CausalDiagnosis.experiments.run_verification.recompute_verdicts --runs RUNS --out OUT.json --check EARLIER.json
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

from CausalDiagnosis.experiments.run_verification.invariant_digests import digest  # noqa: E402  (imports common, which puts this checkout's src first on sys.path)

FORMAT = 1
NEEDED = ("hidden/log.json", "hidden/oracle.json", "hidden/verdict.json", "released.json")


def episode_folders(runs):
    """Every folder under runs that holds the files a verdict needs, sorted by path."""
    return sorted({p.parents[1] for p in Path(runs).rglob("log.json") if p.parent.name == "hidden"})


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def reverify(folder):
    """The recomputed verdict's digest and the keys in which it differs from the stored verdict, or the files missing."""
    from CausalDiagnosis.environment.Causal_Diagnosis_Environment import verifier

    folder = Path(folder)
    missing = [rel for rel in NEEDED if not (folder / rel).exists()]
    if missing:
        return {"skipped": f"missing {', '.join(missing)}"}
    stored = read(folder / "hidden/verdict.json")
    predictions_path = folder / "predictions.json"
    verdict = verifier.verify(read(folder / "hidden/log.json"), read(folder / "released.json"), read(folder / "hidden/oracle.json"),
                              stored.get("cost_oracle"), grade=stored.get("grade"), scope=stored.get("scope", "writes"),
                              predictions=read(predictions_path) if predictions_path.exists() else None)
    differs = sorted(k for k in set(verdict) | set(stored) if verdict.get(k) != stored.get(k))
    return {"digest": digest(verdict), "stored_equal": not differs, "differs": differs}


def run(runs, label=None):
    runs = Path(runs)
    episodes = {str(folder.relative_to(runs)): reverify(folder) for folder in episode_folders(runs)}
    verified = [e for e in episodes.values() if "digest" in e]
    return {"format": FORMAT, "label": label, "runs": str(runs), "episodes": episodes,
            "summary": {"episodes": len(episodes), "recomputed": len(verified), "stored_equal": sum(e["stored_equal"] for e in verified),
                        "skipped": len(episodes) - len(verified)}}


def check(current, earlier):
    """The folders whose recomputed digest differs from an earlier output, and the folders only one of the two holds."""
    a, b = earlier["episodes"], current["episodes"]
    changed = sorted(k for k in a if k in b and a[k].get("digest") != b[k].get("digest"))
    only = sorted(set(a) ^ set(b))
    return changed, only


def main(argv=None):
    ap = argparse.ArgumentParser(description="recompute stored verdicts with this checkout's verifier")
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--label", default=None)
    ap.add_argument("--check", type=Path, default=None, help="an earlier output to compare the digests with")
    a = ap.parse_args(argv)
    result = run(a.runs, a.label)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    s = result["summary"]
    print(f"{s['episodes']} episode folders, {s['recomputed']} recomputed, {s['stored_equal']} equal to the stored verdict, {s['skipped']} skipped")
    if a.check is not None:
        changed, only = check(result, read(a.check))
        print(f"{len(changed)} digests changed and {len(only)} folders in one output only, against {a.check}")
        for name in changed + only:
            print(name)
        return 1 if changed or only else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
