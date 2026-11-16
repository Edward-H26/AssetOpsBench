"""uv.lock and the package group rl of pyproject.toml (CausalDiagnosis/docs/environment.md, section Gym and OpenEnv): the group
adds packages and moves no pin outside it. Every package of the lock before the group (data/locked_pins_outside_rl.tsv) keeps
its version, every package added since is reached from the group alone, and every exact pin of [tool.uv]
constraint-dependencies holds in the lock, so a later lock run that moves a pin of the registered runs fails here."""
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
PINS = Path(__file__).resolve().parent / "data" / "locked_pins_outside_rl.tsv"

if not (ROOT / "uv.lock").exists():
    pytest.skip("no uv.lock in this tree", allow_module_level=True)


def locked():
    return {p["name"]: p for p in tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))["package"]}


def reached(packages, names):
    """The packages of the lock that the named packages need, themselves included."""
    seen, todo = set(), list(names)
    while todo:
        name = todo.pop()
        if name not in seen and name in packages:
            seen.add(name)
            todo.extend(d["name"] for d in packages[name].get("dependencies", []))
    return seen


def test_the_rl_group_moves_no_pin_outside_it():
    packages = locked()
    before = dict(line.split("\t") for line in PINS.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#"))
    moved = sorted(f"{name} {version} to {packages[name]['version']}" for name, version in before.items()
                   if name in packages and packages[name]["version"] != version)
    removed = sorted(name for name in before if name not in packages)
    root = next(p for p in packages.values() if p.get("source", {}).get("editable") == ".")
    group = reached(packages, [d["name"] for d in root["dev-dependencies"]["rl"]])
    stray = sorted(name for name in set(packages) - set(before) if name not in group)
    assert not moved and not removed and not stray, {"moved": moved, "removed": removed, "added outside the group": stray}


def test_the_exact_constraints_hold_in_the_lock():
    packages = locked()
    constraints = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["uv"]["constraint-dependencies"]
    pins = [c.split("==") for c in constraints if "==" in c]
    assert pins and all(packages[name]["version"] == version for name, version in pins), pins
