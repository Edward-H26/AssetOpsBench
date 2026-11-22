"""Opt-in debug overlay; the normal python -m CausalDiagnosis entry stays unchanged."""
import importlib.util
import hashlib
import json
from pathlib import Path
import runpy
import sys
import subprocess


def install():
    root = next(p for p in Path(__file__).resolve().parents if (p / 'CausalDiagnosis/__main__.py').is_file())
    sys.path.insert(0, str(root))
    # Initialize the package first: its __init__ eagerly imports the controller.
    import CausalDiagnosis.agents.Causal_Diagnosis_Agent  # noqa: F401
    name = "CausalDiagnosis.agents.Causal_Diagnosis_Agent.prompts"
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("prompts_v2.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    protocol_spec = importlib.util.spec_from_file_location('maintenance_debug_protocol', Path(__file__).with_name('protocol.py'))
    protocol = importlib.util.module_from_spec(protocol_spec)
    protocol_spec.loader.exec_module(protocol)
    protocol.install(module)


if __name__ == "__main__":
    install()
    sys.argv = ["CausalDiagnosis", *sys.argv[1:]]
    manifest = {"debug_variant": "maintenance-loop-v4", "official_entry": "CausalDiagnosis",
                "one_proposed_action_per_turn": True, "preserve_pending_steps": True, "native_verifier": True,
                "base_commit": subprocess.check_output(
                    ["git", "-C", str(next(p for p in Path(__file__).resolve().parents if (p / 'CausalDiagnosis/__main__.py').is_file())), "rev-parse", "HEAD"], text=True).strip(),
                "overlay_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in Path(__file__).parent.glob("*.py")}}
    if "--out" in sys.argv:
        output = Path(sys.argv[sys.argv.index("--out") + 1])
        output.mkdir(parents=True, exist_ok=True)
        (output / "debug_variant.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest), flush=True)
    runpy.run_module("CausalDiagnosis", run_name="__main__")
