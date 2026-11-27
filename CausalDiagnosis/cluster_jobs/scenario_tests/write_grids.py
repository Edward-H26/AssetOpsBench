"""Write one three-agent development grid per scenario; no model calls or job submissions."""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def write(out):
    manifest = json.loads((HERE / "manifest.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    for scenario in manifest["scenarios"]:
        lines = []
        for mode in manifest["modes"]:
            words = ["--config", Path(scenario["config"]).stem, "--environment-mode", scenario["environment_mode"],
                     "--servers", "aob", "--store", "couchdb", "--agent", mode["agent"]]
            if not mode["diagnostics"]:
                words += ["--no-diagnostics"]
            if scenario["scenario_id"] is not None:
                words += ["--scenario-ids", str(scenario["scenario_id"])]
            lines.append(" ".join(words))
        target = out / f"local_environment_scn{scenario['scenario']}.txt"
        target.write_text("\n".join(lines) + "\n")
        print(f"scenario={scenario['scenario']} mode={scenario['environment_mode']} first_seed={scenario['seed']} grid={target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    write(parser.parse_args().out)
