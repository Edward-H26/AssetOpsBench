"""A thinking clock pinned for the tests whose subject is not the clock. On the rows with the real-time clock the servers
charge the wall seconds between two calls as thinking in the generation time mode, so how fast a test
machine answers, or a pause of the process, would move the clock and change the episode. The pinned clock is the instant
mode of the same row, one cycle per step, so time still passes on every call and on waits, and an episode depends on its
calls alone."""
import dataclasses
import json
from pathlib import Path

PINNED_MODE = "instant"
from CausalDiagnosis.utils.paths import CONFIGS, preset_path


def pinned_preset(folder, name):
    """The path of a copy of the preset in the pinned mode where its row has the real-time clock, under the preset's own name,
    for the driver's --config. A row without that layer keeps its clock."""
    preset = json.loads(preset_path(name).read_text(encoding="utf-8"))
    if "real_time" in preset.get("layers", []):
        preset.setdefault("switches", {})["clock_mode"] = PINNED_MODE
    path = Path(folder) / "pinned_clock" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(preset, indent=2), encoding="utf-8")
    return str(path)


def pinned(config):
    """The configuration in the pinned mode where it charges generation time, as it is elsewhere, for a runner built in the test."""
    return dataclasses.replace(config, clock_mode=PINNED_MODE) if config.clock_mode == "generation_time" else config
