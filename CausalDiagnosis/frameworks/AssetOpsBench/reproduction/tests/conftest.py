"""The reproduction modules import the fork's src and the rig's drivers by path, as they do when run as scripts, and the
tests run from the fork root with the same paths. This folder has no __init__.py, so pytest imports its modules under their
own names, which the test_aob_ prefix keeps apart from every other test module of the fork."""
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]

from CausalDiagnosis.frameworks.AssetOpsBench.reproduction import run_folder_names, scenario_file, scenario_sets  # noqa: E402,F401  puts src and the rig's drivers on the import path
