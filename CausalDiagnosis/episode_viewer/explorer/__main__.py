"""python -m CausalDiagnosis.episode_viewer.explorer serve|export: calls main of cli.py, which holds both commands."""
import sys

from .cli import main


if __name__ == "__main__":
    main(sys.argv[1:])
