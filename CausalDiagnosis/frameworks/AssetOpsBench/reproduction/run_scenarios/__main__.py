"""python -m run_scenarios: calls main of the submodule that holds it."""
import sys
from . import command_line


if __name__ == "__main__":
    command_line.main(sys.argv[1:])
