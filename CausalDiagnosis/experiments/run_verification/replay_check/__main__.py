"""python -m replay_check: calls main of the submodule that holds it."""
import sys
from . import command_line


if __name__ == "__main__":
    sys.exit(command_line.main(sys.argv[1:]))
