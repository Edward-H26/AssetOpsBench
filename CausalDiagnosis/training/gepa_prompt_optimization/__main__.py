"""python -m gepa_prompt_optimization: calls main of the submodule that holds it."""
import sys
from . import run_loop


if __name__ == "__main__":
    run_loop.main(sys.argv[1:])
