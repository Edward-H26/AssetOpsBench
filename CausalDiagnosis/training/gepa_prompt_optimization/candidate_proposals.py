import re
from pathlib import Path
from CausalDiagnosis.utils.json_files import read_json


BLOCK = "task_block"
MUTABLE = (BLOCK,)  # the components a child changes, one per child and in turn, as GEPA's round robin over modules
FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
NO_BLOCK = "the reply holds no complete block within ```"
CUT_REPLY = "the reply reached the token limit before its block closed, so the new text must be shorter"
# The meta prompt of GEPA, appendix C of the GEPA paper, word for word, with the closing paragraph of this protocol.
PROMPT_META = """I provided an assistant with the following instructions to perform a task for me:

```
{current}
```

The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:

```
{examples}
```

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.

The assistant keeps its own instructions, and the new instruction is appended to the task it receives, after an empty line. An empty block above means that the assistant has received no such instruction so far. Provide the new instruction within ``` blocks."""
RETRY_HEADING = "Your previous reply was refused for the reasons below. Return a corrected reply within ``` blocks."


def seed_texts():
    """The texts of the starting candidate, the empty block, under which the agent plays without a block."""
    return {BLOCK: ""}


def fenced(reply):
    """The text of the last complete block within ``` of a reply, or None."""
    blocks = FENCE.findall(reply or "")
    return blocks[-1].strip() if blocks else None


def proposal(parent, component, reply):
    """(texts, problems) of the child that a reply proposes, the texts being the parent's with the reply's block in place of the
    component, and the problems every reason that refuses the child."""
    block = fenced(reply)
    if block is None:
        return None, [NO_BLOCK]
    texts = {**parent["texts"], component: block}
    problems = []
    if block == parent["texts"][component].strip():
        problems.append(f"the reply repeats the current {component.replace('_', ' ')} unchanged")
    return texts, problems


def agent_outputs(ep_dir):
    """What the agent wrote in an episode, read alike for the three agents from the run folder: its written plan, plan.txt of
    Plan-Execute and of the CausalDiagnosis agent with every plan, its source and the reasons of a refused or unparsed reply, or
    the supervisor's steps of Agent-As-Tool, then the closing answer."""
    ep_dir = Path(ep_dir)
    written = _text(ep_dir / "plan.txt") or supervisor_steps(ep_dir)
    return f"plan:\n{written}\nsummary: {_text(ep_dir / 'answer.txt')}"


def supervisor_steps(ep_dir):
    """The ReAct steps of the Agent-As-Tool supervisor: its thought, the sub-agent it called or Finish, the request or the final
    answer, and whether ReActXen took the step as valid, from react_trajectory.json under ReActXen's own key trajectroy_log. The
    sub-agents' replies stay out, as the tool results of the other two agents do, since the feedback card shows the tool calls.
    A run that raised before the file was written leaves only its model calls, so the supervisor's replies stand in."""
    steps = (read_json(ep_dir / "react_trajectory.json", {}) or {}).get("trajectroy_log")
    if steps is not None:
        return "\n".join(f"step {s.get('step')}: {s.get('thought') or ''}\n  {s.get('action') or ''}: {s.get('action_input') or ''} "
                         f"({s.get('state') or ''})" for s in steps)
    calls = read_json(ep_dir / "llm_calls.json", []) or []
    return "\n".join(f"{c.get('kind')}: {c.get('reply') or ''}" for c in calls if c.get("agent") == "supervisor")


def _text(path):
    return Path(path).read_text(encoding="utf-8") if Path(path).exists() else ""
