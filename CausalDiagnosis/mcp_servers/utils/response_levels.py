"""The levels of the parameter response_format that the servers of this project share: the level a call asks for, the text of a
shorter level as JSON without indentation, and the key that opens a repeated answer. A repeated answer is one whose text an
earlier call of the same tool in the episode returned; the transcript of the episode (transcript.py) holds the hash of every
answer, so a process of a server per call, as the Plan-Execute runner starts, finds it as a long lived one does, and a replay of
the episode, which writes the transcript again in order, writes the same key."""
import json

from CausalDiagnosis.mcp_servers.utils import transcript

LEVELS = ("headline", "concise", "detailed", "raw")
DEFAULT_LEVEL = "concise"
UNCHANGED = "unchanged since your call at step {}"


def level_of(response_format):
    """The level a call asks for, the default for an empty string as branch_detail reads it, or None for any other value."""
    level = response_format.strip() or DEFAULT_LEVEL if isinstance(response_format, str) else None
    return level if level in LEVELS else None


def level_error():
    return f"response_format must be one of {', '.join(LEVELS)}"


def text_of(answer, ensure_ascii=True):
    return json.dumps(answer, separators=(",", ":"), ensure_ascii=ensure_ascii)


def first_step(folder, server, tool, text, key="answer_sha256"):
    """The step of the first call of the tool in the transcript of the episode folder whose answer, under key of its line, was
    this text, counting every call of every server of the episode, one step per tool call, or None."""
    if folder is None:
        return None
    lines, _ = transcript.read_since(folder, 0)
    calls = [line for line in lines if line.get("kind") == "call"]
    sha = transcript.text_sha256(text)
    return next((k for k, line in enumerate(calls, start=1)
                 if line.get("server") == server and line.get("tool") == tool and line.get(key) == sha), None)


def marked(answer, step):
    """A repeated answer: the same answer, opened by the key unchanged, so that its text stays one JSON object."""
    return {"unchanged": UNCHANGED.format(step), **answer}
