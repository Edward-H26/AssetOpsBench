import csv
import json
from pathlib import Path


def write_rows(path, rows, fieldnames, lineterminator="\r\n"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), restval="", lineterminator=lineterminator)
        writer.writeheader()
        writer.writerows(rows)


def flat(value):
    """A CSV cell: dictionaries and lists as JSON, None as an empty string, everything else as it is."""
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, separators=(",", ":"), default=str)
    return value


def as_bool(value):
    """A CSV cell read back as a flag; empty cells are None."""
    if value is None or value == "":
        return None
    return str(value).strip().lower() in ("true", "1", "yes")


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
