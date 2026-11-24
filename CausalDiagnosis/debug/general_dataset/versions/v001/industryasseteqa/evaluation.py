"""Structural and released-provenance checks, separate from answer quality."""

import math
import re
from decimal import Decimal, InvalidOperation


NUMBER = re.compile(r"(?<![\w.])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?!\w)")


def direct_numeric_match(answer, reference):
    """Match a single numeric direct answer at the reference's displayed precision.

    Decimal equality allows equivalent spellings, with no chosen tolerance. A
    reference without exactly one number is outside this metric. Multiple numbers
    in a prediction are refused rather than selecting whichever matches gold.
    """
    expected_text = reference.get("direct_answer", "")
    expected = NUMBER.findall(expected_text) if isinstance(expected_text, str) else []
    if len(expected) != 1:
        return None
    text = answer.get("direct_answer", "") if isinstance(answer, dict) else ""
    values = NUMBER.findall(text) if isinstance(text, str) else []
    try:
        return len(values) == 1 and Decimal(values[0]) == Decimal(expected[0])
    except InvalidOperation:
        return False


def check_answer(answer, fact):
    """Check the answer envelope and citations against the episode's own fact.

    These checks do not score semantic correctness or reproduce upstream metrics.
    """
    structure, provenance = [], []
    if not isinstance(answer, dict):
        return {"structure_ok": False, "provenance_ok": False,
                "structure_issues": ["answer must be an object"], "provenance_issues": []}
    for field in ("direct_answer", "reasoning_answer"):
        if not isinstance(answer.get(field), str) or not answer[field].strip():
            structure.append(f"{field} must be a nonempty string")
    confidence = answer.get("confidence")
    if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
            or not 0 <= confidence <= 1 or not math.isfinite(confidence)):
        structure.append("confidence must be a finite number between zero and one")
    cited = answer.get("provenance")
    if not isinstance(cited, dict):
        structure.append("provenance must be an object")
        provenance.append("no provenance object")
    else:
        if cited.get("fact_id") != fact["fact_id"]:
            provenance.append("fact_id must name the fact of this question")
        features = cited.get("features")
        if not isinstance(features, list) or not features or any(not isinstance(name, str) for name in features):
            structure.append("provenance.features must be a nonempty list of strings")
            provenance.append("no valid feature citations")
        else:
            allowed = {feature["name"] for feature in fact["features"]}
            if any(name not in allowed for name in features):
                provenance.append("a cited feature is absent from this fact")
        if "file" in cited and ("source_file" not in fact or cited["file"] != fact["source_file"]):
            provenance.append("file disagrees with source_file")
        if "row" in cited and (isinstance(cited["row"], bool) or "row_index" not in fact
                               or cited["row"] != fact["row_index"]):
            provenance.append("row disagrees with row_index")
    return {"structure_ok": not structure, "provenance_ok": not provenance,
            "structure_issues": structure, "provenance_issues": provenance}
