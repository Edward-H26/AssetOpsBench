"""Read declared observation forms without asset field names or private state."""

import math


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _records(result, ranges):
    served = result.get("served") if isinstance(result, dict) else result
    if not isinstance(served, list):
        return [], "records form needs a list of served records"
    rows = []
    for record in served:
        if not isinstance(record, dict) or not _number(record.get("n")) or not _number(record.get("t")):
            return [], "records form needs a number and a time on each record"
        values = record.get("values")
        label = record.get("label", record["n"])
        if isinstance(label, bool) or not isinstance(label, (str, int, float)) or not str(label).strip():
            return [], "records form needs a nonempty label when a label is present"
        if not isinstance(label, str) and not _number(label):
            return [], "records form needs a finite numeric label"
        if not isinstance(values, dict) or not all(isinstance(key, str) and key.strip() and _number(value)
                                                   for key, value in values.items()):
            return [], "records form needs named finite numeric values"
        for field, value in values.items():
            row = {"id": f"obs/{label}/{field}", "n": record["n"], "t": record["t"],
                   "field": field, "value": value}
            band = ranges.get(field)
            if isinstance(band, (list, tuple)) and len(band) == 2 and all(_number(bound) for bound in band):
                low, high = band
                row.update(inside_band=low <= value <= high, band_low=low, band_high=high)
            rows.append(row)
    return rows, None


def observations(form, result, ranges):
    """Return observation rows and an optional form error, using public ranges."""
    if form == "object":
        return ([], None) if isinstance(result, dict) else ([], "object form needs a JSON object")
    if form == "records":
        return _records(result, ranges if isinstance(ranges, dict) else {})
    if form == "text":
        text = result.get("text") if isinstance(result, dict) else result
        if not isinstance(text, str):
            return [], "text form needs text or an object carrying text"
        number = result.get("n") if isinstance(result, dict) else None
        if number is not None and not _number(number):
            return [], "text form needs a finite answer number when a number is present"
        return [{"text": text, **({"n": number} if number is not None else {})}], None
    return [], f"unknown answer form {form!r}"
