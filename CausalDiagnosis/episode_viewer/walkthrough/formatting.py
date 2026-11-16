PARTS = ("1. Start", "2. Steps", "3. What happened in the environment that the agent could not see", "4. Verdict",
         "5. In plain words")


# ----- formatting -----
def num(value):
    """A value as the files hold it, integers as they are, other numbers at six significant digits and flags in words."""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:.6g}"
    return str(value)


def prob(value):
    return "none" if value is None else f"{value:.3f}"


def plural(n, word, words=None):
    """A count as num prints it with its noun, the singular only when the printed count is 1."""
    text = num(n)
    return f"{text} {word if text == '1' else words or word + 's'}"


def effective_count(n):
    return plural(n, "effective hypothesis", "effective hypotheses")


def quoted(text, limit=None):
    """File text in quotation marks, on one line, cut at the limit."""
    text = " ".join(str(text).split())
    if limit and len(text) > limit:
        text = text[:limit - 3] + "..."
    return f'"{text}"'


def joined(items):
    """Items as a sentence lists them, as in a, b and c."""
    items = [str(i) for i in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1] if items else ""


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    return lines + ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]


def levels_of(combination, components):
    """The levels of a combination by component, from a dict or from a list in component order."""
    if isinstance(combination, dict):
        return {c: combination.get(c) for c in components}
    return dict(zip(components, combination or []))


def short_levels(combination, components):
    """The compact form of parts 2 and 5, (3, 80, 0, 130), the levels in component order."""
    levels = levels_of(combination, components)
    return "(" + ", ".join(num(levels.get(c)) for c in components) + ")"


def long_levels(combination, components):
    """The written out form of parts 3 and 4, cooler 3, valve 80, pump 0, accumulator 130."""
    levels = levels_of(combination, components)
    return ", ".join(f"{c} {num(levels.get(c))}" for c in components)


def tag(field):
    """The channel of a served field, such as CE for CE_CoolingEfficiency_percent_1Hz_mean."""
    return field.split("_")[0]


def outside_band(field, value, bands):
    band = bands.get(field)
    return band is not None and not band["low"] <= value <= band["high"]


def id_list(value):
    items = value.split(",") if isinstance(value, str) else (value or [])
    return [str(i).strip() for i in items if str(i).strip()]
