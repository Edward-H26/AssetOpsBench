def n_cycles(n):
    """A count of cycles in its number, the singular only for one: 1 cycle, 0 cycles, 10 cycles."""
    return f"{n} cycle" if n == 1 else f"{n} cycles"


def fmt(value, digits=3):
    """A number for the board: masses and probabilities at a few digits, prices and cycles as they are."""
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}" if abs(value) < 10 else f"{value:g}"
    return str(value)


def table(headers, rows):
    """A pipe table; cells are stringified as given."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def cycles_text(cycles):
    """Sorted cycles as their runs, such as -2 to 3, 5, so a gap is never shown as read."""
    runs = []
    for cycle in cycles:
        if runs and cycle == runs[-1][1] + 1:
            runs[-1][1] = cycle
        else:
            runs.append([cycle, cycle])
    return ", ".join(f"{a} to {b}" if a != b else str(a) for a, b in runs)


def combination_text(combination):
    return ", ".join(f"{c} {v}" for c, v in combination.items())


def scored_combination_text(verdict):
    """The most probable combination after the repair, which the verdict scores the reading against, with its mass,
    or nothing when the result names none, as an error does."""
    combination = verdict.get("predicted_combination")
    if not combination:
        return ""
    return f", predicted combination ({combination_text(combination)}), the most probable after the repair, with mass {fmt(verdict.get('predicted_mass'))}"


def confirmation_text(verdict):
    """Whether the belief after the verifying reading confirms the predicted combination, with the mass it puts there and, when it
    does not, its most probable combination with that mass and, where the verifying readings favor its strongest rival, that
    ratio; nothing for a verdict that carries no confirmation, as one of a monitor prediction or of an earlier version does."""
    confirmed = verdict.get("confirmed")
    if confirmed is None:
        return ""
    said = f"the belief now puts {fmt(verdict.get('predicted_posterior'))} on the predicted combination"
    if confirmed:
        return f", confirmed: {said}"
    top = combination_text(verdict.get("belief_top_combination") or {})
    ratio = verdict.get("reading_ratio")
    # a likelihood ratio below 1 is a reading that favors the rival, whatever ratio the server confirms at
    favored = f", and the verifying readings favor its strongest rival over it, likelihood ratio {ratio:.3g}" if ratio is not None and ratio < 1 else ""
    return f", not confirmed: {said} and {fmt(verdict.get('belief_top_mass'))} on its most probable combination ({top}){favored}"
