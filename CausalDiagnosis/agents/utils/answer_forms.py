"""How the result of a tool becomes observations, by the key answer of its row: object, records or text
(CausalDiagnosis/docs/general-agent-plan.md, section Reading answers). The module names no key of the facts: the normal ranges
come from the ledger of the kind. A stub of the freeze; the reading of answers builds it."""


def observations(form, result, ranges):
    """(rows, note or None): the observation rows of a result in the form of its row, each with its identifier, with ranges the
    public normal ranges, field -> (low, high) (Ledger.ranges). The form object gives no row and keeps the result whole; records
    gives one row per value, obs/<label or n>/<field>, marked inside or outside where ranges holds the field; text gives one row
    per answer, obs/<n>, with the declared words that the text holds as whole words. A result that does not have the form gives
    no row and a note that names the tool and the form."""
    raise NotImplementedError
