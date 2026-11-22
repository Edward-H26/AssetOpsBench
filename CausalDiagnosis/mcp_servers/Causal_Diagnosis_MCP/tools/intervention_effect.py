import math
import sys
from collections import namedtuple


MONITOR = "monitor"
REPLACE = "replace:"
# E13 and E14 of one action, with the number of resampled fits its recovery interval is taken over
Predicted = namedtuple("Predicted", "after addresses recovery low high band summaries combination mass fits")
# the mass that the belief after the reading must put on the predicted combination for a consistent verdict to confirm a repair, a
# declared threshold of the server like the error level of the verdict
CONFIRMATION_MASS = 0.95
# the least likelihood ratio, of the readings that verify the repair for the predicted combination against its strongest rival, that
# confirms it: 1 asks only that the readings do not favor the rival
CONFIRMATION_RATIO = 1.0


def reading_ratio(before, after, predicted):
    """(the strongest rival, the likelihood ratio of the readings between the two beliefs for the predicted combination against it).
    The rival is the combination other than the predicted one with the most mass in the belief after the readings, a tie going to
    the first of the belief, and the ratio is the posterior odds of the predicted combination against it after the readings divided
    by their odds before them. The ratio is None where one of the four masses is zero, since no odds follow, and a ratio beyond
    the largest float is that float."""
    rival = max((h for h in after if h != predicted), key=after.get)
    masses = (after[predicted], after[rival], before[predicted], before[rival])
    if not all(mass > 0.0 for mass in masses):
        return rival, None
    log_ratio = math.log(masses[0]) - math.log(masses[1]) - math.log(masses[2]) + math.log(masses[3])
    try:
        return rival, math.exp(log_ratio)
    except OverflowError:
        return rival, sys.float_info.max


def confirms(consistent, mass, ratio):
    """Whether a verified repair is confirmed: its verdict consistent, the belief after the readings putting at least
    CONFIRMATION_MASS on the predicted combination, and the readings that verify the repair having at least CONFIRMATION_RATIO for
    it against its strongest rival (reading_ratio). A ratio that is undefined confirms nothing, and an inconsistent verdict refutes the
    prediction whatever the mass and the ratio."""
    return consistent is True and mass >= CONFIRMATION_MASS and ratio is not None and ratio >= CONFIRMATION_RATIO
