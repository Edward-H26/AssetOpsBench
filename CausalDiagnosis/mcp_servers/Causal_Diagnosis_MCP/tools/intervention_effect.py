from collections import namedtuple


MONITOR = "monitor"
REPLACE = "replace:"
# E13 and E14 of one action, with the number of resampled fits its recovery interval is taken over
Predicted = namedtuple("Predicted", "after addresses recovery low high band summaries combination mass fits")
