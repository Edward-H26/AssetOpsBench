from collections import namedtuple


PREDICTIONS_NAME = "predictions.json"
UNIT = "cycle"
Released = namedtuple("Released", "raw view episode status events")


class NoEpisode(Exception):
    """A tool needs released evidence that this server does not have."""


def observation_ids(readings):
    return [f"obs/{r.n}/{field}" for r in readings for field in sorted(r.values)]


def order_ids(closures):
    return [f"wo/{c.id}" for c in closures]
