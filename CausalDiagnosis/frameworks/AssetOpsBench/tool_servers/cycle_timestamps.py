"""The cycle clock of an episode as AssetOpsBench's documents spell it.

A record of cycle c carries the timestamp epoch + c * seconds_per_cycle as a naive ISO date time, since the iot server
refuses bounds whose timezone awareness differs from the records, and one format per episode, so that string order equals
time order across the prehistory at negative cycles. wo stamps its documents through a
conversion to UTC that reads a naive datetime as local time, so wo gets the same instant as an aware UTC datetime.
"""
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

_TOLERANCE = 1e-9  # cycles; a bound on a record's own timestamp must not round past it


@dataclass(frozen=True)
class Timeline:
    epoch: datetime
    seconds_per_cycle: float

    def __post_init__(self):
        if self.epoch.tzinfo is not None:
            raise ValueError("the epoch is a naive datetime, as the records' timestamps are")
        if not self.seconds_per_cycle > 0:
            raise ValueError("seconds_per_cycle must be positive")

    def at(self, cycle):
        return self.epoch + timedelta(seconds=self.seconds_per_cycle * cycle)

    def utc(self, cycle):
        """The cycle's instant as wo needs it, the record timestamp read as UTC."""
        return self.at(cycle).replace(tzinfo=timezone.utc)

    def timestamp(self, cycle):
        """Whole seconds when every cycle of the episode falls on one, microseconds otherwise."""
        whole = float(self.seconds_per_cycle).is_integer() and self.epoch.microsecond == 0
        return self.at(cycle).strftime("%Y-%m-%dT%H:%M:%S" if whole else "%Y-%m-%dT%H:%M:%S.%f")

    def cycle_of(self, moment):
        """The cycle whose record carries this naive timestamp, or the fraction between two cycles."""
        return (moment - self.epoch).total_seconds() / self.seconds_per_cycle

    def first_cycle_from(self, moment):
        """The first cycle whose timestamp is at or after the moment, which turns history's inclusive start and
        exclusive end into cycle bounds."""
        return math.ceil(self.cycle_of(moment) - _TOLERANCE)
