# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of the source tree, whose MIT License text THIRD_PARTY_NOTICES.md at the root of this repository holds.
#
# Adapted from are/simulation/apps/system.py and from Environment.wait_for_next_notification in
# are/simulation/environment.py of facebookresearch/meta-agents-research-environments at commit
# a9fd2edccb4bc7e22fbf45c8482a5fef30a2a5a4. Changes: the time is an integer cycle of a deterministic clock instead of a
# wall clock timestamp, the tool descriptions speak of the cycles of the asset, the rig's by default, the timeout defaults
# to the cycles left before the horizon, and the wait runs the clock one cycle at a time instead of jumping between
# timestamps, since every cycle of the rig runs processes whose events exist only once the cycle has run.
"""The two time tools an agent has, after ARE's System app: get_current_time, which reads the clock and never moves it,
and wait_for_notification, which lets the clock run until the next notification or a timeout. As in ARE, no agent tool
sets or advances time on its own terms (the System app keeps its wait method off the tool list "because the agent should
not be able to modify the time")."""
from dataclasses import dataclass, field

GET_CURRENT_TIME = ("Get the current time without moving the clock: the cycle, the horizon, the budget left, the seconds of wall "
                    "clock time that one cycle stands for and the seconds of thinking that make one cycle. In the generation_time "
                    "clock mode the seconds between your calls, and in the generated_tokens clock mode the seconds_per_token of "
                    "every token you generate between them, add up, and every full think_seconds_per_cycle of them moves the "
                    "clock one cycle before your next call runs. Free, takes no time. Like every answer of this server, it also "
                    "lists the notifications since the previous call.")


def wait_for_notification_description(noun):
    """The description of wait_for_notification for an asset whose short noun is noun."""
    return ("Wait for a specified number of cycles or until the next notification is received, whichever comes first, "
            f"and return what ended the wait. The {noun} runs while it waits, so downtime accrues. Use it only when there "
            "are no other tasks to perform, for example while work orders are open. timeout is the largest number of "
            "cycles to wait, the cycles left before the horizon when it is left empty.")


@dataclass
class WaitForNotificationTimeout:
    time_created: int
    timeout: int = 0
    timeout_cycle: int = field(init=False)

    def __str__(self):
        return f"Wait for notification timeout after {self.timeout} cycles"

    def __post_init__(self):
        self.timeout_cycle = self.time_created + self.timeout


def wait_for_next_notification(wait, now, run_cycle, notified):
    """Run the clock until the next notification or the timeout of `wait`, whichever comes first. now() is the current
    cycle, run_cycle() runs one cycle and returns False when the episode cannot run further (the horizon, or a failure that
    ends the episode), and notified() tells whether a notification is pending, one that arose during the agent's thinking
    time included, which ends the wait at once as a queued notification does in ARE. Returns "notification", "timeout" or
    "stopped"."""
    while not notified():
        if now() >= wait.timeout_cycle:
            return "timeout"
        if not run_cycle():
            return "stopped"
    return "notification"
