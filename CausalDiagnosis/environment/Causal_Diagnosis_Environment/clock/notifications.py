# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of the source tree, whose MIT License text THIRD_PARTY_NOTICES.md at the root of this repository holds.
#
# Adapted from are/simulation/notification_system.py of facebookresearch/meta-agents-research-environments at commit
# a9fd2edccb4bc7e22fbf45c8482a5fef30a2a5a4. Changes: the message carries the cycle of a deterministic clock instead of a
# datetime and no attachments, the user message type is left out since the rig has no user channel, the message queue is
# replaced by the notifications field that every tool answer carries, and the timeout message counts cycles.
"""The notification messages of the hydraulic rig, after ARE's notification system: an environment notification for a
public event of the rig and a stop message when the episode ends. ARE delivers messages by a pull at the start of each
agent step, while the AssetOpsBench frameworks have no message queue, so every tool answer of the action server carries
the messages since the previous call."""
from dataclasses import dataclass
from enum import Enum


class MessageType(Enum):
    ENVIRONMENT_NOTIFICATION = "ENVIRONMENT_NOTIFICATION"
    ENVIRONMENT_STOP = "ENVIRONMENT_STOP"


@dataclass
class Message:
    message_type: MessageType
    message: str
    t: int  # the cycle of the event, ARE's timestamp

    def as_dict(self, **fields):
        """The message as the tool answer lists it, with the structured fields of its event."""
        return {"message_type": self.message_type.value, "t": self.t, "message": self.message, **fields}


def timeout_message(wait, t):
    """The message that tells the agent a wait reached its timeout, which ARE adds only when no other notification arrived
    in the same tick (handle_timeout_after_events), so the caller adds it only to a wait that ended by its timeout."""
    return Message(MessageType.ENVIRONMENT_NOTIFICATION, f"Wait for notification timeout reached after {wait.timeout} cycles", t)
