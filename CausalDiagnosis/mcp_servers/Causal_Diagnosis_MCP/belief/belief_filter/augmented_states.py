import numpy as np


# ----- the augmented states -----
class States:
    """Rows of augmented states: degradation positions (n, J), entry labels (n,), the schedule draw (n, S, 3), hold
    positions and expiry cycles (n, J), log weights (R, n) per fit, and payload arrays copied with the rows."""

    def __init__(self, rules, pos, entry, sched, hold_pos, hold_exp, logw, payload=None):
        self.rules = rules
        self.pos, self.entry, self.sched = pos, entry, sched
        self.hold_pos, self.hold_exp = hold_pos, hold_exp
        self.logw = logw
        self.payload = payload or {}
        self.changed = np.zeros(len(pos), dtype=bool)

    def __len__(self):
        return len(self.pos)

    def take(self, rows):
        taken = States(self.rules, self.pos[rows].copy(), self.entry[rows].copy(), self.sched[rows].copy(), self.hold_pos[rows].copy(),
                       self.hold_exp[rows].copy(), self.logw[:, rows].copy(), {k: v[:, rows, :] for k, v in self.payload.items()})
        taken.changed = self.changed[rows].copy()
        return taken

    def extend(self, other):
        self.pos = np.concatenate([self.pos, other.pos])
        self.entry = np.concatenate([self.entry, other.entry])
        self.sched = np.concatenate([self.sched, other.sched])
        self.hold_pos = np.concatenate([self.hold_pos, other.hold_pos])
        self.hold_exp = np.concatenate([self.hold_exp, other.hold_exp])
        self.logw = np.concatenate([self.logw, other.logw], axis=1)
        self.changed = np.concatenate([self.changed, other.changed])
        for k in self.payload:
            self.payload[k] = np.concatenate([self.payload[k], other.payload[k]], axis=1)

    def clear_hold(self, rows, j):
        self.hold_pos[rows, j] = -1
        self.hold_exp[rows, j] = -1

    def set_position(self, rows, j, value):
        """Set the position of component j on the given rows and mark the rows whose position changed."""
        before = self.pos[rows, j].copy()
        self.pos[rows, j] = value
        self.changed[rows] |= before != self.pos[rows, j]

    def branch(self, rows, probability):
        """Split the given rows: a copy with weight times probability is returned, and the rows keep one minus it."""
        probability = np.broadcast_to(np.asarray(probability, dtype=float), (len(rows),))
        copy = self.take(rows)
        with np.errstate(divide="ignore"):
            copy.logw += np.log(probability)[None, :]
            self.logw[:, rows] += np.log1p(-probability)[None, :]
        return copy

    def merge_key(self):
        """One integer per row that is equal exactly for rows with equal (h, e, m) and an equal changed flag, which marks a
        row whose combination changed in the cycle being run and whose offset restarts when the cycle ends."""
        columns = np.concatenate([self.pos, self.entry[:, None], self.sched.reshape(len(self), -1), self.hold_pos, self.hold_exp,
                                  self.changed[:, None]], axis=1).astype(np.int64)
        low = columns.min(axis=0)
        span = columns.max(axis=0) - low + 1
        if np.prod(span.astype(float)) < 2.0 ** 62:
            radix = np.concatenate([[1], np.cumprod(span[:-1])]).astype(np.int64)
            return (columns - low) @ radix
        return np.unique(columns, axis=0, return_inverse=True)[1].reshape(-1)

    def merge(self):
        """Merge the rows with equal states: log weights add up per fit, and the payload of the first row is kept, which is
        equal across the merged rows, or is restarted when the cycle ends for rows whose combination changed. Rows of zero
        weight under every fit are dropped."""
        alive = np.isfinite(self.logw).any(axis=0)
        if not alive.all():
            self.__dict__.update(self.take(np.nonzero(alive)[0]).__dict__)
        unique, first, inverse = np.unique(self.merge_key(), return_index=True, return_inverse=True)
        if len(unique) < len(self):
            order = np.argsort(inverse, kind="stable")
            starts = np.r_[0, np.nonzero(np.diff(inverse[order]))[0] + 1]
            logw = np.logaddexp.reduceat(self.logw[:, order], starts, axis=1)
            merged = self.take(first)
            merged.logw = logw
            self.__dict__.update(merged.__dict__)
