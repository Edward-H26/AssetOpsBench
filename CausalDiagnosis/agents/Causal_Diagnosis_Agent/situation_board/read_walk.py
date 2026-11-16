from . import asset_rules


class ReadWalk:
    """Reads taken one after the other with every paid cycle counted once. The walk keeps, per sensor group, the cycles it has
    delivered to that group through a read of the group or of a group that covers it, and the price, the duration and the tool
    calls of its reads together. On a row that records ahead of its reads a read takes the newest recorded cycles that its
    group has not paid for and that no earlier read of the walk delivered to it, as many as it asks for, counted as
    Board.new_cycles counts them, so a cycle paid earlier in the walk through a covering group is not paid again. On a row that
    records a cycle only by playing it every cycle of a read is new. The duration of a read is Board.read_duration."""

    def __init__(self, board):
        self.board = board
        self.price = self.duration = 0
        self.calls = 0
        self.new = 0  # the cycles that the last read paid for anew
        self._delivered = {}  # sensor group -> the cycles delivered to it: a set of cycles, or a count where every cycle is new

    def take(self, group, cycles, forward=False):
        """The price and the duration of the next read of the walk, a read of the group for the cycles it asks for; a forward read
        (Board.reads_forward) plays and pays every cycle it asks for."""
        board = self.board
        covered = [g for g in board.profile.get("sensor_groups") or {} if board.covers(group, g)]
        unpaid = board.unpaid_cycles(group, cycles)
        if unpaid is None:
            self.new = cycles
            for g in covered:
                self._delivered[g] = self._delivered.get(g, 0) + cycles
        elif forward or (not unpaid and board.forward_read):  # a forward window plays its cycles at the declared read duration
            self.new = cycles
            price = cycles * asset_rules.read_prices(board.profile).get(group, 0)
            duration = cycles * board.forward_read
            self.price, self.duration, self.calls = self.price + price, self.duration + duration, self.calls + 1
            return price, duration
        elif not unpaid and board.record_grows:  # the open window, priced at the most it may serve (Board.new_cycles)
            self.new = cycles
        else:
            fresh = [c for c in unpaid if c not in self._delivered.get(group, set())][:cycles]
            self.new = len(fresh)
            for g in covered:
                self._delivered.setdefault(g, set()).update(fresh)
        price = self.new * asset_rules.read_prices(board.profile).get(group, 0)
        duration = board.read_duration(cycles)
        self.price += price
        self.duration += duration
        self.calls += 1
        return price, duration

    def delivered(self, group):
        """The cycles the walk has delivered to the group, through a read of it or of a group that covers it."""
        cycles = self._delivered.get(group, 0)
        return cycles if isinstance(cycles, int) else len(cycles)
