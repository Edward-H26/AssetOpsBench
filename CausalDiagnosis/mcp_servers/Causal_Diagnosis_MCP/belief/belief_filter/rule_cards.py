import itertools
import math
import numpy as np


# ----- the rule cards of the asset view -----
class Rules:
    """The rule parameters of the asset view as arrays over the component index: degradation positions, dependencies,
    priorities, the emergency hold, the hazard rates, the schedule and the history rule."""

    def __init__(self, view, space):
        self.space = space
        self.components = space.components
        self.J = len(self.components)
        order = [view.degradation[c] for c in self.components]
        self.worst = np.array([len(o) - 1 for o in order])
        # position in the degradation order -> index of the level in the space, and back
        self.level_index_of_position = [np.array([space.levels[j].index(level) for level in o]) for j, o in enumerate(order)]
        self.position_of_level = [{level: p for p, level in enumerate(o)} for o in order]
        sizes = [len(levels) for levels in space.levels]
        self.strides = np.array([int(np.prod(sizes[j + 1:])) for j in range(self.J)])
        self.deps = [[self.components.index(d) for d in view.dependencies(c)] for c in self.components]
        self.priority = [view.priority(c) for c in self.components]
        self.downtime = [view.repair_downtime(c) for c in self.components]
        processes = view.processes
        self.hold = int((processes.get("emergency") or {}).get("hold", 0))
        hazard = processes.get("hazard") or {}
        self.stochastic = hazard.get("mode") == "stochastic"
        self.collateral = float(hazard.get("collateral", 0.0))
        base, factor = float(hazard.get("base_rate", 0.0)), float(hazard.get("depth_factor", 0.0))
        self.hazard_rate = [np.array([min(1.0, base * (1 + factor * p)) if p < self.worst[j] else 0.0 for p in range(self.worst[j] + 1)])
                            for j in range(self.J)]
        schedule = processes.get("schedule") or {}
        self.scheduled = schedule.get("mode") == "seeded" and int(schedule.get("components", 0)) > 0
        self.schedule_count = int(schedule.get("components", 0)) if self.scheduled else 0
        self.first_drop = tuple(schedule.get("first_drop") or (0, 0))
        self.interval = int(schedule.get("interval", 1))
        self.history_rule = processes.get("fix_history") or {}

    def positions(self, combo):
        return [self.position_of_level[j][v] for j, v in enumerate(combo)]

    def combo_index(self, pos):
        """The index in space.combinations of every row of positions (n, J)."""
        idx = np.zeros(len(pos), dtype=int)
        for j in range(self.J):
            idx += self.level_index_of_position[j][pos[:, j]] * self.strides[j]
        return idx

    def history_log_likelihood(self, combo, history):
        """log P(released history | combination at the start) under the history rule: one entry or none per component,
        an entry closed a cycles before the start with probability p over the width of the window of the component's kind,
        no entry with probability one minus p."""
        total = 0.0
        for j, c in enumerate(self.components):
            kind = "healthy" if combo[j] == self.space.healthy[j] else "degraded"
            p = float(self.history_rule.get(f"{kind}_probability", 0.0))
            low, high = self.history_rule.get(f"{kind}_ago", (0, -1))
            entries = [e for e in history if e.get("component") == c]
            if not entries:
                factor = 1.0 - p
            elif len(entries) == 1 and low <= -int(entries[0]["closed_t"]) <= high:
                factor = p / (high - low + 1)
            else:
                factor = 0.0
            total += math.log(factor) if factor > 0 else -math.inf
        return total

    def schedule_draws(self, combo):
        """[(log probability, ((component, first drop, start position), ...))] of the schedule rule's draw for a start: the
        components healthy at the start first, uniformly, then the faulty ones, each with a first drop uniform in the
        window, as draw_schedule draws them."""
        if not self.scheduled:
            return [(0.0, ())]
        healthy = [j for j in range(self.J) if combo[j] == self.space.healthy[j]]
        faulty = [j for j in range(self.J) if combo[j] != self.space.healthy[j]]
        m = min(self.schedule_count, len(healthy))
        rest = self.schedule_count - m
        subsets = [a + b for a in itertools.combinations(healthy, m) for b in itertools.combinations(faulty, rest)]
        if not subsets:  # the card asks for more components than the asset has, so no draw exists to weigh
            raise ValueError(f"the schedule card draws {self.schedule_count} components, and the asset has {self.J}")
        low, high = self.first_drop
        width = high - low + 1
        log_subset = -math.log(len(subsets))
        pos = self.positions(combo)
        draws = []
        for subset in subsets:
            for taus in itertools.product(range(low, high + 1), repeat=len(subset)):
                draws.append((log_subset - len(subset) * math.log(width), tuple((j, tau, pos[j]) for j, tau in zip(subset, taus))))
        return draws
