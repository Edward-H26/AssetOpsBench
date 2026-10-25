"""The question that opens an episode: what docs/hydraulic_rig/environment.md lets the agent know at reset, and nothing else.

The planner, the argument filler and the summariser of AssetOpsBench's Plan-Execute runner all read the question
(docs/hydraulic_rig/action-server.md), so it carries the legal value of every tool parameter. Its wording is part
of the environment definition. Change BRIEFING_VERSION with it, since every run records that version in run.json
(docs/hydraulic_rig/action-server.md, section The briefing).
"""
from .names import tier_fields

BRIEFING_VERSION = "v0.3"


def briefing(observation):
    """Compose the question from the observation that reset(seed) returns."""
    obs = observation
    levels = "; ".join(f"{c}: {', '.join(map(str, obs['levels'][c]))} (healthy {obs['healthy'][c]})" for c in obs["components"])
    tiers = "\n".join(
        f"  - tier {t} costs {obs['prices']['measure'][t]} and returns {', '.join(tier_fields(t))}" for t in sorted(obs["tiers"])
    )
    parts = ", ".join(f"{c} {p}" for c, p in obs["prices"]["parts"].items())
    return f"""A hydraulic test rig has four components: {', '.join(obs['components'])}. Each component is at one of its condition levels, and {_fault_count_rule(obs)}. Find the level of every component, then repair the rig.

Condition levels: {levels}.

Tools and prices:
- measure(tier) reads the next 60-second operating cycle of the rig and returns the mean, std, min and max of every channel of the tier, each named <channel>_<statistic>:
{tiers}
- submit(cooler, valve, pump, accumulator) gives the diagnosis, one condition level per component. It costs nothing.
- submit_insufficient_evidence(next_action) states that the evidence cannot separate the remaining candidates and names the one action that would. It costs nothing.
- replace(component) restores one component to its healthy level. It costs the part ({parts}) plus {obs['prices']['downtime']} for downtime.
- finish() ends the episode after the diagnosis. It costs nothing.

Rules: measure first, then submit exactly once, with either submit or submit_insufficient_evidence; a replacement before the submission is refused. After the submission you may measure again, replace up to {obs['repairs_left']} components and finish. The episode also ends when every component is healthy, when the replacements are used up, when an action would exceed the budget, or after {obs['steps_left']} steps; a refused call counts as a step, and a call with a missing argument or an argument of the wrong type is refused. Budget: {obs['budget_left']}."""


def _fault_count_rule(obs):
    """The public fault-count rule in words. Version v0.3 added the lower bound (min_faulty), so a scenario class with
    one exact count reads "exactly"; under the default rule of 1 to 2 the question is the same as in v0.2."""
    low, high = obs["min_faulty"], obs["max_faulty"]
    if low < high:
        return f"between {low} and {high} components start off their healthy level"
    if low == 1:
        return "exactly 1 component starts off its healthy level"
    return f"exactly {low} components start off their healthy level"
