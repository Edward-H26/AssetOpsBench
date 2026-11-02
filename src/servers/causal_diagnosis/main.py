"""The diagnostics MCP server: four read only tools over the evidence an episode has released (docs/hydraulic_rig/causal-diagnosis-server.md).

One server process per episode, bound to the episode folder by a launcher file beside the released evidence, as for the
action server. Every tool is declared read only through its ToolAnnotations, since none changes the asset or the case
record; estimate_intervention_effect stores its prediction in predictions.json beside the released evidence, which is the
server's own record. Parameters are flat, typed and required, since the Plan-Execute argument filler sees only their names,
JSON types and required flags, so the legal values are stated in the descriptions. Every tool runs deterministic code over
the released evidence and the fitted models: no hidden label, no future observation, no language model.
"""
from typing import Union

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from .adapter import DEFAULT_ABSTENTION_CUT, CausalTools, load_models
from .models import EpisodeEvidence, ErrorResult, InterventionEffect, OutcomeVerification, RootCauses
from .posterior import DEFAULT_RANKING_MODEL

TOOLS = ("get_episode_evidence", "rank_root_causes", "estimate_intervention_effect", "verify_outcome")
READ_ONLY = ToolAnnotations(readOnlyHint=True)


def build_server(released_path=None, ranking_model=DEFAULT_RANKING_MODEL, abstention_rule="decision", abstention_cut=DEFAULT_ABSTENTION_CUT):
    """ranking_model is joint_block (default), joint or discriminant_urn (the A12 arms); abstention_rule is decision (E10) or
    fixed_cut with abstention_cut on the top posterior (ablation A8)."""
    tools = CausalTools(load_models(), released_path, ranking_model=ranking_model, abstention_rule=abstention_rule, abstention_cut=abstention_cut)
    mcp = FastMCP(
        "causal_diagnosis",
        instructions=("Diagnostics over the released evidence of one hydraulic rig episode: the evidence with anomaly scores, "
                      "root causes ranked by an interventional likelihood with a decision rule for insufficient evidence, the "
                      "predicted effect of a repair with calibrated intervals, and the check of a prediction against the "
                      "recordings after the repair. Reading sensors, advancing time, diagnosing and work orders are actions of "
                      "the hydraulic_rig server; these tools never change the rig and take no simulated time."),
        log_level="WARNING",
    )

    @mcp.tool(title="Get Episode Evidence", annotations=READ_ONLY)
    def get_episode_evidence(sensor_group: str) -> Union[EpisodeEvidence, ErrorResult]:
        """The sensor readings released so far as evidence items, each with an evidence ID, the cycle it was read at, the value, its unit, the public healthy band of the channel mean and an anomaly score against the healthy recordings. sensor_group is all, tier1, tier2, tier3, a channel such as PS1 or its field name such as PS1_Pressure_bar_100Hz. Reads released values only; it never reads the rig."""
        return tools.get_episode_evidence(sensor_group)

    @mcp.tool(title="Rank Root Causes", annotations=READ_ONLY)
    def rank_root_causes(top_k: int) -> Union[RootCauses, ErrorResult]:
        """Rank the candidate start combinations by their posterior under the interventional effect model, with the fault probability of every component, the channels each component explains, the stability of the top component under resampling, and the decision whether the evidence is insufficient: true when reading one more cycle of some sensor group lowers the expected cost of the repair decision below acting now, with that separating test and the expected saving. top_k is the number of candidates to return, at least 1."""
        return tools.rank_root_causes(top_k)

    @mcp.tool(title="Estimate Intervention Effect", annotations=READ_ONLY)
    def estimate_intervention_effect(action: str) -> Union[InterventionEffect, ErrorResult]:
        """Predict what an action would do before it is taken: the probability that it addresses a fault, the probability that the rig recovers, its price, its duration, and the predicted summaries of the first reading after it with conformal intervals at the declared error level. action is replace:<component> such as replace:valve, or monitor for one more cycle without a repair. Returns a prediction ID to pass to create_work_order and to verify_outcome."""
        return tools.estimate_intervention_effect(action)

    @mcp.tool(title="Verify Outcome", annotations=READ_ONLY)
    def verify_outcome(prediction_id: str) -> Union[OutcomeVerification, ErrorResult]:
        """Compare a stored prediction with the first reading after the work order closed: a conformal p value per channel, the Bonferroni and Holm verdicts at the declared error level, and the recovery probability read from the post repair readings against the predicted one. prediction_id is the ID that estimate_intervention_effect returned, such as pred/1."""
        return tools.verify_outcome(prediction_id)

    return mcp


def main(released_path=None, ranking_model=DEFAULT_RANKING_MODEL, abstention_rule="decision", abstention_cut=DEFAULT_ABSTENTION_CUT):
    """Serve the four tools over stdio with the options of build_server."""
    build_server(released_path, ranking_model=ranking_model, abstention_rule=abstention_rule, abstention_cut=abstention_cut).run(transport="stdio")


if __name__ == "__main__":
    main()
