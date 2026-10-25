"""The causal-diagnosis MCP server: seven tools over the answers an episode has released (docs/hydraulic_rig/causal-diagnosis-server.md).

One server process per episode, bound to the episode folder by a launcher file beside the released answers, as for the
environment-action server; without an episode (the causal-diagnosis-mcp-server entry point) only the two static tools
are served. Parameters are flat, typed and required, since the Plan-Execute argument filler sees only their names, JSON
types and required flags, so the legal values are stated in the descriptions. Every tool runs deterministic code over
the released answers and the fitted models: no hidden label, no future observation, no language model.
"""
from typing import Union

from mcp.server.fastmcp import FastMCP

from .adapter import CausalTools, load_models
from .models import AssetContext, EpisodeEvidence, ErrorResult, FmeaPaths, InterventionEffect, OutcomeVerification, RootCauses, SymptomReport


def build_server(released_path=None):
    tools = CausalTools(load_models(), released_path)
    mcp = FastMCP(
        "causal_diagnosis",
        instructions=("Causal diagnosis over the released evidence of one hydraulic rig episode: asset context, symptom report, evidence "
                      "with healthy bands, failure-mode paths, root-cause ranking, intervention effects and outcome verification. "
                      "Measuring and repairing are actions of the hydraulic_rig server; these tools never change the unit."),
        log_level="WARNING",
    )

    @mcp.tool(title="Get Asset Context")
    def get_asset_context(asset_id: str) -> Union[AssetContext, ErrorResult]:
        """Describe an asset for diagnosis: its class, its components with their condition levels and healthy level, its channels with unit, sampling rate and measurement group (tier1, tier2 or tier3), and how many failure-mode codes it has. asset_id is the asset's identifier, such as hyd_1."""
        return tools.get_asset_context(asset_id)

    @mcp.tool(title="Query FMEA Paths")
    def query_fmea_paths(asset_id: str, node: str) -> Union[FmeaPaths, ErrorResult]:
        """Failure-mode paths of the asset's knowledge base. node is a component (cooler, valve, pump or accumulator), a failure-mode code such as valve_switching_critical, or a channel such as PS1 or P1_MotorPower_W_100Hz. Returns each path's component, code, severity, recommended action, the channels that indicate the component with their source, and the held-out accuracy of reading the component per measurement group."""
        return tools.query_fmea_paths(asset_id, node)

    if released_path is not None:
        @mcp.tool(title="Get Symptom Report")
        def get_symptom_report() -> Union[SymptomReport, ErrorResult]:
            """The complaint that opened the episode: the task text, the public rules (how many components are off their healthy level, budget, step cap, repairs, prices) and the current counters, reported at measurement 0, before the first measurement."""
            return tools.get_symptom_report()

        @mcp.tool(title="Get Episode Evidence")
        def get_episode_evidence(sensor_group: str) -> Union[EpisodeEvidence, ErrorResult]:
            """The measurements released so far as evidence items, each with an evidence ID, the value, its unit and the healthy band of that summary. sensor_group is all, tier1, tier2, tier3, a channel such as PS1 or its field name such as PS1_Pressure_bar_100Hz. Reads released values only; it never measures."""
            return tools.get_episode_evidence(sensor_group)

        @mcp.tool(title="Rank Root Causes")
        def rank_root_causes(top_k: int) -> Union[RootCauses, ErrorResult]:
            """Rank the candidate fault combinations by posterior probability given the released measurements, with the fault probability and level probabilities of every component, and whether the evidence is insufficient (top candidate below 0.95), naming the unseparated components and the channels that bear on them. top_k is the number of candidates to return, at least 1."""
            return tools.rank_root_causes(top_k)

        @mcp.tool(title="Estimate Intervention Effect")
        def estimate_intervention_effect(action: str) -> Union[InterventionEffect, ErrorResult]:
            """Estimate what an action would do before it is taken: the probability that it addresses a fault, the probability that the unit recovers, its price, and the predicted summaries of the next measurement with bands. action is monitor or replace:<component>, such as replace:valve. Returns a prediction ID for verify_outcome."""
            return tools.estimate_intervention_effect(action)

        @mcp.tool(title="Verify Outcome")
        def verify_outcome(prediction_id: str) -> Union[OutcomeVerification, ErrorResult]:
            """Compare a stored prediction with what the unit did after the action: the residual of every observed summary against its prediction, inside or outside the band, and whether the unit recovered where that is observable. prediction_id is the ID that estimate_intervention_effect returned, such as pred/1."""
            return tools.verify_outcome(prediction_id)

    return mcp


def main(released_path=None):
    build_server(released_path).run(transport="stdio")


if __name__ == "__main__":
    main()
