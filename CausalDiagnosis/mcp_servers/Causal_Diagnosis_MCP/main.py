"""The diagnostics MCP server: four read only tools over the evidence an episode has released (CausalDiagnosis/docs/causal-diagnosis-mcp.md).

One server process per episode, bound to the episode folder by a launcher file beside the released evidence, as for the
action server. Every tool is declared read only through its ToolAnnotations, since none changes the asset or the case
record; estimate_intervention_effect and the troubleshooting plans of rank_root_causes store their predictions in predictions.json
beside the released evidence, which is the server's own record. Parameters are flat and typed, and required except
rank_root_causes' top_k (5 when left out), its view option branch_detail and the level response_format of every tool, which a caller may
leave out, since the Plan-Execute argument filler sees only their names, JSON types and required flags, so the legal values are stated in
the descriptions. raw returns the answer of CausalTools as FastMCP writes it, and the shorter levels of tools/answer_levels.py, concise
by default, are JSON without indentation. Every tool runs deterministic code over
the released evidence, the public rules and the fitted models: no hidden label, no future observation, no fault rule of the
starts, no language model. The texts of the tools take the nouns, the sensor groups, a channel and a component of the asset of
the released evidence, the default asset before any episode, so the server names no asset of its own.
"""
import json
from pathlib import Path
from typing import Union

import pydantic_core
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent, ToolAnnotations

from CausalDiagnosis.mcp_servers.utils import ready_line as ready
from CausalDiagnosis.mcp_servers.utils import transcript
from CausalDiagnosis.mcp_servers.utils.transcript_hooks import transcribed

from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools import DEFAULT_ABSTENTION_CUT, DEFAULT_TOP_K, CausalTools, answer_levels
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tools.fitted_models import load_models
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import tool_words
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.troubleshooting_plans import DEFAULT_BRANCH_DETAIL
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.tool_result_schemas import (EffectConcise, EffectDetailed, EpisodeEvidence, ErrorResult, EvidenceConcise,
                                                                           EvidenceDetailed, Headline, InterventionEffect, OutcomeVerification, RankingConcise,
                                                                           RankingDetailed, RankingHeadline, RootCauses, VerificationConcise, VerificationDetailed)
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.belief_at_one_cycle import DEFAULT_RANKING_MODEL

TOOLS = ("get_episode_evidence", "rank_root_causes", "estimate_intervention_effect", "verify_outcome")
READ_ONLY = ToolAnnotations(readOnlyHint=True)
LEVELS_LINE = answer_levels.LEVELS_LINE  # the one line on the levels in every tool description, which carries no assumption sentence


def asset_of(released_path):
    """The asset the released evidence names, or None while there is none, as for a tool listing before any episode, so that
    the texts take the default asset's words then."""
    try:
        return json.loads(Path(released_path).read_text(encoding="utf-8")).get("asset_id")
    except (OSError, TypeError, ValueError):
        return None


def descriptions(words):
    """The texts of the four tools and of the server, with the nouns, the sensor groups, a channel and a component of the asset in
    place of any asset of the server's own, and the write tools named by their role, since every row of the
    environment table serves them through its own servers."""
    noun, groups = words["short_noun"], ", ".join(words["groups"])
    return {
        "instructions": (f"Diagnostics over the released evidence of one {words['noun']} episode: the evidence with anomaly scores, "
                         "the belief over every combination of the asset now, filtered through the public rule cards and the "
                         "readings, with a decision rule for reading first and the information gain per unit cost of every read, the "
                         "predicted effect of a repair with calibrated intervals, and the check of a prediction against the "
                         "recordings after the repair. Reading sensors, waiting, diagnosing and work orders are actions of "
                         f"the action servers of the episode; these tools never change the {noun} and move no clock themselves. "
                         "Where the clock of the episode counts the seconds between two calls of the action servers, the seconds of "
                         "a call to these tools are counted with them."),
        "get_episode_evidence": (f"The sensor readings released so far as evidence items, each with an evidence ID, the cycle it was read at, the value, its unit, "
                                 f"the public normal range of the channel mean and an anomaly score against the healthy recordings. sensor_group is all, {groups}, "
                                 f"a channel such as {words['channel']} or its field name such as {words['field']}. Reads released values only; it never reads the {noun}."
                                 + LEVELS_LINE),
        "rank_root_causes": (f"Rank every level combination of the asset as the state of the {noun} now, by a belief that replays the released evidence from the start: "
                             "a uniform prior times the released fix history, the public rule cards applied for every cycle that passed (closed work orders take only when "
                             "the components they depend on are healthy), and every reading weighed by the interventional effect model. Returns the mass of the top "
                             "combinations, the fault probability of every component now, the effective number of hypotheses and the cycle of the belief, and the decision whether to read first, read_first: "
                             "true exactly when the rule of the server names one read of the recorded cycles before a diagnosis, through the sensor group that can "
                             "still show the most summaries and, where readings have prices, that the budget pays beside the repair of every component, with that "
                             "read as the separating test and the expected saving of the best single reading beside it. elimination gives, for every sensor group, the "
                             "information gain of one cycle of it, the expected drop of the log of the effective number of hypotheses, and that gain per unit cost, "
                             "which ranks the reads, with the expected number of hypotheses the cycle eliminates, its one step edge cutting score over repair plan "
                             "classes (ec2, after the EC2 objective of Golovin, Krause and Ray, not their algorithm) and both per unit cost reported beside them, the "
                             "cost being the test cost where reads, parts, repairs or visits cost money, else one call, and read_batch holds the reads of the rule in "
                             "order, that one read where the sensor groups are nested, with left_to_read, the recorded cycles the row can still serve each group. The result "
                             "also carries advice, the step that the scoring of the episode favors before a diagnosis: that read, else the diagnosis of the first "
                             "candidate with one work order per faulty component. The details summary and full add e10_rule, the earlier rule with the margin, the "
                             "cap, none unless the launch names one, and the reserve the server was launched with: read_first when the best single reading lowers the expected cost below acting now "
                             "by more than the margin times its cost, that reading as its separating test, and its read_batch, the reads best first by gain per "
                             "unit cost whose choice no outcome of the reads before them could change. top_k is the number of combinations to return, "
                             f"{DEFAULT_TOP_K} unless the caller names more, with no maximum: the first top_k with positive probability by probability, ties by the combination in component order, "
                             "and more_candidates gives how many combinations with positive probability the reply leaves out and their summed probability; set_masses gives, per faulty set with positive probability, the probability summed over every combination of that set, returned or not. Each distinct faulty set among the returned combinations also comes with a "
                             "troubleshooting plan, one per set: the sensor group that would confirm or refute the set and by how much, the repair plan of "
                             "the set as one technician visit with one visit charge, its lead time, the repairs in priority order, their dependencies, the "
                             f"cycles to a restored {noun} and the prediction ID every work order of the plan cites, the predicted recovery probability after that plan, and the "
                             "predicted expected loss and cost to a restored state of acting on it now against testing it first. branch_detail is compact (the default: the plans of summary, without e10_rule, the attribution, the level probabilities, the "
                             "stability, the Monte Carlo settings and the edge cutting scores, and with one assumption text), summary (adds e10_rule, the channels "
                             "each component explains, the stability of the top component under resampling, the edge cutting scores and every assumption text) or "
                             "full (summary with the predicted summaries after each plan and the separation of every sensor group)." + LEVELS_LINE),
        "estimate_intervention_effect": (f"Predict what an action would do before it is taken: the probability that it addresses a fault, the probability that the {noun} "
                                         "is restored at the first reading after it, with the lead time, the repair downtime, the dependencies of the component and the "
                                         "rule cards acting meanwhile, its price with one visit charge, its duration, and the predicted summaries of that reading, those of "
                                         "the predicted combination, the most probable after the action, which the message names with its mass, with "
                                         f"the intervals of the leave one block out conformal calibration at the declared error level. action is replace:<component> "
                                         f"such as replace:{words['component']}, a standard order, or monitor for one more cycle without a repair. Returns a prediction ID to cite in the work order and to pass to verify_outcome."
                                         + LEVELS_LINE),
        "verify_outcome": ("Compare a stored prediction with the first reading after the work order closed: the largest standardized deviation over the channels "
                           "read against the predicted combination, the most probable after the action, which the result names with its mass, its conformal p value and the "
                           "verdict at the declared error level, which over a calibration block drawn at random and a cycle drawn in it finds a reading "
                           "inconsistent at a rate of at most that level and promises no rate for one named combination, and the mass of the "
                           "healthy combination in the belief now against the predicted recovery. "
                           "prediction_id is the ID that estimate_intervention_effect returned, such as pred/1."),
    }


def build_server(released_path=None, ranking_model=DEFAULT_RANKING_MODEL, abstention_rule="decision", abstention_cut=DEFAULT_ABSTENTION_CUT,
                 models_path=None, left_out=None, **thresholds):
    """ranking_model is joint_block (default), joint or discriminant_urn (the ranking model ablation En9); abstention_rule is
    decision (E10) or fixed_cut with abstention_cut on the top mass (the abstention rule ablation En5). thresholds holds the
    server fields of a decision threshold card, e10_margin, batch_cap and reserve_plans, which keep today's constants when absent;
    an absent batch_cap means no count.
    models_path names a models.json fitted on the recordings of another asset, such as the tiny asset of the asset layer
    tests, and the rig's fitted models serve when it is None. left_out, the combination the episode started in, which the driver
    names at launch, is the block every fit leaves out and has no other use; without it the server serves no
    episode, as for a listing of its tools."""
    models = None if left_out is None else (load_models() if models_path is None else load_models(models_path)).without(left_out)
    tools = CausalTools(models, released_path, ranking_model=ranking_model, abstention_rule=abstention_rule, abstention_cut=abstention_cut,
                        **thresholds)
    worded = descriptions(tool_words(asset_of(released_path)))
    mcp = FastMCP("causal_diagnosis", instructions=worded["instructions"], log_level="WARNING")
    folder = None if released_path is None else Path(released_path).parent

    def answered(tool, response_format, call, shorten):
        """The answer of a tool at the level response_format names: the result of CausalTools as it is at raw and for an error,
        else its shorter level as JSON text without indentation beside the same structured content, opened by the key unchanged
        where an earlier call of the episode returned the same text. The level is checked before the tool runs, since a ranking
        and an estimate store predictions. A shorter answer leaves its raw level, the text FastMCP writes of the result, to the
        transcript of the episode, which keeps it beside the line of the call (transcript.keep_raw)."""
        level = answer_levels.level_of(response_format)
        if level is None:
            return ErrorResult(error=answer_levels.level_error())
        result = call()
        if level == "raw" or isinstance(result, ErrorResult):
            return result
        transcript.keep_raw(pydantic_core.to_json(result, fallback=str, indent=2).decode())
        answer = shorten(result.model_dump(mode="json"), level)
        step = answer_levels.first_step(folder, mcp.name, tool, answer_levels.text_of(answer))
        if step is not None:
            answer = answer_levels.marked(answer, step)
        # FastMCP wraps the structured content of a tool whose answer is a union under the key result
        return CallToolResult(content=[TextContent(type="text", text=answer_levels.text_of(answer))], structuredContent={"result": answer})

    @mcp.tool(title="Get Episode Evidence", annotations=READ_ONLY, description=worded["get_episode_evidence"])
    def get_episode_evidence(sensor_group: str, response_format: str = answer_levels.DEFAULT_LEVEL) -> Union[
            EpisodeEvidence, EvidenceDetailed, EvidenceConcise, Headline, ErrorResult]:
        return answered("get_episode_evidence", response_format, lambda: tools.get_episode_evidence(sensor_group),
                        lambda full, level: answer_levels.evidence(full, tools.released_view(), level))

    @mcp.tool(title="Rank Root Causes", annotations=READ_ONLY, description=worded["rank_root_causes"])
    def rank_root_causes(top_k: int = DEFAULT_TOP_K, branch_detail: str = DEFAULT_BRANCH_DETAIL, response_format: str = answer_levels.DEFAULT_LEVEL) -> Union[
            RootCauses, RankingDetailed, RankingConcise, RankingHeadline, ErrorResult]:
        return answered("rank_root_causes", response_format, lambda: tools.rank_root_causes(top_k, branch_detail),
                        lambda full, level: answer_levels.ranking(full, tools.released_view(), level))

    @mcp.tool(title="Estimate Intervention Effect", annotations=READ_ONLY, description=worded["estimate_intervention_effect"])
    def estimate_intervention_effect(action: str, response_format: str = answer_levels.DEFAULT_LEVEL) -> Union[
            InterventionEffect, EffectDetailed, EffectConcise, Headline, ErrorResult]:
        return answered("estimate_intervention_effect", response_format, lambda: tools.estimate_intervention_effect(action),
                        lambda full, level: answer_levels.effect(full, tools.released_view(), level))

    @mcp.tool(title="Verify Outcome", annotations=READ_ONLY, description=worded["verify_outcome"])
    def verify_outcome(prediction_id: str, response_format: str = answer_levels.DEFAULT_LEVEL) -> Union[
            OutcomeVerification, VerificationDetailed, VerificationConcise, Headline, ErrorResult]:
        return answered("verify_outcome", response_format, lambda: tools.verify_outcome(prediction_id), answer_levels.verification)

    # FastMCP hands a CallToolResult that a tool returns to the protocol as it is, so a caller in process, such as a test harness
    # of an agent, gets of a shorter level the pair (content, structured content) that every answer of a tool with an output
    # schema gives; the protocol handler keeps the method FastMCP bound when it was built
    serve_call = mcp.call_tool

    async def call_tool(name, arguments):
        result = await serve_call(name, arguments)
        return (result.content, result.structuredContent) if isinstance(result, CallToolResult) else result

    mcp.call_tool = call_tool

    # the transcript of the episode keeps every answer of this server, text included, which no other file of a server holds
    return transcribed(mcp, None if released_path is None else Path(released_path).parent, mcp.name, keep_text=True)


def main(released_path=None, ranking_model=DEFAULT_RANKING_MODEL, abstention_rule="decision", abstention_cut=DEFAULT_ABSTENTION_CUT,
         started=None, **thresholds):
    """Serve the four tools over stdio with the options of build_server, the ready line of the process immediately before
    the stdio loop (CausalDiagnosis/mcp_servers/utils/ready_line.py). started is the launcher's stamp."""
    app = build_server(released_path, ranking_model=ranking_model, abstention_rule=abstention_rule, abstention_cut=abstention_cut,
                       **thresholds)
    ready.announce(None if released_path is None else Path(released_path).parent, app.name, started)
    app.run(transport="stdio")


if __name__ == "__main__":
    main()
