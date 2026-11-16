# one text for every episode: it names no estimator, since the estimator is part of the form of the start
A_E1 = ("E1: the mean of each summary (the mean and the standard deviation of a channel over one recorded cycle) under a combination is "
        "an intercept plus one main effect per non healthy level of each component plus the pairwise interactions that raised the held "
        "out log likelihood, with the pooled within combination covariance of one cycle from the covariance estimator of the form of the "
        "model, one of the estimators registered before any fit")
A_BLOCK = ("joint_block: the cycles of one recorded block share a latent offset from the model mean with the covariance of the leave one "
           "block out misfits of the fit that scores the combination, so repeated readings of one block add less than independent draws "
           "would; a combination entered again starts a fresh offset; the default ranking model")
A_JOINT = "joint: every reading is an independent draw around the model mean, without the block offset"
# one text for every episode, true whatever the start and silent on which blocks the fits leave out
A_RULE = ("the fits of the episode: every combination is scored by a fit on the stable cycles of the recorded combinations that leaves "
          "out two blocks, the block of the combination the episode started in and its own, or one second block where it has no block "
          "of its own among the others, for its mean, its within covariance and its block covariance, so no fit has read the block the "
          "episode started in and every combination is predicted from the same number of blocks; the form of the model, its "
          "interactions and its covariance estimator, was chosen before any episode by a registered rule that reads no recording of "
          "that block, and the conformal calibration is computed once on the blocks without the start's")
A_URN = ("discriminant_urn: the v1 ranking model, one linear discriminant per sensor group and component read on the same summaries, "
         "fitted on the recorded blocks without the one the episode started in, repeated readings combined by a Polya urn around the "
         "confusion row of a cross fit on those blocks, every closed work order restoring its component; it has no transition model, so "
         "it serves static episodes, and E9 and the predictions keep the joint model's Gaussian predictive around this belief")
A_DEDUPE = ("a recording enters once with the union of its summaries: two readings show one recording when a record names both by one "
            "cycle, as when two groups read one recorded cycle, or when they agree on every summary they share and share one, as when "
            "the recording window wrapped")
A_E3 = ("E3: the anomaly score of a summary is its absolute deviation from the predicted mean of the healthy combination over the within "
        "standard deviation of the fit that scores that combination; it enters no decision")
A_E4 = ("E4: a summary's predicted deviation under the top combination splits into the main effects of its faulty components and halves "
        "of their interactions, from the coefficients of the fit that scores that combination, and a component's explained anomaly is the "
        "sum over summaries of the anomaly score times its share")
A_E6 = ("E6: the filter is replayed under mean tables refitted on resamples of the recorded blocks drawn by block, every row predicted "
        "without its own block's draws as in the main table, and an interval of E6 is the range over the resampled fits of the call, "
        "resampled_fits of them; the range of R fits holds a further fit with probability (R - 1) / (R + 1), so it stands for the "
        "declared level only when R is at least (2 - alpha) / alpha; no band constant is used")
A_E14 = ("E14: the predicted mean of a summary after the action is the mean of the most probable combination of the belief after it, "
         "a tie going to the combination first in the candidate order, and its interval is the leave one block out conformal interval "
         "mean plus or minus q times the within standard deviation, with q the smallest calibration score, the cycles of every block "
         "without the start's scored against the mean of the fit without that block, such that at least (1 - alpha)(n + 1) units of "
         "block weight lie at or below it, n the number of those blocks; over a calibration block drawn at random and a cycle drawn at "
         "random in it, a summary lies in its interval at a rate of at least 1 - alpha when (1 - alpha)(n + 1) is at most n "
         "(level_reachable), and no rate is promised for one named combination; the mass of the combination says how sure the belief is")
A_E16 = ("E16 and E17: the combined score of the reading is the largest standardized deviation over the channels it holds, its p value is "
         "(1 + weight of calibration cycles whose combined score over the same channels is at or above it) / (n + 1), n the calibration "
         "blocks, and the outcome is inconsistent when that p value is at or below alpha, one test per reading and no correction over "
         "channels, scored against the predicted combination, the most probable after the action; over a calibration block drawn at "
         "random and a cycle drawn at random in it, the verdict finds the reading inconsistent at a rate of at most alpha when the level "
         "is reachable, it promises no rate for one named combination, and a reading of the block the episode started in, which a "
         "monitor prediction scores, is scored by a fit that holds one block more than the fits of the calibration; the recovery check "
         "reports the mass of the healthy combination in the belief now, which holds every released reading and closure; a prediction "
         "whose work orders were created after its estimate, after thinking time or other calls, is made again for the cycle of "
         "their creation from the evidence the estimate held, since the rule cards act on the cycles in between")
A_PREDICT_STEP = ("the predict step of E9, E21, E22 and the troubleshooting plans starts from the belief now, closes the open work orders "
               "on their released cycles and a plan's orders as schedule_visit times one visit. The next reading meets the belief now "
               "where the row serves it without moving the clock, a read of a recorded cycle or a read without a declared duration, after "
               "the open work orders of a row without a clock, which settle before the next call that is not an order, and the belief "
               "after the rule cards of the current cycle where the read plays its cycle")
UNIT_TEXT = {"money": "the test cost of E7, the read price plus its declared duration times the expected downtime rate",
             "call": "one call, since the rules price nothing on this row and every cost of E7 to E10 counts calls"}
A_TIME = "time counts rig cycles from the start of the episode; refused calls release nothing and no cycle index of the recording is released"
A_COMPACT = ("branch_detail compact leaves out attribution, level_probability, stability, monte_carlo and the edge cutting scores of "
             "elimination, and gives each candidate the newest reading of every summary that bears on it; branch_detail summary returns "
             "them with every assumption text, the rules that every number of this reply follows")
A_SIX_DIGITS = "values are the released ones, at six significant digits"
A_BANDS = "the normal range of a channel mean is the public normal range of the asset profile, served by the action server, not a fitted threshold"


def cycles_text(count):
    return f"{count} cycle" + ("" if count == 1 else "s")


def combo_text(view, combo):
    return "(" + ", ".join(f"{c} {v}" for c, v in zip(view.components, combo)) + ")"


def sources_text(view):
    """Which rule parameters come from the released rules and which from the defaults of the asset."""
    released = [k for k, v in view.processes_source.items() if v == "released"]
    defaults = [k for k, v in view.processes_source.items() if v != "released"]
    parts = []
    if released:
        parts.append(f"from the released rules: {', '.join(released)}")
    if defaults:
        parts.append(f"from the defaults of the asset, until the released rules carry them: {', '.join(defaults)}")
    return "the parameters of the rule cards come " + "; ".join(parts)
