from .intervention_effect import CONFIRMATION_MASS, CONFIRMATION_RATIO

# one text for every episode: it names no estimator, since the estimator is part of the form of the start
A_E1 = ("E1: the mean of each summary (the mean and the standard deviation of a channel over one recorded cycle) under a combination is "
        "an intercept plus one main effect per non healthy level of each component plus the pairwise interactions that raised the held "
        "out log likelihood, with the pooled within combination covariance of one cycle from the covariance estimator of the form of the "
        "model, one of the estimators that the server declares, mixed, where the form groups the covariances, with the centred sample "
        "covariances of the recorded blocks at the combination's level of one component by the weight of the form")
A_BLOCK = ("joint_block: the cycles of one recorded block share a latent offset from the model mean with the covariance of the leave one "
           "block out misfits of the fit that scores the combination, so repeated readings of one block add less than independent draws "
           "would; a combination entered again starts a fresh offset; the default ranking model")
A_JOINT = "joint: every reading is an independent draw around the model mean, without the block offset"
# one text for every episode, true whatever the start and the blocks served, and silent on which blocks the fits leave out
A_RULE = ("the fits of the episode: the filter scores every combination by a fit on the stable cycles of the recorded combinations that "
          "leaves out the block of the combination the episode started in, the blocks of the combinations whose recordings the released "
          "readings hold, in the order of their first reading for as long as the remaining blocks fit the model, and its own, or one second "
          "block where it has no block of its own among the others, for its mean, its within covariance and its block covariance, so no fit "
          "that weighs a reading has read a block the episode has served and every combination is predicted from the same number of blocks; "
          "the anomaly scores, the attribution, the predicted summaries and the verdicts of E17 come from the fits without the start's block "
          "alone, which stay the same for the whole episode, and the confirmation of a verdict reads the belief; "
          "the form of the model, its interactions, its covariance estimator and the grouping of "
          "its covariances, was chosen "
          "before any episode by a rule of the server that reads no recording of the start's block, and the conformal calibration is "
          "computed once on the blocks without the start's")
A_URN = ("discriminant_urn: the v1 ranking model, one linear discriminant per sensor group and component read on the same summaries, "
         "fitted on the recorded blocks without the one the episode started in, repeated readings combined by a Polya urn around the "
         "confusion row of a cross fit on those blocks, every closed work order restoring its component; it has no transition model, so "
         "it serves static episodes, and E9 and the predictions keep the joint model's Gaussian predictive around this belief")
MODEL_TEXTS = {"joint_block": A_BLOCK, "joint": A_JOINT, "discriminant_urn": A_URN}  # the text of each ranking model
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
A_E13 = ("E13: the prediction is for a standard order, and no emergency order is predicted; the belief after the action runs the "
         "predict step through the order's lead time and repair downtime with its close on the cycle schedule_visit gives one order alone, so the replacement takes only when every component it depends on is healthy "
         "at the close and the rule cards act meanwhile; for monitor it is the belief one cycle ahead; the recovery probability is the "
         "mass of the healthy combination at the first reading after the action, and addresses_fault is the mass of the combinations in "
         "which the component is faulty now")
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
A_CONFIRM = (f"confirmed: a verified repair is confirmed when its verdict is consistent, the belief now, which holds every released "
             f"reading and closure and weighs the readings by the fits of the belief, puts at least {CONFIRMATION_MASS:g} on the predicted "
             "combination, and the likelihood ratio of the readings that verify the repair for the predicted combination against its strongest "
             "rival, the combination other than the predicted one with the most mass in the belief now, is at least "
             f"{CONFIRMATION_RATIO:g}, the ratio being the posterior odds of the two in the belief now divided by their odds in the belief "
             "of the same fits without those readings; the result names that ratio, which is undefined where a mass is zero and then "
             "confirms nothing, and the most probable combination of the belief with its mass, an inconsistent verdict refutes the "
             "prediction whatever the mass and the ratio, and a monitor prediction, a verdict on no reading after the repair and one that "
             "another repair made not comparable carry no confirmation")
UNIT_TEXT = {"money": "the test cost of E7, the read price plus its declared duration times the expected downtime rate",
             "call": "one call, since the rules price nothing on this row and every cost of E7 to E10 counts calls"}
A_TIME = "time counts rig cycles from the start of the episode; refused calls release nothing and no cycle index of the recording is released"
A_COMPACT = ("branch_detail compact leaves out attribution, level_probability, stability, monte_carlo and the edge cutting scores of "
             "elimination, and gives each candidate the newest reading of every summary that bears on it; branch_detail summary returns "
             "them with every assumption text, the rules that every number of this reply follows")
A_SIX_DIGITS = "values are the released ones, at six significant digits"
A_BANDS = "the normal range of a channel mean is the public normal range of the asset profile, served by the action server, not a fitted threshold"

# The key of each assumption text in the shorter levels of an answer (response_format.py). A fixed text has its own key, and
# its sentence stands in the raw level and in CausalDiagnosis/docs/causal-diagnosis-mcp.md. A text that holds numbers of the call
# is keyed by the words every form of it starts with, and the raw level gives it with those numbers. Any other text, a note on an
# open work order, on a work order before a diagnosis or on the end of the episode, stays as it is, since it tells what the
# environment will refuse.
FIXED_KEYS = {A_RULE: "fits", A_E1: "E1", A_DEDUPE: "dedupe", A_BLOCK: "joint_block", A_JOINT: "joint", A_URN: "discriminant_urn",
              A_PREDICT_STEP: "predict_step", A_E3: "E3", A_E4: "E4", A_E6: "E6", A_E13: "E13", A_E14: "E14", A_E16: "E16", A_CONFIRM: "confirmed", A_TIME: "time",
              A_COMPACT: "compact", A_BANDS: "bands", A_SIX_DIGITS: "six_digits"}
CALL_KEYS = (("E2: ", "E2"), ("the parameters of the rule cards come ", "rule_cards"), ("the filter carried ", "states"), ("E7 to E10", "E7_E10"),
             ("abstention_rule ", "abstention"), ("E21: ", "E21_E22"), ("candidates: the first ", "plans"), ("E15: ", "E15"),
             ("the price is the part plus ", "price"), ("e10_rule, the earlier rule", "e10_rule"))
RANKING_EVIDENCE_RULE = ("obs/<n>/<field> for every field of the sensor group that readings names for reading n, or of the fields it lists, "
                         "and every wo/<id>; evidence_ids keeps the top channels of the attribution at their newest reading, of every "
                         "other reading its first identifier that bears on the first candidate or else its first, and the work orders; "
                         "raw lists every identifier")
EFFECT_EVIDENCE_RULE = ("obs/<n>/<field> for every field of the sensor group that readings names for reading n, or of the fields it lists, "
                        "every wo/<id> and the prediction; evidence_ids keeps one identifier of every reading, the work orders and the "
                        "prediction; raw lists every identifier")
EVIDENCE_TABLE_RULE = ("obs/<n>/<field> for the n of a reading of readings and the field of an entry of fields whose value for it is not "
                       "null, and every wo/<id>; a value below the normal range of its field ends in low and one above it in high; "
                       "evidence_ids keeps one identifier of every reading, outside its range where the reading has one, and the work "
                       "orders; raw gives one item per value with its anomaly score")
CONCISE_TABLE_RULE = EVIDENCE_TABLE_RULE + ("; for all or a sensor group concise lists only the fields with a value outside their range "
                                            "and counts the others in not_listed by the sensor group of their first reading, which "
                                            "detailed lists, and for a channel or a field every field")
VERIFICATION_EVIDENCE_RULE = ("obs/<observation>/<field> for the field of every channel, and the prediction; evidence_ids keeps the "
                              "prediction and the first of them; raw lists every identifier")
VERIFICATION_CONCISE_RULE = VERIFICATION_EVIDENCE_RULE + "; concise leaves out the channels, which detailed lists"


def assumption_key(text):
    """The key of an assumption text in a shorter level, the text itself where it has none."""
    if text in FIXED_KEYS:
        return FIXED_KEYS[text]
    return next((key for lead, key in CALL_KEYS if text.startswith(lead)), text)


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
