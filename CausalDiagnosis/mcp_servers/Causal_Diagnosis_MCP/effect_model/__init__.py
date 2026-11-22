"""The interventional effect model E1 of the CausalDiagnosis MCP, fitted for one episode without the block of its start
(CausalDiagnosis/docs/causal-diagnosis-mcp.md, section E1).

The experimenters set every component level, so the recordings under a combination are draws from the interventional
distribution of the channel summaries under that combination, and the effect of a component condition on a channel is
estimated from the recordings without causal discovery. The mean of each summary (the mean and the standard deviation of a
channel over one recorded cycle) is an intercept plus one main effect per non healthy level of each component plus the
pairwise interactions that raise the held out log likelihood.

Every combination has one recorded block and an episode serves recordings of its start's block, so a fit that read that block
would recognize it, and a fit that left out only the scored hypothesis's own block would still have read the start's block for
every other hypothesis. Each hypothesis h is therefore scored by a fit that leaves out two blocks, the start's and h's own: its
mean is the prediction of h by that fit, its within covariance pools those blocks, and its block covariance is the covariance
of the leave one block out misfits among them. A hypothesis without a block of its own among the others, the start itself and
any combination without a recording, is scored by the fit without the start's block and one second block, picked by a hash of
the start's levels among the recorded combinations one component away from the start (second_block). So no fit reads the
start's block, every hypothesis, the true one included, is extrapolated from the same number of blocks, and the fit of the
start's own hypothesis loses a neighbor of the start as the fit of each near rival loses the start's block. The conformal calibration leaves out the start's block once per start, and the
resampled tables of E6 follow the rule of the means, every row without its own block or without the second block.

The form of the model is chosen before any episode (model_forms). The
development blocks are the recorded combinations outside the test set of the asset. A part of the form that they identify is
chosen on them: the estimator of the covariance of one cycle, among the registered ones, by the held out density of window
means. A part that they do not identify is chosen once per start on every recorded block but the two that the fit of the
start's own hypothesis leaves out: the interaction pairs, on an asset whose development blocks do not identify every pair. The
space of the model, its components, levels, fields and sensor groups, comes from the asset view and is written into
models.json, so the module names no asset.
"""

from . import model_terms
from . import covariance_estimators
from . import recorded_blocks
from . import model_form
from . import episode_fit

from .model_terms import EffectSpace, design_matrix, full_rank, parse_combo, terms_of, weighted_fit
from .covariance_estimators import (
    COVARIANCE_ESTIMATORS, DEFAULT_COVARIANCE, ledoit_wolf_diagonal, one_cycle_covariance, refuse_constant_summaries,
    shrunk_toward_diagonal, window_means)
from .recorded_blocks import (
    LEAVE_OUT_SEED, PoolData, RecordedBlocks, combo_key, fit_pool, leave_one_out, leave_two_out_misfits, second_block)
from .model_form import (
    BLOCK_EFFECTS, FOLDS, FOLD_SEED, block_log_density, cross_fitted_log_likelihood, development_identifies,
    folds_of_pool, form_window_likelihood, gaussian_log_density, held_out_window_dof, identifies_every_pair,
    model_forms, select_form, select_interactions, select_pool_pairs, window_block_likelihood,
    window_scatter_log_density)
from .episode_fit import EffectModel, Resampler, fit_episode, group_tables
