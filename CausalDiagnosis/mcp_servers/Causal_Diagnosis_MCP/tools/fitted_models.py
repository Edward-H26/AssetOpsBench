import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.asset_view import data_variable
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.advice.conformal_intervals import Calibration
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.belief.discriminant_urn_ranking import fit_discriminant_urn
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.effect_model import EffectSpace, PoolData, RecordedBlocks, combo_key, fit_episode, parse_combo
from CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.fit_models import MODELS_FORMAT, MODELS_PATH


class Models:
    """models.json parsed once per process: the space, the stable cycles of every recorded combination, the folds and the form
    of the model of each recorded combination. The fits of an episode come from without(start), which leaves the block of
    the start's combination out of every one of them."""

    def __init__(self, raw):
        if raw.get("format") != MODELS_FORMAT:
            raise ValueError(f"models.json has format {raw.get('format')}, this server reads format {MODELS_FORMAT}; refit with "
                             "`python -m CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.fit_models`")
        self.raw = raw
        self.space = EffectSpace.from_json(raw["space"])
        self.fit = raw["fit"]
        self.blocks = {parse_combo(key): np.asarray(cycles, dtype=float) for key, cycles in raw["blocks"].items()}
        self.recorded = RecordedBlocks(self.space, self.blocks, window=raw["form"]["window"])
        self.fold_of = {parse_combo(key): fold for key, fold in raw["folds"].items()}

    def pairs_without(self, start):
        """The interaction pairs of the start's form, chosen on blocks that hold no recording of the start (fit.FORM_RULE)."""
        names = self.raw["selections"][combo_key(start)]["pairs"]
        return [(self.space.components.index(a), self.space.components.index(b)) for a, b in names]

    def covariance_without(self, start):
        """The covariance estimator of the start's form, chosen on development blocks other than the start's (fit.FORM_RULE)."""
        return self.raw["selections"][combo_key(start)]["covariance"]

    def without(self, start, also_without=()):
        """The fits of an episode that started in the combination, whose block none of them reads, nor the blocks of the
        combinations of also_without, none by default."""
        return EpisodeModels(self, tuple(start), tuple(tuple(h) for h in also_without))


class EpisodeModels:
    """The fits of one episode: the effect model that scores every hypothesis without the start's block and its own or the
    second block, the conformal calibration of E14 and E16 with its scale, and the v1 urn model, fitted when first asked for,
    all on the recorded blocks without the start's block and without the blocks of also_without. The start names the blocks to
    leave out and the form of the model, and has no other use."""

    def __init__(self, models, start, also_without=()):
        episode = fit_episode(models.recorded, models.pairs_without(start), start, models.fit["resamples"], models.fit["resample_seed"],
                              covariance=models.covariance_without(start), also_left_out=also_without)
        self.space, self.effect, self.sigma, self.fit = models.space, episode.effect, episode.sigma, models.fit
        self.calibration = Calibration(episode.scores, episode.weights, episode.n_blocks)
        self._blocks, self._fold_of, self._pool, self._urn = models.blocks, models.fold_of, episode.pool, None

    @property
    def urn(self):
        if self._urn is None:
            self._urn = fit_discriminant_urn(self.space, PoolData(self.space, self._blocks, self._pool), {h: self._fold_of[h] for h in self._pool})
        return self._urn


@lru_cache(maxsize=1)
def load_models(path=MODELS_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(f"{Path(path).name} is generated from the recordings of an asset and is not part of the repository: run "
                                f"`python -m CausalDiagnosis.mcp_servers.Causal_Diagnosis_MCP.offline.fit_models`, which fits it on the data file of the asset, its release file or the file that "
                                f"{data_variable()} names when it is set")
    return Models(json.loads(Path(path).read_text(encoding="utf-8")))
