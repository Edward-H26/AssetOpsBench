"""The wrappers that play in process refuse a clock that charges thinking time."""
import dataclasses

import pytest

from CausalDiagnosis.environment.Causal_Diagnosis_Environment.episode_config import EpisodeConfig
from CausalDiagnosis.datasets.hyd_1.hyd_1_tables import default_facts_path
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.gymnasium_wrapper import HydraulicRigGym
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.layers.layer_2_real_time import THINKING_MODES
from CausalDiagnosis.environment.Causal_Diagnosis_Environment.openenv_wrapper import HydraulicRigOpenEnv


@pytest.mark.parametrize("mode", THINKING_MODES)
def test_a_clock_that_charges_thinking_is_refused_by_both_wrappers(mode):
    config = dataclasses.replace(EpisodeConfig(), clock_mode=mode)
    for build in (lambda: HydraulicRigGym(config, default_facts_path()), lambda: HydraulicRigOpenEnv(config)):
        with pytest.raises(ValueError, match="charges thinking time"):
            build()
