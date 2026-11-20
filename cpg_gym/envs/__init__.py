# License: see [LICENSE, LICENSES/legged_gym_license]

from cpg_gym import CPG_GYM_ROOT_DIR, CPG_GYM_ENVS_DIR
from .base.quadruped_gaits import QuadrupedGaits
from .base.quadruped_gaits_config import QuadrupedGaitsCfg, QuadrupedGaitsCfgPPO
from .base.quadruped_gait_selector import QuadrupedGaitSelector
from .base.quadruped_gait_selector_config import QuadrupedGaitSelectorCfg, QuadrupedGaitSelectorCfgPPO
import os

from cpg_gym.utils.task_registry import task_registry

task_registry.register( "quadruped_gaits", QuadrupedGaits, QuadrupedGaitsCfg(), QuadrupedGaitsCfgPPO())
task_registry.register( "quadruped_gait_selector", QuadrupedGaitSelector, QuadrupedGaitSelectorCfg(), QuadrupedGaitSelectorCfgPPO())