from dataclasses import MISSING

from omni.isaac.lab.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.lab.utils import configclass

from . import discreteaction

##
# Joint actions.
##


@configclass
class DiscreteActionCfg(ActionTermCfg):
    """Configuration for the base joint action term.

    See :class:`JointAction` for more details.
    """

    grasping_expr: tuple = MISSING

    sweeping_right_expr: tuple = MISSING

    sweeping_left_expr: tuple = MISSING

    class_type: type[ActionTerm] = discreteaction.DiscreteAction