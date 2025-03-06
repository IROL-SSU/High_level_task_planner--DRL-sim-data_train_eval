from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.managers import CommandTermCfg
from omni.isaac.lab.markers import VisualizationMarkersCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from omni.isaac.lab.markers.config import FRAME_MARKER_CFG

from .position_command import ObjectGoalPosCommand, EEGoalPosCommand, DynamicObjectGoalPosCommand

@configclass
class DynamicObjectGoalPosCommandCfg(CommandTermCfg):
    class_type: type = DynamicObjectGoalPosCommand
    resampling_time_range: tuple[float, float] = (1e6, 1e6) # no resampling based on time

    # Rigidcollection asset name
    asset_name: str = MISSING

    asset_dict: dict = MISSING

    object_id_dict_rev: dict = MISSING

    init_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    marker_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
    """The configuration for the goal pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/body_pose"
    )
    """The configuration for the current pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.1, 0.1, 0.1)
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)



@configclass
class ObjectGoalPosCommandCfg(CommandTermCfg):

    """
    Configuration for the position command term

    Please refer to the :class: 'ObjectGaolPosCommand' class for more details. 
    """

    class_type: type = ObjectGoalPosCommand
    resampling_time_range: tuple[float, float] = (1e6, 1e6) # no resampling based on time

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated"""

    init_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position offset of the asset from its default position
    
    This is used to account for the offset   
    """

    position_success_threshold: float = MISSING
    """Threshold for the position error to consider the goal position to be reach"""

    update_goal_on_success: bool = MISSING
    """Whether t o update the goal position when the goal position is reached"""

    marker_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """
    Position offset of the marker from the object's desired position.

    This is useful to position the marker at a height above the object's desired position.
    Otherwise, the marker may occluded the object in the visualization.
    """

@configclass
class EEGoalPosCommandCfg(CommandTermCfg):

    """
    Configuration for the position command term

    Please refer to the :class: 'ObjectGaolPosCommand' class for more details. 
    """

    class_type: type = EEGoalPosCommand
    resampling_time_range: tuple[float, float] = (1e6, 1e6) # no resampling based on time

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated"""

    init_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position offset of the asset from its default position
    
    This is used to account for the offset   
    """

    position_success_threshold: float = MISSING
    """Threshold for the position error to consider the goal position to be reach"""

    update_goal_on_success: bool = MISSING
    """Whether t o update the goal position when the goal position is reached"""

    marker_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)





