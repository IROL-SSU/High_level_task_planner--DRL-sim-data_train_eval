## 사용 방법
## ./isaaclab.sh -p source/standalone/shelf_env/ground_truth_generator.py --target_object can_3 --enable_camera --save --row 4

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script shows how to use the camera sensor from the Isaac Lab framework.

The camera sensor is created and interfaced through the Omniverse Replicator API. However, instead of using
the simulator or OpenGL convention for the camera, we use the robotics or ROS convention.

.. code-block:: bash

    # Usage with GUI
    ./isaaclab.sh -p source/standalone/tutorials/04_sensors/run_usd_camera.py --enable_cameras

    # Usage with headless
    ./isaaclab.sh -p source/standalone/tutorials/04_sensors/run_usd_camera.py --headless --enable_cameras

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates how to use the camera sensor."
)
parser.add_argument(
    "--draw",
    action="store_true",
    default=False,
    help="Draw the pointcloud from camera at index specified by ``--camera_id``.",
)
parser.add_argument(
    "--save",
    action="store_true",
    default=False,
    help="Save the data from camera at index specified by ``--camera_id``.",
)
parser.add_argument(
    "--camera_id",
    type=int,
    choices={0, 1},
    default=0,
    help=(
        "The camera ID to use for displaying points or saving the camera data. Default is 0."
        " The viewport will always initialize with the perspective of camera 0."
    ),
)
parser.add_argument(
    "--target_object",
    type=str,
    default="cup_1",
    help="Name of the target object",
)
parser.add_argument("--row", type=int, default=1)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import os
import random
import torch

import omni.isaac.core.utils.prims as prim_utils
import omni.replicator.core as rep

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import RAY_CASTER_MARKER_CFG
from omni.isaac.lab.sensors.camera import Camera, CameraCfg
from omni.isaac.lab.sensors.camera.utils import create_pointcloud_from_depth
from omni.isaac.lab.utils import convert_dict_to_backend, math

# 원래 버전용
# usd_path_mapping = {
#     "cup_1": "omniverse://localhost/Library/Shelf/Object/Cup_1.usd",
#     "cup_2": "omniverse://localhost/Library/Shelf/Object/Cup_2.usd",
#     "cup_3": "omniverse://localhost/Library/Shelf/Object/Cup_3.usd",
#     "cup_4": "omniverse://localhost/Library/Shelf/Object/Cup_4.usd",
#     "cup_5": "omniverse://localhost/Library/Shelf/Object/Cup_5.usd",
#     "mug_1": "omniverse://localhost/Library/Shelf/Object/Mug_1.usd",
#     "mug_2": "omniverse://localhost/Library/Shelf/Object/Mug_2.usd",
#     "mug_3": "omniverse://localhost/Library/Shelf/Object/Mug_3.usd",
#     "mug_4": "omniverse://localhost/Library/Shelf/Object/Mug_4.usd",
#     "mug_5": "omniverse://localhost/Library/Shelf/Object/Mug_5.usd",
#     "bottle_1": "omniverse://localhost/Library/Shelf/Object/Bottle_1.usd",
#     "bottle_2": "omniverse://localhost/Library/Shelf/Object/Bottle_2.usd",
#     "bottle_3": "omniverse://localhost/Library/Shelf/Object/Bottle_3.usd",
#     "bottle_4": "omniverse://localhost/Library/Shelf/Object/Bottle_4.usd",
#     "bottle_5": "omniverse://localhost/Library/Shelf/Object/Bottle_5.usd",
#     "can_1": "omniverse://localhost/Library/Shelf/Object/Can_1.usd",
#     "can_2": "omniverse://localhost/Library/Shelf/Object/Can_2.usd",
#     "can_3": "omniverse://localhost/Library/Shelf/Object/Can_3.usd",
#     "can_4": "omniverse://localhost/Library/Shelf/Object/Can_4.usd",
#     "can_5": "omniverse://localhost/Library/Shelf/Object/Can_5.usd",
# }

# 실증용
usd_path_mapping = {
    "cup_1": "omniverse://localhost/Library/Shelf/Object/Cup_1.usd",
    "cup_2": "omniverse://localhost/Library/Shelf/Object/Cup_2.usd",
    "cup_3": "omniverse://localhost/Library/Shelf/Object/Cup_4.usd",
    "mug_1": "omniverse://localhost/Library/Shelf/Object/Mug_2.usd",
    "mug_2": "omniverse://localhost/Library/Shelf/Object/Mug_3.usd",
    "mug_3": "omniverse://localhost/Library/Shelf/Object/Mug_4.usd",
    "bottle_1": "omniverse://localhost/Library/Shelf/Object/Bottle_6.usd",
    "bottle_2": "omniverse://localhost/Library/Shelf/Object/Bottle_7.usd",
    "bottle_3": "omniverse://localhost/Library/Shelf/Object/Bottle_8.usd",
    "can_1": "omniverse://localhost/Library/Shelf/Object/Can_6.usd",
    "can_2": "omniverse://localhost/Library/Shelf/Object/Can_7.usd",
    "can_3": "omniverse://localhost/Library/Shelf/Object/Can_8.usd",
}


def define_sensor() -> Camera:
    """Defines the camera sensor to add to the scene."""
    # Setup camera sensor
    # In contrast to the ray-cast camera, we spawn the prim at these locations.
    # This means the camera sensor will be attached to these prims.
    prim_utils.create_prim("/World/Origin_00", "Xform")
    camera_cfg = CameraCfg(
        prim_path="/World/Origin_.*/CameraSensor",
        update_period=0,
        height=480,
        width=640,
        data_types=[
            "rgb",
            "distance_to_image_plane",
            "distance_to_camera",
            "semantic_segmentation",
        ],
        colorize_semantic_segmentation=True,
        colorize_instance_id_segmentation=True,
        colorize_instance_segmentation=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
    )
    # Create camera
    camera = Camera(cfg=camera_cfg)

    return camera


def design_scene() -> dict:
    """Design the scene."""
    # Populate scene
    # -- Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # -- Lights
    # spawn distant light
    cfg_light_dome = sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(1.0, 1.0, 1.0),
    )

    cfg_light_dome.func("/World/lightDistant", cfg_light_dome, translation=(-5, 0, 10))
    
    
    cfg_light_dome2 = sim_utils.DomeLightCfg(
        intensity=2700.0,
        color=(1.0, 1.0, 1.0),
    )
    cfg_light_dome2.func("/World/lightDistant2", cfg_light_dome2, translation=(1.2, 0.0, 1.4))


    # Create a dictionary for the scene entities
    scene_entities = {}

    # spawn a usd file of a shelf into the scene
    rack_cfg = RigidObjectCfg(
        prim_path="/World/Rack",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"omniverse://localhost/Library/Shelf/Arena/speedrack.usd",
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        debug_vis=False,
    )

    rack = RigidObject(cfg=rack_cfg)

    prim_utils.create_prim(
        "/World/Objects", "Xform", translation=(-0.12, -0.40, 1.05)
    )  # 원래 버전 (-0.24, -0.38, 0.66) / 5x3 실증용 (-0.12, -0.40, 0.66) / 4x3 실증용 (-0.12, -0.40, 0.66)

    scene_entities = object_spawn(scene_entities=scene_entities)
    # Sensors
    camera = define_sensor()

    # return the scene information
    scene_entities["camera"] = camera
    return scene_entities


def object_spawn(
    scene_entities: dict,
    pos=(0.0, 0.0, 0.0),
    ori=(1.0, 0.0, 0.0, 0.0),
    prim_path="/World/Objects/target",
    common_properties={
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(),
        "mass_props": sim_utils.MassPropertiesCfg(mass=1.0),
        "collision_props": sim_utils.CollisionPropertiesCfg(),
    },
) -> dict:
    # Xform to hold objects

    object_name = args_cli.target_object
    usd_path = usd_path_mapping[object_name]

    obj = RigidObject(
        cfg=RigidObjectCfg(
            prim_path=prim_path,
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd_path,
                scale=(1.0, 1.0, 1.0),
                semantic_tags=[("class", "mug"), ("color", "red")],
                **common_properties,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=ori),
        )
    )

    scene_entities["target"] = obj
    scene_entities["target_pos"] = torch.tensor(pos)

    return scene_entities


def scene_update(
    sim: sim_utils.SimulationContext, scene_entities: dict, scene_count: int
):
    target: RigidObject = scene_entities["target"]
    state_w = torch.zeros(7)

    quat_w = math.quat_from_euler_xyz(
        torch.tensor([0]), torch.tensor([0]), torch.tensor([(scene_count % 8) * 0.79])
    )

    state_w[0:3] = scene_entities["target_pos"]
    state_w[3:7] = quat_w
    if (scene_count % 8) == 0:
        if state_w[1] < 0.80:  # 원래 0.76 / 5x3 실증용 0.80 / 4x3 실증용 0.80
            state_w[1] = state_w[1] + 0.01
        elif state_w[1] >= 0.80:  # 원래 0.76 / 5x3 실증용 0.80 / 4x3 실증용 0.80
            state_w[0] = state_w[0] + 0.065  # 원래 0.12 / 5x3 실증용 0.06 / 4x3 실증용 0.065
            state_w[1] = 0.0

    for key in list(scene_entities.keys()):
        if key == "target":
            prim_utils.delete_prim(scene_entities[key].cfg.prim_path)
            del scene_entities[key]

    new_entities = object_spawn(scene_entities, pos=state_w[0:3], ori=state_w[3:7])
    scene_entities.update(new_entities)

    if (scene_count > 100) & (state_w[0] == 0.065 * args_cli.row):  # 원래 0.12 / 5x3 실증용 0.06 / 4x3 실증용 0.065
        raise RuntimeError


def run_simulator(sim: sim_utils.SimulationContext, scene_entities: dict):
    """Run the simulator."""

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    scene_count = 0
    # extract entities for simplified notation
    camera: Camera = scene_entities["camera"]

    # Create replicator writer
    output_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "output",
        "camera",
        f"{args_cli.target_object}",
        "target",
    )
    rep_writer = rep.BasicWriter(
        output_dir=output_dir,
        frame_padding=0,
        colorize_instance_id_segmentation=camera.cfg.colorize_instance_id_segmentation,
        colorize_instance_segmentation=camera.cfg.colorize_instance_segmentation,
        colorize_semantic_segmentation=camera.cfg.colorize_semantic_segmentation,
    )

    # Camera positions, targets, orientations
    camera_positions = torch.tensor([[1.18, 0.0, 1.27]], device=sim.device)
    camera_targets = torch.tensor([[0.0, 0.0, 1.31]], device=sim.device)

    # Set pose: There are two ways to set the pose of the camera.
    # -- Option-1: Set pose using view
    camera.set_world_poses_from_view(camera_positions, camera_targets)

    # Index of the camera to use for visualization and saving
    camera_index = args_cli.camera_id

    # Create the markers for the --draw option outside of is_running() loop
    if sim.has_gui() and args_cli.draw:
        cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/CameraPointCloud")
        cfg.markers["hit"].radius = 0.002
        pc_markers = VisualizationMarkers(cfg)

    # Simulate physics
    while simulation_app.is_running():
        # Step simulation
        sim.step()
        # Update camera data
        camera.update(dt=sim.get_physics_dt())

        # update step count
        count += 1

        if count % 10 == 0:

            # Extract camera data
            if args_cli.save:
                # Save images from camera at camera_index
                # note: BasicWriter only supports saving data in numpy format, so we need to convert the data to numpy.
                # tensordict allows easy indexing of tensors in the dictionary
                single_cam_data = convert_dict_to_backend(
                    {k: v[camera_index] for k, v in camera.data.output.items()},
                    backend="numpy",
                )

                # Extract the other information
                single_cam_info = camera.data.info[camera_index]

                # Pack data back into replicator format to save them using its writer
                rep_output = {"annotators": {}}
                for key, data, info in zip(
                    single_cam_data.keys(),
                    single_cam_data.values(),
                    single_cam_info.values(),
                ):
                    if info is not None:
                        rep_output["annotators"][key] = {
                            "render_product": {"data": data, **info}
                        }
                    else:
                        rep_output["annotators"][key] = {
                            "render_product": {"data": data}
                        }
                # Save images
                # Note: We need to provide On-time data for Replicator to save the images.
                rep_output["trigger_outputs"] = {"on_time": camera.frame[camera_index]}
                rep_writer.write(rep_output)

            scene_update(
                sim=sim, scene_entities=scene_entities, scene_count=scene_count
            )
            scene_count += 1


def main():
    """Main function."""
    # Load simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[1.5, 0.0, 1.5], target=[0.0, 0.0, 1.3])
    # design the scene
    scene_entities = design_scene()
    # Play simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run simulator
    run_simulator(sim, scene_entities)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
