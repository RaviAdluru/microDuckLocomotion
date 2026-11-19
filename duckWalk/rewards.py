"""Reward functions for walking, grouped into sets so we can ablate them.
minimal -> posture -> full, each set adds terms on top of the previous one.
"""

import math

from mjlab.envs import mdp as envMdp
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as velMdp
from mjlab_microduck.tasks import mdp as duckMdp

FEET = ["left_foot", "right_foot"]
LEG_JOINTS = r"^(?!passive_|.*neck.*|.*head.*).*"

# how loose each joint group is allowed to be around the home pose
STD_STANDING = {
    r".*hip_yaw.*": 0.1,
    r".*hip_roll.*": 0.05,
    r".*hip_pitch.*": 0.15,
    r".*knee.*": 0.15,
    r".*ankle.*": 0.1,
}
STD_WALKING = {
    r".*hip_yaw.*": 0.3,
    r".*hip_roll.*": 0.05,
    r".*hip_pitch.*": 0.4,
    r".*knee.*": 0.4,
    r".*ankle.*": 0.25,
}


def minimalRewards():
    return {
        "trackLinVel": RewardTermCfg(
            func=velMdp.track_linear_velocity,
            weight=2.0,
            params={"command_name": "twist", "std": math.sqrt(0.1)},
        ),
        "trackAngVel": RewardTermCfg(
            func=velMdp.track_angular_velocity,
            weight=2.0,
            params={"command_name": "twist", "std": math.sqrt(0.5)},
        ),
        "alive": RewardTermCfg(func=envMdp.is_alive, weight=0.5),
        "fell": RewardTermCfg(func=envMdp.is_terminated, weight=-10.0),
        # the neck/head motors are in the action space too, this just keeps them still
        "headHold": RewardTermCfg(
            func=duckMdp.head_pose_tracking,
            weight=2.0,
            params={"command_name": "head_pose", "std": 0.5},
        ),
    }


def postureRewards():
    rewards = minimalRewards()
    rewards.update(
        {
            "upright": RewardTermCfg(
                func=velMdp.upright,
                weight=2.0,
                params={
                    "std": math.sqrt(0.05),
                    "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                },
            ),
            "pose": RewardTermCfg(
                func=velMdp.variable_posture,
                weight=1.0,
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=(LEG_JOINTS,)),
                    "command_name": "twist",
                    "std_standing": STD_STANDING,
                    "std_walking": STD_WALKING,
                    "std_running": STD_WALKING,
                    "walking_threshold": 0.01,
                    "running_threshold": 1.5,
                },
            ),
            "jointLimits": RewardTermCfg(func=envMdp.joint_pos_limits, weight=-1.0),
            "actionRate": RewardTermCfg(func=envMdp.action_rate_l2, weight=-0.5),
        }
    )
    return rewards


def fullRewards():
    rewards = postureRewards()
    rewards.update(
        {
            # reward steps that stay in the air between 0.125 and 0.3 s
            "airTime": RewardTermCfg(
                func=velMdp.feet_air_time,
                weight=3.0,
                params={
                    "sensor_name": "feet_ground_contact",
                    "threshold_min": 0.125,
                    "threshold_max": 0.300,
                    "command_name": "twist",
                    "command_threshold": 0.01,
                },
            ),
            # the duck can only lift its feet ~2 cm, so the targets are small
            "footClearance": RewardTermCfg(
                func=velMdp.feet_clearance,
                weight=-2.0,
                params={
                    "target_height": 0.02,
                    "height_sensor_name": "foot_height_scan",
                    "command_name": "twist",
                    "command_threshold": 0.01,
                    "asset_cfg": SceneEntityCfg("robot", site_names=FEET),
                },
            ),
            "footSwingHeight": RewardTermCfg(
                func=velMdp.feet_swing_height,
                weight=-0.25,
                params={
                    "sensor_name": "feet_ground_contact",
                    "height_sensor_name": "foot_height_scan",
                    "target_height": 0.02,
                    "command_name": "twist",
                    "command_threshold": 0.01,
                },
            ),
            "footSlip": RewardTermCfg(
                func=velMdp.feet_slip,
                weight=-0.1,
                params={
                    "sensor_name": "feet_ground_contact",
                    "command_name": "twist",
                    "command_threshold": 0.01,
                    "asset_cfg": SceneEntityCfg("robot", site_names=FEET),
                },
            ),
            "selfCollision": RewardTermCfg(
                func=velMdp.self_collision_cost,
                weight=-1.0,
                params={"sensor_name": "self_collision"},
            ),
            "bodyAngVel": RewardTermCfg(
                func=velMdp.body_angular_velocity_penalty,
                weight=-0.05,
                params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
            ),
            "angularMomentum": RewardTermCfg(
                func=velMdp.angular_momentum_penalty,
                weight=-0.02,
                params={"sensor_name": "robot/root_angmom"},
            ),
        }
    )
    return rewards


REWARD_SETS = {
    "minimal": minimalRewards,
    "posture": postureRewards,
    "full": fullRewards,
}


def buildRewards(name):
    if name not in REWARD_SETS:
        raise ValueError(f"unknown reward set '{name}', pick from {list(REWARD_SETS)}")
    return REWARD_SETS[name]()
