"""Builds the mjlab environment we train in.
We borrow the robot, actuator model, sensors and sim settings from Pollen's microduck_rl,
then swap in our own rewards, observations, terrains and randomization.
"""

from mjlab.envs import ManagerBasedRlEnv
from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg

from duckWalk.rewards import buildRewards
from duckWalk.terrains import trainTerrains

# events that randomize the robot/world. everything else (resets, BAM bookkeeping) always stays
DOMAIN_RAND_EVENTS = [
    "push_robot",
    "foot_friction",
    "encoder_bias",
    "base_com",
    "randomize_com",
    "randomize_head_com",
    "randomize_mass_inertia",
    "randomize_joint_friction",
    "randomize_armature",
]


def makeEnvCfg(
    rewardSet="full",
    terrainSet="mixed",
    historyLength=1,
    domainRand=True,
    numEnvs=4096,
    play=False,
    terrainOverride=None,
):
    isFlat = terrainSet == "flat" and terrainOverride is None
    cfg = make_microduck_velocity_env_cfg(play=play, rough=not isFlat)
    cfg.scene.num_envs = numEnvs

    # Pollen's config also commands head and body poses. we only care about walking,
    # so pin those commands to zero and hide them from the networks
    for name in ["head_pose", "body_pose"]:
        cfg.commands[name].zero_command_prob = 1.0
    for group in ["actor", "critic"]:
        for term in ["head_command", "body_command"]:
            cfg.observations[group].terms.pop(term, None)

    # actor only gets what the real robot can measure (IMU, encoders, last action, command).
    # the critic additionally sees true base velocity and foot contact info, which only
    # exists in sim. it's thrown away after training so this is fine
    # observation history: the actor sees its last N frames instead of just the current one.
    # mjlab keeps a ring buffer per term, oldest to newest, and flattens it, so each term
    # becomes N copies of itself (joint_pos: 14 -> 14*N) and the actor input is 51*N.
    # on reset the buffer is filled with the first frame of the new episode, so nothing
    # carries over between episodes. only the actor gets history: the critic already sees
    # the privileged state it would otherwise have to infer
    if historyLength > 1:
        cfg.observations["actor"].history_length = historyLength

    cfg.rewards = buildRewards(rewardSet)

    if terrainOverride is not None:
        cfg.scene.terrain.terrain_generator = terrainOverride
    elif not isFlat:
        cfg.scene.terrain.terrain_generator = trainTerrains(terrainSet)

    # their curricula ramp reward weights and head commands we removed. keep only the
    # terrain curriculum, and only when there's a curriculum terrain to move through
    keepTerrainCurriculum = not isFlat and terrainOverride is None and not play
    terrainCurriculum = cfg.curriculum.get("terrain_levels")
    cfg.curriculum = {"terrain_levels": terrainCurriculum} if keepTerrainCurriculum else {}

    if not domainRand:
        for name in DOMAIN_RAND_EVENTS:
            cfg.events.pop(name, None)

    return cfg


def makeEnv(device="cuda:0", **kwargs):
    return ManagerBasedRlEnv(makeEnvCfg(**kwargs), device=device)
