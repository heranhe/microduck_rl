"""Jumper lab prototype for mjlab 1.3.0; separate from the duck's BAM contract.

71 observations, 20 gait actions, six-foot contacts, no history or servo thermal
model. This is a port of the robot facts, not the upstream trained-policy layout.
"""

import math

import torch
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.config.go1.rl_cfg import unitree_go1_ppo_runner_cfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from ..robot import jumper_model as J


def get_spec():
    spec = J.robot_spec()
    # Entity creates its own actuators. Do not double-actuate each joint.
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    return spec


def gait_phase(env):
    angle = (env.episode_length_buf % 16).float() * (2 * math.pi / 16)
    command = env.command_manager.get_command("twist")
    moving = torch.linalg.vector_norm(command[:, :3], dim=1) > 0.05
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1) * moving[:, None]


def tripod_match(env, sensor_name="feet_ground_contact"):
    command = env.command_manager.get_command("twist")
    moving = torch.linalg.vector_norm(command[:, :3], dim=1) > 0.05
    found = env.scene[sensor_name].data.found.reshape(env.num_envs, 6, -1).any(dim=-1)
    group = torch.tensor([True, False, False, True, True, False], device=env.device)
    swing = torch.where(((env.episode_length_buf % 16) < 8)[:, None], group, ~group)
    desired = torch.where(moving[:, None], ~swing, torch.ones_like(swing))
    return (found == desired).float().mean(dim=1)


def make_jumper_env_cfg(play=False, stand=False):
    cfg = make_velocity_env_cfg()
    cfg.decimation = 20
    cfg.sim.mujoco.timestep = J.PHYSICS_DT
    cfg.sim.mujoco.integrator = "implicitfast"
    cfg.sim.njmax = 512
    cfg.sim.nconmax = 128
    cfg.sim.contact_sensor_maxmatch = 128
    cfg.scene.entities = {
        "robot": EntityCfg(
            spec_fn=get_spec,
            init_state=EntityCfg.InitialStateCfg(
                pos=(0.0, 0.0, J.STAND_Z + 0.002),
                joint_pos=dict(J.HOME),
                joint_vel={".*": 0.0},
            ),
            articulation=EntityArticulationInfoCfg(
                actuators=(
                    BuiltinPositionActuatorCfg(
                        target_names_expr=(".*_joint",),
                        stiffness=J.FACTS["stiffness"],
                        damping=J.FACTS["damping"],
                        effort_limit=J.FACTS["effort_limit"],
                    ),
                ),
                soft_joint_pos_limit_factor=0.9,
            ),
        )
    }
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.sensors = (
        ContactSensorCfg(
            name="feet_ground_contact",
            primary=ContactMatch(mode="geom", pattern=J.FOOT_GEOMS, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
            track_air_time=True,
        ),
    )
    joints = SceneEntityCfg("robot", joint_names=J.JOINT_NAMES, preserve_order=True)
    # Declare the same ordered contract the CPU env fills, with no history.
    base = cfg.observations["actor"].terms
    terms = {
        name: base[name]
        for name in (
            "base_ang_vel",
            "projected_gravity",
            "joint_pos",
            "joint_vel",
            "actions",
            "command",
        )
    }
    for name in ("joint_pos", "joint_vel"):
        terms[name].params = {"asset_cfg": joints}
    terms["gait_phase"] = ObservationTermCfg(func=gait_phase)
    from copy import deepcopy

    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms=terms, concatenate_terms=True, enable_corruption=not play
        ),
        "critic": ObservationGroupCfg(
            terms=deepcopy(terms), concatenate_terms=True, enable_corruption=False
        ),
    }
    cfg.actions["joint_pos"].actuator_names = J.JOINT_NAMES
    cfg.actions["joint_pos"].scale = J.ACTION_SCALE
    cfg.actions["joint_pos"].preserve_order = True
    keep = (
        "track_linear_velocity",
        "track_angular_velocity",
        "upright",
        "pose",
        "dof_pos_limits",
        "action_rate_l2",
    )
    cfg.rewards = {name: cfg.rewards[name] for name in keep}
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("base_link",)
    cfg.rewards["pose"].params.update(
        asset_cfg=joints,
        std_standing={".*": 0.1},
        std_walking={".*": 0.3},
        std_running={".*": 0.3},
    )
    cfg.rewards["tripod_gait"] = RewardTermCfg(func=tripod_match, weight=1.0)
    cfg.terminations = {
        "time_out": cfg.terminations["time_out"],
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation, params={"limit_angle": math.radians(50.0)}
        ),
    }
    cfg.curriculum = {}
    cfg.events.pop("push_robot", None)
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = J.FOOT_GEOMS
    cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
    cfg.events["base_com"].params["ranges"] = {
        0: (-0.002, 0.002),
        1: (-0.002, 0.002),
        2: (-0.002, 0.002),
    }
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.0, 0.002)
    cmd = cfg.commands["twist"]
    cmd.rel_heading_envs = 0.0
    cmd.rel_forward_envs = 0.0
    cmd.heading_command = False
    cmd.ranges.heading = None
    cmd.rel_standing_envs = 1.0 if stand else 0.2
    cmd.ranges.lin_vel_x = (-0.2, 0.2)
    cmd.ranges.lin_vel_y = (-0.2, 0.2)
    cmd.ranges.ang_vel_z = (-0.5, 0.5)
    cfg.viewer.body_name = "base_link"
    cfg.viewer.distance = 1.0
    if play:
        for key in ("base_com", "encoder_bias", "foot_friction"):
            cfg.events.pop(key, None)
    return cfg


def jumper_rl_cfg(stand=False):
    cfg = unitree_go1_ppo_runner_cfg()
    cfg.clip_actions = 4.0
    cfg.actor.hidden_dims = cfg.critic.hidden_dims = (128, 128)
    cfg.actor.obs_normalization = cfg.critic.obs_normalization = True
    cfg.experiment_name = "jumper_stand" if stand else "jumper_tripod"
    return cfg


def register_jumper_tasks():
    from mjlab.tasks.registry import register_mjlab_task
    from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

    for task, stand in (("Tripod", False), ("Stand", True)):
        register_mjlab_task(
            task_id=f"Mjlab-{task}-Flat-Jumper",
            env_cfg=make_jumper_env_cfg(stand=stand),
            play_env_cfg=make_jumper_env_cfg(play=True, stand=stand),
            rl_cfg=jumper_rl_cfg(stand),
            runner_cls=VelocityOnPolicyRunner,
        )
