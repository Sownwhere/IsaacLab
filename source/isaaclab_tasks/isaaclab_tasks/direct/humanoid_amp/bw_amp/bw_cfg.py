import isaaclab.sim as sim_utils

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


BW_CFG = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(CURRENT_DIR, "../usd/bw.usd"),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=3.0,
                max_angular_velocity=3.0,
                max_depenetration_velocity=10.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.4),
            joint_pos={
                ".*_leg_pitch_joint": 0.0,
                ".*_knee_joint": 0.0,
                ".*_ankle_pitch_joint": 0.0,
                # ".*_elbow_pitch_joint": 0.87
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_leg_pitch_joint",
                    ".*_leg_roll_joint",
                    ".*_leg_yaw_joint",
                    ".*_knee_joint",
                ],
                effort_limit=300,
                velocity_limit=100.0,
                stiffness={
                    ".*_leg_yaw_joint": 150.0,
                    ".*_leg_roll_joint": 150.0,
                    ".*_leg_pitch_joint": 200.0,
                    ".*_knee_joint": 200.0,
                },
                damping={
                    ".*_leg_pitch_joint": 5.0,
                    ".*_leg_roll_joint": 5.0,
                    ".*_leg_yaw_joint": 5.0,
                    
                    ".*_knee_joint": 5.0,
                },
                # armature={
                #     ".*_leg_.*": 0.01,
                #     ".*_knee_joint": 0.01,
                # },
            ),
            "feet": ImplicitActuatorCfg(
                effort_limit=20,
                joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                stiffness=20.0,
                damping=2.0,
                # armature=0.01,
            ),
        },
    )   