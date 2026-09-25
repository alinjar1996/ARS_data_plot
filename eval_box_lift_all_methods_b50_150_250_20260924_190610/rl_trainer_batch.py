"""
ARS/PPO trainer for optimizing CEM planner cost weights — lift-cube task.

What is being learned:
  The 10 cost_weights: collision, theta, z-axis, velocity, distance,
  orientation, eef_to_obj, obj_to_targ, object_orientation, smoothness.

ARS reward signal (per episode):
  +grasp_bonus  EEFs reach box grasp sites (pick → move transition)
  +success      cube reaches target pos+rot
  -fall         cube dropped during move
  -collision    raw collision penalty from CEM
  + shaped:     -current_cost_g (EEF→box sites, pick)
                -cost_r         (EEF rotation)
                -cost_g_ball    (cube pos→target, move)
                -cost_r_ball    (cube rot→target, move)

PPO-v3 keeps the same sparse task-event scales but uses geometric progress,
an all-step time cost, a move-phase grasp-hold cost, and separate pre/post-
grasp timeout penalties. See PPO.py and PPO_BOX_LIFT_PIPELINE.md.

Usage:
    python rl_trainer_batch.py --episodes 500
    python rl_trainer_batch.py --load checkpoint.json --episodes 500
"""
import jax
import jax.numpy as jnp
import os
import sys
import time
import argparse
import json
import numpy as np
import mujoco
import mujoco.mjx as mjx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ARS import ARS, ARSV2, LinearPolicy, _COST_KEYS, ACT_DIM_COST, GRASP_BOUND
from PPO import PPO
from sampling_based_planner.mpc_planner import run_cem_planner


# ---------------------------------------------------------------------------
# JAX-native quaternion helpers (safe inside jax.lax.scan)
# ---------------------------------------------------------------------------

def _jax_rotvec_to_quat(rv):
    """Rotation vector (axis*angle) → unit quaternion [w,x,y,z]. rv=[0,0,0] → identity."""
    angle = jnp.linalg.norm(rv) + 1e-8
    axis  = rv / angle
    half  = angle / 2.0
    return jnp.concatenate([jnp.expand_dims(jnp.cos(half), 0), jnp.sin(half) * axis])

def _jax_quat_multiply(q1, q2):
    """Hamilton product q1 * q2, both [w,x,y,z]."""
    w1,x1,y1,z1 = q1[0],q1[1],q1[2],q1[3]
    w2,x2,y2,z2 = q2[0],q2[1],q2[2],q2[3]
    return jnp.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

# These constants are retained for evaluation metadata. The 4216 dynamics
# below always use fixed wrist targets, the +/-x grasp-site pair, and physical
# (non-slaved) cube motion.
LEGACY_EEF_ROT_TARGETS = True
_LEGACY_BASE_Q0 = [0.183, -0.683, -0.683,  0.183]
_LEGACY_BASE_Q1 = [0.183, -0.683,  0.683, -0.183]
FORCE_XY_FACE_PAIR = True
DISABLE_CUBE_SLAVING = True


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

class EpisodeRunner:
    """
    Runs one episode of the lift-cube planner in MuJoCo (headless, no viewer).
    Observation: 53-D
      joint_pos(12) + ball_pos(3) + ball_quat(4) + joint_vel(12)
      + eef_0(7) + eef_1(7) + target(7) + task_flag(1)
    """

    def __init__(self, model_path: str, num_batch=None, num_steps=None,
                 maxiter_cem=None, maxiter_projection=None, max_steps=None):

        self.key = jax.random.PRNGKey(42)
        self.MAX_STEPS = max_steps
        self.timestep  = 0.1
        self.num_dof   = 12

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model_path = model_path
        self.model.opt.timestep = self.timestep
        self.data = mujoco.MjData(self.model)

        self.mjx_model = mjx.put_model(self.model)
        self.mjx_data  = mjx.put_data(self.model, self.data)
        self.mjx_data  = jax.jit(mjx.forward)(self.mjx_model, self.mjx_data)
        self.jit_step    = jax.jit(mjx.step)
        self.jit_forward = jax.jit(mjx.forward)

        # Joint masks
        joint_names_pos, joint_names_vel = [], []
        for i in range(self.model.njnt):
            jt    = self.model.jnt_type[i]
            n_pos = 7 if jt == mujoco.mjtJoint.mjJNT_FREE else \
                    4 if jt == mujoco.mjtJoint.mjJNT_BALL else 1
            n_vel = 6 if jt == mujoco.mjtJoint.mjJNT_FREE else \
                    3 if jt == mujoco.mjtJoint.mjJNT_BALL else 1
            name  = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            joint_names_pos.extend([name] * n_pos)
            joint_names_vel.extend([name] * n_vel)

        robot_joints = [
            'shoulder_pan_joint_1',  'shoulder_lift_joint_1',
            'elbow_joint_1',         'wrist_1_joint_1',
            'wrist_2_joint_1',       'wrist_3_joint_1',
            'shoulder_pan_joint_2',  'shoulder_lift_joint_2',
            'elbow_joint_2',         'wrist_1_joint_2',
            'wrist_2_joint_2',       'wrist_3_joint_2',
        ]
        self.joint_mask_pos = np.isin(joint_names_pos, robot_joints)
        self.joint_mask_vel = np.isin(joint_names_vel, robot_joints)

        self.ball_qpos_idx = self.model.body_dofadr[self.model.body(name="ball").id]
        self.ball_id       = self.model.body(name="ball").id

        self.home_pos = np.array([
            1.5, -1.8,  1.75, -1.25, -1.6, 0,
           -1.5, -1.8,  1.75, -1.25, -1.6, 0,
        ])

        self.init_noise    = np.full(self.num_dof, 0.3)
        self.ball_pos_noise = np.full(3, 0.01)
        # Initial cube z-rotation randomization (deg). 0.0 = legacy (fixed orientation).
        # x-face grasp is IK-reachable to ~50deg, so up to ~40deg is safe with one grasp.
        self.init_cube_rot_deg = 0.0

        self.data.qpos[self.joint_mask_pos] = self.home_pos
        self.data.qvel[self.joint_mask_vel] = 0
        mujoco.mj_forward(self.model, self.data)
        self.ball_base_pose = self.data.qpos[
            self.ball_qpos_idx:self.ball_qpos_idx+7].copy()

        self.robot_geom_ids = {
            i for i in range(self.model.ngeom)
            if (n := mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i))
            and n.startswith("robot")
        }

        self.hande_id_0 = self.model.body(name="hande_0").id
        self.tcp_id_0   = self.model.site(name="tcp_0").id
        self.hande_id_1 = self.model.body(name="hande_1").id
        self.tcp_id_1   = self.model.site(name="tcp_1").id

        # Target mocap body (for rendering in save_best_rollout_video)
        self.target_mocap_id = self.model.body_mocapid[
            self.model.body(name='target_0').id]

        # Planner
        self.planner = run_cem_planner(
            model=self.model, data=self.data,
            num_dof=self.num_dof, num_batch=num_batch,
            num_steps=num_steps, maxiter_cem=maxiter_cem,
            maxiter_projection=maxiter_projection,
            num_elite=0.2, timestep=self.timestep,
            position_threshold=0.06, rotation_threshold=0.1,
        )

        # Box grasp site IDs (fixed on the cube surface, updated by FK)
        self.box_site_1_id = self.planner.cem.box_site_1_id
        self.box_site_2_id = self.planner.cem.box_site_2_id
        # Lift-cube grab thresholds (from dual_arm_demo.py)
        self.grab_pos_thresh = 0.07   # avg EEF→box-site distance
        self.grab_rot_thresh = 0.13   # per-EEF rotation error
        self.success_pos_thresh = 0.04
        self.success_rot_thresh = 0.1  # quaternion L2 distance, not radians
        self.fall_pos_thresh = 0.2

    # ============================
    # JAX FUNCTIONS
    # ============================

    def reset_jax(self, key, max_yaw_deg=None):
        # The zero-yaw path deliberately uses the exact five-way key split from
        # 4216. A nonzero explicit yaw range is retained only for standalone
        # compatibility experiments; it is not used by the 4280 command.
        fixed_4216_reset = max_yaw_deg is None and self.init_cube_rot_deg == 0.0
        if fixed_4216_reset:
            key1, key2, key3, key4, key5 = jax.random.split(key, 5)
        else:
            if max_yaw_deg is None:
                max_yaw_deg = self.init_cube_rot_deg
            key1, key2, key3, key4, key5, key6 = jax.random.split(key, 6)

        # Ball position noise
        ball_noise = jax.random.uniform(
            key1, (2,), minval=-self.ball_pos_noise[:2], maxval=self.ball_pos_noise[:2]
        )
        ball_pose = jnp.array(self.ball_base_pose).at[:2].add(ball_noise)

        if not fixed_4216_reset:
            init_ang = jax.random.uniform(
                key6, (), minval=-max_yaw_deg, maxval=max_yaw_deg)
            _half = jnp.deg2rad(init_ang) / 2.0
            rz_quat = jnp.stack([
                jnp.cos(_half), jnp.zeros(()), jnp.zeros(()), jnp.sin(_half)
            ])
            ball_init_quat = _jax_quat_multiply(
                rz_quat, ball_pose[3:7])
            ball_pose = ball_pose.at[3:7].set(ball_init_quat)

        # Joint position noise
        init_noise = jax.random.uniform(
            key2, (self.num_dof,), minval=-self.init_noise, maxval=self.init_noise
        )
        qpos = jnp.array(self.data.qpos)
        qpos = qpos.at[self.joint_mask_pos].set(self.home_pos + init_noise)
        qpos = qpos.at[self.ball_qpos_idx:self.ball_qpos_idx+7].set(ball_pose)
        qvel = jnp.array(self.data.qvel)

        # Target position (matches pyt_dual_demo.generate_targets area)
        center = jnp.array([-0.3, -0.1, 0.3])
        size   = jnp.array([0.05, 0.05, 0.05])
        # size   = jnp.array([0.06, 0.07, 0.1])
        target_pos = center + jax.random.uniform(key3, (3,), minval=-size, maxval=size)

        # Target rotation: ±[5, 15] degrees around z-axis → quaternion [w, 0, 0, sin]
        magnitude  = jax.random.uniform(key4, (), minval=5.0, maxval=15.0)  # degrees
        flip       = jax.random.uniform(key5, ()) > 0.5
        angle_deg  = jax.lax.select(flip, magnitude, -magnitude)
        half_rad   = jnp.deg2rad(angle_deg) / 2.0
        zero       = jnp.zeros(())
        target_rot = jnp.stack([jnp.cos(half_rad), zero, zero, jnp.sin(half_rad)])

        target = jnp.concatenate([target_pos, target_rot])  # 7-D

        return {
            "qpos":     qpos,
            "qvel":     qvel,
            "thetadot": jnp.zeros(self.num_dof),
            "task":     jnp.array(0.0),
            "target":   target,           # 7-D: [pos(3), rot(4)]
            "xi_mean":  jnp.array(self.planner.xi_mean),
            "xi_cov":   jnp.array(self.planner.xi_cov),
        }

    def get_obs_jax(self, state):
        """
        53-D observation:
          joint_pos(12) + ball_pos(3) + ball_quat(4) + joint_vel(12)
          + eef_0(7) + eef_1(7) + target(7) + task_flag(1)
        """
        joint_pos = state["qpos"][self.joint_mask_pos]                            # 12
        ball_pos  = state["qpos"][self.ball_qpos_idx:self.ball_qpos_idx + 3]     #  3
        ball_quat = state["qpos"][self.ball_qpos_idx+3:self.ball_qpos_idx + 7]   #  4
        joint_vel = state["qvel"][self.joint_mask_vel]                            # 12

        mjx_data  = self.mjx_data.replace(qpos=state["qpos"], qvel=state["qvel"])
        mjx_data  = self.jit_forward(self.mjx_model, mjx_data)

        eef_0 = jnp.concatenate([mjx_data.site_xpos[self.tcp_id_0],              #  7
                                  mjx_data.xquat[self.hande_id_0]])
        eef_1 = jnp.concatenate([mjx_data.site_xpos[self.tcp_id_1],              #  7
                                  mjx_data.xquat[self.hande_id_1]])

        target    = state["target"]                                               #  7
        task_flag = jnp.array([state["task"]])                                   #  1

        return jnp.concatenate([
            joint_pos, ball_pos, ball_quat, joint_vel, eef_0, eef_1, target, task_flag
        ])  # 53

    def _task_costs_from_mjx(self, mjx_data, target, delta_pos_1,
                             delta_pos_2, grasp_delta_rot_0,
                             grasp_delta_rot_1):
        """Return the four box-task geometric errors for ARS/PPO rewards."""
        eef_0_pos = mjx_data.site_xpos[self.tcp_id_0]
        eef_0_quat = mjx_data.xquat[self.hande_id_0]
        eef_1_pos = mjx_data.site_xpos[self.tcp_id_1]
        eef_1_quat = mjx_data.xquat[self.hande_id_1]

        target_1_pos = mjx_data.site_xpos[self.box_site_1_id] + delta_pos_1
        target_2_pos = mjx_data.site_xpos[self.box_site_2_id] + delta_pos_2
        current_cost_g = (
            jnp.linalg.norm(eef_0_pos - target_1_pos)
            + jnp.linalg.norm(eef_1_pos - target_2_pos)
        ) / 2.0

        target_q0 = _jax_quat_multiply(
            grasp_delta_rot_0, jnp.array(_LEGACY_BASE_Q0))
        target_q1 = _jax_quat_multiply(
            grasp_delta_rot_1, jnp.array(_LEGACY_BASE_Q1))
        cost_r_0 = 2 * jnp.arccos(jnp.clip(
            jnp.abs(jnp.dot(eef_0_quat, target_q0)), 0.0, 1.0))
        cost_r_1 = 2 * jnp.arccos(jnp.clip(
            jnp.abs(jnp.dot(eef_1_quat, target_q1)), 0.0, 1.0))
        cost_r = 0.5 * (cost_r_0 + cost_r_1)

        ball_pos = mjx_data.qpos[
            self.ball_qpos_idx:self.ball_qpos_idx + 3]
        ball_quat = mjx_data.qpos[
            self.ball_qpos_idx + 3:self.ball_qpos_idx + 7]
        cost_g_ball = jnp.linalg.norm(ball_pos - target[:3])
        cost_r_ball = jnp.linalg.norm(ball_quat - target[3:])
        return (current_cost_g, cost_r, cost_g_ball, cost_r_ball,
                cost_r_0, cost_r_1)

    def get_task_costs_jax(self, state):
        """Geometric costs used for PPO progress shaping.

        PPO is cost-only, so its grasp position/rotation residuals are exactly
        zero and the fixed 4216 grasp targets are used here.
        """
        mjx_data = self.mjx_data.replace(
            qpos=state["qpos"], qvel=state["qvel"])
        mjx_data = self.jit_forward(self.mjx_model, mjx_data)
        identity_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
        costs = self._task_costs_from_mjx(
            mjx_data, state["target"], jnp.zeros(3), jnp.zeros(3),
            identity_quat, identity_quat)
        return jnp.array(costs[:4])

    def step_jax(self, state, policy_output, key):
        """
        policy_output: 23-D JAX array from _policy_forward.
          [0:10]  log cost weights (clipped, to be exponentiated)
          [10:13] grasp site 1 position delta (metres)
          [13:16] grasp site 2 position delta (metres)
          [16:19] EEF-0 rotation vector perturbation (rad, axis*angle)
          [19:22] EEF-1 rotation vector perturbation (rad, axis*angle)
          [22]    grasp-side logit (combined with geometric prior -> +/-x vs +/-y pair)
        """
        qpos     = state["qpos"]
        qvel     = state["qvel"]
        thetadot = state["thetadot"]
        task_flag = state["task"]

        # Extract parts from policy output
        cost_weights = jnp.exp(policy_output[:ACT_DIM_COST])
        delta_pos_1  = policy_output[ACT_DIM_COST    :ACT_DIM_COST + 3]
        delta_pos_2  = policy_output[ACT_DIM_COST + 3:ACT_DIM_COST + 6]
        rv_eef_0     = policy_output[ACT_DIM_COST + 6:ACT_DIM_COST + 9]
        rv_eef_1     = policy_output[ACT_DIM_COST + 9:ACT_DIM_COST + 12]

        # Task flag (0 = pick, 1 = move)
        is_pick = task_flag < 0.5
        task_id = jax.lax.select(is_pick, 0, 1)

        # Convert flat array → named dicts expected by the lift-cube planner
        cost_weights_dict = {k: cost_weights[i] for i, k in enumerate(_COST_KEYS)}
        cost_task_weights_dict = {
            'pick': (task_id == 0).astype(jnp.float32),
            'move': (task_id == 1).astype(jnp.float32),
        }

        # Ball pose from state (7-D: pos + quat)
        ball_pose = qpos[self.ball_qpos_idx:self.ball_qpos_idx + 7]

        # target_0_state is already 7-D (pos + rot) for the lift-cube task
        target_0_state = state["target"]

        xi_samples, new_key = self.planner.cem.compute_xi_samples(
            key, state["xi_mean"], state["xi_cov"]
        )

        # Convert policy rotation-vector perturbations to unit quaternions for CEM
        grasp_delta_rot_0 = _jax_rotvec_to_quat(rv_eef_0)
        grasp_delta_rot_1 = _jax_rotvec_to_quat(rv_eef_1)

        # CEM call — grasp deltas applied inside the cost function before physics
        (cost, best_cost_list, best_vels, best_traj, xi_mean, xi_cov,
         thetadot_all, theta_all, avg_res_primal, avg_res_fixed,
         primal_residuals, fixed_point_residuals, idx_min, ball_out,
         eef_0_planned, eef_1_planned, eef_0, eef_1, cem_batch_costs,
        ) = self.planner.cem.compute_cem(
            state["xi_mean"],
            state["xi_cov"],
            qpos[self.joint_mask_pos],
            thetadot,
            jnp.zeros_like(thetadot),
            target_0_state,
            ball_pose,
            ball_pose,           # ball_pick_init = current ball pose
            self.planner.lamda_init,
            self.planner.s_init,
            xi_samples,
            cost_weights_dict,
            cost_task_weights_dict,
            delta_pos_1,
            delta_pos_2,
            grasp_delta_rot_0,
            grasp_delta_rot_1,
        )

        thetadot = jnp.mean(best_vels[1:6], axis=0)

        # Physics step
        qvel_full = jnp.zeros_like(state["qvel"])
        qvel_full = qvel_full.at[self.joint_mask_vel].set(thetadot)

        mjx_data = self.mjx_data.replace(qpos=qpos, qvel=qvel_full)
        mjx_data = self.jit_step(self.mjx_model, mjx_data)

        new_qpos = mjx_data.qpos
        new_qvel = mjx_data.qvel

        # Post-step geometric errors. These are the exact errors used by the
        # 4216 event logic and exposed to PPO for progress shaping.
        (current_cost_g, cost_r, cost_g_ball, cost_r_ball,
         cost_r_0, cost_r_1) = self._task_costs_from_mjx(
            mjx_data, state["target"], delta_pos_1, delta_pos_2,
            grasp_delta_rot_0, grasp_delta_rot_1)

        # Collision from CEM (unscaled by dividing out the collision weight)
        cost_collision  = best_cost_list[0]
        cost_collision /= cost_weights[0] + 1e-8   # cost_weights[0] = collision

        # Pick → move transition (grasp made when both EEFs are on their box sites, aligned)
        grab_cond = (
            (task_flag < 0.5)                     &
            (current_cost_g < self.grab_pos_thresh) &
            (cost_r_0       < self.grab_rot_thresh) &
            (cost_r_1       < self.grab_rot_thresh)
        )
        new_task = jax.lax.select(grab_cond, jnp.array(1.0), task_flag)

        # Success: cube reaches target position AND orientation
        success = ((new_task > 0.5)
                   & (cost_g_ball < self.success_pos_thresh)
                   & (cost_r_ball < self.success_rot_thresh))

        # Policy 4216 used a binary fall when the EEFs lost the cube in move.
        fall = (new_task > 0.5) & (current_cost_g > self.fall_pos_thresh)

        done = success | fall

        # Exact 4216 sparse-event reward layout.
        move_flag = (new_task > 0.5).astype(jnp.float32)
        reward_terms = jnp.array([
            -current_cost_g,                      # [0] EEF→box sites (pick)
            -cost_r,                              # [1] EEF rotation
            -cost_g_ball * move_flag,             # [2] cube pos→target (move)
            -cost_r_ball * move_flag,             # [3] cube rot→target (move)
            grab_cond.astype(jnp.float32),        # [4] grasp event
            success.astype(jnp.float32),          # [5] success event
            -cost_collision,                      # [6] collision from CEM
            -fall.astype(jnp.float32),            # [7] binary fall event
            fall.astype(jnp.float32),             # [8] logging-only fall flag
            success.astype(jnp.float32),          # [9] logging-only success flag
            grab_cond.astype(jnp.float32),        # [10] logging-only grasp flag
        ])

        new_state = {
            "qpos":     new_qpos,
            "qvel":     new_qvel,
            "thetadot": thetadot,
            "task":     new_task,
            "target":   state["target"],
            "xi_mean":  xi_mean,
            "xi_cov":   xi_cov,
        }

        # 4th return value is a DICT (was the bare cem_batch_costs array).
        # `cost_batch` keeps the old contents under a name; the two extra
        # entries are the sampled control candidates, so an offline analysis can
        # re-score the SAME samples under different cost weights instead of
        # redrawing them (which would depend on reproducing a random seed).
        # P = identity in mjx_planner, so thetadot IS the projected xi and
        # thetadot_all is a complete record of what each CEM iteration scored.
        cem_diag = {
            "cost_batch":   cem_batch_costs,   # (maxiter_cem, num_batch)
            "xi_samples":   xi_samples,        # (num_batch, nvar) raw draw
            "thetadot_all": thetadot_all,      # (maxiter_cem, num_batch, dof*num)
            "reward_costs": jnp.array([
                current_cost_g, cost_r, cost_g_ball, cost_r_ball]),
        }
        return new_state, reward_terms, done, cem_diag

    def _collision_penalty(self):
        pen = 0
        num_penetration = 0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 in self.robot_geom_ids or c.geom2 in self.robot_geom_ids:
                if c.dist < -1e-6:
                    num_penetration += 1
                    pen = 0.2 * num_penetration
        return pen

    def save_best_rollout_video(self, params, policy_obj, iteration,
                                LOG_LB, LOG_UB, obs_dim, act_dim, folder):

        import imageio

        if not os.path.exists(folder):
            os.makedirs(folder)

        renderer = mujoco.Renderer(self.model, height=480, width=640)

        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.distance  = 3.0
        cam.lookat    = np.array([0.0, 0.0, 0.5])
        cam.elevation = -30
        cam.azimuth   = 90

        opt = mujoco.MjvOption()
        opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

        frames = []

        state = self.reset_jax(jax.random.PRNGKey(iteration))

        for _ in range(self.MAX_STEPS):
            obs      = self.get_obs_jax(state)
            norm_obs = (obs - policy_obj.mu) / (jnp.sqrt(policy_obj.var) + 1e-8)
            policy_output = policy_obj._policy_forward(params, norm_obs)

            state, _, done, _ = self.step_jax(state, policy_output, key=jax.random.PRNGKey(42))

            # Sync JAX state → MuJoCo C struct for rendering
            self.data.qpos[:] = np.array(state["qpos"])
            self.data.qvel[:] = np.array(state["qvel"])

            # Sync target mocap body so the target cube is visible
            target_np = np.array(state["target"])
            self.data.mocap_pos[self.target_mocap_id]  = target_np[:3]
            self.data.mocap_quat[self.target_mocap_id] = target_np[3:]

            mujoco.mj_forward(self.model, self.data)

            renderer.update_scene(self.data, camera=cam, scene_option=opt)
            frames.append(renderer.render())

            if done:
                break

        video_path = f"{folder}/best_dir_ep_{iteration:04d}.mp4"
        imageio.mimsave(video_path, frames, fps=int(1 / self.timestep))
        print(f"  [Video] Saved rollout to {video_path}")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent.parent
    DEFAULT_MODEL_PATH = str(BASE_DIR / 'ur5e_hande_mjx' / 'scene.xml')

    p = argparse.ArgumentParser()
    p.add_argument('--algo',       choices=['ppo', 'ars'], default='ars')
    p.add_argument('--episodes',   type=int, default=100,
                   help='ARS: number of iterations; PPO: number of updates')
    p.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument('--num_batch',  type=int, default=50,
                   help='CEM samples per planner iteration (PPO-5003 uses 50)')
    p.add_argument('--num_steps',  type=int, default=15,
                   help='Number of steps per planning horizon')
    p.add_argument('--max_steps',  type=int, default=300,
                   help='Max steps per episode (each step calls the CEM planner once)')
    p.add_argument('--maxiter_cem', type=int, default=3,
                   help='Number of CEM iterations per planner call')
    p.add_argument('--policy',     choices=['mlp', 'linear'], default='linear',
                   help='Policy architecture: mlp (two-hidden-layer) or linear')
    p.add_argument('--seed',       type=int, default=None,
                   help='Policy, rollout, and optimizer seed (default: random)')
    p.add_argument('--n_directions', type=int, default=128,
                   help='Number of ARS perturbation directions (4216 used 128)')
    p.add_argument('--b_top', type=int, default=32,
                   help='Number of top ARS directions (4216 used 32)')
    p.add_argument('--ars_alpha', type=float, default=0.025,
                   help='ARS step size (4216 used 0.025)')
    p.add_argument('--ars_noise_std', type=float, default=0.03,
                   help='ARS perturbation standard deviation (4216 used 0.03)')
    p.add_argument('--ppo_num_envs', type=int, default=64,
                   help='PPO parallel closed-loop episodes per update')
    p.add_argument('--ppo_hidden_dim', type=int, default=10,
                   help='PPO critic hidden width (and actor width for --policy mlp)')
    p.add_argument('--ppo_epochs', type=int, default=10,
                   help='PPO optimization epochs over each rollout batch')
    p.add_argument('--ppo_minibatch_size', type=int, default=512)
    p.add_argument('--ppo_learning_rate', type=float, default=1e-4)
    p.add_argument('--ppo_clip_eps', type=float, default=0.2)
    p.add_argument('--ppo_gamma', type=float, default=0.99,
                   help='PPO/GAE discount; environment rewards are not pre-discounted')
    p.add_argument('--ppo_gae_lambda', type=float, default=0.95)
    p.add_argument('--ppo_value_coef', type=float, default=0.5)
    p.add_argument('--ppo_entropy_coef', type=float, default=1e-4)
    p.add_argument('--ppo_max_grad_norm', type=float, default=0.5)
    p.add_argument('--ppo_initial_log_std', type=float, default=-1.5)
    p.add_argument('--ppo_min_log_std', type=float, default=-2.75,
                   help='Lower bound for learned pick-phase log standard deviation')
    p.add_argument('--ppo_max_log_std', type=float, default=-0.5)
    p.add_argument('--ppo_max_log_weight_delta', type=float, default=2.0,
                   help='Smooth bound on PPO residual log cost weights')
    p.add_argument('--ppo_collision_scale', type=float, default=2.0,
                   help='Box PPO scale for the negative predicted CEM collision cost')
    p.add_argument('--ppo_move_hold_scale', type=float, default=10.0,
                   help='Move-phase penalty per step for EEF-to-grasp error')
    p.add_argument('--ppo_time_step_scale', type=float, default=0.5,
                   help='Time penalty on every active PPO environment step')
    p.add_argument('--ppo_timeout_before_grasp_penalty', type=float,
                   default=250.0,
                   help='Horizon penalty when the episode never grasped')
    p.add_argument('--ppo_timeout_after_grasp_penalty', type=float,
                   default=500.0,
                   help='Horizon penalty after grasp without success/fall')
    p.add_argument('--ppo_fall_scale', type=float, default=750.0,
                   help='Fall penalty scale. ARS uses 1500, but since fall '
                        'can only happen after a grasp, that scale makes a '
                        'grasp-then-fall episode net worse than never '
                        'grasping; halved by default so falling stays '
                        'costly without discouraging grasp attempts')
    p.add_argument('--ppo_approach_shortfall_scale', type=float, default=5.0,
                   help='Per-step pick-phase penalty on eef-to-box cost '
                        'above grab_pos_thresh. Dense progress alone only '
                        'rewards the rate of approach, so a policy can '
                        'settle into closing the gap asymptotically '
                        'without ever crossing the grasp threshold; this '
                        'penalizes the remaining shortfall directly')
    p.add_argument('--ppo_move_sample_fraction', type=float, default=0.5,
                   help='Desired move-phase fraction in each PPO epoch')
    p.add_argument('--ppo_phase_oversample_cap', type=float, default=4.0,
                   help='Maximum reuse factor for either phase per PPO epoch')
    p.add_argument('--ppo_move_std_scale', type=float, default=0.5,
                   help='Move-phase exploration std divided by pick-phase std')
    p.add_argument('--ppo_target_kl', type=float, default=0.03,
                   help='Exact per-phase KL early-stop target (0 disables)')
    p.add_argument('--ppo_kl_rollback_multiplier', type=float, default=2.5,
                   help='Reject an epoch above target KL times this value')
    p.add_argument('--ppo_eval_interval', type=int, default=20,
                   help='Fixed-seed deterministic evaluation interval; 0 disables')
    p.add_argument('--ppo_eval_episodes', type=int, default=64)
    p.add_argument('--ppo_eval_seed', type=int, default=None)
    p.add_argument('--save',       type=str, default=None)
    p.add_argument('--load',       type=str, default=None,
                   help='Path to a saved policy JSON to resume training from')
    p.add_argument('--reset_best', action='store_true', default=False,
                   help='Reset best_reward to -inf after loading (use when reward function changed)')
    p.add_argument('--init_cube_rot_deg', type=float, default=0.0,
                   help='Randomize initial cube YAW (z-axis) over +/- this many degrees '
                        '(0=legacy fixed; x-face grasp reaches ~50deg, so ~40 is safe)')
    p.add_argument('--log_idx', type=int, default=None,
                   help='Run index for saved policy/history/tensorboard '
                        '(e.g. 4220 -> ars_v2_<policy>_policy_4220.json). Default: LOG_IDX in ARS.py')
    return p.parse_args()


def main():
    args = parse_args()

    if args.seed is None:
        args.seed = int.from_bytes(os.urandom(4), 'big')
    print(f"[seed] {args.seed}  (pass --seed {args.seed} to reproduce this run)")

    runner = EpisodeRunner(
        model_path=args.model_path,
        num_batch=args.num_batch,
        num_steps=args.num_steps,
        maxiter_cem=args.maxiter_cem,
        maxiter_projection=5,
        max_steps=args.max_steps,
    )
    runner.init_cube_rot_deg = args.init_cube_rot_deg
    print(f"[env] init_cube_rot_deg = {runner.init_cube_rot_deg} (initial cube rotation randomization)")

    # Both algorithms learn ONLY the ten state-dependent CEM cost weights.
    # The runner-facing grasp pose and grasp side fields remain exactly zero.
    if args.algo == 'ppo':
        trainer = PPO(
            runner,
            seed=args.seed,
            hidden_dim=args.ppo_hidden_dim,
            policy_type=args.policy,
            num_envs=args.ppo_num_envs,
            learning_rate=args.ppo_learning_rate,
            clip_eps=args.ppo_clip_eps,
            gae_lambda=args.ppo_gae_lambda,
            gamma=args.ppo_gamma,
            value_coef=args.ppo_value_coef,
            entropy_coef=args.ppo_entropy_coef,
            update_epochs=args.ppo_epochs,
            minibatch_size=args.ppo_minibatch_size,
            max_grad_norm=args.ppo_max_grad_norm,
            initial_log_std=args.ppo_initial_log_std,
            min_log_std=args.ppo_min_log_std,
            max_log_std=args.ppo_max_log_std,
            max_log_weight_delta=args.ppo_max_log_weight_delta,
            collision_scale=args.ppo_collision_scale,
            move_hold_scale=args.ppo_move_hold_scale,
            time_step_scale=args.ppo_time_step_scale,
            timeout_before_grasp_penalty=
                args.ppo_timeout_before_grasp_penalty,
            timeout_after_grasp_penalty=
                args.ppo_timeout_after_grasp_penalty,
            fall_scale=args.ppo_fall_scale,
            approach_shortfall_scale=args.ppo_approach_shortfall_scale,
            move_sample_fraction=args.ppo_move_sample_fraction,
            phase_oversample_cap=args.ppo_phase_oversample_cap,
            move_std_scale=args.ppo_move_std_scale,
            target_kl=args.ppo_target_kl,
            kl_rollback_multiplier=args.ppo_kl_rollback_multiplier,
            eval_interval=args.ppo_eval_interval,
            eval_episodes=args.ppo_eval_episodes,
            eval_seed=args.ppo_eval_seed,
        )
        print(f"[ppo] actor={args.policy}, cost outputs={trainer.act_dim}, "
              f"envs={trainer.num_envs}, epochs={trainer.update_epochs}, "
              f"cem_batch={args.num_batch}, target_kl={trainer.target_kl}, "
              f"pick/move std ratio=1:{trainer.move_std_scale}")
    else:
        trainer = ARSV2(
            runner,
            policy_type=args.policy,
            seed=args.seed,
            n_directions=args.n_directions,
            b_top=args.b_top,
            alpha=args.ars_alpha,
            noise_std=args.ars_noise_std,
            act_dim=ACT_DIM_COST,
        )
        if trainer.policy.act_dim != ACT_DIM_COST:
            raise RuntimeError(
                f"Cost-only training invariant violated: policy act_dim is "
                f"{trainer.policy.act_dim}, expected {ACT_DIM_COST}."
            )
        print(f"[ars] directions={trainer.n_directions}, top={trainer.b_top}, "
              f"alpha={trainer.alpha}, noise_std={trainer.noise_std}, "
              f"coupled_seed={trainer.seed}")
    print(f"[policy] learning {ACT_DIM_COST} cost weights only; "
          "grasp pose and grasp side are fixed")
    if args.log_idx is not None:
        trainer.log_idx = args.log_idx
        print(f"[run] log_idx = {trainer.log_idx}")

    if args.load:
        trainer.load(args.load)
    if args.reset_best:
        trainer.best_reward = -np.inf
        if args.algo == 'ppo':
            trainer.best_eval_reward = -np.inf
        print("  [reset_best] best_reward reset to -inf")

    id = f"{args.policy}_policy_{trainer.log_idx}"

    save = args.save or (
        f"ppo_{id}.json" if args.algo == 'ppo' else f"ars_v2_{id}.json")

    best_params, history = trainer.train(
        n_iterations=args.episodes,
        save_path=save
    )

    history_prefix = "ppo_history" if args.algo == 'ppo' else "ars_history"
    with open(f"{history_prefix}_{trainer.log_idx}.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == '__main__':
    main()
