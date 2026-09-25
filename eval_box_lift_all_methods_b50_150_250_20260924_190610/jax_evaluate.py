"""
JAX-only inference evaluation — uses the same MJX physics as training.
No MuJoCo C simulation is involved in the rollout, eliminating the
MJX/C discrepancy present in pyt_dual_demo.py.

Usage:
    python jax_evaluate.py --load ars_v2_linear_policy_4208.json --n_eval 20
    python jax_evaluate.py --load ars_v2_linear_policy_4208.json --n_eval 20 --video
    python jax_evaluate.py --load ars_v2_linear_policy_4208.json --n_eval 20 --rl_move_only
"""

import os, sys, argparse, time, json, hashlib
from datetime import datetime
import jax
import jax.numpy as jnp
import numpy as np
import mujoco

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl_trainer_batch import (EpisodeRunner, LEGACY_EEF_ROT_TARGETS,
                              FORCE_XY_FACE_PAIR, DISABLE_CUBE_SLAVING)
from ARS import (ARSV2, DEFAULT_LOG_WEIGHTS, DEFAULT_LOG_HANDTUNEDWEIGHTS,
                 ACT_DIM_COST, ACT_DIM_GRASP, LOG_LB, LOG_UB, _COST_KEYS)

_WEIGHT_SETS = {
    'bayesian':  DEFAULT_LOG_WEIGHTS,
    'handtuned': DEFAULT_LOG_HANDTUNEDWEIGHTS,
}

_PPO_ACTION_MAPPING = (
    "10_cost_log_residuals_bounded_tanh;grasp_fields_zero")
_PLANNER_CONFIG_KEYS = (
    "num_batch", "num_steps", "maxiter_cem", "maxiter_projection",
    "max_steps", "model_path", "init_cube_rot_deg")

# obs layout:  joint_pos[0:12]  ball_pos[12:15]  ball_quat[15:19]  joint_vel[19:31]
#              eef_0_pos[31:34] eef_0_quat[34:38] eef_1_pos[38:41] eef_1_quat[41:45]
#              target_pos[45:48] target_quat[48:52] task_flag[52]
# 23-D, matching what step_jax reads: 10 log cost weights + 12 grasp params +
# 1 grasp-side logit at index ACT_DIM_COST + ACT_DIM_GRASP. The trailing zero
# was missing, so `jnp.where(use_default, _DEFAULT_POLICY_OUTPUT, ...)` tried to
# broadcast (22,) against the policy's (23,) and raised. A zero learned side
# score leaves grasp_side to the geometric prior (the square-on rule from the
# cube's yaw), which is the intended default.
_DEFAULT_POLICY_OUTPUT = jnp.concatenate(
    [jnp.array(DEFAULT_LOG_WEIGHTS), jnp.zeros(ACT_DIM_GRASP), jnp.zeros(1)]
)


class PPOPolicyWrapper:
    """Deterministic box-PPO actor with the exact training action mapping."""

    def __init__(self, checkpoint):
        if checkpoint.get("algorithm") != "ppo":
            raise ValueError("PPOPolicyWrapper requires a PPO checkpoint")
        if checkpoint.get("task", "box_lift") != "box_lift":
            raise ValueError("Checkpoint is not a box-lift PPO policy")
        self.algorithm = "ppo"
        self.policy_type = checkpoint.get("policy_type", "linear")
        self.params = jnp.asarray(checkpoint["params"])
        self.mu = jnp.asarray(checkpoint["mu"])
        self.var = jnp.asarray(checkpoint["var"])
        self.obs_dim = int(self.mu.shape[0])
        self.act_dim = int(checkpoint.get("act_dim", ACT_DIM_COST))
        self.hidden_dim = int(checkpoint.get("hidden_dim", 10))
        self.max_log_weight_delta = float(
            checkpoint.get("max_log_weight_delta", 2.0))
        self.checkpoint_version = int(checkpoint.get("checkpoint_version", 1))
        self.reward_profile = checkpoint.get("reward_profile", "legacy")
        self.checkpoint_type = checkpoint.get("checkpoint_type", "legacy")
        self.action_mapping = checkpoint.get(
            "action_mapping", _PPO_ACTION_MAPPING)
        self.planner_config = checkpoint.get("planner_config", {})
        if self.act_dim != ACT_DIM_COST:
            raise ValueError(
                f"Box PPO must have {ACT_DIM_COST} cost actions, got {self.act_dim}")
        if self.action_mapping != _PPO_ACTION_MAPPING:
            raise ValueError(
                "Unsupported PPO action_mapping="
                f"{self.action_mapping!r}; evaluator implements "
                f"{_PPO_ACTION_MAPPING!r}")

        if self.policy_type == "linear":
            expected = self.obs_dim * self.act_dim + self.act_dim
        elif self.policy_type == "mlp":
            h = self.hidden_dim
            expected = (self.obs_dim*h + h + h*h + h + h*self.act_dim
                        + self.act_dim)
        else:
            raise ValueError(f"Unsupported PPO policy_type={self.policy_type!r}")
        if self.params.size != expected:
            raise ValueError(
                f"PPO actor size mismatch: expected {expected}, got {self.params.size}")

    def _actor(self, params, obs):
        d, h, a = self.obs_dim, self.hidden_dim, self.act_dim
        if self.policy_type == "linear":
            weights = params[:d*a].reshape(d, a)
            bias = params[d*a:d*a+a]
            return obs @ weights + bias
        i = 0
        w1 = params[i:i+d*h].reshape(d, h); i += d*h
        b1 = params[i:i+h]; i += h
        w2 = params[i:i+h*h].reshape(h, h); i += h*h
        b2 = params[i:i+h]; i += h
        w3 = params[i:i+h*a].reshape(h, a); i += h*a
        b3 = params[i:i+a]
        x = jnp.tanh(obs @ w1 + b1)
        x = jnp.tanh(x @ w2 + b2)
        return x @ w3 + b3

    def _policy_forward(self, params, normalized_obs):
        mean = self._actor(params, normalized_obs)
        limit = self.max_log_weight_delta
        residual = limit * jnp.tanh(mean / limit)
        log_weights = jnp.clip(
            jnp.asarray(DEFAULT_LOG_WEIGHTS) + residual, LOG_LB, LOG_UB)
        return jnp.concatenate([
            log_weights,
            jnp.zeros(ACT_DIM_GRASP + 1, dtype=log_weights.dtype),
        ])


def _runtime_planner_config(runner):
    """Return the planner/dynamics settings that this evaluator will run."""
    planner = runner.planner
    return {
        "num_batch": int(planner.num_batch),
        "num_steps": int(planner.num_steps),
        "maxiter_cem": int(planner.maxiter_cem),
        "maxiter_projection": int(planner.maxiter_projection),
        "max_steps": int(runner.MAX_STEPS),
        "model_path": os.path.abspath(runner.model_path),
        "init_cube_rot_deg": float(runner.init_cube_rot_deg),
    }


def _planner_config_mismatches(checkpoint_config, runtime_config):
    """Compare fields present in a checkpoint with the active evaluator."""
    mismatches = []
    for key in _PLANNER_CONFIG_KEYS:
        if key not in checkpoint_config:
            continue
        saved = checkpoint_config[key]
        active = runtime_config[key]
        if key == "model_path":
            equal = (os.path.realpath(str(saved))
                     == os.path.realpath(str(active)))
        elif key == "init_cube_rot_deg":
            equal = bool(np.isclose(float(saved), float(active)))
        else:
            equal = saved == active
        if not equal:
            mismatches.append(
                f"{key}: checkpoint={saved!r}, runtime={active!r}")
    return mismatches


def _ars_action_mapping(act_dim):
    """Describe ARS deployment without changing its existing forward pass."""
    if act_dim >= ACT_DIM_COST + ACT_DIM_GRASP + 1:
        suffix = "grasp_pose_and_side_learned"
    elif act_dim > ACT_DIM_COST:
        suffix = "grasp_fields_zero;grasp_side_learned"
    else:
        suffix = "grasp_fields_zero"
    return f"10_cost_log_residuals_clipped;{suffix}"


def _fold_step_keys(mode, algorithm):
    """Resolve a shared benchmark RNG schedule or the legacy algorithm default."""
    if mode == "checkpoint":
        return algorithm == "ppo"
    if mode not in ("episode", "fold_in"):
        raise ValueError(f"Unknown CEM key mode: {mode}")
    return mode == "fold_in"


def run_episode(runner, params, policy_obj, seed,
                rl_move_only=False,
                rl_pick_only=False,
                fold_step_keys=False,
                render=False, folder=None,
                save_samples=False, sample_stride=1,
                timed=False,
                ep_idx=0,
                video_tag=''):
    """
    One episode fully on MJX. Returns (outcome, per_step_data, target_0).
    outcome: 'success' | 'fall' | 'timeout'
    per_step_data: dict of lists, one value per step.

    Render and timed modes use an early-stopping Python loop with per-step
    timings. Otherwise a full-horizon scan runs and its logs are trimmed after
    the first done; that legacy path does not provide per-step timings.
    """
    key = jax.random.PRNGKey(seed)
    key_reset, ppo_step_root = jax.random.split(key)
    state = runner.reset_jax(key_reset)

    def env_step_key(t):
        if fold_step_keys:
            # PPO rollout: fold the post-reset split key by closed-loop step,
            # then use split[1] for the stochastic CEM environment/planner.
            return jax.random.split(jax.random.fold_in(ppo_step_root, t))[1]
        # 4216-compatible ARS rollout: reuse the episode key every MPC step.
        return key

    des_dist = float(runner.planner.cem.des_dist_bet_eef)

    # ------------------------------------------------------------------ render
    if render:
        import imageio
        os.makedirs(folder, exist_ok=True)
        # reset_jax reads runner.data. Rendering must not feed the previous
        # episode's final velocities back into the next episode's reset.
        render_data = mujoco.MjData(runner.model)
        render_data.mocap_pos[:] = runner.data.mocap_pos
        render_data.mocap_quat[:] = runner.data.mocap_quat
        renderer = mujoco.Renderer(runner.model, height=480, width=640)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.distance, cam.elevation, cam.azimuth = 2.0, -30, 90
        cam.lookat = np.array([0.0, 0.0, 0.5])
        opt = mujoco.MjvOption()
        opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        frames = []

        outcome = 'timeout'
        step_data = {k: [] for k in [
            'step_time_ms', 'theta', 'thetadot', 'task_flag', 'cost_r',
            'cost_eef_to_obj', 'cost_obj_to_targ', 'cost_dist', 'cost_zy', 'cost_weights',
            'cem_batch_costs', 'qpos_pre', 'qvel_pre', 'thetadot_cmd',
            'task_pre', 'target', 'policy_output', 'sample_steps',
            'xi_samples', 'thetadot_all',
            'reward_terms', 'step_done',
        ]}

        for step_idx in range(runner.MAX_STEPS):
            t0 = time.perf_counter()
            obs = runner.get_obs_jax(state)

            is_pick = float(state["task"]) < 0.5
            use_default = (rl_move_only and is_pick) or (rl_pick_only and not is_pick)
            if use_default:
                policy_output = _DEFAULT_POLICY_OUTPUT
            else:
                norm_obs = (obs - policy_obj.mu) / (jnp.sqrt(policy_obj.var) + 1e-8)
                policy_output = policy_obj._policy_forward(params, norm_obs)

            step_data['qpos_pre'].append(np.array(state['qpos']))
            step_data['qvel_pre'].append(np.array(state['qvel']))
            step_data['thetadot_cmd'].append(np.array(state['thetadot']))
            step_data['task_pre'].append(float(state['task']))
            step_data['target'].append(np.array(state['target']))
            step_data['policy_output'].append(np.array(policy_output))

            state, reward_terms, done, cem_diag = runner.step_jax(
                state, policy_output, key=env_step_key(step_idx))
            cem_costs = cem_diag['cost_batch']
            _ = reward_terms.block_until_ready()
            step_ms = (time.perf_counter() - t0) * 1000

            eef_0_pos = np.array(obs[31:34])
            eef_1_pos = np.array(obs[38:41])
            eef_dist  = float(np.linalg.norm(eef_0_pos - eef_1_pos))

            step_data['step_time_ms'].append(step_ms)
            step_data['theta'].append(np.array(state["qpos"])[runner.joint_mask_pos].copy())
            step_data['thetadot'].append(np.array(state["thetadot"]).copy())
            step_data['task_flag'].append(float(state["task"]))
            step_data['cost_r'].append(float(-reward_terms[1]))
            step_data['cost_eef_to_obj'].append(float(-reward_terms[0]))
            step_data['cost_obj_to_targ'].append(float(jnp.linalg.norm(obs[12:15] - obs[45:48])))
            step_data['cost_dist'].append(abs(eef_dist - des_dist))
            step_data['cost_zy'].append(float(np.linalg.norm(obs[32:34] - obs[39:41])))
            step_data['cost_weights'].append(np.exp(np.array(policy_output[:ACT_DIM_COST])))
            step_data['cem_batch_costs'].append(np.array(cem_costs))
            step_data['reward_terms'].append(np.array(reward_terms))
            step_data['step_done'].append(bool(done))

            render_data.qpos[:] = np.array(state["qpos"])
            render_data.qvel[:] = np.array(state["qvel"])
            t = np.array(state["target"])
            render_data.mocap_pos[runner.target_mocap_id]  = t[:3]
            render_data.mocap_quat[runner.target_mocap_id] = t[3:]
            mujoco.mj_forward(runner.model, render_data)
            renderer.update_scene(render_data, camera=cam, scene_option=opt)
            frames.append(renderer.render())

            if bool(done):
                # [5] is the rewarded success event; [9] is its logging duplicate.
                outcome = 'success' if float(reward_terms[9]) > 0.5 else 'fall'
                break

        renderer.close()
        prefix = f"{video_tag}_" if video_tag else ""
        path = f"{folder}/{prefix}ep{ep_idx:04d}_{outcome}.mp4"
        imageio.mimsave(path, frames, fps=int(1 / runner.timestep))
        print(f"  saved {path}")
        step_data['sample_steps'] = []
        step_data['xi_samples'] = []
        step_data['thetadot_all'] = []
        # The 4216 physical-box dynamics have no cube-slaving state. Retain
        # empty compatibility fields so older NPZ readers can detect absence.
        step_data['grabbed'] = []
        step_data['rel_pos'] = []
        step_data['rel_quat'] = []
        target_0 = np.array(state["target"])
        return outcome, step_data, target_0

    # ------------------------------------------- timed Python loop (no render)
    if timed:
        outcome = 'timeout'
        step_data = {k: [] for k in [
            'step_time_ms', 'theta', 'thetadot', 'task_flag', 'cost_r',
            'cost_eef_to_obj', 'cost_obj_to_targ', 'cost_dist', 'cost_zy', 'cost_weights',
            'cem_batch_costs', 'qpos_pre', 'qvel_pre', 'thetadot_cmd',
            'task_pre', 'target', 'policy_output', 'sample_steps',
            'xi_samples', 'thetadot_all',
            'reward_terms', 'step_done',
        ]}

        for step_idx in range(runner.MAX_STEPS):
            t0 = time.perf_counter()
            obs = runner.get_obs_jax(state)

            is_pick = float(state["task"]) < 0.5
            use_default = (rl_move_only and is_pick) or (rl_pick_only and not is_pick)
            if use_default:
                policy_output = _DEFAULT_POLICY_OUTPUT
            else:
                norm_obs = (obs - policy_obj.mu) / (jnp.sqrt(policy_obj.var) + 1e-8)
                policy_output = policy_obj._policy_forward(params, norm_obs)

            step_data['qpos_pre'].append(np.array(state['qpos']))
            step_data['qvel_pre'].append(np.array(state['qvel']))
            step_data['thetadot_cmd'].append(np.array(state['thetadot']))
            step_data['task_pre'].append(float(state['task']))
            step_data['target'].append(np.array(state['target']))
            step_data['policy_output'].append(np.array(policy_output))

            state, reward_terms, done, cem_diag = runner.step_jax(
                state, policy_output, key=env_step_key(step_idx))
            cem_costs = cem_diag['cost_batch']
            _ = reward_terms.block_until_ready()
            step_ms = (time.perf_counter() - t0) * 1000

            eef_0_pos = np.array(obs[31:34])
            eef_1_pos = np.array(obs[38:41])
            eef_dist  = float(np.linalg.norm(eef_0_pos - eef_1_pos))

            step_data['step_time_ms'].append(step_ms)
            step_data['theta'].append(np.array(state["qpos"])[runner.joint_mask_pos].copy())
            step_data['thetadot'].append(np.array(state["thetadot"]).copy())
            step_data['task_flag'].append(float(state["task"]))
            step_data['cost_r'].append(float(-reward_terms[1]))
            step_data['cost_eef_to_obj'].append(float(-reward_terms[0]))
            step_data['cost_obj_to_targ'].append(float(jnp.linalg.norm(obs[12:15] - obs[45:48])))
            step_data['cost_dist'].append(abs(eef_dist - des_dist))
            step_data['cost_zy'].append(float(np.linalg.norm(obs[32:34] - obs[39:41])))
            step_data['cost_weights'].append(np.exp(np.array(policy_output[:ACT_DIM_COST])))
            step_data['cem_batch_costs'].append(np.array(cem_costs))
            step_data['reward_terms'].append(np.array(reward_terms))
            step_data['step_done'].append(bool(done))

            if bool(done):
                # [5] is the rewarded success event; [9] is its logging duplicate.
                outcome = 'success' if float(reward_terms[9]) > 0.5 else 'fall'
                break

        step_data['sample_steps'] = []
        step_data['xi_samples'] = []
        step_data['thetadot_all'] = []
        step_data['grabbed'] = []
        step_data['rel_pos'] = []
        step_data['rel_quat'] = []
        target_0 = np.array(state["target"])
        return outcome, step_data, target_0

    # ----------------------------------------------- scan (deterministic eval)
    rl_move_only_jax = jnp.array(rl_move_only)
    rl_pick_only_jax = jnp.array(rl_pick_only)

    def step_fn(carry, _t):
        state, done_mask = carry

        obs      = runner.get_obs_jax(state)
        norm_obs = (obs - policy_obj.mu) / (jnp.sqrt(policy_obj.var) + 1e-8)
        policy_output_rl = policy_obj._policy_forward(params, norm_obs)

        is_pick     = state["task"] < 0.5
        use_default = (rl_move_only_jax & is_pick) | (rl_pick_only_jax & (~is_pick))
        policy_output = jnp.where(use_default, _DEFAULT_POLICY_OUTPUT, policy_output_rl)

        # PRE-step planning state: this is what the CEM at this step consumed,
        # and what an offline re-scoring has to restore. The post-step values
        # emitted below are a different thing and are kept for the existing
        # diagnostics.
        pre = (state["qpos"], state["qvel"], state["thetadot"], state["task"],
               state["target"])

        state, reward_terms, done, cem_diag = runner.step_jax(
            state, policy_output, key=env_step_key(_t))
        cem_costs = cem_diag['cost_batch']

        new_done_mask = jnp.maximum(done_mask, done.astype(jnp.float32))
        samples = ((cem_diag['xi_samples'], cem_diag['thetadot_all'])
                   if save_samples else (jnp.zeros(()), jnp.zeros(())))
        return (state, new_done_mask), (obs, policy_output, reward_terms, done,
                                        state["qpos"], state["thetadot"], cem_costs,
                                        state["task"], pre, samples)

    ((final_state, _),
     (obs_hist, pol_hist, rt_hist, done_hist, qpos_hist, td_hist, cem_hist,
      task_hist, pre_hist, sample_hist)) = jax.lax.scan(
        step_fn, (state, jnp.array(0.0)), jnp.arange(runner.MAX_STEPS)
    )

    # Determine outcome and actual episode length
    any_done       = bool(jnp.any(done_hist))
    first_done_idx = int(jnp.argmax(done_hist)) if any_done else runner.MAX_STEPS - 1
    n_steps        = first_done_idx + 1 if any_done else runner.MAX_STEPS

    if any_done:
        # [5] is the rewarded success event; [9] is its logging duplicate.
        outcome = 'success' if float(rt_hist[first_done_idx, 9]) > 0.5 else 'fall'
    else:
        outcome = 'timeout'

    # Convert scan outputs to numpy and trim to actual episode length
    obs_np  = np.array(obs_hist[:n_steps])    # (n_steps, obs_dim)
    pol_np  = np.array(pol_hist[:n_steps])    # (n_steps, act_dim)
    rt_np   = np.array(rt_hist[:n_steps])     # (n_steps, n_reward_terms)
    qpos_np = np.array(qpos_hist[:n_steps])   # (n_steps, n_qpos)
    td_np   = np.array(td_hist[:n_steps])     # (n_steps, n_joints)
    cem_np  = np.array(cem_hist[:n_steps])    # (n_steps, maxiter_cem, num_batch)
    task_np = np.array(task_hist[:n_steps])   # (n_steps,)

    eef_0    = obs_np[:, 31:34]
    eef_1    = obs_np[:, 38:41]
    eef_dist = np.linalg.norm(eef_0 - eef_1, axis=1)

    jmask = np.array(runner.joint_mask_pos)
    step_data = {
        'step_time_ms':     [0.0] * n_steps,   # not meaningful in scan mode
        'reward_terms':     list(rt_np),
        'step_done':        list(np.asarray(done_hist[:n_steps], dtype=bool)),
        'theta':            [qpos_np[i, jmask] for i in range(n_steps)],
        'thetadot':         [td_np[i] for i in range(n_steps)],
        'task_flag':        list(task_np),
        'cost_r':           list(-rt_np[:, 1]),
        'cost_eef_to_obj':  list(-rt_np[:, 0]),
        'cost_obj_to_targ': list(np.linalg.norm(obs_np[:, 12:15] - obs_np[:, 45:48], axis=1)),
        'cost_dist':        list(np.abs(eef_dist - des_dist)),
        'cost_zy':          list(np.linalg.norm(obs_np[:, 32:34] - obs_np[:, 39:41], axis=1)),
        'cost_weights':     [np.exp(pol_np[i, :ACT_DIM_COST]) for i in range(n_steps)],
        'cem_batch_costs':  [cem_np[i] for i in range(n_steps)],
    }

    # ── PRE-step planning state, one entry per MPC step ─────────────────────
    # Indexed by t, unlike the candidate samples below, which are thinned by
    # --sample_stride and are indexed by their position in `sample_steps`.
    qpos_pre, qvel_pre, tdcmd_pre, task_pre, target_pre = pre_hist
    step_data['qpos_pre']     = [np.array(qpos_pre[i])   for i in range(n_steps)]
    step_data['qvel_pre']     = [np.array(qvel_pre[i])   for i in range(n_steps)]
    step_data['thetadot_cmd'] = [np.array(tdcmd_pre[i])  for i in range(n_steps)]
    step_data['task_pre']     = [float(task_pre[i])      for i in range(n_steps)]
    step_data['target']       = [np.array(target_pre[i]) for i in range(n_steps)]
    # Full 23-D policy output: the analysis needs the grasp deltas and the side
    # logit as well as the 10 cost weights, if only to zero them deliberately.
    step_data['policy_output'] = [np.array(pol_np[i]) for i in range(n_steps)]
    step_data['grabbed'] = []
    step_data['rel_pos'] = []
    step_data['rel_quat'] = []

    # ── candidate samples (empty unless save_samples) ────────────────────────
    if save_samples:
        xs_hist, td_all_hist = sample_hist
        keep = list(range(0, n_steps, max(1, sample_stride)))
        step_data['sample_steps'] = keep
        step_data['xi_samples']   = [np.asarray(xs_hist[i], np.float32) for i in keep]
        step_data['thetadot_all'] = [np.asarray(td_all_hist[i], np.float32) for i in keep]
    else:
        step_data['sample_steps'] = []
        step_data['xi_samples']   = []
        step_data['thetadot_all'] = []

    target_0 = np.array(final_state["target"])
    return outcome, step_data, target_0


def save_npz(all_data, n_eval, num_batch, num_steps, save_path, meta=None,
             timestep=0.1):
    n_steps = np.array([len(ep) for ep in all_data['task_flag']], dtype=int)
    measured = np.array([
        len(ep) > 0 and np.all(np.isfinite(ep)) and np.all(np.asarray(ep) > 0)
        for ep in all_data['step_time_ms']
    ], dtype=bool)
    np.savez(
        save_path,
        batch_size         = np.array([num_batch]),
        horizon            = np.array([num_steps]),
        total_time         = np.array(all_data['total_time_s']),
        wall_time          = np.array(all_data['total_time_s']),
        task_time          = n_steps * timestep,  # simulated duration, seconds
        timestep           = np.array([timestep]),
        n_steps            = n_steps,
        sim_time           = np.array([
            (np.arange(n) + 1) * timestep for n in n_steps], dtype=object),
        episode_seed       = np.array(all_data['episode_seed'], dtype=np.uint32),
        execution_mode     = np.array(all_data['execution_mode']),
        step_time_valid    = measured,
        step_time_unit     = np.array(['ms']),
        task_time_unit     = np.array(['s_simulated']),
        total_time_unit    = np.array(['s_wall_including_episode_overhead']),
        success_rate       = np.array([np.mean(all_data['success'])]),
        success            = np.array(all_data['success']),
        reward_terms       = np.array(all_data['reward_terms'], dtype=object),
        step_done          = np.array(all_data['step_done'], dtype=object),
        step_success       = np.array([
            [bool(rt[9] > 0.5) for rt in ep]
            for ep in all_data['reward_terms']], dtype=object),
        reason             = np.array(all_data['reason']),
        target_0           = np.array(all_data['target_0'], dtype=object),
        step_time          = np.array(all_data['step_time_ms'], dtype=object),
        theta              = np.array(all_data['theta'],         dtype=object),
        thetadot           = np.array(all_data['thetadot'],      dtype=object),
        task_flag          = np.array(all_data['task_flag'],     dtype=object),
        cost_r             = np.array(all_data['cost_r'],         dtype=object),
        cost_eef_to_obj    = np.array(all_data['cost_eef_to_obj'], dtype=object),
        cost_obj_to_targ   = np.array(all_data['cost_obj_to_targ'], dtype=object),
        cost_dist          = np.array(all_data['cost_dist'],     dtype=object),
        cost_zy            = np.array(all_data['cost_zy'],       dtype=object),
        cost_weights       = np.array(all_data['cost_weights'],  dtype=object),
        cem_batch_costs    = np.array(all_data['cem_batch_costs'], dtype=object),
        cost_weights_keys  = np.array(_COST_KEYS),
        # ── pre-step planning state, one entry per MPC step ─────────────────
        qpos_pre           = np.array(all_data['qpos_pre'],     dtype=object),
        qvel_pre           = np.array(all_data['qvel_pre'],     dtype=object),
        thetadot_cmd       = np.array(all_data['thetadot_cmd'], dtype=object),
        task_pre           = np.array(all_data['task_pre'],     dtype=object),
        target             = np.array(all_data['target'],       dtype=object),
        grabbed            = np.array(all_data['grabbed'],      dtype=object),
        rel_pos            = np.array(all_data['rel_pos'],      dtype=object),
        rel_quat           = np.array(all_data['rel_quat'],     dtype=object),
        policy_output      = np.array(all_data['policy_output'], dtype=object),
        # ── candidate samples (empty without --save_samples) ────────────────
        sample_steps       = np.array(all_data['sample_steps'], dtype=object),
        xi_samples         = np.array(all_data['xi_samples'],   dtype=object),
        thetadot_all       = np.array(all_data['thetadot_all'], dtype=object),
        # ── frozen planner config, so the analysis can rebuild the planner ──
        **(meta or {}),
    )
    print(f"Data saved to {save_path}")


def main():
    from pathlib import Path
    BASE_DIR      = Path(__file__).resolve().parent.parent
    DEFAULT_MODEL = str(BASE_DIR / 'ur5e_hande_mjx' / 'scene.xml')

    p = argparse.ArgumentParser()
    p.add_argument('--load',        default=None, help='Policy checkpoint JSON (omit to use default CEM weights)')
    p.add_argument('--n_eval',      type=int, default=20, help='Number of episodes')
    p.add_argument('--seed',        type=int, default=0,  help='Starting random seed')
    p.add_argument('--video',       action='store_true',  help='Save mp4 per episode')
    p.add_argument('--video_stride', type=int, default=1,
                   help='Only render/save video every Nth episode (ep_idx %% N == 0); '
                        'other episodes use --timed if set, otherwise the full scan; data is '
                        'still saved to the npz. Default 1 = every episode.')
    p.add_argument('--video_dir',   type=str, default='eval_videos')
    p.add_argument('--video_tag',   type=str, default='',
                   help='Prefix added to each video filename, e.g. "handtuned_batch250_seed0"')
    p.add_argument('--save_npz',    type=str, default=None,
                   help='Path to save .npz data (default: auto-named next to checkpoint)')
    p.add_argument('--rl_move_only', action='store_true', default=False,
                   help='Use RL policy only in move phase; default weights during pick')
    p.add_argument('--rl_pick_only', action='store_true', default=False,
                   help='Use RL policy only in pick phase; default weights during move')
    p.add_argument('--timed', action='store_true', default=False,
                   help='Run Python loop (no video) to measure per-step computation time')
    p.add_argument('--cem_key_mode', choices=['checkpoint', 'episode', 'fold_in'],
                   default='checkpoint',
                   help='CEM RNG: checkpoint = legacy algorithm-specific schedule; '
                        'episode = reuse episode key (legacy ARS); fold_in = fresh '
                        'per-step keys (PPO). Set the SAME mode for all benchmark methods.')
    p.add_argument('--warmup_episodes', type=int, default=0,
                   help='Unreported headless warm-up episodes before timed measurement; '
                        'requires --timed. Use 1 for the matched benchmark.')
    p.add_argument('--weights', type=str, default='bayesian',
                   choices=['bayesian', 'handtuned'],
                   help='Default cost weights to use when no policy is loaded (default: bayesian)')
    p.add_argument('--init_cube_rot_deg', type=float, default=0.0,
                   help='Randomize initial cube YAW over +/- this many deg (match training, '
                        'e.g. 40, to evaluate the wider-initial-pose capability)')
    p.add_argument('--seed_stride', type=int, default=0,
                   help='Advance the episode seed by this much per episode within '
                        'one invocation (0 = legacy behaviour, where every episode '
                        'reuses --seed and is therefore identical). Use e.g. 1000 '
                        'to get n_eval distinct episodes from one process.')
    p.add_argument('--save_samples', action='store_true',
                   help='Store the CEM candidate samples (xi_samples + '
                        'thetadot_all for every CEM iteration) so the analysis '
                        'can re-cost the SAME samples under other weight sets. '
                        'Large: ~(num_batch*nvar + maxiter_cem*num_batch*'
                        'num_dof*num_steps)*4 bytes per saved step.')
    p.add_argument('--sample_stride', type=int, default=1,
                   help='Store samples every Nth MPC step (default 1 = all)')
    p.add_argument('--model_path',  type=str, default=DEFAULT_MODEL)
    p.add_argument('--num_batch',   type=int, default=50)
    p.add_argument('--num_steps',   type=int, default=15)
    p.add_argument('--maxiter_cem', type=int, default=3)
    p.add_argument('--max_steps',   type=int, default=300)
    p.add_argument('--strict_checkpoint_planner', action='store_true',
                   help='Fail if evaluator planner settings differ from '
                        'planner_config saved in the checkpoint. Without this '
                        'flag a mismatch is reported and recorded, allowing '
                        'intentional CEM batch/horizon sweeps.')
    args = p.parse_args()
    if args.n_eval < 1 or args.max_steps < 1 or args.warmup_episodes < 0:
        p.error('n_eval/max_steps must be positive and warmup_episodes nonnegative')
    if args.warmup_episodes and not args.timed:
        p.error('--warmup_episodes requires --timed')
    seeds = [args.seed + i * args.seed_stride for i in range(args.n_eval)]
    if min(seeds) < 0 or max(seeds) >= 2**32:
        p.error('episode seeds must fit in an unsigned 32-bit integer')

    runner = EpisodeRunner(
        model_path=args.model_path,
        num_batch=args.num_batch,
        num_steps=args.num_steps,
        maxiter_cem=args.maxiter_cem,
        maxiter_projection=5,
        max_steps=args.max_steps,
    )
    runner.init_cube_rot_deg = args.init_cube_rot_deg
    print(f"[eval] init_cube_rot_deg = {runner.init_cube_rot_deg}")

    global _DEFAULT_POLICY_OUTPUT
    # Must be 23-D like the module-level default and like the policy's own
    # output: step_jax reads the grasp-side logit at index
    # ACT_DIM_COST + ACT_DIM_GRASP. This rebind (which selects the --weights
    # set) was still 22-D, so it shadowed the correct constant and every
    # jnp.where against the policy output failed to broadcast.
    _DEFAULT_POLICY_OUTPUT = jnp.concatenate(
        [jnp.array(_WEIGHT_SETS[args.weights]),
         jnp.zeros(ACT_DIM_GRASP), jnp.zeros(1)]
    )

    checkpoint_algorithm = "default"
    checkpoint_version = 0
    checkpoint_reward_profile = "none"
    checkpoint_type = "none"
    checkpoint_action_mapping = "fixed_default_cem_weights"
    checkpoint_policy_type = "none"
    checkpoint_planner_config = {}
    if args.load:
        with open(args.load) as _f:
            _ck = json.load(_f)
        checkpoint_algorithm = _ck.get("algorithm", "ars")
        checkpoint_version = int(_ck.get(
            "checkpoint_version", 1 if checkpoint_algorithm == "ppo" else 0))
        checkpoint_reward_profile = _ck.get("reward_profile", "legacy")
        checkpoint_type = _ck.get("checkpoint_type", "legacy")
        checkpoint_policy_type = _ck.get("policy_type", "linear")
        checkpoint_planner_config = _ck.get(
            "planner_config", _ck.get("planner", {}))
        if checkpoint_algorithm == "ppo":
            trainer = PPOPolicyWrapper(_ck)
            checkpoint_action_mapping = trainer.action_mapping
            print(f"  [eval] PPO {trainer.policy_type} actor: "
                  f"{trainer.obs_dim} observations -> {trainer.act_dim} "
                  "cost-weight residuals; grasp outputs are ZERO")
            print(f"  [eval] PPO checkpoint v{trainer.checkpoint_version}, "
                  f"type={trainer.checkpoint_type!r}, "
                  f"reward_profile={trainer.reward_profile!r}")
            print(f"  [eval] action_mapping={trainer.action_mapping}")
        else:
            policy_type = _ck.get("policy_type", "linear")
            _act_dim = _ck.get("act_dim")
            if _act_dim is None:
                if policy_type != "linear":
                    raise ValueError(
                        "Legacy MLP ARS checkpoint is missing act_dim metadata")
                _n = len(_ck["params"])
                _obs = len(_ck.get("mu", [])) or 53
                _act_dim = _n // (_obs + 1)
            trainer = ARSV2(
                runner, policy_type=policy_type, act_dim=int(_act_dim))
            trainer.load(args.load)
            checkpoint_action_mapping = _ck.get(
                "action_mapping", _ars_action_mapping(int(_act_dim)))
            print(f"  [eval] ARS checkpoint act_dim = {_act_dim} "
                  f"({'full 23-D: learns grasp pose + side' if int(_act_dim) >= 23 else 'cost weights only: grasp deltas and side are ZERO'})")
    else:
        trainer = ARSV2(runner, policy_type='linear', act_dim=ACT_DIM_COST)
        args.rl_move_only = True
        args.rl_pick_only = True

    runtime_planner_config = _runtime_planner_config(runner)
    planner_mismatches = _planner_config_mismatches(
        checkpoint_planner_config, runtime_planner_config)
    print("  [eval] runtime planner_config="
          + json.dumps(runtime_planner_config, sort_keys=True))
    if checkpoint_planner_config:
        print("  [eval] checkpoint planner_config="
              + json.dumps(checkpoint_planner_config, sort_keys=True))
        if planner_mismatches:
            mismatch_text = "; ".join(planner_mismatches)
            message = ("Evaluator planner differs from checkpoint: "
                       + mismatch_text)
            if args.strict_checkpoint_planner:
                raise ValueError(message)
            print("  [eval] WARNING: " + message)
            print("  [eval] This is allowed for an intentional planner sweep; "
                  "use --strict_checkpoint_planner to reject it.")
        else:
            print("  [eval] planner configuration matches checkpoint")
    elif args.load:
        print("  [eval] checkpoint has no planner_config; planner compatibility "
              "cannot be verified")

    if args.load is None:
        print("Mode: default CEM weights in ALL phases (no policy loaded)")
    elif args.rl_move_only:
        print("Mode: RL policy active in MOVE phase only (default weights during PICK)")
    elif args.rl_pick_only:
        print("Mode: RL policy active in PICK phase only (default weights during MOVE)")
    else:
        print("Mode: RL policy active in ALL phases (matches training)")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # auto npz path
    npz_path = args.save_npz
    if npz_path is None:
        base = os.path.splitext(os.path.basename(args.load))[0] if args.load else f"default_{args.weights}"
        suffix = "" if args.load is None else "_move_only" if args.rl_move_only else "_pick_only" if args.rl_pick_only else ""
        npz_path = f"eval_{base}{suffix}_b{args.num_batch}_n{args.n_eval}_{ts}.npz"

    video_dir = f"{args.video_dir}/{ts}" if args.video else args.video_dir

    fold_step_keys = _fold_step_keys(args.cem_key_mode, checkpoint_algorithm)
    resolved_key_mode = 'fold_in' if fold_step_keys else 'episode'
    print(f"[eval] CEM key mode: {resolved_key_mode}; episode seeds: {seeds}")
    for warmup in range(args.warmup_episodes):
        print(f"[eval] warm-up {warmup + 1}/{args.warmup_episodes} (not saved)")
        run_episode(
            runner, trainer.params, trainer, seed=args.seed,
            rl_move_only=args.rl_move_only, rl_pick_only=args.rl_pick_only,
            fold_step_keys=fold_step_keys, timed=True)

    all_data = {k: [] for k in [
        'total_time_s', 'success', 'reason', 'target_0',
        'episode_seed', 'execution_mode', 'reward_terms', 'step_done',
        'step_time_ms', 'theta', 'thetadot', 'task_flag',
        'cost_r', 'cost_eef_to_obj', 'cost_obj_to_targ',
        'cost_dist', 'cost_zy', 'cost_weights', 'cem_batch_costs',
        'qpos_pre', 'qvel_pre', 'thetadot_cmd', 'task_pre', 'target',
        'grabbed', 'rel_pos', 'rel_quat', 'policy_output',
        'sample_steps', 'xi_samples', 'thetadot_all',
    ]}

    counts = {'success': 0, 'fall': 0, 'timeout': 0}

    for i, seed in enumerate(seeds):
        ep_start = time.perf_counter()

        render_this = args.video and (i % max(1, args.video_stride) == 0)

        outcome, step_data, target_0 = run_episode(
            runner, trainer.params, trainer,
            seed=seed,
            rl_move_only=args.rl_move_only,
            rl_pick_only=args.rl_pick_only,
            fold_step_keys=fold_step_keys,
            render=render_this,
            folder=video_dir,
            timed=args.timed,
            ep_idx=i,
            video_tag=args.video_tag,
            save_samples=args.save_samples,
            sample_stride=args.sample_stride,
        )

        ep_time = time.perf_counter() - ep_start
        counts[outcome] += 1
        n = i + 1

        all_data['total_time_s'].append(ep_time)
        all_data['episode_seed'].append(seed)
        all_data['execution_mode'].append(
            'render' if render_this else 'timed' if args.timed else 'scan')
        all_data['success'].append(1 if outcome == 'success' else 0)
        all_data['reason'].append('na' if outcome == 'success' else outcome)
        all_data['target_0'].append(target_0)
        for k in ['step_time_ms', 'theta', 'thetadot', 'task_flag', 'cost_r',
                  'cost_eef_to_obj', 'cost_obj_to_targ', 'cost_dist',
                  'cost_zy', 'cost_weights', 'cem_batch_costs',
                  'qpos_pre', 'qvel_pre', 'thetadot_cmd', 'task_pre', 'target',
                  'grabbed', 'rel_pos', 'rel_quat', 'policy_output',
                  'sample_steps', 'xi_samples', 'thetadot_all',
                  'reward_terms', 'step_done']:
            all_data[k].append(step_data[k])

        print(f"[{n:3d}/{args.n_eval}] seed={seed}  outcome={outcome:8s}  "
              f"success={counts['success']}/{n} ({100*counts['success']/n:.1f}%)")

    print("\n=== Results ===")
    for k, v in counts.items():
        print(f"  {k:8s}: {v}/{args.n_eval} ({100*v/args.n_eval:.1f}%)")

    save_npz(all_data, args.n_eval, args.num_batch, args.num_steps, npz_path,
             timestep=runner.timestep,
             meta=dict(
                 maxiter_cem        = np.array([args.maxiter_cem]),
                 maxiter_projection = np.array([5]),
                 max_steps          = np.array([args.max_steps]),
                 model_path         = np.array([args.model_path]),
                 eval_seed          = np.array([args.seed]),
                 seed_stride        = np.array([args.seed_stride]),
                 cem_key_mode       = np.array([resolved_key_mode]),
                 warmup_episodes    = np.array([args.warmup_episodes]),
                 termination_conditions = np.array([json.dumps({
                     'grab_pos_m_lt': runner.grab_pos_thresh,
                     'grab_rot_rad_each_lt': runner.grab_rot_thresh,
                     'success_pos_m_lt': runner.success_pos_thresh,
                     'success_quaternion_l2_lt': runner.success_rot_thresh,
                     'fall_eef_box_distance_m_gt': runner.fall_pos_thresh,
                     'success_and_fall_require_move_phase': True,
                     'timeout_steps': runner.MAX_STEPS,
                 }, sort_keys=True)]),
                 checkpoint_path    = np.array([args.load or '']),
                 checkpoint_sha256  = np.array([
                     hashlib.sha256(Path(args.load).read_bytes()).hexdigest()
                     if args.load else '']),
                 actor_parameter_count = np.array([
                     int(trainer.params.size) if args.load else 0]),
                 default_weight_set = np.array([args.weights]),
                 checkpoint_algorithm = np.array([checkpoint_algorithm]),
                 checkpoint_version = np.array([checkpoint_version]),
                 reward_profile     = np.array([checkpoint_reward_profile]),
                 checkpoint_type    = np.array([checkpoint_type]),
                 checkpoint_policy_type = np.array([checkpoint_policy_type]),
                 action_mapping     = np.array([checkpoint_action_mapping]),
                 planner_config     = np.array([
                     json.dumps(checkpoint_planner_config, sort_keys=True)]),
                 runtime_planner_config = np.array([
                     json.dumps(runtime_planner_config, sort_keys=True)]),
                 planner_config_match = np.array([not planner_mismatches]),
                 planner_config_mismatches = np.array([
                     json.dumps(planner_mismatches)]),
                 init_cube_rot_deg  = np.array([args.init_cube_rot_deg]),
                 # Compatibility switches in force during collection. The
                 # analysis reads these rather than assuming, so its reward can
                 # never disagree with the environment that produced the data
                 # (e.g. applying cube-slaving to a run that had it disabled).
                 legacy_eef_rot_targets = np.array([LEGACY_EEF_ROT_TARGETS]),
                 force_xy_face_pair     = np.array([FORCE_XY_FACE_PAIR]),
                 disable_cube_slaving   = np.array([DISABLE_CUBE_SLAVING]),
             ))


if __name__ == '__main__':
    main()
