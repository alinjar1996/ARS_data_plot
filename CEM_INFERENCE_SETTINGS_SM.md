# CEM settings for the small-batch inference results

Verified 2026-10-07 against the sources selected by [benchmark_stat_batch_sm.ipynb](benchmark_stat_batch_sm.ipynb).

This documents **inference**, not the training settings of the saved policies. Scope: Ball Lift, Box Lift, and Tray Push; Hand-Tuned, Bayesian, PPO, and ARS; CEM batches **50, 150, and 250**. No simulation was rerun for this audit.

All **180 selected aggregate NPZ files** were checked: 3 tasks × 4 methods × 3 batches × 5 files. Each file contains 20 episodes, giving 100 recorded episodes per task/method/batch and 3,600 overall. These are recorded episode counts, not a claim of 100 independent random scenarios; see the seed schedules below. The notebook, figures, simulator code, and `supp.tex` were not changed.

## 1. Runtime budgets: applicable to every method within each task

`batch_size` is the number of candidate trajectories **per CEM iteration**. It is not the number of evaluation episodes, PPO environments, or optimizer minibatch size. `horizon` is the number of prediction steps in one plan; `max_steps` is the maximum number of executed MPC calls in an episode.

| Task | CEM batch | Prediction horizon | CEM iterations per MPC call | Projection iterations | Timestep | Maximum episode MPC steps | Nominal candidate evaluations per MPC call |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ball Lift | 50 | 15 | 2 | 5 | 0.1 s | 300 | 100 |
| Ball Lift | 150 | 15 | 2 | 5 | 0.1 s | 300 | 300 |
| Ball Lift | 250 | 15 | 2 | 5 | 0.1 s | 300 | 500 |
| Box Lift | 50 | 15 | 3 | 5 | 0.1 s | 300 | 150 |
| Box Lift | 150 | 15 | 3 | 5 | 0.1 s | 300 | 450 |
| Box Lift | 250 | 15 | 3 | 5 | 0.1 s | 300 | 750 |
| Tray Push | 50 | 10 | 1 | 5 | 0.1 s | 600 | 50 |
| Tray Push | 150 | 10 | 1 | 5 | 0.1 s | 600 | 150 |
| Tray Push | 250 | 10 | 1 | 5 | 0.1 s | 600 | 250 |

The final column is `batch_size × maxiter_cem`, counting candidate cost evaluations inside the planning loop, not physics steps or wall-clock time. Projection iterations apply to candidate feasibility projection, not the MuJoCo contact solver.

The planner's nominal prediction durations (`horizon × timestep`) are **1.5 s, 1.5 s, and 1.0 s**. Episode simulation-time caps are **30 s, 30 s, and 60 s**, respectively; success/fall can terminate earlier. The selected Ball run explicitly disables wall-clock timeout (`timeout_secs=0`); Box and Tray use step-limited evaluation loops, not a 30/60-second wall-clock deadline.

Evidence: all selected NPZs store the horizon, CEM/projection iteration counts, episode cap, and batch. Ball/Box also store `timestep=0.1`. The selected Tray NPZs omit timestep, but **all 60 corresponding Tray logs** print `Timestep: 0.1`; their embedded `planner_config_json` agrees with the other runtime settings. All 60 Box `runtime_planner_config` records were also checked. No within-task/method budget discrepancy was found.

## 2. Exact runs and methods covered

Source roots, relative to this results repository:

- **B:** [September 29 Ball sweep](eval_sweep_100_ball_lift_bayes_retrained_20260929_191418/npz/).
- **X:** [October 1 all-contacts Box sweep](eval_box_lift_all_contacts_b50_150_250_20261001_154416_3370028/npz/).
- **XB:** [Separate post-October-1 Box Bayesian5001 sweep](eval_box_lift_bayesian5001_b50_150_250_after_20261001_154416/npz/).
- **T:** [September 25 randomized common-JIT Tray sweep](eval_sweep_100_tray_push_common_jit_randomized_xy001_yaw2_b50_150_250-20260925T081018Z-1-001/eval_sweep_100_tray_push_common_jit_randomized_xy001_yaw2_b50_150_250/npz/).

In this table `{b}` is separately 50, 150, and 250. Each row covers 15 files: five per batch. Every method in a task uses that task's runtime budgets in Section 1.

| Task | Plotted method | Selected weights/checkpoint | Source root + subfolder |
|---|---|---|---|
| Ball Lift | Hand-Tuned | Historical fixed hand-tuned weights | B / `handtuned_batch{b}` |
| Ball Lift | Bayesian | September 29 retrained `bayesian_ball_lift_best.json`, frozen as `bayesian_weights.json` | B / `bayesian_batch{b}` |
| Ball Lift | PPO | PPO4951 **best-evaluation**, update 1410 | B / `ppo_best_eval_batch{b}` |
| Ball Lift | ARS | ARS3950 | B / `ars_batch{b}` |
| Box Lift | Hand-Tuned | Fixed Box hand-tuned weights | X / `handtuned_batch{b}` |
| Box Lift | Bayesian | Bayesian5001, not Bayesian5002 | XB / `bayesian5001_batch{b}` |
| Box Lift | PPO | PPO5005 **best-evaluation**, update 2900 | X / `ppo5005besteval_batch{b}` |
| Box Lift | ARS | ARS4218 | X / `ars4218_batch{b}` |
| Tray Push | Hand-Tuned | Fixed Tray hand-tuned weights | T / `handtuned_batch{b}` |
| Tray Push | Bayesian | Historical Bayesian4104 | T / `bayesian_batch{b}` |
| Tray Push | PPO | PPO5307 **best-evaluation**, update 2180; directory calls it `best_val` | T / `ppo5307_best_val_batch{b}` |
| Tray Push | ARS | Historical nested ARS4124 actor, iteration 806 | T / `ars4124_batch{b}` |

One checkpoint per learned method/task is reused across the three inference batches. PPO uses its deterministic mean; ARS uses its deterministic actor. CEM still samples candidate trajectories. Hand-Tuned/Bayesian supply fixed weights, whereas the learned actors supply state-dependent weights.

PPO best-training folders, Box Bayesian5002, earlier small-batch runs, individual Box episode dumps, and the large-batch notebook's inputs are **outside this document's scope**. See [BENCHMARK_NPZ_FILES.md](BENCHMARK_NPZ_FILES.md#checkpoint-identity-audit-2026-10-06) for the existing checkpoint identity/hash audit and complete filename patterns.

## 3. Active sampling implementation, not just CLI settings

These details were read from the historical task-specific `mjx_planner.py`, `mpc_planner.py`, and runner used by the selected evaluations. They distinguish the executed algorithm from unused configuration fields.

| Setting | Ball Lift | Box Lift | Tray Push |
|---|---|---|---|
| Robot planning DoFs | 12 | 12 | 12 |
| Trajectory parameterization | Direct velocity samples, `P = I_15` | Direct velocity samples, `P = I_15` | Order-10 Bernstein basis, `P` shape 10 × 11 |
| Sample-vector dimension | 180 | 180 | 132 |
| Initial proposal mean | Zero | Zero | Zero |
| Initial `xi_cov` | `10 I` | `10 I` | `1 I` |
| Sampling covariance passed to Gaussian sampler | `xi_cov + 0.003 I` | Same | Same |
| Active cost-to-sampling weights | `softmax(-10 × cost)` over **all candidates** | Same | Same |
| Variance stabilizer inside active update | `1e-4` | `1e-4` | `1e-4` |
| Proposal across MPC calls | Carry returned mean/matrix to next call | Same | Same |
| Returned trajectory | Minimum-cost candidate from **last** CEM iteration | Same | Same |
| Applied joint velocity | `mean(best_vels[1:6], axis=0)` | Same | Same |
| Configured elite fraction, **not used by active moment update** | 0.20 | 0.20 | 0.15 |
| Configured `alpha_mean`, `alpha_cov`, **not used by active moment update** | 0.6, 0.6 | 0.6, 0.6 | 0.6, 0.6 |

The averaged velocity is applied for one 0.1-second simulation step, then the planner is called again. It does not execute five open-loop steps merely because five planned velocity rows are averaged. The proposal is carried directly; these runner paths do not shift its entries along the horizon before the next call.

### Exact active proposal update

Let `x_i` be an original sampled parameter vector and `J_i` the cost of its projected rollout. The active `compute_adaptive_mean_cov` evaluates:

```text
p_i       = softmax(-10 * J)_i
mu_new    = sum_i p_i * x_i
v_new     = sum_i p_i * (x_i - mu_new)^2       # element-wise
xi_cov    = diag(sqrt(v_new + 1e-4))
next_draw ~ Normal(mu_new, xi_cov + 0.003 I)
```

Important reproduction details:

- The cost comes from the feasibility-projected trajectory, but the proposal moments use the original `xi_samples`.
- Despite its name, the returned `xi_cov` contains **square-root variances** on the diagonal and is then passed as the Gaussian **covariance** argument. This is the historical implementation, not the conventional `diag(v_new)` covariance update. No correction was made during this audit.
- `mean_control_prev` and `sigma_diag_prev` are accepted by the active update but unused. The previous proposal still matters because it generated the current samples; it is not blended into the new moments using the stored 0.6 coefficients.
- `compute_ellite_samples` is called, but its outputs do not feed the active proposal update or final trajectory choice. The elite-based `compute_mean_cov(...)` call is commented out. Therefore **do not describe these runs as updating from only the top 20%/15%, or as using 0.6 mean/covariance smoothing**.
- For Tray's single-iteration planner, the updated proposal affects the next MPC call, not a second evaluated iteration in the same call.

### Projection configuration

The historical runner-to-planner construction supplies the following bound magnitudes. These are planner constraint settings, not independently verified guarantees about realized simulator motion or absence of contact penetration.

| Configured bound / penalty | Ball Lift | Box Lift | Tray Push |
|---|---:|---:|---:|
| Joint position magnitude | π rad | π rad | π rad |
| Joint velocity magnitude | 10 rad/s | 1 rad/s | 1 rad/s |
| Joint acceleration magnitude | 20 rad/s² | 2 rad/s² | 2 rad/s² |
| Joint jerk magnitude | 40 rad/s³ | 4 rad/s³ | 4 rad/s³ |
| `rho_ineq` | 5 | 5 | 5 |
| `rho_projection` | 1 | 1 | 1 |
| Projection iterations | 5 | 5 | 5 |

Thus matching the batch number across different tasks does not make their planning problems identical: horizons, iteration counts, parameterization, initial sampling spread, and constraint settings differ.

## 4. Episode seeds and CEM random-key schedules

| Task | Five block seeds | Episodes per block | Initial candidate-draw key schedule |
|---|---|---:|---|
| Ball Lift | 0, 1, 2, 3, 4 | 20 | Split a new episode key within each block; fold/split the step key per MPC call (`fold_step_keys=True`) |
| Box Lift | 0, 5, 4, 10, 15 | 20 | `seed_stride=0`; repeat the block seed for each episode, reuse its CEM key within the episode (`cem_key_mode=episode`) |
| Tray Push | 99, 6, 77, 27, 114 | 20 | Repeat the block seed for each episode; reuse its CEM key within the episode (`cem_key_schedule=episode`) |

These schedules are shared by all four selected methods **within each task**. They are not the same across tasks. Repeated-seed Box/Tray episodes must not be described as 100 independently sampled scenarios; exact numerical outcomes need not be identical despite repeated seeds.

There is also an **inner CEM key** distinct from the evaluator's initial candidate-draw key: all three inspected `compute_cem` implementations start internal resampling from `split(self.key)`, with `self.key = PRNGKey(42)`. Thus Ball's per-step-folded evaluator key does not imply that every inner CEM resampling key is fresh across MPC calls. Samples still depend on the changing proposal mean/matrix.

The selected Box runs use the October 1 all-contacts model. The selected Tray runs use `evaluation_protocol=sweep8`, `observation_type=ars4124`, initial XY/yaw perturbations ±0.01 m/±2°, joint reset noise 0.01 rad, and five settling steps. These identify the inference conditions; they should not be substituted for the PPO checkpoint's training protocol.

## 5. Evidence and manual verification

Use historical revisions, not whatever defaults happen to be present on the currently checked-out branch.

| Task | Simulator branch lineage | Source revision used for this audit | Evidence strength |
|---|---|---|---|
| Ball Lift | `issue_49_ball_lift_ppo` | `f4202cc` | All 60 NPZs agree; run manifest and saved evaluator/runner/`mjx_planner.py` hashes match this revision. `mpc_planner.py` details are recovered from the corresponding historical source, not a separately saved hash. |
| Box Lift, including separate Bayesian5001 run | `issue_50_box_lift_ppo` | `00898ab` | All 60 NPZs agree; all-contacts run records this HEAD. Both runs' evaluator, runner, `mjx_planner.py`, and `mpc_planner.py` hashes match this revision. |
| Tray Push | Historical issue-53 Tray lineage | `6117e21` | All 60 NPZs/runtime configs and corresponding logs agree. Internal implementation settings are recovered from the matching historical source; this older run does not provide the same source-hash manifest coverage as Ball/Box. |

Local evidence:

- Ball: [run manifest](eval_sweep_100_ball_lift_bayes_retrained_20260929_191418/run_manifest.txt), [artifact hashes](eval_sweep_100_ball_lift_bayes_retrained_20260929_191418/artifact_hashes.sha256).
- Box: [recorded commit](eval_box_lift_all_contacts_b50_150_250_20261001_154416_3370028/git_commit.txt), [all-contacts input hashes](eval_box_lift_all_contacts_b50_150_250_20261001_154416_3370028/input_sha256.txt), [Bayesian-run input hashes](eval_box_lift_bayesian5001_b50_150_250_after_20261001_154416/input_sha256.txt), [dedicated Bayesian5001 hash](eval_box_lift_bayesian5001_b50_150_250_after_20261001_154416/bayesian_5001.sha256).
- Example Box runtime log: [PPO5005 best-evaluation, batch 50, seed 0](eval_box_lift_all_contacts_b50_150_250_20261001_154416_3370028/logs/ppo5005besteval_batch50_seed0.log).
- Example Tray runtime log: [Hand-Tuned, batch 50, seed 99](eval_sweep_100_tray_push_common_jit_randomized_xy001_yaw2_b50_150_250-20260925T081018Z-1-001/eval_sweep_100_tray_push_common_jit_randomized_xy001_yaw2_b50_150_250/logs/eval_handtuned_b50_ino0.log).

The Box Bayesian rerun's shared `input_sha256.txt` also lists Bayesian5002 from the parent comparison setup; that line is **not** proof of which Bayesian was executed. Its dedicated Bayesian5001 hash, selected NPZ checkpoint hashes, and actual run logs identify Bayesian5001. Both runs nevertheless identify the same planner/evaluator source bytes.

Read-only source inspection (run in `manipulator_mujoco`; no checkout required):

```bash
# Ball: active update, sampling, final-candidate selection.
git show f4202cc:real_demo/sampling_based_planner/mjx_planner.py
git show f4202cc:real_demo/sampling_based_planner/mpc_planner.py
git show f4202cc:real_demo/real_demo/rl_trainer_batch.py
git show f4202cc:real_demo/real_demo/jax_evaluate.py

# Box: the corresponding October 1 implementation.
git show 00898ab:real_demo/sampling_based_planner/mjx_planner.py
git show 00898ab:real_demo/sampling_based_planner/mpc_planner.py
git show 00898ab:real_demo/real_demo/rl_trainer_batch.py
git show 00898ab:real_demo/real_demo/jax_evaluate.py
git show 00898ab:run_eval_box_lift_matched.sh

# Tray: corresponding common-JIT implementation and randomized launcher.
git show 6117e21:real_demo/sampling_based_planner/mjx_planner.py
git show 6117e21:real_demo/sampling_based_planner/mpc_planner.py
git show 6117e21:real_demo/real_demo/rl_trainer_batch.py
git show 6117e21:real_demo/real_demo/jax_evaluate.py
git show 6117e21:run_eval_common_jit_randomized_tray_push.sh
git show 6117e21:run_eval_common_jit_b50_150_250_tray_push.sh
```

In the planner, inspect `compute_xi_samples`, `compute_adaptive_mean_cov`, `cem_iter`, and `compute_cem`. In the runner, inspect planner construction, `reset_jax`, and `step_jax`. In the evaluator, inspect runtime argument propagation, seed construction, and metadata saving. For Box/Tray, use the **runtime** configuration, not `planner_config`/`checkpoint_planner_config_json` describing training. The notebook's explicit `SOURCE_PATTERNS` remains the authority for which result folders enter the plots.
