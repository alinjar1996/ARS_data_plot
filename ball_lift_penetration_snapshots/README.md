# Ball-lift successful worst-case snapshots

[Open combined nine-panel figure](successful_worst_cases_top3.jpg)

[Full findings and video-coverage report](../BALL_LIFT_PENETRATION_FINDINGS.md)

Rows: robot–ball, ball–table, robot–table. Columns: first, second, third largest episode maximum among the 758 episodes marked successful. Each row contains three distinct episodes. The sampled state can precede the eventual success event.

**All nine panels are native MuJoCo re-renders of saved states, not original video frames.** Their original episode recordings are unavailable locally. A contact-focused camera is used. No physical rollout was rerun, and no scene content was generated with an image-synthesis model. Saved full qpos/qvel, target position, and randomized table-1 body position were restored. The transparent cube is the target marker.

Times are elapsed simulation time at the saved pre-step state (index × 0.1 s), not video playback times. Episode indices are zero-based; the original evaluator used one-based episode numbers in video filenames.

| Contact / rank | Depth (mm) | Method / batch / seed / episode | Saved state time | Phase | Source NPZ (local) |
|---|---:|---|---:|---:|---|
| robot_ball / 1 | 54.838 | ARS / 150 / 3 / 0003 | 9.8 s | 1 | [NPZ](../eval_sweep_100_ball_lift_h15_20260924_140128/npz/policy_batch150/eval_policy_batch150_n20_20260924_160343.npz) |
| robot_ball / 2 | 53.847 | ARS / 250 / 3 / 0003 | 8.1 s | 1 | [NPZ](../eval_sweep_100_ball_lift_h15_20260923_152226/policy_batch250/eval_policy_batch250_n20_20260923_182246.npz) |
| robot_ball / 3 | 51.310 | ARS / 250 / 2 / 0004 | 8.2 s | 1 | [NPZ](../eval_sweep_100_ball_lift_h15_20260923_152226/policy_batch250/eval_policy_batch250_n20_20260923_182011.npz) |
| ball_table / 1 | 57.332 | Bayesian / 150 / 2 / 0017 | 5.5 s | 0 | [NPZ](../eval_sweep_100_ball_lift_h15_20260924_140128/npz/bayesian_batch150/eval_bayesian_batch150_n20_20260924_152825.npz) |
| ball_table / 2 | 53.773 | Bayesian / 150 / 2 / 0005 | 10.9 s | 0 | [NPZ](../eval_sweep_100_ball_lift_h15_20260924_140128/npz/bayesian_batch150/eval_bayesian_batch150_n20_20260924_152825.npz) |
| ball_table / 3 | 51.230 | Bayesian / 250 / 0 / 0012 | 9.3 s | 0 | [NPZ](../eval_sweep_100_ball_lift_h15_20260923_152226/bayesian_batch250/eval_bayesian_batch250_n20_20260923_170405.npz) |
| robot_table / 1 | 6.733 | PPO / 50 / 1 / 0003 | 8.4 s | 0 | [NPZ](../eval_sweep_100_ball_lift_h15_20260924_140128/npz/ppo_batch50/eval_policy_batch50_n20_20260924_161300.npz) |
| robot_table / 2 | 2.974 | Bayesian / 250 / 3 / 0018 | 9.1 s | 0 | [NPZ](../eval_sweep_100_ball_lift_h15_20260923_152226/bayesian_batch250/eval_bayesian_batch250_n20_20260923_171354.npz) |
| robot_table / 3 | 1.554 | PPO / 250 / 4 / 0008 | 4.8 s | 0 | [NPZ](../ppo_results/Ball_lift/ppo4951_best_eval/batch250/block4/eval_policy_batch250_n20_20260829_131024.npz) |

Phase 0: approach/pick. Phase 1: move/lift. Depth is collision-shape overlap, not measured real material deformation. Table positions are recorded for this task, unlike the missing tray-task collision-table coordinates.

Only the combined figure is retained for Git. Individual renders, full audit reports, video checks, and scripts are stored in the ignored `archive/` folder. Source model: `issue_49_ball_lift_ppo`, commit `4ec9b0bf65f791bd10def2d13071d0457e6a2b24`; MuJoCo 3.3.1.
