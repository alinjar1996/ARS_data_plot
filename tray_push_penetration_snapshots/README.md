# Tray-push selected evidence

Two figures are retained for Git. Detailed reports, individual raw/labeled snapshots, and audit scripts are in `archive/`, which is ignored by Git. No benchmark or simulator files were changed.

[Full findings and threshold counts](../TRAY_PUSH_PENETRATION_FINDINGS.md)

## Top three worst successful episodes per contact type

[Open combined nine-panel figure](successful_worst_cases_top3.jpg)

Rows: robot–tray, tray–fixed-table reference, robot–fixed-table reference. Columns: worst, second-worst, third-worst among the 266 episodes marked successful. Each row uses three distinct successful episodes, ranked by their maximum depth anywhere in the episode, not necessarily at the instant success is declared. Table rows are reference overlaps against the fixed visible tables, NOT recovered contacts with the movable simulation table colliders.

## Top three successful robot–tray episodes

[Open successful-episode comparison](robot_tray_successful_top3.jpg)

Three distinct successful episodes, ordered left to right by maximum robot–tray depth. These illustrate extremes, not typical outcomes.

## Panel provenance

Snapshots come from original video frames; headers were added and JPEG compression applied. No scene content was synthesized. The camera can occlude contact regions. Depths come from reconstructed collision geometry, not image measurements. Videos and NPZ files are local data and are not included in Git.

Episode/frame indices are zero-based. Observations are pre-step, videos post-step: saved observation s corresponds to video frame s−1. Times below are playback times. Phase 0 is approach/pick; phase 1 is move/push.

| Figure / panel | Depth (mm) | Method / batch / seed / episode | Playback time | Original video (local) |
|---|---:|---|---:|---|
| Successful worst cases: robot_tray, rank 1 | 32.121 | ARS / 250 / 27 / 0015 | 22.1 s | [MP4](../eval_sweep_100_tray_push_common_jit_randomized_xy001_yaw2_b50_150_250-20260925T081018Z-1-001/eval_sweep_100_tray_push_common_jit_randomized_xy001_yaw2_b50_150_250/videos/ars4124_batch250/20260924_214610_914234/ep_0015_seed27.mp4) |
| Successful worst cases: robot_tray, rank 2 | 31.756 | PPO / 250 / 6 / 0008 | 8.5 s | [MP4](../eval_sweep_100_tray_push_common_jit_randomized_xy001_yaw2_b50_150_250-20260925T081018Z-1-001/eval_sweep_100_tray_push_common_jit_randomized_xy001_yaw2_b50_150_250/videos/ppo5307_best_val_batch250/20260924_223440_574286/ep_0008_seed6.mp4) |
| Successful worst cases: robot_tray, rank 3 | 31.365 | ARS / 50 / 27 / 0003 | 23.2 s | [MP4](../eval_sweep_100_tray_push_common_jit_randomized_xy001_yaw2_b50_150_250-20260925T081018Z-1-001/eval_sweep_100_tray_push_common_jit_randomized_xy001_yaw2_b50_150_250/videos/ars4124_batch50/20260924_211204_541923/ep_0003_seed27.mp4) |
| Successful worst cases: tray_table_reference, rank 1 | 36.657 | Bayesian / 150 / 6 / 0017 | 8.3 s | [MP4](../eval_sweep_100_tray_push_common_jit_randomized_xy001_yaw2_b50_150_250-20260925T081018Z-1-001/eval_sweep_100_tray_push_common_jit_randomized_xy001_yaw2_b50_150_250/videos/bayesian_batch150/20260924_202324_280940/ep_0017_seed6.mp4) |
| Successful worst cases: tray_table_reference, rank 2 | 35.288 | ARS / 250 / 99 / 0005 | 16.5 s | [MP4](../eval_sweep_100_tray_push_common_jit_randomized_xy001_yaw2_b50_150_250-20260925T081018Z-1-001/eval_sweep_100_tray_push_common_jit_randomized_xy001_yaw2_b50_150_250/videos/ars4124_batch250/20260924_213610_513801/ep_0005_seed99.mp4) |
| Successful worst cases: tray_table_reference, rank 3 | 35.212 | Hand-Tuned / 250 / 99 / 0011 | 44.9 s | [MP4](../eval_sweep_100_tray_push_common_jit_randomized_xy001_yaw2_b50_150_250-20260925T081018Z-1-001/eval_sweep_100_tray_push_common_jit_randomized_xy001_yaw2_b50_150_250/videos/handtuned_batch250/20260924_193717_725806/ep_0011_seed99.mp4) |
| Successful worst cases: robot_table_reference, rank 1 | 30.807 | Bayesian / 250 / 77 / 0007 | 2.6 s | [MP4](../eval_sweep_100_tray_push_common_jit_randomized_xy001_yaw2_b50_150_250-20260925T081018Z-1-001/eval_sweep_100_tray_push_common_jit_randomized_xy001_yaw2_b50_150_250/videos/bayesian_batch250/20260924_204807_653183/ep_0007_seed77.mp4) |
| Successful worst cases: robot_table_reference, rank 2 | 19.824 | Bayesian / 250 / 99 / 0001 | 2.1 s | [MP4](../eval_sweep_100_tray_push_common_jit_randomized_xy001_yaw2_b50_150_250-20260925T081018Z-1-001/eval_sweep_100_tray_push_common_jit_randomized_xy001_yaw2_b50_150_250/videos/bayesian_batch250/20260924_203958_664978/ep_0001_seed99.mp4) |
| Successful worst cases: robot_table_reference, rank 3 | 16.392 | Bayesian / 150 / 77 / 0019 | 2.6 s | [MP4](../eval_sweep_100_tray_push_common_jit_randomized_xy001_yaw2_b50_150_250-20260925T081018Z-1-001/eval_sweep_100_tray_push_common_jit_randomized_xy001_yaw2_b50_150_250/videos/bayesian_batch150/20260924_202733_218612/ep_0019_seed77.mp4) |
| Successful comparison: 1 | 32.121 | ARS / 250 / 27 / 0015 | 22.1 s | [MP4](../eval_sweep_100_tray_push_common_jit_randomized_xy001_yaw2_b50_150_250-20260925T081018Z-1-001/eval_sweep_100_tray_push_common_jit_randomized_xy001_yaw2_b50_150_250/videos/ars4124_batch250/20260924_214610_914234/ep_0015_seed27.mp4) |
| Successful comparison: 2 | 31.756 | PPO / 250 / 6 / 0008 | 8.5 s | [MP4](../eval_sweep_100_tray_push_common_jit_randomized_xy001_yaw2_b50_150_250-20260925T081018Z-1-001/eval_sweep_100_tray_push_common_jit_randomized_xy001_yaw2_b50_150_250/videos/ppo5307_best_val_batch250/20260924_223440_574286/ep_0008_seed6.mp4) |
| Successful comparison: 3 | 31.365 | ARS / 50 / 27 / 0003 | 23.2 s | [MP4](../eval_sweep_100_tray_push_common_jit_randomized_xy001_yaw2_b50_150_250-20260925T081018Z-1-001/eval_sweep_100_tray_push_common_jit_randomized_xy001_yaw2_b50_150_250/videos/ars4124_batch50/20260924_211204_541923/ep_0003_seed27.mp4) |

Model: `origin/issue_53_tray_push_ppo`, commit `6117e21f30808697a60a3d7f33e857062dfaae77`; MuJoCo 3.3.1. All 1,200 videos decoded without reported errors and matched saved episode lengths. Detailed measurement limitations are in the findings summary.
