# Tray Push: contact-penetration and video findings

Date: 2026-09-29

## Summary

Audited all 1,200 small-batch Tray Push episodes, including 266 recorded successes, across 610,938 saved observations. All 1,200 corresponding videos decoded without reported errors and matched their episode lengths.

Robot–tray collision-geometry penetration exceeds 20 mm in 539/1,200 episodes (44.9%), including 112/266 successful episodes (42.1%). Ten successful episodes exceed 30 mm; none exceeds 40 mm. Maximum robot–tray depth is 64.91 mm overall and 32.12 mm among successful episodes.

**Important difference from the box audit:** the tray model has fixed visible tables but movable collision-table proxies. The two proxy slide coordinates are not recorded. Table values below are therefore geometric overlap against the **fixed visible tabletop reference**, not recovered penetration into the actual moving simulation colliders. Robot–tray reconstruction does not require those missing table coordinates.

## Scope and reconstruction

- Inputs: all 60 Tray Push NPZ files selected by [benchmark_stat_batch_sm.ipynb](benchmark_stat_batch_sm.ipynb); batches 50, 150, and 250; four methods, five seed blocks per method/batch, 20 episodes per file.
- Input root: `eval_sweep_100_tray_push_common_jit_randomized_xy001_yaw2_b50_150_250-20260925T081018Z-1-001/eval_sweep_100_tray_push_common_jit_randomized_xy001_yaw2_b50_150_250/`.
- Task source: `origin/issue_53_tray_push_ppo`, snapshot commit `6117e21f30808697a60a3d7f33e857062dfaae77`; native MuJoCo 3.3.1 used for geometric reconstruction, not a rerun of MJX dynamics.
- Saved 68-dimensional observations contain 12 robot joint positions and the 7-coordinate tray pose, sufficient for robot–tray geometry. The missing two coordinates belong to the collision-table sliders.
- Reconstructed end-effector positions agree with the saved observation positions to within 0.00051 mm over all audited states. Consecutive pre-step robot positions exactly match the previous saved post-step `theta`. Tray collision geometry also matches the runtime XML template.
- All files specify `sweep8`, `ars4124` observations, five initial settling steps, and a maximum of 600 execution steps. The runner timestep is 0.1 s.
- The five seed blocks repeat their seed across 20 episodes. These are 1,200 recorded trials, not 1,200 independently randomized initial/goal poses. The 20 saved observation trajectories in each file are nevertheless distinct.

Depth means the magnitude of a negative MuJoCo collision-contact distance in millimeters, using enabled collision pairs. It is not measured material deformation, a pixel-derived depth, or necessarily the minimum translation needed to separate whole objects. Collision approximations and rendered robot meshes can differ.

Each count below uses the maximum over an episode’s saved observations. Thresholds are strictly `>` and cumulative. They do not measure duration or force. The initial settling sequence is not in these recorded observations; unsaved between-step and final post-step states may contain additional overlap. A recorded success is the task’s outcome flag, not a contact-validity judgment.

## Robot–tray penetration

| Episode maximum | All episodes (n=1,200) | Successful episodes (n=266) |
|---|---:|---:|
| >5 mm | 1,167 (97.3%) | 266 (100.0%) |
| >10 mm | 1,088 (90.7%) | 265 (99.6%) |
| >20 mm | 539 (44.9%) | 112 (42.1%) |
| >30 mm | 117 (9.8%) | 10 (3.8%) |
| >40 mm | 25 (2.1%) | 0 (0.0%) |
| >50 mm | 4 (0.3%) | 0 (0.0%) |

### Method breakdown

Each method has 300 recorded episodes. Threshold columns below count only successful episodes.

| Method | Successes / 300 | Successful >20 mm | Successful >30 mm | Largest successful robot–tray depth |
|---|---:|---:|---:|---:|
| Hand-Tuned | 23 (7.7%) | 8 | 2 | 30.84 mm |
| Bayesian | 43 (14.3%) | 10 | 2 | 30.78 mm |
| PPO | 85 (28.3%) | 37 | 3 | 31.76 mm |
| ARS | 115 (38.3%) | 57 | 3 | 32.12 mm |

Among successful episodes, robot–tray depth exceeds 20 mm during approach/pick in 1 episode and during move/push in 111 episodes. For >30 mm, the corresponding counts are 1 and 9. The top three successful maxima occur during move/push. These phase counts describe observed states, not causal attribution to a particular cost weight.

## Table-reference overlap

The source model defines `table_geom_0`/`table_geom_1` as fixed visible boxes with collision disabled. Separate `table_geom_0_col`/`table_geom_1_col` boxes sit on slide joints with weld constraints. Their motion is part of the simulation state but is omitted from the NPZ observations.

For this reference check only, those two slide coordinates were set to zero, placing each collision box at its fixed visible-table geometry. Original table contact masks were retained. No physical rollout was performed. The resulting values quantify overlap with the nominal tabletop among those enabled pairs; they cannot establish the actual dynamic table-contact distances or forces.

| Contact / episode maximum | All episodes (n=1,200) | Successful episodes (n=266) |
|---|---:|---:|
| Tray–fixed-table >5 mm | 449 (37.4%) | 37 (13.9%) |
| Tray–fixed-table >20 mm | 301 (25.1%) | 35 (13.2%) |
| Tray–fixed-table >30 mm | 169 (14.1%) | 22 (8.3%) |
| Tray–fixed-table >40 mm | 26 (2.2%) | 0 (0.0%) |
| Robot–fixed-table >5 mm | 55 (4.6%) | 3 (1.1%) |
| Robot–fixed-table >20 mm | 35 (2.9%) | 1 (0.4%) |
| Robot–fixed-table >30 mm | 15 (1.3%) | 1 (0.4%) |
| Robot–fixed-table >40 mm | 1 (0.1%) | 0 (0.0%) |

Maximum tray–table-reference overlap is 70.95 mm overall and 36.66 mm among successes. Maximum robot–table-reference overlap is 40.92 mm overall and 30.81 mm among successes.

These are not solely initial-loading events. Excluding the first ten saved states (one second after the recorded episode starts) leaves each episode’s >20/>30/>40/>50 mm threshold membership unchanged for robot–tray and both table-reference categories. The successful table-reference maxima occur at video playback times 8.3 s (tray–table) and 2.6 s (robot–table).

Separately, robot–robot penetration exceeds 5 mm in 75 episodes, including 3 successes. Maximum robot–robot depth is 49.19 mm overall and 8.48 mm among successes. This category includes both self-contact and contact between arms.

## Video evidence

All 1,200 videos were fully decoded and uniquely matched by method/batch, block seed, and episode index; frame counts agree with the saved episode lengths. This is an automated integrity/matching check plus state-based analysis, not a claim that every video was watched manually. The selected extreme frames were visually inspected.

Only two figures are included for Git:

1. [Combined top-three worst successful-episode snapshots](tray_push_penetration_snapshots/successful_worst_cases_top3.jpg): nine panels, with one row per contact category (robot–tray and the two fixed-table reference categories) and columns ranked first through third among the 266 episodes marked successful. Each row shows three distinct successful episodes. Maxima are taken anywhere in those episodes, not necessarily at the instant success is declared. Table panels explicitly identify the reference limitation. Earlier all-episode and six-panel overviews are preserved in the ignored archive.
2. [Top three successful robot–tray episodes](tray_push_penetration_snapshots/robot_tray_successful_top3.jpg): 32.12, 31.76, and 31.37 mm, from ARS b250/seed27/ep0015, PPO b250/seed6/ep0008, and ARS b50/seed27/ep0003, respectively.

See the [evidence index](tray_push_penetration_snapshots/README.md) for all panel identities, timestamps, and original local-video links. All frames come from the original videos; only headers and JPEG compression were added. Observations are pre-step and videos post-step, so observation index `s` corresponds to video frame `s−1`. Episode/frame indices are zero-based; timestamps are playback times.

Detailed per-episode/per-state reports, video checks, raw snapshots, metadata, and audit scripts are preserved in `tray_push_penetration_snapshots/archive/`, excluded from Git.

## Relevant code behavior and interpretation

The following observations refer to the issue-53 snapshot above, not a modification of the current simulator:

- `real_demo/sampling_based_planner/mjx_planner.py` constructs a move-phase collision mask that excludes contacts involving any of the five tray geoms. Thus robot–tray contacts are omitted from that move-phase collision cost. Tray–table contacts also lack a robot/object-prefixed geom and are not included in the base collision mask.
- `real_demo/real_demo/rl_trainer_batch.py` constructs a zero velocity vector each execution step and writes in the robot command velocities; tray/table velocities are not carried into that step. Execution uses one MJX step at 0.1 s. The scene specifies one solver iteration.
- The success test checks task phase and tray position/orientation error, not a maximum-penetration limit. The saved `collision` signal is a planner-horizon cost, not a measured execution penetration in millimeters.

These implementation choices help explain why task success is not sufficient evidence of physically admissible contact, but this audit does not isolate their individual causal effects. The missing table states prevent an exact retrospective audit of the dynamic table contacts.

The defensible conclusion is that the saved tray trials contain substantial robot–tray overlap, including successful episodes, and overlap with the nominal rigid-table geometry. A softer tray is not established as a remedy: translating overlap into a real material requirement would require calibrated deformation, force, and grasp/push-stability measurements.

No notebook, simulation source, model, or benchmark data was changed. Only analysis outputs/documentation and the local-archive ignore rule were added.
