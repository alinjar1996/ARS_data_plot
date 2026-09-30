# Ball Lift: contact-penetration, video coverage, and successful worst cases

**Historical dataset:** This penetration audit and its snapshots cover the September 23/24 inputs (plus the earlier PPO4951 best-evaluation batch-250 files), not the September 29 retrained-Bayesian sweep selected in the small-batch notebook on 2026-09-30. These findings have not been recomputed for the new runs.

Date: 2026-09-29

## Summary

Audited all 1,200 small-batch Ball Lift episodes across 60 NPZ files and 205,761 saved states. There are 758 recorded successful episodes. Every successful episode exceeds 20 mm of robot–ball collision-shape overlap; 715 (94.3%) exceed 30 mm, 170 (22.4%) exceed 40 mm, and 8 (1.1%) exceed 50 mm.

The largest robot–ball depth is 54.84 mm, in a successful ARS episode. Ball–table depth reaches 57.33 mm, also in a successful episode and not during initialization. Robot–table depth is much smaller among successful episodes: maximum 6.73 mm, with only one success exceeding 5 mm.

[Open the combined top-three worst successful-episode snapshots](ball_lift_penetration_snapshots/successful_worst_cases_top3.jpg).

**These nine snapshots are native MuJoCo re-renders of saved states, not original video frames.** None of the selected worst-case episodes has a surviving original video in this local dataset. Every panel explicitly states its provenance. The poses come from recorded data; no physical trajectory was rerun and no image-synthesis model was used.

## Scope and measurement

- Inputs: the Ball Lift `SOURCE_PATTERNS` as selected on 2026-09-29 in [benchmark_stat_batch_sm.ipynb](benchmark_stat_batch_sm.ipynb); four methods, batches 50/150/250, five seed blocks per method/batch, 20 episodes per file.
- Batches 50/150: `eval_sweep_100_ball_lift_h15_20260924_140128/npz/`.
- Batch 250, Hand-Tuned/Bayesian/ARS: `eval_sweep_100_ball_lift_h15_20260923_152226/`.
- Batch 250, PPO: `ppo_results/Ball_lift/ppo4951_best_eval/batch250/block*/`.
- Model: `issue_49_ball_lift_ppo`, commit `4ec9b0bf65f791bd10def2d13071d0457e6a2b24`; geometric audit with native MuJoCo 3.3.1. Model assets on this branch last changed on August 29 before the earliest selected PPO recordings.
- Full saved pre-step `qpos`/`qvel`, ball pose, target position, and per-episode `table1_pos` were restored. All saved ball-pose slices match full qpos; subsequent pre-step robot positions match the preceding saved post-step joint positions exactly.
- Unlike Tray Push, the Ball Lift tables are rigid bodies without unrecorded table-slider coordinates. The randomized table-1 body position is explicitly saved and was restored, including the attached second robot. No nominal-table substitution was needed.
- All files use a 0.1 s timestep, horizon 15, two CEM iterations, and a 300-step episode limit. The modeled sphere radius is 0.11 m.

Depth means the magnitude of a negative MuJoCo collision-contact distance, in millimeters, for enabled collision pairs. These are reconstructed collision-shape overlaps, not measured material deformation or pixel-derived depths. Robot collision approximations can differ from visible meshes. The audit does not reconstruct contact forces or rerun the original MJX dynamics.

Counts use each episode’s maximum over saved pre-step states. Thresholds are strictly `>` and cumulative, not separate bands. Each episode counts once at each threshold. Unsaved intermediate/final post-step states may contain additional overlap. “Successful” means the recorded task outcome, not a contact-validity certification.

## Robot–ball penetration

| Maximum per episode | All episodes (n=1,200) | Successful episodes (n=758) |
|---|---:|---:|
| >5 mm | 826 (68.8%) | 758 (100.0%) |
| >10 mm | 798 (66.5%) | 758 (100.0%) |
| >20 mm | 778 (64.8%) | 758 (100.0%) |
| >30 mm | 729 (60.8%) | 715 (94.3%) |
| >40 mm | 178 (14.8%) | 170 (22.4%) |
| >50 mm | 8 (0.7%) | 8 (1.1%) |

### Successful episodes by method

Each method has 300 episodes. Penetration columns below count only successful episodes.

| Method | Successful episodes | >20 mm | >30 mm | >40 mm | >50 mm | Largest successful depth |
|---|---:|---:|---:|---:|---:|---:|
| Hand-Tuned | 21 | 21 | 14 | 0 | 0 | 36.74 mm |
| Bayesian | 156 | 156 | 129 | 8 | 0 | 43.82 mm |
| PPO | 286 | 286 | 278 | 52 | 1 | 50.56 mm |
| ARS | 295 | 295 | 294 | 110 | 7 | 54.84 mm |

All 758 successful episodes exceed 20 mm during move/lift; 11 also exceed it during approach/pick. For >30 mm, 715 exceed it during move/lift and 2 during approach/pick. Phase counts can overlap. This association does not establish that overlap caused success.

## Table contacts

| Contact / episode maximum | All episodes (n=1,200) | Successful episodes (n=758) |
|---|---:|---:|
| Ball–table >10 mm | 1,200 (100.0%) | 758 (100.0%) |
| Ball–table >20 mm | 683 (56.9%) | 474 (62.5%) |
| Ball–table >30 mm | 149 (12.4%) | 116 (15.3%) |
| Ball–table >40 mm | 34 (2.8%) | 26 (3.4%) |
| Ball–table >50 mm | 3 | 3 |
| Robot–table >5 mm | 8 (0.7%) | 1 (0.1%) |
| Robot–table >10 mm | 7 (0.6%) | 0 |
| Robot–table >20 mm | 4 (0.3%) | 0 |
| Robot–table >40 mm | 1 (0.1%) | 0 |

Ball–table maximum: 57.33 mm overall and among successes. Robot–table maximum: 45.33 mm overall, 6.73 mm among successes.

The successful ball–table maxima occur at saved-state times 5.5, 10.9, and 9.3 s. They are not simply initial loading. Excluding the first ten saved states (one simulated second) leaves every episode’s >5/>10/>20/>30/>40/>50 mm threshold membership unchanged for all three contact categories. This cutoff is a sensitivity check, not a claim that every transient has settled after exactly one second.

## Top three worst cases among successful episodes

Each row ranks three distinct episodes by the maximum anywhere in an episode ultimately marked successful, not necessarily the state at which success was declared. Indices below are zero-based.

| Contact | Rank 1 | Rank 2 | Rank 3 |
|---|---|---|---|
| Robot–ball | 54.84 mm; ARS b150, seed3, ep0003 | 53.85 mm; ARS b250, seed3, ep0003 | 51.31 mm; ARS b250, seed2, ep0004 |
| Ball–table | 57.33 mm; Bayesian b150, seed2, ep0017 | 53.77 mm; Bayesian b150, seed2, ep0005 | 51.23 mm; Bayesian b250, seed0, ep0012 |
| Robot–table | 6.73 mm; PPO b50, seed1, ep0003 | 2.97 mm; Bayesian b250, seed3, ep0018 | 1.55 mm; PPO b250, seed4, ep0008 |

The [combined figure](ball_lift_penetration_snapshots/successful_worst_cases_top3.jpg) uses a contact-focused camera to show the saved geometry. The transparent cube is the target marker. Scene geometry was not deformed to illustrate the penetration. Table and robot placements use the saved randomized table position. Times are elapsed simulation times at the saved pre-step state (`index × 0.1 s`), not original-video playback times.

The [snapshot index](ball_lift_penetration_snapshots/README.md) records panel identities, source NPZ paths, times, and phases. Only this one combined figure is retained for Git; individual renders and detailed reports are in the ignored archive.

## Original-video coverage and verification

Only 202/1,200 selected episodes have surviving, attributable local videos:

| Method | Batch 50 | Batch 150 | Batch 250 |
|---|---:|---:|---:|
| Hand-Tuned | 20 | 20 | 0 |
| Bayesian | 29 | 37 | 0 |
| PPO | 27 | 25 | 0 |
| ARS | 23 | 21 | 0 |

The 50/150 evaluator repeatedly saved to `epNNN_outcome.mp4` without a seed/block component. Its 40 run logs record 800 video writes into just 202 surviving filenames. Later runs overwrote earlier recordings when episode number and outcome matched; different-outcome leftovers can survive from earlier blocks. Matching by episode number alone would therefore be incorrect.

Each surviving filename was attributed to its latest logged writer using the timestamped NPZ output, then checked against that episode’s outcome and frame count. All 202 videos fully decoded without reported errors, with no matching discrepancies. For one surviving recording per method, a restored state rendered with the original camera was compared with the corresponding video frame: mean absolute pixel difference was approximately 1.55–1.60 on a 0–255 scale. This is a four-case spot check, not exhaustive visual verification of all recordings.

No selected batch-250 videos are available locally. PPO batch-250 metadata requests every fifth episode rather than every episode, but those recordings are also absent here. Videos under the older `eval_sweep_100_ball_lift/videos/` root belong to a different run and were not substituted. None of the nine ranked successful extremes has a surviving attributable recording.

Video frames are post-step while saved qpos is pre-step, so saved state `s` corresponds to video frame `s−1` when the correct recording exists. The original filenames use one-based episode numbering; this report uses zero-based NPZ indices.

## Relevant implementation and physical interpretation

At the issue-49 snapshot, `real_demo/sampling_based_planner/mjx_planner.py` excludes ball-related contacts from its move-phase collision mask. Ball–table contact also does not enter the base robot/object-prefixed collision mask. In `real_demo/real_demo/rl_trainer_batch.py`, execution resets the velocity vector before inserting robot command velocities, then takes one 0.1 s MJX step; ball velocity is not carried into that step. The scene uses one solver iteration.

The success condition requires move phase and ball-to-target distance below 0.05 m, not a penetration-depth limit. The saved `collision` signal is a planner-horizon cost rather than measured execution depth. These code observations help explain why success does not certify contact validity, but this audit does not isolate each mechanism’s causal contribution.

The recorded successful motions consistently use overlapping robot/ball collision geometry. This limits confidence in rigid-contact physical replication. It does not directly specify the stiffness, softness, or deformation required of a real ball; that would require calibrated force–deformation and grasp-stability measurements.

No notebook, simulator code, source model, or benchmark data was changed. Added outputs are this report, one combined figure, its index, and local-only audit artifacts under `ball_lift_penetration_snapshots/archive/`, excluded via `.gitignore`.
