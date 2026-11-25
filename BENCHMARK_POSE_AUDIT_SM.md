# Initial object pose, target pose, and robot posture

**Starting conditions matched within each task.** For the same seed block and saved episode index, every plotted method at batches 50, 150, and 250 had the same initial object position and orientation, target position and orientation, and 12 starting robot joint angles. The comparison covered all 360 NPZ files selected on 2026-11-24, including the ESS, phase-mean, normalized ARS, and trained constant ARS runs. All 7,200 episode records matched their task’s Hand-Tuned batch-50 reference exactly after treating opposite quaternion signs as the same orientation.

| Task | Records checked | Distinct starting configurations per 100 episodes | What varies between configurations |
|---|---:|---:|---|
| Ball Lift | 2,400 | 100 | Each of five seed blocks contains 20 distinct configurations. Ball position, target position, and robot posture vary. The recorded ball and target orientations are identity throughout. |
| Box Lift | 2,400 | 5 | Each seed repeats one configuration for 20 episodes. Box position, target position and orientation, and robot posture vary across seeds. The initial box orientation is identity throughout. |
| Tray Push | 2,400 | 5 | Each seed repeats one configuration for 20 episodes. Tray position and orientation, target orientation, and robot posture vary across seeds. The target position stays at approximately (−0.25, −0.30, 0.05) m. |

**How this was checked.** Episodes were paired by task, file seed, and zero-based episode index, including failures. For Ball Lift, the first pre-action `qpos` supplied the 12 joint angles and the seven-component ball pose at `ball_qpos_idx`. The field `tray_target` supplied the saved goal. For Box Lift, the first `qpos_pre` supplied joints 0–11 and the object pose at indices 12–18. The field `target_0` supplied the goal. For Tray Push, the first pre-action `observations` vector supplied joints 0–11 and the tray pose at indices 12–18 after settling. Its target at indices 46–52 was cross-checked against `tray_target`. Positions are in metres, joint angles in radians, and orientations are quaternions.

The same numbered episode in different tasks represents a different task-specific scene. For Box Lift and Tray Push, each reported 100-episode group therefore represents 20 trials at each of five starting configurations.

## Direct comparison of normalized ARS with every other plotted method

Each cell counts episodes whose initial robot joints, object pose, and target pose matched the normalized ARS episode exactly at the same batch, seed block, and saved episode index. Every 300/300 entry includes 100/100 matches at each of batches 50, 150, and 250. A dash means the method was not plotted for that task.

| Comparison method | Ball Lift | Box Lift | Tray Push |
|---|---:|---:|---:|
| Hand-Tuned | 300/300 | 300/300 | 300/300 |
| Hand-Tuned (ESS 0.01) | 300/300 | — | 300/300 |
| Hand-Tuned (ESS 0.05) | — | 300/300 | 300/300 |
| Bayesian | 300/300 | 300/300 | 300/300 |
| PPO | 300/300 | 300/300 | 300/300 |
| Adaptive ARS | 300/300 | 300/300 | 300/300 |
| ARS phase mean (3950 Ball, 4218 Box, 4124 Tray) | 300/300 | 300/300 | 300/300 |
| Trained constant ARS (Ball ARS3990, Box ARS4292) | 300/300 | 300/300 | — |

All 6,300 direct pairings matched. The recorded planning horizon, CEM iterations, projection iterations, and episode step limit also matched within each task and batch. For Ball Lift the initial table position and generalized velocities matched across methods; for Tray Push the entire first 68-component observation matched. These checks establish comparability of the recorded starting conditions. The methods still differ in their intended controllers and weight or temperature settings.

The [benchmark LaTeX table](benchmark_stat_batch_sm.tex) includes every selected method and lists success rate and task time at each batch size. The normalized ARS row is labeled `Normalized ARS` in each task section.

## Ball Lift trained constant ARS (ARS3990)

The best checkpoint from training iteration 590 stores 18 learned log-weight residuals. All 15 evaluation files identify this checkpoint by SHA-256. Adding the two nine-element residual vectors to the logged baseline log weights and exponentiating reproduced every logged cost weight (relative tolerance 0.000001). The linear weights were identical within each phase across all 300 episodes. The pre-step phase selected the first vector at 56,640 approach steps and the second at 4,965 lift steps. All nine entries differed between the two phases. [ARS3990 checkpoint](ball_lift_phase_3990_eval.oblv3lZt/ars_v2_phase_policy_3990.json)

All 2,100 direct episode pairings against the seven other Ball Lift methods matched exactly on initial joint and ball state, generalized velocities, table position, and target pose at the same batch, seed, and episode index. Each batch used the same 100 distinct starts. All files matched on scene, robot, and object XML hashes, planning horizon (15), CEM iterations (2), projection iterations (5), timestep (0.1 s), solver iterations (1 with 5 line-search iterations), per-step key schedule, and 300-step limit with wall-clock timeout disabled. The new run used fixed inverse temperature 10. Applying the historical lift-phase success and fall thresholds to the logged distances reproduced the recorded success flags, termination reasons, and stopping steps across all eight methods.

| Batch | Successes | Successful task time (simulated s, mean ± SD of file means) |
|---:|---:|---:|
| 50 | 42/100 | 15.58 ± 3.01 |
| 150 | 65/100 | 13.53 ± 1.05 |
| 250 | 69/100 | 13.91 ± 0.88 |

Times follow the notebook rule of excluding episode 0 in each file and equally weighting files with eligible successes. The archive summary pools successful episodes and therefore reports different times. ARS3990 is a separately trained controller. The benchmark supports comparing success and simulated task time under matched evaluation conditions. The new evaluator has a different source hash, and identical execution hardware has not been established for wall-clock computation times. [Archived results](ball_lift_phase_3990_eval.oblv3lZt/comparison_summary.csv)

## Box Lift trained constant ARS (ARS4292)

ARS4292 learns ten cost weights for approach and ten for lift. All 15 evaluation files identify the same iteration-356 checkpoint by SHA-256. Reconstructing its two weight vectors from the stored log-weight residuals and the evaluator’s Bayesian baseline reproduced the recorded weights at all 49,523 approach steps and 20,727 lift steps. Within each phase the recorded vector was identical at every step across all batches. All ten weights differed between phases. The separate ARS4218 phase-mean row uses weights averaged from successful trajectories. [Checkpoint](box_phase4292_kekh_dez/ars_v2_phase_policy_4292.json) · [Phase selection in the actor](box_phase4292_kekh_dez/source/real_demo/real_demo/ARS.py#L425)

At each batch size (50, 150, 250), all 100 episodes matched each of the seven other plotted Box Lift methods on starting robot joints, box pose, and target pose when paired by seed and episode index. All files also agreed on the 15-step horizon, three CEM iterations, five projection iterations, 0.1 s timestep, 300-step limit, CEM key schedule, and success and termination criteria. Successes were 6/100, 24/100, and 22/100 respectively. Each batch covers five starting configurations with 20 trials each.

This supports comparing the evaluated policies on the same simulated Box Lift benchmark. ARS4292 has its own warm-start training run, so its performance difference from adaptive ARS4218 includes both training and policy design. [Run manifest](box_phase4292_kekh_dez/manifest.json)

## Ball Lift and Tray Push normalized ARS controls

Each new archive contains both a normalized run and an unnormalized rerun. For each task, all 300 normalized episodes matched both the historical ARS series and the rerun exactly on initial robot joints, object pose, and target pose when paired by batch, seed block, and episode index. Ball also matched on initial joint/object velocities and table position. Tray matched on the complete first 68-component observation after settling.

The original and new checkpoint files have identical SHA-256 digests within each task (Ball ARS3950 and Tray ARS4124). Across all batches the planning horizon, CEM and projection iteration counts, and step limit match. Ball’s recorded XML hashes and solver settings match the historical run. The new Ball arms share an evaluator hash that differs from the historical evaluator. Tray retains the same reset and target randomization, settling steps, task protocol, and CEM key schedule. The Tray archives do not establish identical historical scene XML or evaluator binaries by hash.

The normalized runs used fixed temperature. At every saved step, applying the recorded phase factors to the linear weights reproduced the reference sum to within one part per million: 132.75736 (approach) and 492.08090 (lift) for Ball, and 778.0 (approach) and 902.5 (push) for Tray.

| Task | Batch | Historical ARS success | Same-run ARS control | Normalized ARS | Change vs same-run control |
|---|---:|---:|---:|---:|---:|
| Ball Lift | 50 | 99% | 97% | 97% | 0 pp |
| Ball Lift | 150 | 99% | 99% | 100% | +1 pp |
| Ball Lift | 250 | 98% | 99% | 99% | 0 pp |
| Tray Push | 50 | 8% | 9% | 8% | −1 pp |
| Tray Push | 150 | 56% | 47% | 46% | −1 pp |
| Tray Push | 250 | 51% | 52% | 58% | +6 pp |

Each percentage covers 100 episodes. Use the same-run controls to assess the effect of normalization. The benchmark figures and exported LaTeX table retain the historical ARS series for continuity. Differences in historical wall-clock timing can also reflect execution hardware and software.

[Ball paired run manifest](ball_lift_normalized_after_tray_3429475/results/run_manifest.txt) · [Tray paired results by seed](eval_sweep_100_tray_push_common_jit_randomized_xy001_yaw2_b50_150_250_normalized_ars_20261123_210616_3429475/paired_success.tsv)

[Notebook and dataset selection](benchmark_stat_batch_sm.ipynb)
