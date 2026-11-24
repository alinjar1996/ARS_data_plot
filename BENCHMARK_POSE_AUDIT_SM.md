# Initial object pose, target pose, and robot posture

**Starting conditions matched within each task.** For the same seed block and saved episode index, every plotted method at batches 50, 150, and 250 had the same initial object position and orientation, target position and orientation, and 12 starting robot joint angles. The comparison covered all 345 NPZ files selected on 2026-11-24, including the ESS, ARS phase-mean, and normalized ARS runs. All 6,900 episode records matched their task’s Hand-Tuned batch-50 reference exactly after treating opposite quaternion signs as the same orientation.

| Task | Records checked | Distinct starting configurations per 100 episodes | What varies between configurations |
|---|---:|---:|---|
| Ball Lift | 2,100 | 100 | Each of five seed blocks contains 20 distinct configurations. Ball position, target position, and robot posture vary. The recorded ball and target orientations are identity throughout. |
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
| ARS4292 (trained phase) | — | 300/300 | — |

All 6,000 direct pairings matched. The recorded planning horizon, CEM iterations, projection iterations, and episode step limit also matched within each task and batch. For Ball Lift the initial table position and generalized velocities matched across methods; for Tray Push the entire first 68-component observation matched. These checks establish comparability of the recorded starting conditions. The methods still differ in their intended controllers and weight or temperature settings.

The [benchmark LaTeX table](benchmark_stat_batch_sm.tex) includes every selected method and lists success rate and task time at each batch size. The normalized ARS row is labeled `ARS norm abalation` in each task section.

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
