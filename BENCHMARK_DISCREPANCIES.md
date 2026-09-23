# Benchmark discrepancies and missing data

Verified: 2026-09-23. Scope: the datasets currently used by [benchmark_stat_batch.ipynb](benchmark_stat_batch.ipynb). This report is a snapshot; it does not update automatically when files or notebook settings change.

## Current data coverage

All 12 task–method combinations currently used by the notebook have **five NPZ files per batch**, for batch sizes 250, 500, 750, and 1000. The selected PPO files each contain 20 episodes, giving 100 episodes per task and batch.

- **Resolved:** Ball PPO 4951 best-evaluation results now include blocks 0–4 for every batch. The previously missing six files are present.
- **Verified:** All selected PPO files contain `success`, `total_time`, and `step_time`, with matching lengths of 20.
- **Still missing:** Box PPO has zero-filled per-step timing arrays for 80 of the 100 episodes in each batch. File coverage is complete, but timing coverage is not.

## Summary of discrepancies

| Finding | Status | Consequence |
|---|---|---|
| Ball PPO uses horizon 15; plotted Ball baselines use horizon 10 | Verified configuration mismatch | Computation-time comparisons use different MPC workloads |
| Box PPO has all-zero step timings in 16 of 20 episodes per file | Verified missing measurements; averaging corrected | Computation time is estimated from only four measured episodes per file |
| Box PPO's episodes without step timings have much larger `total_time` | Verified association; cause unresolved | Large Task Time bars cannot be attributed confidently to slower task execution |
| Tray PPO has much lower computation time despite matching recorded core planner settings | Verified timing difference; cause unresolved | Speedup cannot yet be attributed to the PPO algorithm |
| Box and Tray PPO timestep is absent from saved metadata/config fields inspected | Missing metadata | Physical task duration cannot be reconstructed from episode lengths without an independently verified timestep |
| Matched evaluator code, timing boundaries, hardware, and software provenance are not established | Comparison evidence incomplete | Cross-run timing equivalence remains unverified |

## 1. Ball Lift: different planning horizons

| Method currently plotted | Horizon | CEM iterations | Projection iterations |
|---|---:|---:|---:|
| Hand-Tuned | 10 | 2 | 5 |
| Bayesian | 10 | 2 | 5 |
| PPO 4951, best eval | 15 | 2 | 5 |
| ARS | 10 | 2 | 5 |

The PPO planner uses a 50% longer horizon. This is a strong explanation for its higher per-step computation time, but the data do not isolate the causal contribution of every runtime difference.

Additional ARS3982 results in `ppo_results/Ball_lift/ars3982/` use horizon 15, CEM iterations 2, and projection iterations 5. Their computation times are close to PPO's:

| Batch | Plotted ARS, horizon 10 (ms) | ARS3982, horizon 15 (ms) | PPO4951, horizon 15 (ms) |
|---:|---:|---:|---:|
| 250 | 24.07 | 31.41 | 31.81 |
| 500 | 29.48 | 43.20 | 43.94 |
| 750 | 35.37 | 56.19 | 55.03 |
| 1000 | 42.31 | 64.36 | 63.01 |

The ARS3982 comparison is indicative: it has incomplete file coverage and is **not** the ARS dataset plotted in the notebook. A controlled comparison needs all methods evaluated with the same horizon and execution setup.

## 2. Box Lift: missing computation-time measurements

Across the selected PPO5005 best-evaluation files, only zero-based episode indices **0, 5, 10, 15** contain nonzero `step_time` values. The remaining 16 episodes contain zero-filled arrays.

The notebook now excludes the first step of each episode, then excludes zero, negative, and non-finite entries. An episode with no valid timings does not contribute to computation time. No negative or non-finite post-first-step timings were found in the selected PPO datasets during this audit.

| Batch | Previous mean including zero-filled episodes (ms) | Current mean from measured episodes (ms) | Measured episodes / total |
|---:|---:|---:|---:|
| 250 | 11.00 | 55.01 | 20 / 100 |
| 500 | 14.00 | 69.98 | 20 / 100 |
| 750 | 17.74 | 88.69 | 20 / 100 |
| 1000 | 21.59 | 107.93 | 20 / 100 |

**Remaining limitation:** filtering corrects the downward bias caused by averaging missing measurements as zeros. It does not recover the missing timings or prove that the measured episodes represent the unmeasured episodes. The evaluator needs inspection to explain why only every fifth episode has step timings.

## 3. Box Lift: unusually large recorded Task Time

Task Time still uses `total_time` from all successful episodes except episode 0, including episodes whose step timings are zero-filled. No computation-time filter is applied to Task Time.

| Batch | Current plotted Task Time (s) | Successful episodes with step timings: mean (s), count | Successful episodes with zero-filled timings: mean (s), count |
|---:|---:|---|---|
| 250 | 46.54 | 6.75, n=3 | 52.90, n=23 |
| 500 | 52.26 | 6.58, n=7 | 59.22, n=39 |
| 750 | 59.51 | 8.02, n=6 | 69.93, n=39 |
| 1000 | 69.40 | 9.21, n=10 | 85.49, n=40 |

The plotted column uses the notebook's equal-weight mean of per-file means. The two diagnostic subgroup columns pool eligible episodes across files, so their weighted average need not exactly reproduce the plotted column.

At batch 1000, the two groups average approximately **80.9 and 81.25 steps**, respectively. The roughly 76-second duration difference therefore cannot be explained by proportionally longer episode trajectories.

**Inference:** the pattern suggests different evaluation overhead or a timing/logging issue. Compilation, synchronization, or another evaluator operation are possible explanations, not established causes. The evaluator's timer boundaries and execution paths must be inspected before interpreting the large bars as slower control. Stored values have been retained as requested.

## 4. Tray Push: unexplained lower timings

Tray PPO has valid nonzero timings throughout the selected files. Its `step_time` and `step_time_ms` fields agree. The plotted ARS and PPO evaluations both report horizon 10, one CEM iteration, and five projection iterations.

| Batch | ARS computation (ms) | PPO computation (ms) | ARS Task Time (s) | PPO Task Time (s) |
|---:|---:|---:|---:|---:|
| 250 | 41.75 | 11.60 | 9.07 | 3.13 |
| 500 | 45.97 | 15.74 | 10.27 | 3.76 |
| 750 | 51.92 | 21.76 | 9.93 | 4.89 |
| 1000 | 56.27 | 26.12 | 10.42 | 5.75 |

Lower per-step times account for much of the lower wall-clock episode duration. PPO does not consistently finish in fewer steps: at batch 250 the mean of per-file successful-episode step-count means is approximately 219 for PPO and 197 for ARS, excluding episode 0.

**Unresolved:** matching these recorded planner settings does not establish identical evaluator implementations, timing boundaries, devices, or software versions. The current data support a recorded speed difference, not an isolated algorithmic speedup.

## 5. Metric definitions and interpretation

- **Success rate:** all episodes, averaged per file and then across files. With equal file sizes this equals the pooled episode success rate.
- **Task Time:** `total_time` in seconds, restricted to successful episodes and excluding episode 0 from each file. File means are averaged equally. This is the chosen wall-clock metric, not simulated task duration. Its exact timer scope still requires evaluator verification, especially for Box PPO.
- **Ball's explicit `task_time`:** a separate simulated-duration field; `total_time` equals `wall_time` in the inspected Ball PPO data. The notebook intentionally uses `total_time` for consistency with the selected definition.
- **Computation Time:** positive, finite `step_time` values in milliseconds after excluding step 0; averaged within each measured episode, then within each file, then across files. Both successful and failed measured episodes contribute. This is an episode-weighted statistic rather than one pooled average of every recorded step.
- **Error bars:** standard deviation across per-file means, not episode-level standard deviation, standard error, or confidence intervals.
- **Best-file check:** the notebook identifies the highest-success file for verification, but the plotted metrics aggregate all discovered candidate files for the configured method and batch. It does not plot only the best file.
- **Training versus evaluation batch size:** Box and Tray PPO checkpoint configs record batch size 50; runtime results use 250, 500, 750, and 1000. This is an explicit evaluation change, not evidence that the current batch labels are wrong.

## 6. Missing files in alternative Ball datasets

These alternatives are **not used by the current plots**. Expected coverage is blocks 0–4 for each batch.

| Alternative | Batch 250: missing blocks | Batch 500: missing blocks | Batch 750: missing blocks | Batch 1000: missing blocks |
|---|---|---|---|---|
| ARS3950 | 1, 4 | 4 | None | 0, 1, 2, 3 |
| ARS3982 | 0 | None | None | 1, 3, 4 |

Ball PPO4951 `best_training` now also has five files per batch, but the notebook uses the `best_eval` variant. Alternatives should not be silently mixed into the current results to fill gaps.

## Evidence locations and follow-up

| Task | Original-method NPZs and logs | Selected PPO NPZ pattern |
|---|---|---|
| Ball Lift | `eval_sweep_100_ball_lift/{npz,logs}/` | `ppo_results/Ball_lift/ppo4951_best_eval/**/*.npz` |
| Box Lift | `eval_sweep_100_box_lift_3/{npz,logs}/` | `ppo_results/Box_lift/ppo5005besteval_batch*/*.npz` |
| Tray Push | `eval_sweep_100_tray_push_8/{npz,logs}/` | `ppo_results/Tray_push/ppo5307_best_val_batch*/*.npz` |

Recommended investigation order:

1. Inspect the Box PPO evaluator for the every-fifth-episode timing pattern and the operations included in `total_time`.
2. Recover or rerun missing Box step timings with a consistent timer for every episode; keep timing provenance with the results.
3. Evaluate all Ball methods with a common horizon and otherwise matched settings.
4. Compare Tray evaluator code and timing boundaries, including accelerator synchronization, and record hardware/software versions for both runs.
5. Save explicit timing units, timer scope, timestep, runtime planner configuration, and evaluator version in future NPZs. Record additional CEM parameters such as elite selection and smoothing to support complete configuration comparisons.

This report documents findings only. It does not replace datasets, change checkpoint selection, or modify plotted statistics.
