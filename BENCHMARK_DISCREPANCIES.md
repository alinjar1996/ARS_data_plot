# Benchmark discrepancies and data limitations

Verified: 2026-09-24. Scope: the datasets currently discovered and aggregated by [benchmark_stat_batch.ipynb](benchmark_stat_batch.ipynb). This is a snapshot; it does not update automatically when files or notebook settings change.

## Current data coverage

The notebook currently aggregates **240 NPZ files**: 3 tasks × 4 methods × 4 batch sizes × 5 files. Every file contains 20 saved episodes, so each task–method–batch result is based on 100 saved episodes before metric-specific filtering.

- All 12 task–method combinations have five files for each batch size: 250, 500, 750, and 1000.
- Every selected file contains `success`, `total_time`, and `step_time` with 20 episode entries.
- Every selected episode has at least one positive finite `step_time` entry. The previously selected Box PPO data with zero-filled timings are no longer plotted.
- The exact active sources and filenames are listed in [BENCHMARK_NPZ_FILES.md](BENCHMARK_NPZ_FILES.md).

## Summary

| Finding | Current status | Consequence |
|---|---|---|
| Ball methods previously used different horizons | Resolved in the active inputs | All four Ball methods now use horizon 15, 2 CEM iterations, and 5 projection iterations |
| Box PPO previously had zero-filled timing arrays in 80% of episodes | Resolved in the active inputs | The matched Box Hand-Tuned, PPO, and ARS files have valid per-step timings for every saved episode |
| Box PPO previously had anomalously large `total_time` values | Resolved by replacing the active source | Current Box PPO Task Time is 6.88–16.32 s rather than 46.54–69.40 s |
| Box Hand-Tuned, PPO, and ARS render and encode a video for every saved episode | Verified current protocol | Their `total_time` includes rendering and MP4 encoding and is not pure controller execution time |
| Box Bayesian still comes from the earlier sweep | Source/protocol mismatch | Its timing should not be treated as fully matched to the other three Box methods |
| Matched Box files use `seed_stride=0` | Verified current protocol | Each 20-episode file records one repeated episode seed; there are five recorded seed values per method/batch, not 100 distinct seed values |
| Tray PPO remains much faster despite matching recorded core planner settings | Verified timing difference; cause unresolved | The speed difference cannot be attributed to PPO alone |
| Tray PPO timestep is absent from the inspected metadata/config fields | Missing metadata | Simulated task duration cannot be reconstructed from its episode lengths without an independently verified timestep |
| Hardware, software, and evaluator provenance are not uniform across all active sources | Comparison evidence incomplete | Cross-source wall-clock comparisons remain conditional |

## 1. Ball Lift: horizon mismatch resolved

The notebook now uses the September 23 horizon-15 sweep for Hand-Tuned, Bayesian, and ARS. PPO4951 best-evaluation already uses horizon 15.

| Active method | Horizon | CEM iterations | Projection iterations | Timestep (s) |
|---|---:|---:|---:|---:|
| Hand-Tuned | 15 | 2 | 5 | 0.1 |
| Bayesian | 15 | 2 | 5 | 0.1 |
| PPO 4951, best eval | 15 | 2 | 5 | 0.1 |
| ARS | 15 | 2 | 5 | 0.1 |

All 80 active Ball NPZs contain these settings in metadata. The older horizon-10 files under `eval_sweep_100_ball_lift/npz/` and the ARS3950/ARS3982 diagnostic directories are not plotted.

The resulting computation-time means are now similar within each batch:

| Batch | Hand-Tuned (ms) | Bayesian (ms) | PPO (ms) | ARS (ms) |
|---:|---:|---:|---:|---:|
| 250 | 32.26 | 31.14 | 31.81 | 32.89 |
| 500 | 43.61 | 43.07 | 43.94 | 46.19 |
| 750 | 54.03 | 54.03 | 55.03 | 57.52 |
| 1000 | 61.74 | 63.61 | 63.01 | 63.83 |

Matching these recorded settings removes the known horizon confound, but does not by itself establish identical evaluator code, devices, or software versions.

## 2. Box Lift: selected sources and timing protocol

The active Box sources are:

| Method | Active source | Protocol status |
|---|---|---|
| Hand-Tuned | `eval_box_lift_matched_20260924_105842/handtuned_batch*/*.npz` | Matched evaluator metadata |
| Bayesian | `eval_sweep_100_box_lift_3/npz/bayesian_batch*/*.npz` | Earlier sweep |
| PPO 5005, best eval | `eval_box_lift_matched_20260923_205507/npz/ppo5005besteval_batch*/*.npz` | Matched evaluator |
| ARS 4218 | `eval_box_lift_matched_20260923_205507/npz/ars4218_batch*/*.npz` | Matched evaluator |

All four methods use horizon 15, 3 CEM iterations, 5 projection iterations, and a 0.1 s timestep according to NPZ metadata and, for the earlier Bayesian files, their evaluation logs.

The matched evaluator fixes the former sparse-timing problem: every selected Hand-Tuned, PPO, and ARS episode contains positive finite per-step timings. Their current computation-time results track closely, while the earlier Bayesian run is consistently lower:

| Batch | Hand-Tuned (ms) | Bayesian, earlier sweep (ms) | PPO (ms) | ARS (ms) |
|---:|---:|---:|---:|---:|
| 250 | 67.29 | 59.34 | 66.67 | 67.87 |
| 500 | 85.61 | 70.81 | 84.14 | 84.83 |
| 750 | 117.65 | 87.00 | 114.71 | 114.77 |
| 1000 | 149.87 | 108.63 | 147.86 | 147.92 |

### What the matched timing fields include

The selected NPZ metadata and the evaluator/launcher copied with the September 23 matched run establish the following:

- `step_time` is milliseconds around observation, policy, CEM/environment step, and device synchronization. Rendering occurs after this timer, so it is excluded from `step_time`.
- `total_time` is seconds around the complete `run_episode` call.
- Every selected matched file records `execution_mode='render'` for all 20 episodes. The copied PPO/ARS launcher passes `--video --video_stride 1`; the separate Hand-Tuned directory records the same execution mode and warm-up metadata but does not contain its launcher.
- The render path creates frames and writes an MP4 with `imageio.mimsave` before returning. Consequently, matched `total_time` includes rendering and video encoding.
- One headless timed warm-up episode is run before the 20 saved episodes in each process and is recorded as `warmup_episodes=1`.

The notebook nevertheless excludes saved episode 0 from Task Time for every dataset. For the matched Box runs this drops an additional measured episode after the evaluator's unsaved warm-up. This is the implemented statistic, but it is not required for compilation warm-up in those runs.

Because Box Bayesian remains from the earlier sweep, the four Box Task Time bars do not share a fully established timer/rendering protocol. The matched Hand-Tuned, PPO, and ARS bars are mutually more comparable, but should be described as end-to-end rendered evaluation time rather than pure task execution time.

### Episode seeds

The matched launcher uses base seeds `0`, `5`, `4`, `10`, and `15`, with `seed_stride=0`. Each file therefore stores the same `episode_seed` for all 20 episodes. The notebook correctly reports 100 saved episode outcomes per method/batch, but those are not 100 distinct recorded seeds. Any uncertainty analysis should account for the five file-level seed groups and possible within-process dependence.

## 3. Tray Push: lower PPO timings remain unexplained

Tray PPO has valid nonzero timings throughout its selected files. Its `step_time` and `step_time_ms` fields agree. All four methods report horizon 10, one CEM iteration, and five projection iterations.

| Batch | ARS computation (ms) | PPO computation (ms) | ARS Task Time (s) | PPO Task Time (s) |
|---:|---:|---:|---:|---:|
| 250 | 41.75 | 11.60 | 9.07 | 3.13 |
| 500 | 45.97 | 15.74 | 10.27 | 3.76 |
| 750 | 51.92 | 21.76 | 9.93 | 4.89 |
| 1000 | 56.27 | 26.12 | 10.42 | 5.75 |

Matching the recorded core planner settings does not establish identical evaluator implementations, timing boundaries, devices, or software versions. The data support a recorded speed difference, not an isolated PPO speedup. Tray PPO's timestep is also not stored in the inspected NPZ metadata/config fields.

## 4. Implemented metric definitions

- **Success rate:** mean over all 20 episodes in each file, then mean over the five files. Because file sizes are equal, this also equals the pooled rate over 100 saved episodes.
- **Task Time:** `total_time` in seconds, restricted to successful saved episodes after removing episode 0 from each file. The notebook averages eligible episodes within each file and then gives each finite file mean equal weight.
- **Computation Time:** positive finite `step_time` values after removing step 0 from each episode. It averages steps within each measured episode, then episodes within each file, then the five file means. Successful and failed episodes both contribute.
- **Error bars:** population standard deviation (`numpy.nanstd`, default `ddof=0`) across the five per-file means. They are not episode-level standard deviations, standard errors, or confidence intervals.
- **Best-file check:** the notebook separately identifies the highest-success candidate for verification, but plots aggregate every candidate discovered for the method and batch. It does not plot only the best file.
- **Overview plots:** use only batches 250 and 500; the per-task plots use all four batches.
- **Training versus evaluation batch size:** Box and Tray PPO checkpoint configs record training batch size 50, while the plotted evaluation batch sizes are 250, 500, 750, and 1000.

## 5. Active NPZ folders

`{b}` is one of 250, 500, 750, or 1000. Each row contributes five files per batch and 20 files total.

| Task | Method | Active folder/pattern |
|---|---|---|
| Ball Lift | Hand-Tuned | `eval_sweep_100_ball_lift_h15_20260923_152226/handtuned_batch{b}/*.npz` |
| Ball Lift | Bayesian | `eval_sweep_100_ball_lift_h15_20260923_152226/bayesian_batch{b}/*.npz` |
| Ball Lift | PPO 4951, best eval | `ppo_results/Ball_lift/ppo4951_best_eval/batch{b}/block*/*.npz` |
| Ball Lift | ARS | `eval_sweep_100_ball_lift_h15_20260923_152226/policy_batch{b}/*.npz` |
| Box Lift | Hand-Tuned, matched | `eval_box_lift_matched_20260924_105842/handtuned_batch{b}/*.npz` |
| Box Lift | Bayesian | `eval_sweep_100_box_lift_3/npz/bayesian_batch{b}/*.npz` |
| Box Lift | PPO 5005 best eval, matched | `eval_box_lift_matched_20260923_205507/npz/ppo5005besteval_batch{b}/*.npz` |
| Box Lift | ARS 4218, matched | `eval_box_lift_matched_20260923_205507/npz/ars4218_batch{b}/*.npz` |
| Tray Push | Hand-Tuned | `eval_sweep_100_tray_push_8/npz/handtuned_batch{b}/*.npz` |
| Tray Push | Bayesian | `eval_sweep_100_tray_push_8/npz/bayesian_batch{b}/*.npz` |
| Tray Push | PPO 5307, best val | `ppo_results/Tray_push/ppo5307_best_val_batch{b}/*.npz` |
| Tray Push | ARS | `eval_sweep_100_tray_push_8/npz/policy_batch{b}/*.npz` |

The old Ball horizon-10 inputs, old Box Hand-Tuned/ARS inputs, Box PPO files under `ppo_results/Box_lift/`, PPO5005 best-training files, and other result directories are not silently pooled into the active plots.

## Follow-up priorities

1. Rerun or replace Box Bayesian using the matched evaluator and record the same timer/rendering protocol as the other Box methods.
2. For wall-clock Task Time comparisons, evaluate without video or store controller time separately from rendering and encoding time.
3. Use distinct episode seeds (`seed_stride > 0`) or explicitly model repeated-seed dependence.
4. Compare Tray evaluator code and timing boundaries, including accelerator synchronization, and record hardware/software versions.
5. Save timing units, timer scope, timestep, complete runtime planner configuration, evaluator version, and hardware/software provenance in every future NPZ.

This report documents the current implementation and data limitations. It does not replace datasets, alter checkpoint selection, or change plotted statistics.
