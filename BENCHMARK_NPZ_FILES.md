# NPZ files used by the benchmark

This manifest covers [benchmark_stat_batch.ipynb](benchmark_stat_batch.ipynb) (batches 250, 500, 750, 1000) and [benchmark_stat_batch_sm.ipynb](benchmark_stat_batch_sm.ipynb) (batches 50, 150, 250). Paths are relative to the directory containing the notebooks.

The original large-batch inventory below was verified on 2026-09-24; its expansion notation and exclusions apply to `benchmark_stat_batch.ipynb`. The [small-batch inventory](#small-batches-50-150-and-250) was verified on 2026-09-29.

The large-batch notebook aggregates **240 NPZ files**: 12 task–method combinations × 4 batches × 5 files. Every listed file exists and contains 20 saved episodes. The method order is Hand-Tuned, Bayesian, PPO, ARS.

## Expansion notation

- `{b}` expands to exactly `250`, `500`, `750`, and `1000`.
- A comma-separated set in braces expands to each listed value. For example, `seed{0,10,15,4,5}` denotes five exact filenames, not a wildcard search.
- For Ball timestamped files, the tables list the five exact timestamp suffixes for every batch.
- No unlisted result directory is part of the current plots.

## Ball Lift

### Hand-Tuned

Folder: `eval_sweep_100_ball_lift_h15_20260923_152226/handtuned_batch{b}/`

Filename form: `eval_handtuned_batch{b}_n20_{timestamp}.npz`

| Batch | Exact timestamp suffixes |
|---:|---|
| 250 | `20260923_152708`, `20260923_153141`, `20260923_153607`, `20260923_154018`, `20260923_154420` |
| 500 | `20260923_154935`, `20260923_155526`, `20260923_160030`, `20260923_160552`, `20260923_161113` |
| 750 | `20260923_161706`, `20260923_162323`, `20260923_162943`, `20260923_163535`, `20260923_164156` |
| 1000 | `20260923_164532`, `20260923_164923`, `20260923_165321`, `20260923_165643`, `20260923_170059` |

### Bayesian

Folder: `eval_sweep_100_ball_lift_h15_20260923_152226/bayesian_batch{b}/`

Filename form: `eval_bayesian_batch{b}_n20_{timestamp}.npz`

| Batch | Exact timestamp suffixes |
|---:|---|
| 250 | `20260923_170405`, `20260923_170656`, `20260923_171033`, `20260923_171354`, `20260923_171704` |
| 500 | `20260923_172011`, `20260923_172338`, `20260923_172720`, `20260923_173104`, `20260923_173424` |
| 750 | `20260923_173755`, `20260923_174122`, `20260923_174457`, `20260923_174846`, `20260923_175214` |
| 1000 | `20260923_175637`, `20260923_180049`, `20260923_180458`, `20260923_180832`, `20260923_181230` |

### PPO 4951 — best evaluation checkpoint

Folder form: `ppo_results/Ball_lift/ppo4951_best_eval/batch{b}/block{block}/`

Filename form: `eval_policy_batch{b}_n20_{timestamp}.npz`

Each table entry is `block: timestamp` and therefore identifies one exact path.

| Batch | Exact block/timestamp pairs |
|---:|---|
| 250 | `0: 20260829_130129`, `1: 20260829_130342`, `2: 20260829_130553`, `3: 20260829_130804`, `4: 20260829_131024` |
| 500 | `0: 20260829_131258`, `1: 20260829_131554`, `2: 20260829_131822`, `3: 20260829_132058`, `4: 20260829_132339` |
| 750 | `0: 20260829_132629`, `1: 20260829_132915`, `2: 20260829_133158`, `3: 20260829_133445`, `4: 20260829_133746` |
| 1000 | `0: 20260829_134041`, `1: 20260829_134336`, `2: 20260829_134627`, `3: 20260829_134922`, `4: 20260829_135231` |

The `policy` token in these filenames does not mean ARS; the parent folder selects PPO4951 best-eval.

### ARS

Folder: `eval_sweep_100_ball_lift_h15_20260923_152226/policy_batch{b}/`

Filename form: `eval_policy_batch{b}_n20_{timestamp}.npz`

| Batch | Exact timestamp suffixes |
|---:|---|
| 250 | `20260923_181449`, `20260923_181731`, `20260923_182011`, `20260923_182246`, `20260923_182517` |
| 500 | `20260923_182827`, `20260923_183132`, `20260923_183437`, `20260923_183747`, `20260923_184053` |
| 750 | `20260923_184415`, `20260923_184730`, `20260923_185049`, `20260923_185413`, `20260923_185730` |
| 1000 | `20260923_190110`, `20260923_190428`, `20260923_190740`, `20260923_191058`, `20260923_191411` |

## Box Lift

For every Box row below, `{s}` expands to exactly `0`, `10`, `15`, `4`, and `5`. Each row therefore identifies 5 files per batch and 20 files total.

| Method | Exact folder template | Exact filename template |
|---|---|---|
| Hand-Tuned, matched | `eval_box_lift_matched_20260924_105842/handtuned_batch{b}/` | `eval_handtuned_batch{b}_n20_seed{s}.npz` |
| Bayesian, earlier sweep | `eval_sweep_100_box_lift_3/npz/bayesian_batch{b}/` | `eval_bayesian_batch{b}_n20_seed{s}.npz` |
| PPO 5005, best eval, matched | `eval_box_lift_matched_20260923_205507/npz/ppo5005besteval_batch{b}/` | `eval_ppo5005besteval_batch{b}_n20_seed{s}.npz` |
| ARS 4218, matched | `eval_box_lift_matched_20260923_205507/npz/ars4218_batch{b}/` | `eval_ars4218_batch{b}_n20_seed{s}.npz` |

The notebook does not use the old Box Hand-Tuned/ARS files in `eval_sweep_100_box_lift_3/npz/`, the PPO files in `ppo_results/Box_lift/`, or the matched `ppo5005best` training-checkpoint folders.

## Tray Push

For every Tray row below, `{s}` expands to exactly `114`, `27`, `6`, `77`, and `99`. Each row therefore identifies 5 files per batch and 20 files total.

| Method | Exact folder template | Exact filename template |
|---|---|---|
| Hand-Tuned | `eval_sweep_100_tray_push_8/npz/handtuned_batch{b}/` | `eval_handtuned_batch{b}_n20_seed{s}.npz` |
| Bayesian | `eval_sweep_100_tray_push_8/npz/bayesian_batch{b}/` | `eval_bayesian_batch{b}_n20_seed{s}.npz` |
| PPO 5307, best validation | `ppo_results/Tray_push/ppo5307_best_val_batch{b}/` | `eval_ppo5307_best_val_batch{b}_n20_seed{s}.npz` |
| ARS | `eval_sweep_100_tray_push_8/npz/policy_batch{b}/` | `eval_policy_batch{b}_n20_seed{s}.npz` |

The other `eval_sweep_100_tray_push_*` directories and PPO5306/PPO5307 training-checkpoint variants are not inputs to the current plots.

## Coverage totals

| Task | Hand-Tuned | Bayesian | PPO | ARS | Task total |
|---|---:|---:|---:|---:|---:|
| Ball Lift | 20 | 20 | 20 | 20 | 80 |
| Box Lift | 20 | 20 | 20 | 20 | 80 |
| Tray Push | 20 | 20 | 20 | 20 | 80 |
| **Total** | **60** | **60** | **60** | **60** | **240** |

## Small batches: 50, 150, and 250

Verified: 2026-09-29 against the `SOURCE_PATTERNS` in [benchmark_stat_batch_sm.ipynb](benchmark_stat_batch_sm.ipynb). This notebook aggregates **180 NPZ files**: 3 tasks × 4 methods × 3 batches × 5 files. Every file exists and contains 20 saved episodes, giving 100 episodes per task/method/batch and 3,600 episode records overall.

**Within this section only**, `{b}` expands to exactly `50`, `150`, and `250`, unless a row explicitly restricts the batches. These inputs are separate from the large-batch inventory above; a shared batch size does not imply the same input files.

### Small-batch Ball Lift

For batches **50 and 150**, the common root is:

`eval_sweep_100_ball_lift_h15_20260924_140128/npz/`

Append the folder and filename forms below to that root.

| Method | Folder | Filename form |
|---|---|---|
| Hand-Tuned | `handtuned_batch{b}/` | `eval_handtuned_batch{b}_n20_{timestamp}.npz` |
| Bayesian | `bayesian_batch{b}/` | `eval_bayesian_batch{b}_n20_{timestamp}.npz` |
| PPO | `ppo_batch{b}/` | `eval_policy_batch{b}_n20_{timestamp}.npz` |
| ARS | `policy_batch{b}/` | `eval_policy_batch{b}_n20_{timestamp}.npz` |

| Method | Batch | Exact timestamp suffixes |
|---|---:|---|
| Hand-Tuned | 50 | `20260924_140614`, `20260924_141058`, `20260924_141544`, `20260924_142032`, `20260924_142516` |
| Hand-Tuned | 150 | `20260924_143034`, `20260924_143554`, `20260924_144112`, `20260924_144630`, `20260924_145148` |
| Bayesian | 50 | `20260924_145634`, `20260924_150107`, `20260924_150604`, `20260924_151105`, `20260924_151605` |
| Bayesian | 150 | `20260924_152013`, `20260924_152404`, `20260924_152825`, `20260924_153235`, `20260924_153619` |
| PPO | 50 | `20260924_161001`, `20260924_161300`, `20260924_161605`, `20260924_161906`, `20260924_162208` |
| PPO | 150 | `20260924_162507`, `20260924_162801`, `20260924_163043`, `20260924_163333`, `20260924_163641` |
| ARS | 50 | `20260924_153917`, `20260924_154226`, `20260924_154530`, `20260924_154826`, `20260924_155129` |
| ARS | 150 | `20260924_155426`, `20260924_155728`, `20260924_160034`, `20260924_160343`, `20260924_160645` |

For batch **250**, all four methods reuse the exact batch-250 files listed in the [Ball Lift inventory above](#ball-lift): Hand-Tuned, Bayesian, and ARS use `eval_sweep_100_ball_lift_h15_20260923_152226/`; PPO uses `ppo_results/Ball_lift/ppo4951_best_eval/batch250/block{block}/` and the five block/timestamp pairs listed above. The copied batch-250 files in the September 24 Ball run are excluded.

For batches 50/150, both PPO and ARS filenames contain `policy`; their parent folders (`ppo_batch{b}` versus `policy_batch{b}`) distinguish the methods.

### Small-batch Box Lift

Here `{s}` expands to exactly `0`, `10`, `15`, `4`, and `5`. Each row identifies five files per batch, or 15 files across batches 50/150/250.

| Method | Exact folder template | Exact filename template |
|---|---|---|
| Hand-Tuned | `eval_box_lift_all_methods_b50_150_250_20260924_190610/npz/handtuned_batch{b}/` | `eval_handtuned_batch{b}_n20_seed{s}.npz` |
| Bayesian 5001 | `eval_box_lift_bayesian5001_b50_150_250_20260926_092000/npz/bayesian5001_batch{b}/` | `eval_bayesian5001_batch{b}_n20_seed{s}.npz` |
| PPO 5005, best eval | `eval_box_lift_all_methods_b50_150_250_20260924_190610/npz/ppo5005besteval_batch{b}/` | `eval_ppo5005besteval_batch{b}_n20_seed{s}.npz` |
| ARS 4218 | `eval_box_lift_all_methods_b50_150_250_20260924_190610/npz/ars4218_batch{b}/` | `eval_ars4218_batch{b}_n20_seed{s}.npz` |

Bayesian uses the September 26 run for all three batches. The older `bayesian_batch{b}` directories in the September 24 all-methods run are excluded.

### Small-batch Tray Push

All methods use this exact nested root:

`eval_sweep_100_tray_push_common_jit_randomized_xy001_yaw2_b50_150_250-20260925T081018Z-1-001/eval_sweep_100_tray_push_common_jit_randomized_xy001_yaw2_b50_150_250/npz/`

Append the folder and filename forms below to that root. Here `{s}` expands to exactly `114`, `27`, `6`, `77`, and `99`. Each row identifies five files per batch, or 15 files across batches 50/150/250.

| Method | Folder | Exact filename template |
|---|---|---|
| Hand-Tuned | `handtuned_batch{b}/` | `eval_handtuned_batch{b}_n20_seed{s}.npz` |
| Bayesian | `bayesian_batch{b}/` | `eval_bayesian_batch{b}_n20_seed{s}.npz` |
| PPO 5307, best validation | `ppo5307_best_val_batch{b}/` | `eval_ppo5307_best_val_batch{b}_n20_seed{s}.npz` |
| ARS 4124 | `ars4124_batch{b}/` | `eval_ars4124_batch{b}_n20_seed{s}.npz` |

### Small-batch coverage totals

| Task | Hand-Tuned | Bayesian | PPO | ARS | Task total |
|---|---:|---:|---:|---:|---:|
| Ball Lift | 15 | 15 | 15 | 15 | 60 |
| Box Lift | 15 | 15 | 15 | 15 | 60 |
| Tray Push | 15 | 15 | 15 | 15 | 60 |
| **Total** | **45** | **45** | **45** | **45** | **180** |

These totals count files, not the subset of episodes eligible for each metric. Success rate uses all 100 episodes per group; task time uses successful episodes after excluding episode 0 in each file; computation time uses episodes with valid timings after excluding their first MPC step. Run `./check_benchmark_data.sh` or the final audit cell in the small-batch notebook to check coverage and metric inclusion counts.

See [BENCHMARK_DISCREPANCIES.md](BENCHMARK_DISCREPANCIES.md) for metric definitions, resolved issues, and remaining comparability limitations.
