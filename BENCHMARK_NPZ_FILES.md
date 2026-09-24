# NPZ files used by the benchmark

Verified: 2026-09-24. This manifest is the exact expansion of the discovery patterns in [benchmark_stat_batch.ipynb](benchmark_stat_batch.ipynb). Paths are relative to the directory containing the notebook.

The notebook aggregates **240 NPZ files**: 12 task–method combinations × 4 batches × 5 files. Every listed file exists and contains 20 saved episodes. The method order is Hand-Tuned, Bayesian, PPO, ARS.

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

See [BENCHMARK_DISCREPANCIES.md](BENCHMARK_DISCREPANCIES.md) for metric definitions, resolved issues, and remaining comparability limitations.
