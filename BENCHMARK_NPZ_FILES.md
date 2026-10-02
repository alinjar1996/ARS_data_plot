# NPZ files used by the benchmark

This manifest covers [benchmark_stat_batch.ipynb](benchmark_stat_batch.ipynb) (batches 250, 500, 750, 1000) and [benchmark_stat_batch_sm.ipynb](benchmark_stat_batch_sm.ipynb) (batches 50, 150, 250). Paths are relative to the directory containing the notebooks.

The large-batch inventory below was verified on 2026-09-29; its expansion notation and exclusions apply to `benchmark_stat_batch.ipynb`. The [small-batch inventory](#small-batches-50-150-and-250) was updated on 2026-09-30 for Ball and 2026-10-02 for Box.

The large-batch notebook aggregates **300 NPZ files**: 15 task–series combinations × 4 batches × 5 files. Every listed file exists and contains 20 saved episodes. The series order is Hand-Tuned, Bayesian, PPO (best train), PPO (best val), ARS. Both PPO series are evaluation rollouts; the labels identify how each checkpoint was selected.

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

### PPO 4951 — best training checkpoint

Folder form: `ppo_results/Ball_lift/ppo4951_best_training/batch{b}/block{block}/`

Filename form: `eval_policy_batch{b}_n20_{timestamp}.npz`

Each entry identifies an exact `block: timestamp` pair.

| Batch | Exact block/timestamp pairs |
|---:|---|
| 250 | `0: 20260829_120727`, `1: 20260829_120939`, `2: 20260829_121150`, `3: 20260829_121406`, `4: 20260829_121626` |
| 500 | `0: 20260829_121900`, `1: 20260829_122139`, `2: 20260829_122409`, `3: 20260829_122700`, `4: 20260829_122939` |
| 750 | `0: 20260829_123229`, `1: 20260829_123536`, `2: 20260829_123821`, `3: 20260829_124125`, `4: 20260829_124426` |
| 1000 | `0: 20260829_124721`, `1: 20260829_125021`, `2: 20260829_125312`, `3: 20260829_125607`, `4: 20260829_125918` |

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
| PPO 5005, best train, matched | `eval_box_lift_matched_20260923_205507/npz/ppo5005best_batch{b}/` | `eval_ppo5005best_batch{b}_n20_seed{s}.npz` |
| PPO 5005, best eval, matched | `eval_box_lift_matched_20260923_205507/npz/ppo5005besteval_batch{b}/` | `eval_ppo5005besteval_batch{b}_n20_seed{s}.npz` |
| ARS 4218, matched | `eval_box_lift_matched_20260923_205507/npz/ars4218_batch{b}/` | `eval_ars4218_batch{b}_n20_seed{s}.npz` |

The notebook does not use the old Box Hand-Tuned/ARS files in `eval_sweep_100_box_lift_3/npz/` or the PPO files in `ppo_results/Box_lift/`.

## Tray Push

For every Tray row below, `{s}` expands to exactly `114`, `27`, `6`, `77`, and `99`. Each row therefore identifies 5 files per batch and 20 files total.

| Method | Exact folder template | Exact filename template |
|---|---|---|
| Hand-Tuned | `eval_sweep_100_tray_push_8/npz/handtuned_batch{b}/` | `eval_handtuned_batch{b}_n20_seed{s}.npz` |
| Bayesian | `eval_sweep_100_tray_push_8/npz/bayesian_batch{b}/` | `eval_bayesian_batch{b}_n20_seed{s}.npz` |
| PPO 5307, best train | `ppo_results/Tray_push/ppo5307_best_train_batch{b}/` | `eval_ppo5307_best_train_batch{b}_n20_seed{s}.npz` |
| PPO 5307, best validation | `ppo_results/Tray_push/ppo5307_best_val_batch{b}/` | `eval_ppo5307_best_val_batch{b}_n20_seed{s}.npz` |
| ARS | `eval_sweep_100_tray_push_8/npz/policy_batch{b}/` | `eval_policy_batch{b}_n20_seed{s}.npz` |

The other `eval_sweep_100_tray_push_*` directories and PPO5306 variants are not inputs to the current plots.

## Coverage totals

| Task | Hand-Tuned | Bayesian | PPO (best train) | PPO (best val) | ARS | Task total |
|---|---:|---:|---:|---:|---:|---:|
| Ball Lift | 20 | 20 | 20 | 20 | 20 | 100 |
| Box Lift | 20 | 20 | 20 | 20 | 20 | 100 |
| Tray Push | 20 | 20 | 20 | 20 | 20 | 100 |
| **Total** | **60** | **60** | **60** | **60** | **60** | **300** |

## Small batches: 50, 150, and 250

Verified: 2026-09-30 against the `SOURCE_PATTERNS` in [benchmark_stat_batch_sm.ipynb](benchmark_stat_batch_sm.ipynb). This notebook aggregates **180 NPZ files**: 3 tasks × 4 methods × 3 batches × 5 files. Every file exists and contains 20 saved episodes, giving 100 episodes per task/method/batch and 3,600 episode records overall.

**Within this section only**, `{b}` expands to exactly `50`, `150`, and `250`, unless a row explicitly restricts the batches. These inputs are separate from the large-batch inventory above; a shared batch size does not imply the same input files.

### Small-batch Ball Lift

Updated 2026-09-30. All four plotted methods at batches **50, 150, and 250** use this root:

`eval_sweep_100_ball_lift_bayes_retrained_20260929_191418/npz/`

PPO uses **PPO4951 best-training**, not best-evaluation; ARS uses **ARS3950**. The checkpoint label refers to selection during training; these NPZs contain evaluation episodes. Each folder contains five complete 20-episode files, with evaluation seeds 0–4 (100 episodes per method/batch).

| Method | Folder | Filename form |
|---|---|---|
| Hand-Tuned | `handtuned_batch{b}/` | `eval_handtuned_batch{b}_n20_{suffix}.npz` |
| Bayesian, retrained | `bayesian_batch{b}/` | `eval_bayesian_batch{b}_n20_{suffix}.npz` |
| PPO 4951, best train | `ppo_best_train_batch{b}/` | `eval_policy_batch{b}_n20_{suffix}.npz` |
| ARS 3950 | `ars_batch{b}/` | `eval_policy_batch{b}_n20_{suffix}.npz` |

The suffix includes the timestamp, evaluation seed, and run-ID token; exact values are listed below.

| Method | Batch | Exact suffixes |
|---|---:|---|
| Hand-Tuned | 50 | `20260929_224013_412335_seed0_90a30b0a`, `20260929_224511_755624_seed1_1331d858`, `20260929_225010_508194_seed2_98a26bb7`, `20260929_225508_915066_seed3_a539ebc8`, `20260929_230006_477525_seed4_d86bb799` |
| Hand-Tuned | 150 | `20260930_001141_253541_seed0_494e321f`, `20260930_001752_314444_seed1_c4558fca`, `20260930_002403_505661_seed2_af06806f`, `20260930_003000_205306_seed3_45bab5e8`, `20260930_003613_187679_seed4_423ea995` |
| Hand-Tuned | 250 | `20260930_015308_673060_seed0_8eec0ac1`, `20260930_015958_978577_seed1_3f5b23a3`, `20260930_020612_186663_seed2_8ef0adfe`, `20260930_021157_839661_seed3_873cf105`, `20260930_021844_459956_seed4_c2118121` |
| Bayesian | 50 | `20260929_230505_015635_seed0_d565dfb8`, `20260929_230926_631522_seed1_703958be`, `20260929_231408_167581_seed2_195f04cf`, `20260929_231848_595373_seed3_cca06b1b`, `20260929_232312_274243_seed4_7c77ce1a` |
| Bayesian | 150 | `20260930_004224_739541_seed0_22fee135`, `20260930_004730_422818_seed1_471adb6f`, `20260930_005249_485300_seed2_7972d6c3`, `20260930_005807_884052_seed3_ba1147f5`, `20260930_010321_611214_seed4_0292d316` |
| Bayesian | 250 | `20260930_022518_250416_seed0_09dea838`, `20260930_023104_563535_seed1_cbd507be`, `20260930_023639_007722_seed2_3b17d46f`, `20260930_024254_681897_seed3_aa22de3d`, `20260930_024834_621822_seed4_b5b14335` |
| PPO | 50 | `20260929_232752_808912_seed0_dea5a62d`, `20260929_233052_253134_seed1_02044a88`, `20260929_233351_444645_seed2_2dec9043`, `20260929_233638_474139_seed3_7d3589cd`, `20260929_233945_708918_seed4_ea4c7423` |
| PPO | 150 | `20260930_010813_934842_seed0_d26ba4ba`, `20260930_011130_186804_seed1_b2a0d17f`, `20260930_011432_894785_seed2_9ea1f11e`, `20260930_011725_312619_seed3_a61c7dfe`, `20260930_012033_796849_seed4_842fb754` |
| PPO | 250 | `20260930_025351_515655_seed0_706739ef`, `20260930_025655_409036_seed1_24c1ea9e`, `20260930_025957_684992_seed2_a96a4bdc`, `20260930_030255_568467_seed3_d69e078b`, `20260930_030604_743893_seed4_8147efb2` |
| ARS | 50 | `20260929_235740_817737_seed0_8bf80c26`, `20260930_000028_387205_seed1_87978dec`, `20260930_000312_374509_seed2_321d1cf3`, `20260930_000556_801099_seed3_6cbfbdc3`, `20260930_000842_435392_seed4_b4d69b59` |
| ARS | 150 | `20260930_013815_682390_seed0_888ebc10`, `20260930_014115_594261_seed1_4dad6a72`, `20260930_014417_361709_seed2_dfae30b1`, `20260930_014716_517551_seed3_049301cc`, `20260930_015010_228176_seed4_528d320d` |
| ARS | 250 | `20260930_032423_710943_seed0_b5030ffc`, `20260930_032746_122653_seed1_a239d777`, `20260930_033054_201306_seed2_a0d54d3e`, `20260930_033407_668705_seed3_9d42a3db`, `20260930_033717_903772_seed4_18d0d63b` |

The earlier September 23/24 Ball sources and this run’s `ppo_best_eval_batch{b}` files are excluded from the small-batch plots. Large-batch sources are unchanged; batch 250 no longer shares Ball input files with the large-batch notebook. Task Time retains the notebook’s wall-clock `total_time` definition and filtering, not simulated `task_time` or the run’s CSV-summary aggregation.

### Small-batch Box Lift

Here `{s}` expands to exactly `0`, `10`, `15`, `4`, and `5`. Each row identifies five files per batch, or 15 files across batches 50/150/250.

| Method | Exact folder template | Exact filename template |
|---|---|---|
| Hand-Tuned | `eval_box_lift_all_contacts_b50_150_250_20261001_154416_3370028/npz/handtuned_batch{b}/` | `eval_handtuned_batch{b}_n20_seed{s}.npz` |
| Bayesian 5001 | `eval_box_lift_bayesian5001_b50_150_250_after_20261001_154416/npz/bayesian5001_batch{b}/` | `eval_bayesian5001_batch{b}_n20_seed{s}.npz` |
| PPO 5005, best eval | `eval_box_lift_all_contacts_b50_150_250_20261001_154416_3370028/npz/ppo5005besteval_batch{b}/` | `eval_ppo5005besteval_batch{b}_n20_seed{s}.npz` |
| ARS 4218 | `eval_box_lift_all_contacts_b50_150_250_20261001_154416_3370028/npz/ars4218_batch{b}/` | `eval_ars4218_batch{b}_n20_seed{s}.npz` |

Updated 2026-10-02. Hand-Tuned, PPO5005 best-evaluation, and ARS4218 use the October 1 all-contacts run. Bayesian uses the separate Bayesian5001 rerun for all three batches. The all-contacts run's `bayesian5002_batch{b}` and `ppo5005best_batch{b}` directories are excluded, as are the previous September Box inputs. Only the aggregate `npz/` files are selected; individual `episodes/` files are not pooled in. All 60 selected Box files contain 20 episodes each. Box penetration findings/snapshots were refreshed for these inputs on 2026-10-02, including terminal states. Previous evidence is preserved in the ignored archive.

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
