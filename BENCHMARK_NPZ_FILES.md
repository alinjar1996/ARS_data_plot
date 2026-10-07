# NPZ files used by the benchmark

This manifest covers [benchmark_stat_batch.ipynb](benchmark_stat_batch.ipynb) (batches 250, 500, 750, 1000) and [benchmark_stat_batch_sm.ipynb](benchmark_stat_batch_sm.ipynb) (batches 50, 150, 250). Paths are relative to the directory containing the notebooks.

For the verified method identities, exact checkpoint files, hashes, and generation-time Git references, see the [checkpoint identity audit](#checkpoint-identity-audit-2026-10-06). That audit covers all 480 currently selected files, not unused/archived sweeps.

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

Checkpoint selection and filenames verified 2026-10-02. All four plotted methods at batches **50, 150, and 250** use this root:

`eval_sweep_100_ball_lift_bayes_retrained_20260929_191418/npz/`

PPO uses **PPO4951 best-evaluation**, not best-training; ARS uses **ARS3950**. The checkpoint label refers to selection during training; these NPZs contain evaluation episodes. Each folder contains five complete 20-episode files, with evaluation seeds 0–4 (100 episodes per method/batch).

| Method | Folder | Filename form |
|---|---|---|
| Hand-Tuned | `handtuned_batch{b}/` | `eval_handtuned_batch{b}_n20_{suffix}.npz` |
| Bayesian, retrained | `bayesian_batch{b}/` | `eval_bayesian_batch{b}_n20_{suffix}.npz` |
| PPO 4951, best eval | `ppo_best_eval_batch{b}/` | `eval_policy_batch{b}_n20_{suffix}.npz` |
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
| PPO | 50 | `20260929_234246_335191_seed0_4244b40c`, `20260929_234543_590106_seed1_d1a58a93`, `20260929_234843_363632_seed2_fbf10b60`, `20260929_235143_747242_seed3_35347958`, `20260929_235425_353375_seed4_2afaca2e` |
| PPO | 150 | `20260930_012339_662020_seed0_7c2f75b7`, `20260930_012642_724376_seed1_74484f99`, `20260930_012931_337089_seed2_1f03599c`, `20260930_013222_613095_seed3_41968dd7`, `20260930_013514_059282_seed4_7395bd9c` |
| PPO | 250 | `20260930_030913_622278_seed0_fc9c5a81`, `20260930_031215_042548_seed1_d378096c`, `20260930_031510_878003_seed2_2dbb482a`, `20260930_031808_450687_seed3_2161eb95`, `20260930_032110_888415_seed4_cc5fcfb1` |
| ARS | 50 | `20260929_235740_817737_seed0_8bf80c26`, `20260930_000028_387205_seed1_87978dec`, `20260930_000312_374509_seed2_321d1cf3`, `20260930_000556_801099_seed3_6cbfbdc3`, `20260930_000842_435392_seed4_b4d69b59` |
| ARS | 150 | `20260930_013815_682390_seed0_888ebc10`, `20260930_014115_594261_seed1_4dad6a72`, `20260930_014417_361709_seed2_dfae30b1`, `20260930_014716_517551_seed3_049301cc`, `20260930_015010_228176_seed4_528d320d` |
| ARS | 250 | `20260930_032423_710943_seed0_b5030ffc`, `20260930_032746_122653_seed1_a239d777`, `20260930_033054_201306_seed2_a0d54d3e`, `20260930_033407_668705_seed3_9d42a3db`, `20260930_033717_903772_seed4_18d0d63b` |

The earlier September 23/24 Ball sources and this run’s `ppo_best_train_batch{b}` files are excluded from the small-batch plots. Large-batch sources are unchanged; batch 250 no longer shares Ball input files with the large-batch notebook. Task Time retains the notebook’s wall-clock `total_time` definition and filtering, not simulated `task_time` or the run’s CSV-summary aggregation.

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

## Checkpoint identity audit 2026-10-06

### Scope and conclusion

Read-only data/code audit of the actual candidate lists in both notebooks: **300 large-batch + 180 small-batch = 480 distinct NPZ files, 9,600 recorded episodes, 27 task/method series**. Every selected file has 20 episodes. All candidates were checked, not just the representative highest-SR file that the notebooks retain for diagnostics. Unselected historical runs, individual-episode files, and newer simulator evaluations are outside this inventory.

**No ARS/PPO/Hand-Tuned/Bayesian method swap or PPO best-training/best-evaluation swap was found in the current selections.** Evidence includes metadata from every selected NPZ, original/frozen checkpoints, generation logs, source/checkpoint SHA256 values where available, Git training and evaluation history, fixed-weight traces, and independent actor-output checks. Older files have weaker provenance, identified below; this is not a claim that every run has an immutable launch manifest.

Only this documentation was updated. No result, checkpoint, notebook selector, figure, or simulator code was changed. Correct identity does not resolve the separate physical-validity and comparison concerns in [concerns.md](concerns.md).

### Checkpoints actually used

The source IDs link to the exact retrievable artifact list below. “Large” means batches 250/500/750/1000; “small” means 50/150/250. Each large row covers 20 files and each small row 15 files.

| Task | Plotted method | Large-batch checkpoint / weight source | Small-batch checkpoint / weight source |
|---|---|---|---|
| Ball Lift | Hand-Tuned | Fixed hand-tuned constants; no learned checkpoint | Same fixed hand-tuned constants; no learned checkpoint |
| Ball Lift | Bayesian | **2100-derived hardcoded constants**, not a JSON loaded at runtime; see transcription caveat below (B0) | Retrained `bayesian_ball_lift_best.json`, frozen as `checkpoints/bayesian_weights.json` (B1) |
| Ball Lift | ARS | **ARS3950**, `ars_v2_linear_policy_3950.json` (B2) | Same exact actor bytes, frozen as `checkpoints/ars3950.json` (B2) |
| Ball Lift | PPO best train | **PPO4951**, `ppo4951_best_training.json`; `checkpoint_type=best`, `updates_completed=1325` (B3) | Not selected; a best-training copy exists but is excluded |
| Ball Lift | PPO best eval/val | **PPO4951**, `ppo4951_best_eval.json`; `checkpoint_type=best_eval`, update **1410** (B4) | Same exact actor bytes, `checkpoints/ppo4951_best_eval.json` (B4) |
| Box Lift | Hand-Tuned | Fixed hand-tuned constants; no learned checkpoint | Same fixed hand-tuned constants; no learned checkpoint |
| Box Lift | Bayesian | **Bayesian3601** constants, plus fixed smoothness 0.01; no runtime JSON path recorded (X0) | **Bayesian5001**, `bayesian_5001.json`, fixed smoothness 0.01 (X1); **not 5002** |
| Box Lift | ARS | **ARS4218**, `ars_v2_linear_policy_4218.json` (X2) | Same exact actor bytes (X2) |
| Box Lift | PPO best train | **PPO5005**, `real_demo/real_demo/ppo_linear_policy_5005.json`; `checkpoint_type=best`, `updates_completed=2920` (X3) | Not selected; the run's `ppo5005best_batch*` files are excluded |
| Box Lift | PPO best eval/val | **PPO5005**, `real_demo/real_demo/ppo_linear_policy_5005_best_eval.json`; update **2900** (X4) | Same exact actor bytes (X4) |
| Tray Push | Hand-Tuned | Fixed hand-tuned constants; no learned checkpoint | Zero-output `handtuned_common_jit.json` adapter implementing the same constants (T0) |
| Tray Push | Bayesian | **Bayesian4104** constants from `cost_weights_optim_4104.json` (T1) | Same weights in zero-output `bayesian4104_common_jit.json` adapter (T1) |
| Tray Push | ARS | **ARS4124**, nested `real_demo/real_demo/ars_v2_mlp_policy_4124.json`, iteration **806** (T2) | Same parameters and normalization copied into `ars4124_common_jit.json` (T2) |
| Tray Push | PPO best train | **PPO5307**, `ppo_mlp_policy_5307.json`; selected training rollout **2323** (T3) | Not selected |
| Tray Push | PPO best eval/val | **PPO5307**, `ppo_mlp_policy_5307_best_eval.json`; update **2180** (T4) | Same actor, verified against recorded outputs (T4) |

PPO “best train” and “best eval” identify checkpoint selection, not whether the plotted episodes are training or evaluation rollouts: all plotted files are evaluations. Do not infer checkpoint type from a `best_eval_update` field alone, because a best-training checkpoint can also retain that historical statistic. Tray's best-training file records `updates_completed=2322` alongside selected rollout/update 2323; those counters have different save-time semantics and do not make it the best-evaluation actor.

### Exact artifact locations and Git recovery

Git references below belong to `/home/aks-lab/colcon_ws/src/manipulator_mujoco`, not this results repository. They are historical blob references and do not require checking out or changing the active branch. For example, inspect B2 with:

```bash
git -C /home/aks-lab/colcon_ws/src/manipulator_mujoco show 255a412:ars_v2_linear_policy_3950.json
```

| ID | Exact artifact / historical reference |
|---|---|
| B0 | `119d1d0:real_demo/real_demo/ARS.py`, `DEFAULT_LOG_WEIGHTS`; derived from `119d1d0:real_demo/sampling_based_planner/cost_weights_optimized/cost_weights_optim_2100.json`, with the collision-value difference below |
| B1 | `f4202cc:real_demo/real_demo/bayesian_ball_lift_matched_20260929_191418_252222/bayesian_ball_lift_best.json`; identical local [frozen Bayesian weights](eval_sweep_100_ball_lift_bayes_retrained_20260929_191418/checkpoints/bayesian_weights.json) |
| B2 | `255a412:ars_v2_linear_policy_3950.json`; identical local [frozen ARS3950](eval_sweep_100_ball_lift_bayes_retrained_20260929_191418/checkpoints/ars3950.json) |
| B3 | `d260301:real_demo/real_demo/eval_ars3950_3982_vs_ppo4951_ball_lift_100_20260829_101625/checkpoints/ppo4951_best_training.json`; identical local [best-training copy](eval_sweep_100_ball_lift_bayes_retrained_20260929_191418/checkpoints/ppo4951_best_train.json) |
| B4 | `d260301:real_demo/real_demo/eval_ars3950_3982_vs_ppo4951_ball_lift_100_20260829_101625/checkpoints/ppo4951_best_eval.json`; identical local [best-evaluation copy](eval_sweep_100_ball_lift_bayes_retrained_20260929_191418/checkpoints/ppo4951_best_eval.json) |
| X0 | `79fa0aa:real_demo/sampling_based_planner/cost_weights_optimized/cost_weights_optim_3601.json`; evaluator constants add smoothness 0.01 |
| X1 | `504aa80:real_demo/real_demo/bayesian_box_lift_matched_20260925_183328_398207/bayesian_5001.json`; a byte-identical local [saved copy](eval_box_lift_bayesian5001_b50_150_250_20260926_092000/bayesian_5001.json) exists in the older run, but the selected episode data are the October rerun |
| X2 | `2d9ee336:ars_v2_linear_policy_4218.json`; unchanged in `79fa0aa` and `00898ab` |
| X3 | `79fa0aa:real_demo/real_demo/ppo_linear_policy_5005.json` |
| X4 | `79fa0aa:real_demo/real_demo/ppo_linear_policy_5005_best_eval.json` |
| T0 | `<TRAY_ROOT>/checkpoints/handtuned_common_jit.json` |
| T1 | `e709694:real_demo/sampling_based_planner/cost_weights_optimized/cost_weights_optim_4104.json`; small-run adapter `<TRAY_ROOT>/checkpoints/bayesian4104_common_jit.json` |
| T2 | `587a453:real_demo/real_demo/ars_v2_mlp_policy_4124.json`; small-run adapter `<TRAY_ROOT>/checkpoints/ars4124_common_jit.json` |
| T3 | `6117e21:ppo_mlp_policy_5307.json` |
| T4 | `6117e21:ppo_mlp_policy_5307_best_eval.json` |

Here `<TRAY_ROOT>` is the existing local directory:

`eval_sweep_100_tray_push_common_jit_randomized_xy001_yaw2_b50_150_250-20260925T081018Z-1-001/eval_sweep_100_tray_push_common_jit_randomized_xy001_yaw2_b50_150_250`

Hand-tuned vectors, in each task's saved `cost_weight_keys` / `cost_weights_keys` order:

- Ball: `[200, 0.2, 0.04, 1.5, 5, 0.01, 0.5, 5, 5]`.
- Box: `[500, 0.3, 10, 0.1, 5, 4, 7, 3, 6, 0.01]`.
- Tray: `[750, 5, 1, 0.5, 1, 12, 10, 10, 15, 20, 50, 50]`.

### Checkpoint hashes

For B1–B4 and X1–X4, the SHA256 values below match the corresponding recorded NPZ checkpoint/weight hashes in every selected file that uses them. T0–T4 hashes are computed from the local adapters or historical Git blobs during this audit: **the Tray NPZs do not themselves record those hashes**. For Tray, tensor equality, metadata, run logs, and actor-output recomputation supply the additional evidence.

| Artifact | SHA256 |
|---|---|
| B1, retrained Ball Bayesian | `c58bf0eb06a79ce269576237b8c0e3c83bc8692971b42c005646ae72605bc2a6` |
| B2, Ball ARS3950 | `0d652353afc7855a7236a64b56c302c16bd5582e2b80a232302c3d93ed0779fd` |
| B3, Ball PPO4951 best train | `18d5d12a575b2d6e3f2cd6bc7d120a5544e8cb294981ec1df67e1579b13e3be8` |
| B4, Ball PPO4951 best eval | `197268ee929868a07a367b1e51ffdee896ea2f5f41dccc21a266b3811c997684` |
| X1, Box Bayesian5001 | `73d3804f0d9ff9e1d2595a73995f713a80ea249d1511a9aa7e787f622aaa022b` |
| X2, Box ARS4218 | `65a9291e5ff106d5e1e9413c90241926f70e001b869b0d921f6c011a3b9ea09c` |
| X3, Box PPO5005 best train | `0c60f8b1a6dd6db3bcf049597e8cfca9d31dfc5e7ccae0f8e12b62352b81e06e` |
| X4, Box PPO5005 best eval | `52ff85446069951289625108334377f08af49b5128d1d09dc8ac3a68326e190d` |
| T0, Tray Hand-Tuned adapter | `9a2f2025a0c833ead4a3ac3cd0f6048bf92e6a485e520fa2fdbc5f62f90f3c67` |
| T1, Tray Bayesian4104 adapter | `047ffccb93d9922face2f4e3451914fca4932c1ebe2c14ca2a32759e424e9036` |
| T2, original Tray ARS4124 iteration 806 | `9166d54ee3cc8ef97605a3ba442e2b968895627fc46aa5bde2a08db855b4d8b7` |
| T2, Tray ARS4124 adapter | `b8577e0bae8e05082ec54f0cc3f83f0f6a725c2c6874a65ceee2df07cf9aa075` |
| T3, Tray PPO5307 best train | `5e75b54f30dc7e91aa8ab9a16f350b60dac6ba9823fac2362992408f95e8b121` |
| T4, Tray PPO5307 best eval | `28535c318a394ddd47bf08a8ba3636eaa4e4d0c1c56b052b1eb02c052c16a2eb` |

The ARS4124 adapter and original JSON have different file hashes because the schema changed. Their `params`, `mu`, and `var` arrays match exactly. The adapter's hand-tuned baseline and clipping match the historical ARS implementation.

### Generation and training history checked

| Dataset / actor | Git and saved evidence |
|---|---|
| Ball ARS3950 training | `255a412` (2026-07-30), original checkpoint and ARS training history; exact hash remains unchanged in both selected sweeps |
| Ball PPO4951 training / large evaluation | `42fcbfb` and `d260301` (2026-08-29), PPO training artifacts, frozen best-training/best-evaluation actors, and evaluation manifest |
| Ball large fixed weights / ARS evaluation | `119d1d0` (2026-09-23), `run_eval_ht_bayes_ball_lift_100.sh`, evaluator `PolicyWrapper`, and ARS constants |
| Ball retrained Bayesian / small evaluation | `f4202cc` (2026-09-30), September 29 training artifacts, `run_manifest.txt`, `artifact_hashes.sha256`, and each selected NPZ's evaluator/checkpoint hashes; all 12 manifest entries match their Git blobs |
| Box ARS4218 training | `2d9ee336` (2026-06-22), ARS training source and saved 4218 checkpoint; exact hash matches both selected sweeps |
| Box PPO5005 training | `1a070555` (2026-09-04), PPO training source and best/best-evaluation actors |
| Box large evaluation | `79fa0aa` and `e08cb00` (2026-09-24), matched evaluator/runner/launcher snapshots; older Bayesian vectors match 3601 constants plus smoothness |
| Box small evaluation | Run records HEAD `00898ab` (2026-10-01). All 13 entries in each selected run's `input_sha256.txt` match historical blobs. Bayesian5001's dedicated hash and all 15 run logs/NPZs additionally confirm X1; see shared-list caveat below |
| Tray ARS4124 training / large baseline sweep | `587a453` (2026-06-23), nested iteration-806 actor; `e709694` (2026-06-26), sweep8 launcher and evaluator. All 20 selected large ARS logs explicitly load the nested actor with `start_it=806` |
| Tray PPO5307 / common-JIT evaluation | `a393e46` (2026-09-20), training artifacts; `49b06eda` and `6117e21` (September 24–25), adapters, shared evaluator, fixed-pose versus randomized launchers, and inference settings |

Training-code inspection distinguished actual algorithms rather than trusting filenames alone: ARS perturbs actor parameters in positive/negative directions and updates from their reward differences; PPO uses policy likelihood ratios and a clipped surrogate. Legacy ARS JSONs often omit `algorithm`; the historical loaders default these to ARS. Their hashes match the original ARS training artifacts, and their outputs match the deployed weight traces in the checks below.

### Numerical identity checks and evidence limits

| Check | Coverage | Result |
|---|---|---|
| Notebook selection and metadata | All 480 files / 9,600 episodes | No conflicting method identity or PPO checkpoint type within a selected series |
| Fixed-weight behavior | All 210 Hand-Tuned/Bayesian files; **1,314,339 saved steps** | Every recorded weight vector matches its selected fixed source within floating-point rounding; maximum absolute log-weight error below `1.4e-7` |
| Tray actor recomputation from saved 68-D observations | All 40 large PPO files plus 15 small PPO and 15 small ARS files; **553,548 steps** | Historical actor parameters/normalization/mapping reproduce logged weights; maximum absolute log-weight error below `8.4e-7` |
| Ball/Box actor spot checks | All 180 selected policy files; first/middle/last step of each of 3,600 episodes = **10,800 samples** | Reconstructed pre-step observations reproduce checkpoint outputs; maximum absolute log-weight error below `3.4e-6`. This is a kinematic/actor check, not dynamic replay or an every-step check |
| Large Tray ARS identity | All 20 logs explicitly name nested ARS4124 iteration 806; corresponding saved trajectories have adaptive weights | Supported by logs/source history, **not** a recorded NPZ hash or full actor replay: these older NPZs omit full observations/object state |

The actor checks used the historical input ordering, baselines, normalization, and output transforms. They did not load today's branch's arbitrary defaults or run new simulation episodes. Ball/Box reconstruction used existing frozen kinematic models; it does not certify every original physics/XML setting. Matching outputs provides stronger actor-identity evidence than checking the `policy` filename token alone.

### Naming and provenance caveats

1. **`policy` is not an algorithm name.** Ball ARS and PPO can both have `eval_policy_*.npz` filenames. The source directory, checkpoint hash, loader metadata, and output checks disambiguate them. Conversely, a `default_weight_set=bayesian` field on Box ARS/PPO identifies the residual baseline, not the plotted algorithm.
2. **Tray adapters say `algorithm="ppo"` to use a common loader.** Their `source_algorithm` is `handtuned`, `bayesian4104`, or `ars4124`. The first two have zero-output actors; the third preserves the trained ARS tensors. The selected NPZs correctly record the source identities. This is not evidence that these three methods were trained with PPO.
3. **Ball's historical Bayesian2100 vector is not an exact JSON copy.** `cost_weights_optim_2100.json` has collision weight `65.32040405273438`; historical `ARS.py` hardcodes `65.320040405273438`, which is what the large-batch Bayesian traces use. Difference: `0.00036364746`, or approximately **0.000557%**. Other entries match. This is a tiny transcription discrepancy, not an ARS/Bayesian swap; call this source **2100-derived constants**. The same historical constants anchor the selected Ball ARS/PPO policies. The retrained small-batch Bayesian JSON is separately hash-verified.
4. **Box's Bayesian5001 rerun carries a shared input list mentioning Bayesian5002.** Its `input_sha256.txt` describes the broader all-method launcher and includes `bayesian_5002.json`. The dedicated `bayesian_5001.sha256`, all 15 evaluation logs, all 15 NPZ checkpoint hashes, and every saved fixed-weight vector instead agree on **5001**. Do not use the general input list alone to identify the selected Bayesian method. The main all-contacts run's actual 5002 files are excluded from the notebook.
5. **Training identifiers and evaluation seeds are different.** `3950`, `4218`, `4124`, `4951`, `5005`, and `5307` identify trained artifacts, not episode seeds or batch sizes. A file renamed to an algorithm-like name would not establish provenance; the cross-checks above are why these selected labels are supported.
6. **Some old paths are historical, not currently accessible paths.** `/home/aks-lab/manipulator_mujoco/...` and the current `colcon_ws/src/...` prefix can refer to byte-identical actors. Use the recorded hash and historical Git artifact, not path existence or today's same-named file alone. The checkpoint hash table distinguishes values saved at generation from hashes computed during this audit.
7. **Identity is not fairness or physical validity.** In particular, the old Tray launcher at `e709694` explicitly comments on choosing the five seeds for low joint noise/small target angles. They should not be described as an independently randomized test set. This does not change which actor produced the results or the matched-start checks, but it limits generalization claims. The simulator/contact concerns remain separate.
