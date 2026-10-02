# Concerns about the current benchmark results

Reviewed: 2026-10-02.

## Scope

This report concerns the data currently selected by [benchmark_stat_batch_sm.ipynb](benchmark_stat_batch_sm.ipynb), with batches 50, 150, and 250. It distinguishes confirmed correctness problems from limitations on fair comparisons. It does not claim that all differences favor one method or that the recorded success-rate arithmetic is wrong.

The review used saved NPZ data, notebook source and outputs, penetration audits, and locally available Git history in `/home/aks-lab/colcon_ws/src/manipulator_mujoco`. Relevant task lineages are issue 49 (Ball Lift), issue 50 (Box Lift), and issue 53 (Tray Push), with later Tray corrections in issues 55/56. Later code changes do not retroactively repair earlier recorded results. This is a dated review, not an exhaustive verification of every branch or the large-batch notebook.

The selected historical Tray results use Bayesian4104, ARS4124, and PPO5307 best-evaluation, not the newer corrected-state Tray checkpoints. The previously inspected simulator snapshots include Box commit `71f381f` and Tray commit `6117e21f30808697a60a3d7f33e857062dfaae77`.

## 1. Historical Tray results use code with confirmed state-handling bugs

**Classification: correctness and simulation-validity problem.**

The issue-53 runner clears tray/passive-body velocities on each execution step by constructing `qvel_full = jnp.zeros_like(state["qvel"])` and then inserting only commanded robot velocities. Planner rollouts do not receive the complete current simulation state. Pose-dependent quantities are also read after stepping without refreshing forward kinematics to the newly integrated configuration.

Commit `321f4dc11b8bef2774bebcfa40e415b5100ada30` subsequently changes the runner and planner to preserve passive-body momentum, pass `initial_data` into rollouts, and refresh derived poses after stepping. These are concrete implementation corrections, not merely a different cost-weight choice.

**Consequence:** the old results measure performance in the older flawed implementation. They cannot establish performance under the corrected simulator. A shared environment bug may affect policies differently, but its existence alone does not establish which method benefited or by how much.

**Recommended action:** evaluate all methods together under one corrected simulator revision. Keep historical and corrected results separate, and distinguish reevaluation of old policies from retraining under corrected dynamics.

Evidence: issue-53 `real_demo/real_demo/rl_trainer_batch.py` and `real_demo/sampling_based_planner/mjx_planner.py`; their changes in commit `321f4dc`. See also [Tray findings](TRAY_PUSH_PENETRATION_FINDINGS.md).

## 2. Recorded task success does not establish physically valid manipulation

**Classification: physical-validity and interpretation problem.**

The success flags do not enforce a maximum-penetration limit. Successful episodes can therefore contain substantial geometric overlap.

The expanded Tray audit found forearm–tray overlap in **13 of 266 successful episodes**, reaching **41.73 mm**. Tray collisions are disabled for arm 2's forearm and both upper arms. These disabled pairs were absent from the original enabled-contact robot–tray statistics. The earlier **32.12 mm** successful maximum remains valid only for that enabled-contact scope; it is not an all-pair geometric maximum.

**Consequence:** calling these physically feasible successful manipulations would overstate the evidence. Common collision rules do not guarantee equal impact across policies: different trajectories encounter different omitted contacts. This does not prove that penetration caused a method's success.

**Recommended action:** report task success and contact validity separately, with an explicit contact set, depth threshold, and initialization policy. Reevaluate all methods under the same corrected collision model. Geometric overlap is not measured material deformation; saved-state checks cannot exclude overlap during unrecorded intermediate motion.

Evidence: [Tray findings](TRAY_PUSH_PENETRATION_FINDINGS.md), [arm–tray snapshots](tray_push_penetration_snapshots/arm_tray_successful_top3.jpg), [Ball findings](BALL_LIFT_PENETRATION_FINDINGS.md), and [Box findings](BOX_LIFT_PENETRATION_FINDINGS.md).

## 3. Ball PPO checkpoint documentation disagrees with the selected data

**Classification: confirmed reporting error.**

The notebook introduction, source comment, settings table, and [NPZ manifest](BENCHMARK_NPZ_FILES.md) describe Ball PPO as **best-training**. The actual source patterns select `ppo_best_eval_batch{batch}` from `eval_sweep_100_ball_lift_bayes_retrained_20260929_191418/npz/`.

Recomputing the selected files gives success rates of **94%, 98%, and 99%** at batches 50, 150, and 250. These agree with the notebook's saved numerical output. The checkpoint description is wrong; the success-rate arithmetic is consistent with best-evaluation inputs.

**Recommended action:** make the prose, manifest, and labels match the intended checkpoint selection, then verify generated figures. Do not silently change the checkpoint merely to match stale text. The penetration audit already explicitly identifies best-evaluation PPO.

Evidence: `SOURCE_PATTERNS` and the saved output in [the small-batch notebook](benchmark_stat_batch_sm.ipynb), compared with its introductory Markdown and the small-batch section of [the manifest](BENCHMARK_NPZ_FILES.md).

## 4. PPO and ARS have different permitted weight-output ranges

**Classification: confound for an optimizer-only comparison.**

For the selected Ball and Box policies, PPO maps its action through a bounded log-weight residual, `delta = 2 * tanh(action / 2)`. Before shared global clipping, this permits approximately **0.135 to 7.39 times** each baseline weight. ARS does not have this additional residual bound; both methods still apply global log-weight clipping.

Their inspected historical residual baselines match within Ball and within Box. The concern is therefore not a demonstrated baseline mismatch, but a difference in output freedom. This particular PPO residual-bound statement should not be generalized to selected Tray PPO5307, whose recorded mapping is `12_cost_log_residuals_clipped`.

**Consequence:** the comparison changes both the learning algorithm and the permitted policy outputs. It is valid as a comparison of these complete systems, but does not isolate the optimizer.

**Recommended action:** disclose the mappings and compare matched output bounds in a controlled retraining experiment. Changing a trained policy's mapping only at inference is not equivalent to training it with the alternative mapping.

Evidence: task-specific `real_demo/real_demo/ARS.py` and `PPO.py`; the selected [Ball PPO checkpoint](eval_sweep_100_ball_lift_bayes_retrained_20260929_191418/checkpoints/ppo4951_best_eval.json) records `max_log_weight_delta=2.0`. The historical mappings and baselines are documented at Git commit `7ff4f3b`, `real_demo/real_demo/docs/bayesian_ars_ppo_all_tasks.tex`.

## 5. Training objectives and planning budgets are not matched

**Classification: training-comparison confound.**

ARS and PPO use different reward formulations, and Bayesian optimization uses a separate episode-level objective. For example, the historical ARS rewards use error penalties and event terms, while PPO uses progress-based shaping with different penalties. Equal evaluation success criteria do not make their training objectives identical.

The selected training setups also differ:

| Task | Bayesian training planning batch | ARS/PPO training planning batch |
|---|---:|---:|
| Ball Lift | 300 | 50 |
| Box Lift | 1000 | 50 |
| Tray Push, historical issue 53 | 1000 | 50 |

These are CEM samples per planning step, not total training-episode or wall-clock budgets. Matching the evaluation batches to 50/150/250 does not remove the training differences.

**Consequence:** the benchmark can compare the resulting trained systems, but does not establish an algorithm-only ranking or training sample efficiency. Fixed versus state-adaptive weights is an intentional method difference, not itself an implementation error.

**Recommended action:** disclose objectives, planning settings, training interactions, and compute budgets. Use controlled objective/budget ablations for claims about the learning algorithms themselves.

Evidence: historical training artifacts and source summarized at commit `7ff4f3b`, `real_demo/real_demo/docs/bayesian_ars_ppo_all_tasks.tex`, particularly the ARS/PPO reward sections and training-versus-inference table. That document is a historical snapshot, not a statement about every newer branch.

## 6. Task Time is not a clean or equally supported task-speed comparison

**Classification: metric-definition and comparison limitation.**

The notebook's Task Time uses wall-clock `total_time`, not simulated completion duration. Box's rendered evaluation includes rendering and video encoding in that duration. It also:

- includes only successful episodes, so methods are timed on different outcome subsets;
- removes saved episode 0 from every file, even though the selected Box files record a separate unsaved warm-up episode;
- gives each finite file-level mean equal weight rather than pooling successful episodes;
- can report a time and zero spread from very little data.

Concrete checks on the current small-batch inputs:

- Box Bayesian batch 150: **8.61 s** with the notebook's mean-of-file-means aggregation, versus **7.61 s** when pooling the same eligible successful episodes.
- Tray PPO batch 50: the time bar uses **one successful episode** after episode-0 filtering. Its zero error bar is not evidence of low variability.

Neither aggregation is inherently invalid, but they answer different questions. Success-only timing is conditional performance, not unconditional efficiency or a paired comparison on equally difficult solved instances.

**Recommended action:** distinguish simulated completion time, controller computation time, and rendered end-to-end wall time. State the aggregation and exclusions; show contributing episode/file counts and avoid presenting insufficient-data error bars as strong evidence. Pair comparisons on common solved instances where appropriate, alongside success rates.

Evidence: the notebook's `comp_time_stats`, `finite_mean_std`, and `data_all` aggregation cells; selected NPZ timing/warm-up metadata. [Benchmark discrepancies](BENCHMARK_DISCREPANCIES.md) explains related timing definitions, but its dataset-specific tables concern the large-batch notebook and must not be substituted for current small-batch results.

## 7. Episode count and error bars overstate evidence if interpreted as independent trials or confidence intervals

**Classification: statistical interpretation limitation.**

The current Box and Tray files repeat one recorded seed across 20 episodes. Each method/batch has five such seed groups. Thus **100 recorded outcomes are not evidence of 100 independently randomized starts**. The trajectories can still differ; repeated seeds do not imply every rollout is identical.

The plotted timing error bars are population standard deviations across finite file-level means, not episode-level standard deviations, standard errors, or confidence intervals. Some groups have fewer than five contributing successful-file means.

**Consequence:** the raw success fractions remain valid descriptions of recorded outcomes, but claims of statistical significance or generalization need additional analysis. Treating all repetitions as independent can exaggerate the effective evidence.

**Recommended action:** use more independent reset seeds, retain paired seeds across methods, and account for seed groups in uncertainty estimates. Clearly label the current standard-deviation bars and disclose their contributing counts. This repeated-seed finding is specific to the inspected Box/Tray inputs, not a blanket statement about Ball's episode-key schedule.

Evidence: selected Box/Tray `episode_seed` or `episode_seeds` arrays, notebook aggregation functions, and the RNG/episode-protocol section of the historical training document at commit `7ff4f3b`.

## Checks that passed and next priorities

- Recomputed small-batch success rates match the saved notebook outputs. All five selected files contribute; the separate best-file check does not mean the plots use only that file.
- Recorded horizon, CEM iterations, and projection iterations match across methods within each task.
- The inspected historical Ball/Box ARS and PPO baseline anchors match; concern 4 is about their different output constraints.

Priority order: correct checkpoint documentation; keep task success distinct from contact validity; evaluate all methods under one corrected simulator revision; then strengthen timing and statistical reporting. Strong optimizer-only claims additionally require controlled training objectives, output mappings, and budgets.

This document records concerns and proposed follow-up work. Creating it does not change checkpoints, notebooks, figures, simulator code, or evaluation data.
