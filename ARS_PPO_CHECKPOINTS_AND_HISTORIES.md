# ARS and PPO checkpoints and training histories

Verified 2026-10-07 for the current [benchmark_stat_batch_sm.ipynb](benchmark_stat_batch_sm.ipynb): Ball Lift, Box Lift, and Tray Push, inference CEM batches **50, 150, and 250**. Each task/method uses one checkpoint across these three batches. PPO means **best-evaluation**, not best-training. New ARS3950/3980 comparison runs on `ball_lift_test_395080` are outside this notebook's selection.

Simulator repository: `/home/aks-lab/colcon_ws/src/manipulator_mujoco`.

All six selected checkpoints are available, either as local files or Git blobs. Training histories are less complete. A checkpoint's `best_reward`/`best_eval_reward` field is a single statistic, not a training history. Likewise, inference NPZs and inference logs are not training histories.

## Availability summary

| Task | Method / selected checkpoint | Matching training history available? |
|---|---|---|
| Ball Lift | ARS3950 | **Yes:** JSON iterations 1–1000, plus an exactly matching TensorBoard reward curve. Maximum recorded training reward is at iteration 607. |
| Ball Lift | PPO4951, best-eval update 1410 | **Not found:** neither matching JSON history nor matching TensorBoard logs. |
| Box Lift | ARS4218 | **Yes, a recorded portion:** TensorBoard iterations 1–390, including the selected checkpoint at logged iteration 287. JSON history not found; this does not establish a complete from-scratch training curve. |
| Box Lift | PPO5005, best-eval update 2900 | **Not found:** neither matching JSON history nor matching TensorBoard logs. |
| Tray Push | ARS4124, iteration 806 | **Selected-checkpoint history not found.** An earlier same-ID TensorBoard log covers iterations 1–161 and matches a different, iteration-132 checkpoint. |
| Tray Push | PPO5307, best-eval update 2180 | **Partial:** JSON updates 1501–3000. The selected update 2180 is included; updates 1–1500 were not found. |

## 1. Exact checkpoints

Paths in Git references are relative to the simulator repository. `git show COMMIT:PATH` reads them without switching branches. Branch names identify the relevant task lineage; the commit references pin the particular saved artifacts.

| Task / method | Relevant branch | Exact checkpoint Git reference |
|---|---|---|
| Ball ARS3950 | `issue_39_ball_lift_flow_RL`; subsequently used on `issue_49_ball_lift_ppo` | `255a412:ars_v2_linear_policy_3950.json` |
| Ball PPO4951 | `issue_49_ball_lift_ppo` | `42fcbfb:ppo_linear_policy_4951_best_eval.json` |
| Box ARS4218 | `issue_42_lift_box_RL`; subsequently used on `issue_50_box_lift_ppo` | `2d9ee336:ars_v2_linear_policy_4218.json` |
| Box PPO5005 | `issue_50_box_lift_ppo` | `1a070555:real_demo/real_demo/ppo_linear_policy_5005_best_eval.json` |
| Tray ARS4124 | `issue_41_move_reorient_box_RL`; subsequently used on `issue_53_tray_push_ppo` | `587a453:real_demo/real_demo/ars_v2_mlp_policy_4124.json` |
| Tray PPO5307 | `issue_53_tray_push_ppo` | `a393e46:ppo_mlp_policy_5307_best_eval.json` |

### Local copies currently available

- Ball ARS original: [ars_v2_linear_policy_3950.json](/home/aks-lab/colcon_ws/src/manipulator_mujoco/ars_v2_linear_policy_3950.json); notebook-run frozen copy: [ars3950.json](eval_sweep_100_ball_lift_bayes_retrained_20260929_191418/checkpoints/ars3950.json).
- Ball PPO original: [ppo_linear_policy_4951_best_eval.json](/home/aks-lab/colcon_ws/src/manipulator_mujoco/ppo_linear_policy_4951_best_eval.json); notebook-run frozen copy: [ppo4951_best_eval.json](eval_sweep_100_ball_lift_bayes_retrained_20260929_191418/checkpoints/ppo4951_best_eval.json).
- Tray ARS inference adapter: [ars4124_common_jit.json](eval_sweep_100_tray_push_common_jit_randomized_xy001_yaw2_b50_150_250-20260925T081018Z-1-001/eval_sweep_100_tray_push_common_jit_randomized_xy001_yaw2_b50_150_250/checkpoints/ars4124_common_jit.json). Its `params`, `mu`, and `var` exactly match the nested iteration-806 checkpoint above. It is an evaluation-format adapter, not a new training checkpoint. `algorithm="ppo"` selects the common loader; `source_algorithm="ars4124"` identifies its origin.
- Box ARS/PPO and original Tray ARS/PPO checkpoints were not found as loose files in the current Ball working tree or searched local downloads. They remain retrievable from Git using the references above; no checkout is necessary.

PPO best-training and latest files also exist in Git, but are **not selected** by the small-batch notebook:

| Task | Same-commit alternatives |
|---|---|
| Ball, `42fcbfb` | `ppo_linear_policy_4951.json`, `ppo_linear_policy_4951_latest.json` |
| Box, `1a070555` | `real_demo/real_demo/ppo_linear_policy_5005.json`, `real_demo/real_demo/ppo_linear_policy_5005_latest.json` |
| Tray, `a393e46` | `ppo_mlp_policy_5307.json`, `ppo_mlp_policy_5307_latest.json` |

The existing [checkpoint identity audit](BENCHMARK_NPZ_FILES.md#checkpoint-identity-audit-2026-10-06) provides full SHA256 values and the connection to recorded inference outputs. The three selected ARS historical blobs were rehashed in this audit and match that inventory.

## 2. ARS histories: what exists and what it proves

### Ball ARS3950 — matching 1,000-iteration history

- Local JSON: [ars_history_linear_policy_3950.json](/home/aks-lab/colcon_ws/src/manipulator_mujoco/ars_history_linear_policy_3950.json).
- Git: `255a412:ars_history_linear_policy_3950.json` (2026-07-30).
- Contains 1,000 contiguous records, iterations **1–1000**, with `iter` and `reward` fields.
- Maximum recorded `reward`: **1243.1702880859375 at iteration 607**.
- Local JSON and checkpoint are byte-identical to their respective blobs at `255a412`.

Matching TensorBoard event file:

```text
/home/aks-lab/Downloads/Tensorboard_logs_b50_80_balllift-20261007T105419Z-1-001/Tensorboard_logs_b50_80_balllift/arsv2_experiment_linear_policy_3950/events.out.tfevents.1785345792.aks-lab-Z790-D-AX
```

Its `Reward/Mean` values match **all 1,000 JSON records exactly**. At iteration 607, `Stats/Observation_Count=46617600`, matching checkpoint `count=46617600.0001` up to the small initial counter offset/float precision. The checkpoint does not store an explicit iteration or best reward; identifying iteration 607 is based on the matching curve, count, and historical best-save logic.

Do not merge the separate event file `events.out.tfevents.1785345366.aks-lab-Z790-D-AX` in that directory into this run: it contains a different two-iteration trial. Git commit `40c9f303` similarly contains an earlier two-row history, not the selected final 1,000-row history.

### Box ARS4218 — TensorBoard history available, JSON absent

Matching local TensorBoard event file:

```text
/home/aks-lab/colcon_ws/src/manipulator_mujoco/runs_linear_policy_4218/arsv2_experiment_linear_policy_4218/events.out.tfevents.1782023698.akslab
```

- Contains **390** `Reward/Mean` records, logged iterations **1–390**.
- Maximum `Reward/Mean` is **1034.224365234375 at iteration 287**, exactly matching the selected checkpoint's `best_reward`.
- At iteration 287, `Stats/Observation_Count=200140800`, matching checkpoint `count=200140800.00010002` up to the small initial counter offset/float precision.
- The checkpoint has no explicit iteration field: **287 is the iteration in this TensorBoard segment**, not a proven global from-scratch iteration number. The large carried observation count is consistent with resumed training; earlier training is not covered by this event file.
- These events are ignored by Git in the simulator repository. Their presence is local, not evidence that they can be recovered from Git later.
- **`ars_history_4218.json` was not found.** The training script at `2d9ee336:real_demo/real_demo/rl_trainer_batch.py:522` specifies this filename. `ars_history_4217.json` is not a substitute: its maximum reward is 1031.394775390625, not the selected 4218 checkpoint's best reward.

### Tray ARS4124 — early history exists, selected iteration-806 history absent

Local TensorBoard event file:

```text
/home/aks-lab/colcon_ws/src/manipulator_mujoco/runs_mlp_policy_4124/arsv2_experiment_mlp_policy_4124/events.out.tfevents.1779704183.akslab
```

- Contains **161** `Reward/Mean` records, iterations **1–161**.
- Maximum is **-399.24493408203125 at iteration 132**; observation count at that iteration is **20275200**.
- These values match `587a453:ars_v2_mlp_policy_4124.json` at the **repository root**, which stores `iteration=132` and `count=20275200.0001`.
- That root checkpoint is **not** the selected actor. Inference uses the **nested** `587a453:real_demo/real_demo/ars_v2_mlp_policy_4124.json`, with **iteration 806**, `best_reward=-248.8690185546875`, and `count=123801600.0001`.
- A second event file in the same directory, `events.out.tfevents.1779704053.akslab`, contains no scalar records. Neither file covers iteration 806. Both are ignored by Git.
- **`ars_history_mlp_policy_4124.json` and the later matching TensorBoard segment were not found.** The normal ARS training path at `587a453:real_demo/real_demo/rl_trainer_batch.py:1103` specifies that JSON name for run ID 4124 and policy type `mlp`.

Also checked the other historical Tray histories, including `ars_history_mlp_policy_42.json` (iterations 516–1515) and `ars_history_linear_policy_4150.json`. They do not establish the selected iteration-806 checkpoint's training history; none has its recorded best reward.

## 3. PPO histories

### Ball PPO4951

Matching history **not found**. Expected JSON name from `42fcbfb:real_demo/real_demo/rl_trainer_batch.py:766`:

```text
ppo_history_linear_policy_4951.json
```

Expected TensorBoard directory, relative to the training process's working directory:

```text
runs_ppo_4951/ppo_experiment_4951/
```

That directory pattern is set in `42fcbfb:real_demo/real_demo/PPO.py:342`. No matching event files were found. `ppo_history_mlp_policy_4900.json` belongs to a different run.

### Box PPO5005

Matching history **not found**. Expected JSON name from `1a070555:real_demo/real_demo/rl_trainer_batch.py:758`:

```text
ppo_history_5005.json
```

Expected TensorBoard directory, relative to the training process's working directory:

```text
runs_ppo_5005/ppo_experiment_5005/
```

That directory pattern is set in `1a070555:real_demo/real_demo/PPO.py:607`. No matching event files were found. `real_demo/real_demo/ppo_history_5002.json` belongs to a different run.

### Tray PPO5307

Matching history exists in Git:

```text
a393e46:ppo_mlp_policy_5307_history.json
```

It contains **1,500 contiguous records, updates 1501–3000**. At update **2180**, `eval_reward=401.0724182128906`, exactly matching the selected best-evaluation checkpoint. This is also the maximum recorded evaluation reward in the available history.

The history includes training reward, losses, KL diagnostics, training success rates, and periodic evaluation metrics. These training/validation success rates are not the notebook's inference success rates.

**Updates 1–1500 were not found.** Look for an earlier saved copy of `ppo_mlp_policy_5307_history.json` or TensorBoard events in `runs_ppo_5307/ppo_experiment_5307/`. No matching local TensorBoard events were found. The history JSON itself is available through Git, not as a loose file in the current Ball checkout.

## 4. Missing files / segments to request or recover

Expected filenames come from the historical trainers; they are **search targets, not assertions that the files were actually written or retained**. History JSON is written after `trainer.train(...)` returns, so a checkpoint or TensorBoard segment can exist without a final history JSON.

| Task / method | Missing filename or segment | TensorBoard alternative / qualification |
|---|---|---|
| Ball PPO4951 | `ppo_history_linear_policy_4951.json` | `runs_ppo_4951/ppo_experiment_4951/events.out.tfevents.*` |
| Box PPO5005 | `ppo_history_5005.json` | `runs_ppo_5005/ppo_experiment_5005/events.out.tfevents.*` |
| Box ARS4218 | `ars_history_4218.json`; earlier training records if a complete from-scratch curve is required | Existing `runs_linear_policy_4218/arsv2_experiment_linear_policy_4218/` events already cover the selected checkpoint; the JSON is not needed just to plot that 390-iteration segment. |
| Tray ARS4124 | `ars_history_mlp_policy_4124.json`, specifically the run/continuation containing iteration **806**, best reward **-248.8690185546875** | Later events under `runs_mlp_policy_4124/arsv2_experiment_mlp_policy_4124/`; the available iteration-1–161 events are insufficient. |
| Tray PPO5307 | Earlier `ppo_mlp_policy_5307_history.json` containing updates **1–1500** | `runs_ppo_5307/ppo_experiment_5307/events.out.tfevents.*`; updates 1501–3000 are already available in Git. |

No missing checkpoint filename needs to be requested for the six selected actors: all six can be retrieved. Ball ARS3950's matching 1,000-iteration history is also available.

## 5. Read-only inspection commands

These commands do not change branches, restore files, or overwrite anything:

```bash
cd /home/aks-lab/colcon_ws/src/manipulator_mujoco

# Ball: selected ARS checkpoint and matching JSON history
git show 255a412:ars_v2_linear_policy_3950.json
git show 255a412:ars_history_linear_policy_3950.json

# Box: selected ARS and PPO checkpoints
git show 2d9ee336:ars_v2_linear_policy_4218.json
git show 1a070555:real_demo/real_demo/ppo_linear_policy_5005_best_eval.json

# Tray: selected nested ARS checkpoint (not the root-level iteration-132 file)
git show 587a453:real_demo/real_demo/ars_v2_mlp_policy_4124.json

# Tray: selected PPO checkpoint and available partial history
git show a393e46:ppo_mlp_policy_5307_best_eval.json
git show a393e46:ppo_mlp_policy_5307_history.json

# Ball: selected PPO checkpoint
git show 42fcbfb:ppo_linear_policy_4951_best_eval.json
```

## Search scope and limitations

Checked the notebook's source selection and existing checkpoint inventory; searched accessible local files under `/home/aks-lab` (including ignored run directories and downloads, excluding cache/node_modules/Git internals from the loose-file search); inspected all reachable Git references in the simulator repository; and parsed all **30 distinct historical ARS/PPO history JSON blobs** exposed by the repository-wide object inventory, including differently named histories. Available relevant TensorBoard event files were parsed with all scalar samples retained, rather than TensorBoard's default sample reservoir.

“Not found” means absent from these searches, not proof the artifact never existed on the training machine, in an external backup, or in an unsearched archive. No remote fetch, branch switch, training, simulation, checkpoint extraction, or notebook/code edit was performed. Only this Markdown inventory was added.
