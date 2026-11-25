# Training budgets at the evaluated ARS and PPO checkpoints

Each figure is the **cumulative training budget up to the checkpoint used in the benchmark**. A run could continue after that checkpoint. The same selected policy was evaluated at planning batches 50, 150, and 250, so its training budget is counted once. The evaluation episodes are separate.

| Task | Unit | ARS checkpoint | ARS budget | PPO checkpoint | PPO budget calculation | ARS / PPO (millions) |
|---|---|---|---:|---|---:|---:|
| Ball Lift | Planned rollout steps | ARS3950 | 46,617,600 | PPO4951 best-eval | 1,410 × 32 × 300 = 13,536,000 | 46.62 / 13.54 |
| Box Lift | Planned rollout steps | ARS4218 | 200,140,800 | PPO5005 best-eval | 2,900 × 64 × 300 = 55,680,000 | 200.14 / 55.68 |
| Tray Push | Planned rollout steps | ARS4124 | 123,801,600 | PPO5307 best-eval | 2,180 × 64 × 600 = 83,712,000 | 123.80 / 83.71 |
| Quadrotor | Policy decisions | ARS wall3 | 9,917,952 | PPO ppowall1 best-eval | 600 × 64 × 66 = 2,534,400 | 9.92 / 2.53 |

## How the counts were obtained

**ARS.** The selected checkpoint stores a cumulative observation-normalizer counter in the JSON field `count`. ARS increments it by the number of observation slots collected from positive and negative perturbation rollouts. Rounding to the nearest integer removes the counter’s initial offset of 0.0001. This preserves carried training history for Box ARS4218, whose selected checkpoint corresponds to iteration 287 of a recorded training segment. [Bimanual checkpoint identities, Git revisions, and counter checks](ARS_PPO_CHECKPOINTS_AND_HISTORIES.md) · [Quadrotor ARS checkpoint](/home/aks-lab/sampling_based_general/real_demo/real_demo/ars_ckpt/ars_quad_best_idxwall3.json)

**PPO.** Each calculation uses the selected checkpoint’s completed training updates × parallel episodes per update × scheduled steps or decisions per episode. For example, Ball PPO’s evaluated checkpoint was saved at update 1,410, with 32 episodes per update and a 300-step episode limit. Early termination can reduce the number of active transitions within those scheduled slots. For Quadrotor, one policy decision spans three simulation steps, giving 66 decisions per 198-step episode. The resulting PPO count also agrees with the Quadrotor checkpoint’s stored counter. [Bimanual checkpoint records](ARS_PPO_CHECKPOINTS_AND_HISTORIES.md#1-exact-checkpoints) · [Supplementary Table 4](/home/aks-lab/Downloads/Bimanual_AAMAS_2348_sup.pdf) · [Quadrotor training settings](/home/aks-lab/sampling_based_general/real_demo/sampling_based_planner/quadrotor_fairness_audit.md) · [Quadrotor PPO checkpoint](/home/aks-lab/sampling_based_general/real_demo/real_demo/ppo_ckpt/ppo_quad_best_idxppowall1_best_eval.json)

The budgets answer the training-interaction question for the selected policies. Hyperparameter tuning effort and wall-clock compute would require separate accounting. The Box ARS4211 and Tray PPO5306 reward curves shown in the rebuttal draft describe different runs from the evaluated Box ARS4218 and Tray PPO5307 checkpoints. [Training reward figure](/home/aks-lab/Downloads/mean_reward_100_by_run.png)

## Rebuttal wording

At the evaluated checkpoints, ARS/PPO budgets were 46.62M/13.54M, 200.14M/55.68M, and 123.80M/83.71M planned rollout steps for Ball Lift, Box Lift, and Tray Push, respectively, and 9.92M/2.53M policy decisions for Quadrotor.
