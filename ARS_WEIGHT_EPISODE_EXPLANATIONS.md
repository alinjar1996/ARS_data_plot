# Interpreting the ARS weight-and-snapshot figures

## What the revised figures show

The figures show **phase-masked, linear policy weights**, using the exact component names printed above the plots. The fixed lifting-spacing gain is **not** applied to these curves. In particular, the planner's existing ×10 lifting gain is omitted from the Ball Lift and Box Lift **Spacing** plots; the simulator, notebook, policies, and recorded inference runs are unchanged.

The plotted value is `active_in_phase * saved_policy_weight`. Inactive components are exactly zero, including Box Lift **Smoothness**, which is unused in this cost implementation. The CSVs additionally retain the actual gain-adjusted CEM coefficients for auditing; those are not the plotted values.

**Spacing still increases overall within lifting without the fixed gain:** Ball goes from **36.4 to 113.3 (about 3.1-fold)**, and Box from **0.813 to 5.74 (about 7.1-fold)**. Removing a gain that is constant throughout lifting changes absolute values, not these within-phase ratios. Neither trajectory is monotonic: both peak higher and then decrease. Box also initially drops from **1.21 before C to 0.813 after C**, before rising within lift. A phase-boundary change and a within-phase trend are different comparisons.

A larger weight makes a given error more expensive to MPC; a smaller weight relaxes that penalty. It does not follow that an increasing weight reduces the error, or that a decreasing weight means the error has disappeared. Each panel has its own linear y-axis. Neither line heights across panels nor coefficient magnitudes establish which term dominates the total cost.

**Evidence and intuition are separated below.** Numerical trends and reported motion diagnostics are observations. The physical explanations are plausible interpretations, not identified reasons for the learned policy or causal demonstrations. The Box Lift collision-input decomposition is a separately verified explanation of the policy output, not a proof of why training learned it. Non-monotonic curves are described as such rather than given an artificial increasing/decreasing story.

## Episodes and timing

These are the same three illustrative recorded-success episodes; no episode was replaced to produce a more convenient explanation. Episode numbers are one-based; NPZ/video indices can be zero-based.

| Task and ARS checkpoint | Batch / seed / episode | A: start | B: approach | C: phase transition | D: successful end |
|---|---|---:|---:|---:|---:|
| Ball Lift, ARS3950 | 50 / 4 / 18 | 0.1 s | 3.5 s | 6.9 s | 9.1 s |
| Box Lift, ARS4218 | 50 / 5 / 5 | 1.5 s | 3.8 s | 6.1 s | 8.1 s |
| Tray Push, ARS4124 | 250 / 27 / 4 | 0.1 s | 4.3 s | 8.6 s | 16.6 s |

Box Lift starts after the initial drop/rebound settles; its graph also starts at 1.5 s, without rebasing time. Ball/Tray A is the first saved frame, not the reset state. Frame `k` shows the post-step state at `(k+1)*dt`; weight `k` applies on `[k*dt,(k+1)*dt)`. C marks the recorded controller phase switch, not independently measured first physical contact. “Start/end of lift/push” below means its first/last executed policy coefficient; peaks are intermediate unless stated otherwise.

## Ball Lift: establish the grasp configuration, then strengthen spacing control

[Open the Ball Lift figure](ars_weights_snapshots/ars_ball_weights_snapshots.png).

### Overall finding and physical intuition

During approach, **End-effector-to-object**, **Orientation**, and **Planar alignment** have pronounced intermediate peaks. An intuitive reading is that approaching the ball requires both moving the hands into place and arranging them into a coordinated grasp configuration. These penalties need not stay high once that configuration is approximately established. During lift, **Spacing** increases strongly even without the fixed planner gain, while **Object-to-target** is active and ends higher than it starts. A plausible interpretation is stronger regulation of hand separation while transporting the ball, rather than continuing to emphasize acquisition of the initial grasp pose.

This reading has some independent support from the saved diagnostics: the reported hand-midpoint approach error falls from **48.60 cm at A to 1.88 cm at C**, and the reported mean hand-orientation error falls from **0.766 to 0.0508 rad**. From C to D, the ball-to-goal distance calculated from saved post-step positions falls from **15.73 to 4.43 cm**, and ball height increases from **0.1154 to 0.2268 m**. These are accompanying changes, not evidence that one weight caused them. The reported approach diagnostic follows the current ball position, whereas the planner's approach feature uses its supplied approach reference.

### Every component, using the figure names

| Name in figure | Observed policy-weight change | Intuition: why that direction could make sense |
|---|---|---|
| **Collision** | Approach peak about **14,590**; lift **126 → 6.6**. | The learned policy puts a much larger coefficient on the approach collision penalty than on the late-lift penalty. Moving from reaching to transport also changes the included collision pairs. Relaxation after the reaching maneuver is plausible, but the curve alone does not show that collision risk has fallen. |
| **Joint deviation** | Early peak **16.8**, falling to **0.58** before transition; lift **0.49 → 1.71**. | Relaxing the initial-posture penalty can allow the arms to leave their starting configuration to reach the ball. Its partial late recovery could discourage unnecessary joint displacement during completion; it does not imply a return to the initial posture. |
| **Relative velocity** | Fluctuating; lift approximately **0.107 → 0.072 → 0.122**. | This feature penalizes changing inter-hand separation, not overall robot speed. Its varying coefficient can regulate coordination without requiring both arms to move slowly. There is no clear monotonic strategy to claim. |
| **Planar alignment** | Approach peak **91.0**; lift starts at **7.9**, dips to **2.5**, ends at **4.7**. | Matching the hands' y/z positions is useful when forming a balanced grasp configuration. A smaller coefficient after acquisition is consistent with less emphasis on correcting that geometry, but saved weights alone do not establish that its error stayed small. |
| **Spacing** | **26.8** immediately before transition; lift **36.4 → 141.3 → 113.3**. | Increasing the policy weight makes deviation from the planner's desired **0.17 m** hand separation more expensive during transport. This is consistent with maintaining the grasp geometry while lifting. The rise shown here is not the omitted fixed gain. |
| **Smoothness** | Mid-approach minimum **0.00064**, rising to **0.0242** near transition; lift approximately **0.0177 → 0.0091**. | Stronger joint-acceleration regularization near grasp acquisition is consistent with gentler changes in commanded motion there. Its reduction during lifting permits a different acceleration/transport trade-off. This is not a measurement that the actual trajectory became smoother. |
| **Orientation** | Approach peak **41.2**; lift approximately **12.0 → 3.0**. | The hands first need to reach the prescribed grasp orientations. Reducing the coefficient later is consistent with those orientations already being approximately established: the reported orientation error is small at C and smaller at D (**0.0292 rad**). This is an association, not proof that the policy explicitly tests that error. |
| **End-effector-to-object** | Approach peak **115**, falling to **16.1** before becoming **zero** in lift. | The intermediate peak is consistent with emphasizing approach to the grasp region. The drop to zero is a programmed phase gate, not evidence that ARS learned to switch this objective off. |
| **Object-to-target** | Zero in approach; lift **4.66 → 7.37 → 6.61**. | Once transport is enabled, a larger coefficient penalizes remaining goal error more strongly. Here the cost uses the hand midpoint plus a **0.05 m vertical offset** as a transport proxy, not direct ball-to-goal distance. |

### Concise wording for the paper

> In this Ball Lift episode, the approach stage exhibits pronounced peaks in End-effector-to-object, Orientation, and Planar alignment, alongside a reduction in the reported grasp-position and orientation errors. During lifting, the policy's Spacing weight increases substantially while Object-to-target remains active, consistent with a shift from grasp acquisition toward coordinated transport. This interpretation concerns state-dependent reweighting within predefined phase objectives; it does not identify the causal contribution of an individual weight.

## Box Lift: a transport-and-coordination trade-off, not uniform weight growth

[Open the Box Lift figure](ars_weights_snapshots/ars_box_weights_snapshots.png).

### Overall finding and physical intuition

During approach, **Box-specific alignment**, **Orientation**, and **Planar alignment** emphasize setting up the hands relative to the box. During lift, **Spacing**, **Relative velocity**, and **Box orientation** become stronger overall, whereas **Object-to-target** decreases. An intuitive reading is a changing balance between transporting the object and regulating the two hands and the box pose. A decreasing transport coefficient is not contradictory: transport remains in the objective, other terms also influence the motion, and the object is already approaching its goal.

From C to D, post-step box-to-goal distance falls from **20.57 to 3.21 cm**, and box height rises from **0.1001 to 0.2622 m**. The instantaneous box-to-target quaternion-difference norm also decreases, from **0.0765 to 0.0556**. That last diagnostic uses the same kind of quaternion difference as **Box orientation**, but it is an instantaneous state error rather than the predicted-horizon cost. It supports an association between the late orientation weighting and pose refinement, not a claim that the weight increase caused the improvement.

### Every component, using the figure names

| Name in figure | Observed policy-weight change | Intuition: why that direction could make sense |
|---|---|---|
| **Collision** | Lift **208 → 11,624 → 5,084**. | This is not a programmed lift multiplier. A verified linear-policy decomposition attributes most of the rise to the joint-velocity inputs. It increases the penalty on retained collision terms during this motion, but cannot be interpreted as the policy detecting box penetration: box-contact pairs are excluded from the lifting collision penalty. |
| **Joint deviation** | Lift approximately **0.86 → 0.62 → 3.58**. | The stronger late penalty could discourage unnecessary departure from the initial joint configuration while finishing the lift. It does not require the arms to return to their starting positions. |
| **Planar alignment** | Approach peak **132**; **30.6 → 106** across transition; ends at **42.9**. | Renewed emphasis on matching the hands' y/z positions is consistent with coordinating support at lift onset. The term is active in both phases, so this jump is a policy change, not activation from zero. |
| **Relative velocity** | Lift approximately **0.0268 → 0.0142 → 0.1152**. | Stronger late penalization of changing inter-hand separation is consistent with reducing differential squeezing/separation motion while transporting the box. It is not a general penalty on joint speed. |
| **Spacing** | **1.21** before transition; lift **0.813 → 14.77 → 5.74**. | Importantly, the policy weight initially **decreases** at the switch when the fixed gain is excluded. It subsequently rises strongly during lifting, consistent with stronger regulation of the desired hand separation. A claim of an immediate learned increase at C would be incorrect. |
| **Orientation** | Approach peak **51.0**; lift **8.26 → 1.61 → 3.25**. | Once the hands have approximately acquired the grasp orientation, the policy can reduce its coefficient while retaining the term. The late recovery is consistent with continued adjustment. This concerns hand orientation, not the box's own orientation. |
| **Box-specific alignment** | Displayed approach peak **52.2**, falling to **19.9**, then zero. | Approaching box-attached grasp sites is a natural acquisition objective. Its later deactivation is programmed; hand-to-box contact maintenance still appears in another lifting term. |
| **Object-to-target** | Zero in approach; lift **40.0 → 4.66 → 7.40**. | This coefficient weights both the hand-midpoint-to-goal term and hand-to-grasp-site contact maintenance. Its decline is consistent with relaxing that combined objective as transport progresses while coordination/box-orientation terms strengthen. It is neither a pure box-goal weight nor evidence that grasp maintenance is no longer needed. |
| **Box orientation** | Lift starts at **1.33**, dips slightly, finishes at **5.59**. | Greater late penalization of box-to-target orientation error is consistent with pose refinement near completion. The observed orientation error decreases, but the association is not a causal test. |
| **Smoothness** | Zero throughout. | This output does not enter this revision's CEM cost. No learned smoothness-control explanation is supported for this panel. |

### Verified reason for the Collision rise

The ARS4218 checkpoint is a linear actor on normalized observations, followed by a baseline log-weight offset, clipping, and exponentiation. Its stored outputs were reproduced from the checkpoint and saved states using forward kinematics, with maximum absolute log-output discrepancy below **0.000004** over all **81 × 10** cost outputs. No closed-loop simulation was rerun.

From the first lifting solve at **6.1 s** to the collision-weight peak at **7.3 s**, the log-weight rises by **4.024**, corresponding to roughly a **56-fold** linear increase. The input-group contributions are **+4.512 from joint velocities**, **−0.489 from all other inputs combined**, and **zero from the unchanged phase flag**. This explains how this actor produces the increase. It does not prove why ARS training learned that mapping or that the increase improved safety.

### Concise wording for the paper

> In the Box Lift example, acquisition-related alignment weights peak during approach, while Spacing, Relative velocity, and Box orientation strengthen overall during transport. Object-to-target, which also weights contact maintenance in this implementation, decreases as the box approaches its goal. The Collision increase is primarily attributable to the linear actor's joint-velocity dependence and applies only to the collision pairs retained during lifting. The trajectories therefore illustrate a changing transport–coordination trade-off rather than uniform growth of all task weights.

## Tray Push: goal progress accompanied by relaxed position maintenance

[Open the Tray Push figure](ars_weights_snapshots/ars_tray_weights_snapshots.png).

### Overall finding and physical intuition

The prescribed transition switches off **Approach position** and **Approach orientation**, and switches on pushing objectives. During pushing, **Contact-maintenance position** is high early but decreases substantially late, while **Tray-to-goal position** and **Tray-to-goal orientation** generally strengthen toward the later part of the episode. A plausible interpretation is reduced emphasis on preserving the initial hand–tray positional relationship while continuing to move the tray toward its goal.

The measured trade-off matters. The reported worst-hand grasp-position error falls from **40.54 cm at A to 1.52 cm at C**, and its reported orientation error falls from **1.564 to 0.0355 rad**: approach successfully establishes the grasp region. From C to D, tray-to-goal distance falls from **19.96 to 3.98 cm**, but the reported worst-hand grasp-position error grows to **12.77 cm**, and its orientation error grows to **0.372 rad**. Thus it would be misleading to say Contact-maintenance position decreases because the grasp has become perfect. The episode instead illustrates goal progress alongside deteriorating reported hand–tray grasp geometry.

The tray-to-goal orientation error also increases from **0.0172 to 0.0352 rad** between C and D, even though its weight ends higher. A stronger orientation coefficient does not guarantee a smaller realized orientation error. These reported grasp diagnostics use the worse hand under the historical ARS protocol; they are not the same aggregation as the CEM contact-maintenance costs and do not measure contact force or certify physical contact.

### Every component, using the figure names

| Name in figure | Observed policy-weight change | Intuition: why that direction could make sense |
|---|---|---|
| **Collision** | Early approach peak **1,468**; push fluctuates about **292–647**, ending at **512**. | The policy continues to adjust the retained collision penalty during pushing. There is no monotonic safety narrative; neither the coefficient nor its fluctuations establish collision risk. |
| **Joint deviation** | Approach peak **13.0**, falling to about **4.0**; push **5.11 → 3.26 → 5.55**. | Relaxation can allow reaching away from the initial arm configuration; later variation is consistent with balancing transport and unnecessary joint displacement. |
| **End-effector height alignment** | Approach peak **2.10**; push ends lower, **0.906 → 0.550** overall. | Less weight on equal hand height could accommodate asymmetric hand motion during transport. This is an interpretation of the penalty, not evidence that height asymmetry actually increased. |
| **Rotation-axis alignment** | Push-only: **0.781 → 1.066 → 0.571**. | This penalizes the hands' relative-position component along the tray rotation axis. Stronger early weighting is consistent with establishing the pushing geometry, followed by relaxed enforcement later. |
| **Relative velocity** | Push-only; fluctuates about **0.479–1.060**, ending at **0.868**. | Varying the penalty on changing inter-hand separation can regulate coordination as the tray moves. No simple monotonic explanation is warranted. |
| **Approach position** | **11.3 → 25.6 → 15.1**, then zero. | Temporarily stronger attraction toward the grasp targets is consistent with closing the approach error. Deactivation at C is prescribed, not learned phase discovery. |
| **Approach orientation** | **6.07 → 10.02 → 8.55**, then zero. | The hands must acquire appropriate orientations before pushing. Their reported orientation errors decrease during approach, supporting this interpretation; switching the term off is programmed. |
| **End-effector distance** | Push-only: **14.0 → 16.9 → 3.87 → 11.6**. | The spacing penalty relaxes late and then recovers. This permits a changing coordination trade-off; it should not be described as maintaining perfectly constant spacing. |
| **Tray-to-goal position** | Push-only: **8.67 → 6.66 → 21.1 → 12.9**. | Stronger weighting in the later push is consistent with emphasizing remaining positional goal error. The tray does approach its goal, but the curve alone cannot establish the weight's causal role. |
| **Tray-to-goal orientation** | Push-only: **16.7 → 13.5 → 23.8 → 22.5**. | Increasing late orientation weighting is consistent with resisting pose error while transporting the tray. The realized orientation error nevertheless increases slightly, so do not claim that the higher coefficient demonstrably improves alignment. |
| **Contact-maintenance position** | **121 → 149 → 18.3 → 30.0**. | Its late reduction is consistent with relaxing the hand–tray positional relationship in favor of continued goal progress. The reported grasp-position error grows, so “less weight because contact is already perfect” is not supported. This does not prove that reduced weighting caused the deviation. |
| **Contact-maintenance orientation** | **25.0 → 16.8 → 46.3 → 28.5**. | The late peak is consistent with renewed emphasis on orienting the hands relative to the tray as their geometry changes. The reported grasp-orientation error still grows overall; a stronger penalty is not proof of successful correction. |

### Concise wording for the paper

> During Tray Push, acquisition terms are replaced by the prescribed transport and contact-maintenance objectives. The later push exhibits lower Contact-maintenance position weighting alongside stronger goal-pose weighting and continued positional progress. However, reported hand–tray grasp errors increase over the same phase. The episode therefore illustrates a possible transport–maintenance trade-off, not uniformly improving grasp quality or a demonstrated causal effect of a particular coefficient.

## Scope, physical validity, and provenance

- These are selected illustrative episodes, not averages or representative statistical trends over all runs. Apparent changes are not claims of statistical significance.
- The transport collision masks exclude contacts involving the manipulated ball/box/tray geoms. Collision-weight plots therefore do not certify prevention of robot–object penetration.
- Existing audits report maximum robot–object depths of **34.88 mm (Ball)**, **23.20 mm (Box)**, and **14.82 mm (Tray)** for these episodes. “Recorded success” must not be rewritten as “penetration-free” or “physically validated.”
- Predicted cost contributions, candidate comparisons, or controlled re-evaluations would be needed to test which weight changes actually cause different actions. In particular, some changes may be state-correlated side effects of a jointly trained actor rather than independently useful adaptations.
- Saved evaluator diagnostics are not interchangeable with CEM horizon features. For example, the Box `cost_obj_to_targ` log is pre-step; the goal-distance evidence here is computed from saved post-step positions instead. Ball's reward diagnostic and transport proxy also differ as described above.

Frozen planner definitions used for interpretation:

| Task | Branch lineage | Commit | Cost definition |
|---|---|---|---|
| Ball Lift | `issue_49_ball_lift_ppo` | `f4202cc` | `real_demo/sampling_based_planner/mjx_planner.py`, lines 735–845 |
| Box Lift | `issue_50_box_lift_ppo` | `00898ab` | Same path, lines 729–864 |
| Tray Push | `issue_53_tray_push_ppo` | `6117e21` | Same path, lines 687–810 |

The [generator](make_ars_weights_snapshots.py) reads the current notebook's source paths without executing the notebook. Exact NPZ/video identities, checkpoint hashes, phase rules, plotted policy-weight summaries, actual CEM coefficients, snapshot diagnostics, and penetration-audit references are recorded in [sources.json](ars_weights_snapshots/sources.json). The full per-step CSVs remain beside the three images. Snapshot diagnostics were also checked against the historical evaluators/runners at the commits above; the exact figure labels are inherited from the notebook.

Regenerate the figures and numerical provenance with:

```bash
/home/aks-lab/manipulator_env/bin/python3 make_ars_weights_snapshots.py --overwrite
```

This prose is a reviewed explanation of the pinned episodes, not automatically regenerated text; changing episodes requires reviewing the explanation again. A matching standalone LaTeX draft is provided in [ARS_WEIGHT_EPISODE_EXPLANATIONS.tex](ARS_WEIGHT_EPISODE_EXPLANATIONS.tex).
