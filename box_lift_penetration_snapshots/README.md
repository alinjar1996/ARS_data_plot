# Box-lift selected penetration evidence

Only two figures are selected for Git (approximately 1.27 MB combined). The complete original extraction is preserved in `archive/`, excluded by the repository `.gitignore`. No original files were deleted.

See the [findings summary](../BOX_LIFT_PENETRATION_FINDINGS.md) for threshold counts, methodology, and physical interpretation.

## Figure 1: maximum-penetration overview

[Open six-panel overview](maximum_penetrations_overview.jpg)

Rows show robot–box, box–table, and robot–table contacts. Columns show all-episode maxima (left) and successful-episode maxima (right). This consolidates the maximum evidence across all three contact categories, including late box–table overlap versus initial settling.

| Panel | Contact / scope | Depth (mm) | Method / batch / seed / episode | Playback time |
|---|---|---:|---|---:|
| Top left | Robot–box / all | 75.97 | Hand-Tuned / 50 / 4 / 0017 | 14.0 s |
| Top right | Robot–box / successful | 55.60 | ARS / 150 / 4 / 0006 | 6.6 s |
| Middle left | Box–table / all | 86.21 | ARS / 150 / 5 / 0017 | 12.6 s |
| Middle right | Box–table / successful | 24.69 | Hand-Tuned / 50 / 4 / 0007 | 0.2 s |
| Bottom left | Robot–table / all | 17.60 | Hand-Tuned / 250 / 0 / 0008 | 9.7 s |
| Bottom right | Robot–table / successful | 8.83 | Hand-Tuned / 250 / 0 / 0018 | 4.9 s |

## Figure 2: top three worst successful episodes for every observed contact category

[Open combined twelve-panel figure](successful_worst_cases_top3.jpg)

Rows show robot–box, box–table, robot–table, and robot–robot (self-contact). Columns show the worst, second-worst, and third-worst distinct successful episodes **within each contact category**, ranked by maximum depth over the saved episode states, not just at task completion. These are extreme examples, not typical successful episodes.

The box–table row contains tied **initial-drop/settling** events at 0.2 s; it is not evidence of manipulation-phase table overlap. No settling cutoff was applied. Equal depths use source NPZ path and episode index as deterministic tie-breakers. The audit's remaining `other` contact category has no negative contact distances in the saved states, so there is no additional nonzero category to illustrate. This is coverage of collision contacts reconstructed under the selected model, not proof that disabled/unmodeled collision pairs cannot intersect.

| Contact | Rank | Depth (mm) | Method / batch / seed / episode | Playback time | Geom pair | Source data / video (local) |
|---|---:|---:|---|---:|---|---|
| Robot–box | 1 | 55.603 | ARS / 150 / 4 / 0006 | 6.6 s | `robot_0 / ball` | [NPZ](../eval_box_lift_all_methods_b50_150_250_20260924_190610/npz/ars4218_batch150/eval_ars4218_batch150_n20_seed4.npz), [MP4](../eval_box_lift_all_methods_b50_150_250_20260924_190610/videos/ars4218_batch150/20260925_004542/ars4218_batch150_seed4_ep0006_success.mp4) |
| Robot–box | 2 | 52.879 | PPO / 150 / 4 / 0016 | 10.8 s | `robot_02 / ball` | [NPZ](../eval_box_lift_all_methods_b50_150_250_20260924_190610/npz/ppo5005besteval_batch150/eval_ppo5005besteval_batch150_n20_seed4.npz), [MP4](../eval_box_lift_all_methods_b50_150_250_20260924_190610/videos/ppo5005besteval_batch150/20260924_232346/ppo5005besteval_batch150_seed4_ep0016_success.mp4) |
| Robot–box | 3 | 51.647 | ARS / 150 / 4 / 0007 | 7.1 s | `robot_02 / ball` | [NPZ](../eval_box_lift_all_methods_b50_150_250_20260924_190610/npz/ars4218_batch150/eval_ars4218_batch150_n20_seed4.npz), [MP4](../eval_box_lift_all_methods_b50_150_250_20260924_190610/videos/ars4218_batch150/20260925_004542/ars4218_batch150_seed4_ep0007_success.mp4) |
| Box–table | 1 (tied) | 24.689 | ARS / 150 / 10 / 0002 | 0.2 s | `table_geom_0 / ball` | [NPZ](../eval_box_lift_all_methods_b50_150_250_20260924_190610/npz/ars4218_batch150/eval_ars4218_batch150_n20_seed10.npz), [MP4](../eval_box_lift_all_methods_b50_150_250_20260924_190610/videos/ars4218_batch150/20260925_004931/ars4218_batch150_seed10_ep0002_success.mp4) |
| Box–table | 2 (tied) | 24.689 | ARS / 150 / 10 / 0006 | 0.2 s | `table_geom_0 / ball` | [NPZ](../eval_box_lift_all_methods_b50_150_250_20260924_190610/npz/ars4218_batch150/eval_ars4218_batch150_n20_seed10.npz), [MP4](../eval_box_lift_all_methods_b50_150_250_20260924_190610/videos/ars4218_batch150/20260925_004931/ars4218_batch150_seed10_ep0006_success.mp4) |
| Box–table | 3 (tied) | 24.689 | ARS / 150 / 10 / 0017 | 0.2 s | `table_geom_0 / ball` | [NPZ](../eval_box_lift_all_methods_b50_150_250_20260924_190610/npz/ars4218_batch150/eval_ars4218_batch150_n20_seed10.npz), [MP4](../eval_box_lift_all_methods_b50_150_250_20260924_190610/videos/ars4218_batch150/20260925_004931/ars4218_batch150_seed10_ep0017_success.mp4) |
| Robot–table | 1 | 8.834 | Hand-Tuned / 250 / 0 / 0018 | 4.9 s | `robot_01 / table_geom_0` | [NPZ](../eval_box_lift_all_methods_b50_150_250_20260924_190610/npz/handtuned_batch250/eval_handtuned_batch250_n20_seed0.npz), [MP4](../eval_box_lift_all_methods_b50_150_250_20260924_190610/videos/handtuned_batch250/20260924_201051/handtuned_batch250_seed0_ep0018_success.mp4) |
| Robot–table | 2 | 5.974 | Bayesian / 250 / 4 / 0007 | 6.4 s | `robot_02 / table_geom_0` | [NPZ](../eval_box_lift_bayesian5001_b50_150_250_20260926_092000/npz/bayesian5001_batch250/eval_bayesian5001_batch250_n20_seed4.npz), [MP4](../eval_box_lift_bayesian5001_b50_150_250_20260926_092000/videos/bayesian5001_batch250/20260926_130700/bayesian5001_batch250_seed4_resume_20260926_130640_ep0007_success.mp4) |
| Robot–table | 3 | 3.433 | ARS / 250 / 0 / 0007 | 5.7 s | `robot_02 / table_geom_0` | [NPZ](../eval_box_lift_all_methods_b50_150_250_20260924_190610/npz/ars4218_batch250/eval_ars4218_batch250_n20_seed0.npz), [MP4](../eval_box_lift_all_methods_b50_150_250_20260924_190610/videos/ars4218_batch250/20260925_010015/ars4218_batch250_seed0_ep0007_success.mp4) |
| Robot–robot (self-contact) | 1 | 3.984 | Bayesian / 250 / 15 / 0016 | 4.0 s | `robot_01 / robot_103` | [NPZ](../eval_box_lift_bayesian5001_b50_150_250_20260926_092000/npz/bayesian5001_batch250/eval_bayesian5001_batch250_n20_seed15.npz), [MP4](../eval_box_lift_bayesian5001_b50_150_250_20260926_092000/videos/bayesian5001_batch250/20260926_131754/bayesian5001_batch250_seed15_resume_20260926_131734_ep0016_success.mp4) |
| Robot–robot (self-contact) | 2 | 3.290 | Bayesian / 250 / 15 / 0018 | 4.1 s | `robot_01 / robot_103` | [NPZ](../eval_box_lift_bayesian5001_b50_150_250_20260926_092000/npz/bayesian5001_batch250/eval_bayesian5001_batch250_n20_seed15.npz), [MP4](../eval_box_lift_bayesian5001_b50_150_250_20260926_092000/videos/bayesian5001_batch250/20260926_131754/bayesian5001_batch250_seed15_resume_20260926_131734_ep0018_success.mp4) |
| Robot–robot (self-contact) | 3 | 3.102 | Bayesian / 250 / 15 / 0013 | 4.8 s | `robot_01 / robot_103` | [NPZ](../eval_box_lift_bayesian5001_b50_150_250_20260926_092000/npz/bayesian5001_batch250/eval_bayesian5001_batch250_n20_seed15.npz), [MP4](../eval_box_lift_bayesian5001_b50_150_250_20260926_092000/videos/bayesian5001_batch250/20260926_131754/bayesian5001_batch250_seed15_resume_20260926_131734_ep0013_success.mp4) |

The model calls the manipulated box collision geom `ball`; the task is still Box Lift. The previous robot–box-only selected figure remains available locally in `archive/box_robot_successful_top3_selected.png`.

## Provenance and limitations

Audit scope: 1,200 episodes in 60 Box Lift NPZ files, batches 50/150/250, four methods, including the newer Bayesian 5001 run; 305 successful episodes. Contacts were reconstructed at 236,549 saved pre-step configurations using MuJoCo 3.3.1 and `issue_50_box_lift_ppo` (model snapshot `71f381f`).

Depth means the magnitude of negative MuJoCo contact distance, not a pixel measurement or calibrated physical deformation. Maxima cover saved states only; contact regions can be occluded by the camera. These selected snapshots illustrate extrema; prevalence is established by the counts in the findings summary.

Images were extracted from original videos, not synthesized. Labels are placed outside the original frames; both selected figures are JPEG-compressed. All 12 selected depths were independently reproduced from their saved states using MuJoCo 3.3.1, and the nine previously extracted frames matched the archived originals pixel-for-pixel. Episode/frame indices are zero-based. Saved pre-step state `s` corresponds to video frame `s−1`; timestamps are video playback times.

The six overview panels use videos under `eval_box_lift_all_methods_b50_150_250_20260924_190610/videos/`. The following overview paths are relative to that directory. Successful top-three source links, including the newer Bayesian run, are in the table above. Videos are local data and are not included in Git.

| Figure / panel | Source video |
|---|---|
| Overview / top left | `handtuned_batch50/20260924_192044/handtuned_batch50_seed4_ep0017_fall.mp4` |
| Overview / top right | `ars4218_batch150/20260925_004542/ars4218_batch150_seed4_ep0006_success.mp4` |
| Overview / middle left | `ars4218_batch150/20260925_004136/ars4218_batch150_seed5_ep0017_fall.mp4` |
| Overview / middle right | `handtuned_batch50/20260924_192044/handtuned_batch50_seed4_ep0007_success.mp4` |
| Overview / bottom left | `handtuned_batch250/20260924_201051/handtuned_batch250_seed0_ep0008_fall.mp4` |
| Overview / bottom right | `handtuned_batch250/20260924_201051/handtuned_batch250_seed0_ep0018_success.mp4` |
