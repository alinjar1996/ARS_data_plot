# Box-lift selected penetration evidence

Only two figures are selected for Git (approximately 0.83 MB combined). The complete original extraction is preserved in `archive/`, excluded by the repository `.gitignore`. No original files were deleted.

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

## Figure 2: substantial overlap in successful robot–box episodes

[Open successful-episode comparison](box_robot_successful_top3.png)

These three distinct successful episodes show that substantial robot–box overlap is not restricted to failed trials. They illustrate extreme cases, not the typical successful episode. Panels are ordered left to right by maximum depth.

| Panel | Depth (mm) | Method / batch / seed / episode | Playback time |
|---|---:|---|---:|
| Left | 55.60 | ARS / 150 / 4 / 0006 | 6.6 s |
| Middle | 52.88 | PPO / 150 / 4 / 0016 | 10.8 s |
| Right | 51.65 | ARS / 150 / 4 / 0007 | 7.1 s |

## Provenance and limitations

Audit scope: 1,200 episodes in 60 Box Lift NPZ files, batches 50/150/250, four methods, including the newer Bayesian 5001 run; 305 successful episodes. Contacts were reconstructed at 236,549 saved pre-step configurations using MuJoCo 3.3.1 and `issue_50_box_lift_ppo` (model snapshot `71f381f`).

Depth means the magnitude of negative MuJoCo contact distance, not a pixel measurement or calibrated physical deformation. Maxima cover saved states only; contact regions can be occluded by the camera. These selected snapshots illustrate extrema; prevalence is established by the counts in the findings summary.

Images were extracted from original videos, not synthesized. Headers were added above the original frames. The overview is JPEG-compressed; the successful-episode comparison is PNG. Episode/frame indices are zero-based. Saved pre-step state `s` corresponds to video frame `s−1`; timestamps are video playback times.

All selected panels use videos under `eval_box_lift_all_methods_b50_150_250_20260924_190610/videos/`. The following paths are relative to that directory. Videos are local data and are not included in Git.

| Figure / panel | Source video |
|---|---|
| Overview / top left | `handtuned_batch50/20260924_192044/handtuned_batch50_seed4_ep0017_fall.mp4` |
| Overview / top right; successful comparison / left | `ars4218_batch150/20260925_004542/ars4218_batch150_seed4_ep0006_success.mp4` |
| Overview / middle left | `ars4218_batch150/20260925_004136/ars4218_batch150_seed5_ep0017_fall.mp4` |
| Overview / middle right | `handtuned_batch50/20260924_192044/handtuned_batch50_seed4_ep0007_success.mp4` |
| Overview / bottom left | `handtuned_batch250/20260924_201051/handtuned_batch250_seed0_ep0008_fall.mp4` |
| Overview / bottom right | `handtuned_batch250/20260924_201051/handtuned_batch250_seed0_ep0018_success.mp4` |
| Successful comparison / middle | `ppo5005besteval_batch150/20260924_232346/ppo5005besteval_batch150_seed4_ep0016_success.mp4` |
| Successful comparison / right | `ars4218_batch150/20260925_004542/ars4218_batch150_seed4_ep0007_success.mp4` |
