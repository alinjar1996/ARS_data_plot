# Box Lift: contact-penetration findings

Date: 2026-09-29

## Summary

Substantial robot–box overlap occurs in the saved simulations, including episodes marked successful. Of 305 successful episodes, 292 (95.7%) exceed 20 mm, 239 (78.4%) exceed 30 mm, and 116 (38.0%) exceed 40 mm of robot–box contact-depth magnitude. Eight successful episodes reach at least 50 mm.

These results establish a simulation-to-reality limitation, not a material specification: a softer box might accommodate some deformation, but these measurements do not demonstrate that changing the box material would reproduce the simulated motions or grasp success.

## Data and measurement scope

- Task/model: Box Lift, `issue_50_box_lift_ppo`, model snapshot commit `71f381f`.
- Inputs: 60 NPZ files selected for Box Lift in [benchmark_stat_batch_sm.ipynb](benchmark_stat_batch_sm.ipynb), covering 1,200 episodes and 236,549 saved configurations.
- Methods: Hand-Tuned, Bayesian, PPO, and ARS; 300 episodes per method across batches 50, 150, and 250.
- Recorded successes: 305/1,200 (25.4%). “Successful” refers to the recorded task outcome, not a contact-validity assessment.
- Hand-Tuned/PPO/ARS run: `eval_box_lift_all_methods_b50_150_250_20260924_190610`.
- Bayesian run: `eval_box_lift_bayesian5001_b50_150_250_20260926_092000`; older Bayesian results are not included.
- Contacts were reconstructed from saved pre-step configurations using native MuJoCo 3.3.1 and the task-specific model. This is a geometric audit, not a rerun of the original MJX dynamics.

Here, **depth** means the magnitude of a negative MuJoCo collision-contact distance, expressed in millimeters. It is not a pixel-based estimate, measured hardware deformation, or necessarily the minimum translation needed to separate the complete objects. Collision geometry and rendered surfaces can differ.

Counts below use each episode’s maximum depth over saved states. Each episode is counted once per threshold; thresholds are cumulative, not disjoint bands. Unsaved intermediate and final post-step states may contain additional overlap. Counts do not measure penetration duration or contact force. No initialization window has been excluded from these statistics.

## Robot–box penetration

| Episode maximum | All episodes (n=1,200) | Successful episodes (n=305) |
|---|---:|---:|
| >20 mm | 864 (72.0%) | 292 (95.7%) |
| >30 mm | 781 (65.1%) | 239 (78.4%) |
| >40 mm | 620 (51.7%) | 116 (38.0%) |
| ≥50 mm | 423 (35.3%) | 8 (2.6%) |
| ≥52 mm | 368 (30.7%) | 2 (0.7%) |
| ≥55 mm | 289 (24.1%) | 1 (0.3%) |
| ≥56 mm | 258 (21.5%) | 0 (0.0%) |

The distinction between strict `>` and inclusive `≥` matches the thresholds used in the discussion.

### Successful episodes by method

| Method | Successful episodes | >20 mm | >30 mm | >40 mm | ≥50 mm |
|---|---:|---:|---:|---:|---:|
| Hand-Tuned | 21 | 20 | 20 | 13 | 0 |
| Bayesian | 41 | 40 | 26 | 9 | 0 |
| PPO | 57 | 57 | 47 | 17 | 1 |
| ARS | 186 | 175 | 146 | 77 | 7 |

These are counts within each method’s successful subset, whose sizes differ; they are not normalized method comparisons. The observed association between penetration and success does not establish that penetration caused success.

## Table contacts and initialization

### Box–table

All 1,200 episodes, including all 305 successes, exceed 20 mm at some saved state. The largest successful-episode depth is 24.69 mm during initial drop/settling, at approximately 0.2 s of video playback.

For interpreting manipulation performance, initialization overlap should be reported separately under a consistent, explicitly defined settling cutoff. Such a cutoff has **not** been applied here, and post-settling threshold counts have not been calculated.

Not all box–table overlap is initialization: the top three all-episode maxima are 86.21, 58.49, and 54.46 mm at playback times 12.6, 19.2, and 18.7 s, respectively. Those later events cannot be dismissed as initial loading.

### Robot–table

Robot–table depth exceeds 5 mm in 18/1,200 episodes (1.5%), including 2/305 successful episodes (0.7%). Maximum depth is 17.60 mm overall and 8.83 mm among successful episodes. These values are smaller than robot–box maxima, but smaller overlap does not establish physical acceptability against a rigid table.

## Selected visual evidence

Only two figures are retained for Git (approximately 0.83 MB combined):

1. [Maximum-penetration overview](box_lift_penetration_snapshots/maximum_penetrations_overview.jpg): six labeled panels comparing all-episode and successful-episode maxima for robot–box, box–table, and robot–table contacts. It includes late box–table overlap and distinguishes it from initial settling.
2. [Top three successful robot–box episodes](box_lift_penetration_snapshots/box_robot_successful_top3.png): 55.60, 52.88, and 51.65 mm. These ARS/PPO examples demonstrate that substantial overlap is not restricted to failed episodes; they are extreme examples, not typical successful episodes.

The full top-three numerical results remain below without duplicating their figures. Each set contains distinct episodes, ranked by episode maximum; successful-only and all-episode sets may overlap.

| Contact | All episodes: top three depths (mm) | Successful episodes: top three depths (mm) |
|---|---|---|
| Robot–box | 75.97, 73.91, 73.29 | 55.60, 52.88, 51.65 |
| Box–table | 86.21, 58.49, 54.46 | 24.69, 24.69, 24.69 |
| Robot–table | 17.60, 12.01, 9.48 | 8.83, 5.97, 3.43 |

The successful box–table examples are tied initialization cases; three representative episodes were selected using a deterministic tie-break, not three uniquely larger events.

See the [selected-evidence index](box_lift_penetration_snapshots/README.md) for panel identities, timestamps, and local video references. Images are extracted from the original videos, not synthesized; only headers were added. Contact regions may be occluded by the camera. The complete original extraction, including raw frames, full top-three sets, and JSON metadata, is preserved in `box_lift_penetration_snapshots/archive/`, which is excluded from Git. Videos are also local data, not included in Git.

Video timestamps are playback times; episode and frame indices are zero-based. Saved pre-step state `s` corresponds to video frame `s−1`, because frames were recorded after simulation steps.

## Physical interpretation

MuJoCo permits overlap through its soft-contact formulation; contact response depends on solver parameters such as `solref` and `solimp`. That does not automatically make geometric overlap a calibrated model of real material deformation. See [MuJoCo’s contact/solver documentation](https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters).

The current results should therefore be described as successful task outcomes **under the existing simulation contact/execution setup**, not demonstrated physically admissible grasps. This audit does not prove that the observed overlap is unavoidable in other configurations.

A softer box or compliant contact surface is a hypothesis to validate, not an established fix. In particular, the approximately 52–56 mm contact depths in the most extreme successful robot–box episodes cannot be translated directly into a required foam thickness or material stiffness. Physical validation would require force–deformation behavior, contact geometry, and grasp stability to be checked with a matching model or measurements.

This summary adds documentation only; it does not change the notebook, simulation code, models, or benchmark data.
