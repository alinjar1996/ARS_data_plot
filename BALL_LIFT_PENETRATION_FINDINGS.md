# Ball Lift: new-run penetration findings

Date: 2026-09-30

## Scope and result

Audited **1,200 episodes** from 60 NPZ files, including **711 successes and 489 failures** (487 timeouts, 2 falls). Reconstructed contacts at **215,716 distinct saved states**: each initial state plus every post-step state, including the terminal state. Consecutive pre-step and preceding post-step qpos/qvel matched exactly.

Run: `eval_sweep_100_ball_lift_bayes_retrained_20260929_191418`. Methods: Hand-Tuned, retrained Bayesian, **PPO4951 best-evaluation**, ARS3950; batches 50/150/250, five 20-episode blocks per method/batch. Best-training PPO is excluded from all statistics and snapshots below. Best-evaluation was explicitly confirmed by the user and matches current notebook source paths, despite stale best-train prose/legends. No notebook change was made.

**The penetration-threshold statistics below describe failed episodes; all selected snapshots are from successful episodes.**

[Open the twelve-panel successful-only snapshots](ball_lift_penetration_snapshots/successful_worst_cases_top3.jpg) · [Panel identities and original-video links](ball_lift_penetration_snapshots/README.md)

## Penetration statistics for failures

Each entry counts failed episodes whose maximum depth strictly exceeds the threshold. Thresholds are cumulative, and one episode may contribute to multiple contact categories. The denominator is 489 failures, not all episodes or all time steps. No initialization window is excluded.

| Contact | >5 mm | >10 mm | >20 mm | >30 mm | >40 mm | >50 mm | Maximum (mm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Robot–ball | 30/489 (6.1%) | 18/489 (3.7%) | 16/489 (3.3%) | 14/489 (2.9%) | 6/489 (1.2%) | 2/489 (0.4%) | 51.542 |
| Ball–table | 489/489 (100.0%) | 489/489 (100.0%) | 138/489 (28.2%) | 6/489 (1.2%) | 3/489 (0.6%) | 2/489 (0.4%) | 91.110 |
| Robot–table | 5/489 (1.0%) | 3/489 (0.6%) | 0/489 (0.0%) | 0/489 (0.0%) | 0/489 (0.0%) | 0/489 (0.0%) | 11.976 |
| Robot–robot | 2/489 (0.4%) | 0/489 (0.0%) | 0/489 (0.0%) | 0/489 (0.0%) | 0/489 (0.0%) | 0/489 (0.0%) | 6.150 |

### Failed episodes by method

| Method | Failures / evaluated | Timeout / fall | Robot–ball >20 mm | Ball–table >20 mm | Robot–table >5 mm | Robot–robot >5 mm |
|---|---:|---|---:|---:|---:|---:|
| Hand-Tuned | 285/300 | 285 / 0 | 0/285 (0.0%) | 113/285 (39.6%) | 1/285 (0.4%) | 0/285 (0.0%) |
| Bayesian | 191/300 | 191 / 0 | 6/191 (3.1%) | 13/191 (6.8%) | 4/191 (2.1%) | 2/191 (1.0%) |
| PPO (best eval) | 9/300 | 9 / 0 | 6/9 (66.7%) | 8/9 (88.9%) | 0/9 (0.0%) | 0/9 (0.0%) |
| ARS | 4/300 | 2 / 2 | 4/4 (100.0%) | 4/4 (100.0%) | 0/4 (0.0%) | 0/4 (0.0%) |

### Worst failed case in each category (numerical only)

| Contact | Depth (mm) | Method / batch / seed / episode | Outcome | Video time | Terminal state? |
|---|---:|---|---|---:|---|
| Robot–ball | 51.542 | ARS / 150 / 3 / 0014 | fall | 7.2 s | no |
| Ball–table | 91.110 | ARS / 250 / 0 / 0007 | fall | 10.0 s | yes |
| Robot–table | 11.976 | Hand-Tuned / 250 / 0 / 0003 | timeout | 20.8 s | no |
| Robot–robot | 6.150 | Bayesian / 250 / 0 / 0007 | timeout | 3.8 s | no |

The two failed episodes with ball–table depth >50 mm are ARS falls. Their episode maxima are 91.11 mm, 76.43 mm. Terminal states are included in this audit; the previous historical report used pre-step-only states, so its coverage differs as well as its inputs.

The failures are dominated by timeouts; these counts do not establish that penetration caused failure or that lower failure penetration implies better physical behavior. Ball–table statistics include initial contact/settling as well as later manipulation.

### Initial-window sensitivity

Excluding states before simulation time 1.0 s changes the following failure counts (full episode → after cutoff):

No threshold memberships change for the listed thresholds/contact categories. This sensitivity cutoff does not establish that settling is complete after one second.

## Successful-only top-three snapshots

| Contact | Rank 1 depth (mm) | Rank 2 depth (mm) | Rank 3 depth (mm) |
|---|---:|---:|---:|
| Robot–ball | 57.910 | 55.313 | 55.015 |
| Ball–table | 39.771 | 37.490 | 37.305 |
| Robot–table | 2.865 | 2.651 | 1.644 |
| Robot–robot | 3.412 | 3.083 | 2.515 |

All 12 panels are original video frames from distinct successful episodes within each row. They show episode extrema, not typical successes. The successful ball–table extrema occur at video times 3.2, 4.4, and 3.9 s, rather than at initial loading.

## Same-arm collision check in successful episodes

**No same-arm collision was detected in any of the 711 successful episodes (0/711)** in the audited saved states, including the terminal robot pose. Both the enabled MuJoCo contact check and a separate nonadjacent-link capsule-overlap check (including disabled collision pairs) found zero positive same-arm penetration in these successful episodes. Contacts within one rigid link and between directly adjacent joint links are excluded.

The successful robot–robot contacts shown above are **between the two different arms**, not between links of the same arm. This result applies to the recorded states and modeled collision geometry; it cannot rule out contact during unsaved intermediate motion or between unmodeled surfaces.

## Measurement and provenance

- Native MuJoCo 3.3.1 reconstructed enabled collision contacts from saved states; no dynamics rollout was rerun. Depth is the magnitude of negative contact distance in millimeters, not measured physical deformation or necessarily the minimum separation translation for complete bodies.
- The model XML files and evaluator/planner source hashes match the recording manifest and per-file XML metadata. The scene sphere radius is 0.11 m; timestep 0.1 s; recorded solver iterations 1 and line-search iterations 5.
- Per-episode randomized table-1 position and target position were restored, together with full qpos/qvel. No nominal-table substitution was used.
- The `other` category has no negative contact distances in any selected saved state. The main penetration audit includes enabled robot–robot contacts. The separate same-arm check above additionally tests disabled nonadjacent-link capsule pairs; unmodeled surfaces and unsaved substeps remain outside the audit.
- All 1,200 selected episodes have distinct existing recorded video paths. For the 12 selected panels, video frame counts match episode MPC-step counts and every depth was recomputed independently. Original-camera render/video mean absolute pixel differences range from 1.53 to 1.62 on a 0–255 scale. This is a selected-panel alignment check, not exhaustive video validation.
- The transparent cube is the target marker. Render meshes and collision shapes can differ; the original camera can occlude contacts.
- Detailed audit data and raw frames are in `ball_lift_penetration_snapshots/archive/new_runs_20260930/`. Prior-run reports and the old selected figure are preserved in the ignored parent archive. Only one combined evidence image remains selected for Git.

No notebook, simulator code, source model, or benchmark NPZ was changed.
