# Box Lift current contact penetration findings

Date: 2026-10-02. Replaces the September-run report; the old report is preserved in the ignored evidence archive.

## Summary

Audited all **1,200 episodes**, including **300 successes and 900 failures**, at **252,200 distinct recorded states**, including every terminal post-step pose. Questionable robot–box overlap remains in successful episodes. The three largest successful depths are **57.85, 52.49, and 52.03 mm**.

No successful episode has detected same-arm contact or upper-arm/forearm–box overlap. Five successful episodes do contain contact between the two different arms. Failure statistics below are numerical only; all refreshed snapshots show successful episodes.

[Top three successful cases](box_lift_penetration_snapshots/successful_worst_cases_top3.jpg) · [Successful initial-versus-later overview](box_lift_penetration_snapshots/maximum_penetrations_overview.jpg) · [Video links and panel index](box_lift_penetration_snapshots/README.md)

## Input and model scope

- Hand-Tuned, PPO5005 best-evaluation and ARS4218: `eval_box_lift_all_contacts_b50_150_250_20261001_154416_3370028/`.
- Bayesian5001: `eval_box_lift_bayesian5001_b50_150_250_after_20261001_154416/`. Bayesian5002 and PPO best-training are excluded.
- Each method has 300 episodes: batches 50/150/250, five files per batch, 20 episodes per file. The 60 aggregate NPZs are selected directly from the notebook; individual episode exports are not counted again.
- All recorded model XML, evaluator, runner and planner hashes inspected agree between the two run sources and the frozen audit copy. The all-contacts run records launch commit `00898abec9be43509d97c03e021673ef354e78ea`; recorded content hashes, not the current checkout alone, establish reconstruction provenance.
- MuJoCo 3.3.1; timestep 0.1 s. Reconstructed native contacts from recorded full qpos; no physical rollout was rerun. Consecutive pre-step qpos exactly match preceding post-step qpos. The final post-step state is added once per episode.
- All 336 checked relevant nonadjacent robot–robot, robot–box and robot–table pairs are collision-enabled, and the model has no explicit body-pair exclusions. Same-rigid-body and directly adjacent same-arm link pairs are not an independent self-intersection audit. Unmodeled surfaces remain outside scope.
- The manipulated box collision geom is named `ball`. The tables are fixed; the Tray task's missing table-slider-state limitation does not apply here.
- Successful NPZ episodes have `success=1` but `reason="na"`; classification uses the success flag, and the matched video filename says `success`. Failed reason fields agree with video outcomes.

## Episode penetration counts

Counts use each episode's maximum across saved states. Thresholds are strictly greater than the stated depth and are cumulative. No initial time window is excluded in this table.

| Contact | Threshold mm | All 1200 | Successful 300 | Failed 900 |
|---|---:|---:|---:|---:|
| Robot–box | >5 | 954 | 300 | 654 |
| Robot–box | >10 | 899 | 300 | 599 |
| Robot–box | >20 | 865 | 294 | 571 |
| Robot–box | >30 | 786 | 243 | 543 |
| Robot–box | >40 | 620 | 120 | 500 |
| Robot–box | >50 | 444 | 7 | 437 |
| Box–table | >0 | 1200 | 300 | 900 |
| Box–table | >5 | 1200 | 300 | 900 |
| Box–table | >20 | 1200 | 300 | 900 |
| Robot–table | >0 | 167 | 7 | 160 |
| Robot–table | >5 | 28 | 1 | 27 |
| Robot–table | >20 | 1 | 0 | 1 |
| Between arms | >0 | 398 | 5 | 393 |
| Between arms | >5 | 216 | 0 | 216 |
| Between arms | >20 | 18 | 0 | 18 |
| Same arm | >0 | 12 | 0 | 12 |
| Same arm | >5 | 4 | 0 | 4 |
| Same arm | >20 | 0 | 0 | 0 |
| Upper arm or forearm–box | >0 | 121 | 0 | 121 |
| Upper arm or forearm–box | >5 | 110 | 0 | 110 |
| Upper arm or forearm–box | >20 | 55 | 0 | 55 |

| Contact | Largest successful depth mm | Largest failed depth mm |
|---|---:|---:|
| Robot–box | 57.850 | 83.753 |
| Box–table | 24.689 | 91.882 |
| Robot–table | 5.214 | 33.925 |
| Between arms | 4.645 | 43.273 |
| Same arm | 0.000 | 15.796 |
| Upper arm or forearm–box | 0.000 | 76.472 |

## Initialization and later table overlap

All 1,200 episodes exceed 20 mm of box–table overlap during initial loading. The successful whole-episode maximum is 24.689 mm at video 0.2 s (recorded state time 0.3 s). These tied initial-drop events are labeled separately in the overview.

The main top-three figure instead ranks successful box–table overlap after recorded state time 1.0 s. The top depths are 12.120, 12.061, and 10.884 mm, at playback times 3.7, 3.2, and 5.9 s respectively. The cutoff is a sensitivity choice, not a physical settling detector. Robot–box and robot–table rows retain whole-episode maxima.

## Method counts

| Method | Successes / 300 | Successful robot–box >20 mm | >30 mm | >40 mm |
|---|---:|---:|---:|---:|
| Hand-Tuned | 22 | 22 | 22 | 13 |
| Bayesian | 42 | 41 | 33 | 6 |
| PPO | 55 | 54 | 47 | 21 |
| ARS | 181 | 177 | 141 | 80 |

## Interpretation and evidence limits

A success flag is the task outcome, not a contact-validity judgment. Depths are magnitudes of negative collision-contact distances, not force, calibrated material deformation, or necessarily the translation needed to separate complete bodies. The new contact masks do not by themselves establish physically admissible grasps. Softer material is not demonstrated as a remedy.

Same-arm overlap occurs in failed episodes but not in the 300 successful episodes. All successful robot–robot examples are between different arms. Likewise, upper-arm/forearm–box overlap occurs only in failures; successful robot–box extrema shown here involve the grippers.

These results cover recorded states, including terminal poses, but not unsaved intermediate integration states. Box/Tray-style repeated block seeds should not be described as 100 independent randomized starts. Differences from the historical audit reflect both new runs/model contacts and expanded terminal-state coverage.

All selected snapshot videos were fully decoded and matched to NPZ step counts; all 1,200 episode video paths are unique and present. Selected depths and camera alignment were independently checked. This is not a claim that every video was manually watched or fully decoded. Only two conclusive combined images remain selected for Git; all detailed evidence is in `box_lift_penetration_snapshots/archive/current_runs_20261002/`.

No notebook calculations or data selections, simulator source, source model, checkpoint, or benchmark NPZ was changed by this audit. Only evidence outputs, documentation, and the notebook audit-status note were updated.
