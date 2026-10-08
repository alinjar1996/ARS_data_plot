#!/usr/bin/env python3
"""Pair one successful ARS episode per task with its weights and video frames.

Run in the project environment (numpy, matplotlib, opencv-python, Pillow):
    python make_ars_weights_snapshots.py --dry-run
    python make_ars_weights_snapshots.py
    python make_ars_weights_snapshots.py --presentation
    python make_ars_weights_snapshots.py --rank
    python make_ars_weights_snapshots.py --tasks box --select box:50:0:4

Sources, paper labels and phase rules are read from benchmark_stat_batch_sm.ipynb
as literal configuration, WITHOUT executing the notebook. Defaults below were
selected from all 900 ARS episodes in its b50/150/250 sources. --rank reproduces
the quantitative shortlist using the existing, local-only penetration reports.
--select is TASK:BATCH:SEED:EPISODE, with ONE-based episode numbering.

Each figure has two blocks: four video frames (start, approach, recorded phase
transition, lift/push at successful end) above linear, independently scaled
panels for ALL phase-masked POLICY weights, with matching A-D time markers.
Fixed planner gains (including the lifting spacing gain) are NOT plotted.
Actual CEM coefficients remain available separately in the CSV/manifest.
Phase banners sit
outside the plotting area so no snapshot line runs through a phase label.
Labels stay OUTSIDE the photographs. Ball/Tray start at the first saved frame;
Box starts after the initial landing/rebound has settled (checked from saved
positions and velocities). Its plot starts then too, retaining absolute time.
Curves are never normalized, smoothed or reshaped to
create a desired trend. Within an active phase, larger weights increase the
penalty on the corresponding cost feature, all else equal; this does not by
itself prove a change in observed motion or a dominant total-cost contribution.
Weights at index k apply over [k*dt, (k+1)*dt); video frame k shows the resulting
POST-step state at (k+1)*dt. Video playback timestamps are not simulation time.
The transition is a controller proximity/alignment flag, not measured contact.

These are illustrative recorded successes, NOT causal ablations or physically
penetration-free demonstrations. All selected episodes have known penetration;
available audit values and selection criteria are retained in sources.json.
No physics reruns, model imports, Git checkouts, or input/notebook edits occur.
Only trusted local NPZs should be used (their ragged arrays require pickle).

--presentation writes separate *_presentation files with larger labels/ticks,
task-only headings and untimed snapshot captions. Episode/method identities and
exact times remain in the separate provenance, not on these figures. Existing
figures, CSVs and sources.json are untouched by this option.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np

from make_ars_video_snapshots import (
    DEFAULT_NOTEBOOK, TASKS, Episode, extract_frames, load_episodes,
    phase_frame_indices, read_sources, resolve_video, sha256,
)


# Explicit, reviewable choices. Indices in NPZ/video filenames may be zero-based.
DEFAULT_SELECTIONS = {"ball": (50, 4, 18), "box": (50, 5, 5), "tray": (250, 27, 4)}
FOCUS_WEIGHTS = {
    "ball": ("eef_to_obj", "obj_to_targ", "distance", "orientation"),
    "box": ("eef_to_obj", "obj_to_targ", "distance", "object_orientation"),
    "tray": ("position_pick", "position_tray", "orientation_tray", "position_move"),
}
SOURCE_REVISIONS = {
    "ball": ("issue_49_ball_lift_ppo", "f4202cc", "735-845"),
    "box": ("issue_50_box_lift_ppo", "00898ab", "729-864"),
    "tray": ("issue_53_tray_push_ppo", "6117e21", "687-810"),
}
PLANNER_PATH = "real_demo/sampling_based_planner/mjx_planner.py"
CHECKPOINT_HASHES = {
    "ball": "0d652353afc7855a7236a64b56c302c16bd5582e2b80a232302c3d93ed0779fd",
    "box": "65a9291e5ff106d5e1e9413c90241926f70e001b869b0d921f6c011a3b9ea09c",
}
AUDIT_PATHS = {
    "ball": "ball_lift_penetration_snapshots/archive/new_runs_20260930/contact_report.json",
    "box": "box_lift_penetration_snapshots/archive/current_runs_20261002/contact_report.json",
    "tray": "tray_push_penetration_snapshots/archive/contact_report.json",
}
TRAY_EXTRA_AUDITS = (
    ("same_arm_contact_report_20260930.json", "records"),
    ("arm_tray_all_pairs_report_20261001.json", "episodes"),
)
COST_MEANINGS = {
    "ball": {
        "collision": "Phase-specific predicted collision-clearance proxy; not penetration depth or contact force.",
        "theta": "Joint-trajectory deviation from the reference initial joint configuration.",
        "eef_vel": "Norm of relative EEF position dot relative EEF velocity: penalizes changing separation.",
        "eef_pos": "Norm of EEF y-z position differences: planar alignment.",
        "eef_to_obj": "Approach: EEF midpoint to ball reference, offset down by 0.05 m.",
        "obj_to_targ": "Lift: EEF midpoint plus 0.05 m z to target (a transport proxy, not direct ball position).",
        "distance": "EEF separation error relative to 0.17 m; coefficient multiplied by 10 during lift.",
        "orientation": "EEF quaternion-angle errors relative to fixed grasp orientations, both phases.",
        "smoothness": "Joint-acceleration trajectory norm; active in both phases.",
    },
    "box": {
        "collision": "Phase-specific predicted collision-clearance proxy; not penetration depth or contact force.",
        "theta": "Joint-trajectory deviation from the reference initial joint configuration.",
        "z-axis": "Norm of EEF y-z position differences, despite the historical z-axis key.",
        "velocity": "Norm of relative EEF position dot relative EEF velocity: penalizes changing separation.",
        "orientation": "Both phases: EEF quaternion-angle errors relative to box-attached grasp-site orientations.",
        "eef_to_obj": "Approach: mean EEF-to-box-attached grasp-site distance (cost_g_move despite its name).",
        "obj_to_targ": "Lift: EEF-midpoint-to-goal PLUS EEF-to-grasp-site contact-maintenance distance.",
        "distance": "EEF separation error relative to desired spacing; coefficient multiplied by 10 during lift.",
        "object_orientation": "Lift: norm of box-to-target quaternion difference (not an angle in radians).",
        "smoothness": "Stored policy output, but absent from this revision's cost sum; always masked to zero.",
    },
    "tray": {
        "collision": "Phase-specific predicted collision-clearance proxy; not penetration depth or contact force.",
        "theta": "Joint-trajectory deviation from the reference initial joint configuration.",
        "eef_height_level": "Both phases: norm of the two EEF z-coordinate differences.",
        "rot_axis_tray_alignment": "Push: relative EEF position projected onto the tray rotation axis.",
        "eef_tray_rigid_body_velocity": "Push: norm of relative EEF position dot relative EEF velocity.",
        "orientation_pick": "Approach: EEF quaternion-angle errors relative to the grasp targets.",
        "distance": "Push: squared EEF separation error relative to the current grasp-point spacing.",
        "position_pick": "Approach: mean EEF distance to tray grasp targets.",
        "position_tray": "Push: tray-to-target position error over the predicted horizon.",
        "orientation_tray": "Push: summed tray-to-target quaternion-angle errors.",
        "position_move": "Push: mean EEF-to-moving-tray-grasp-site distance (contact maintenance).",
        "orientation_move": "Push: EEF quaternion-angle errors relative to moving tray-attached grasp orientations.",
    },
}
LIMITATIONS = (
    "Weights are coefficients, not weighted cost contributions; comparing their magnitudes does not establish cost dominance.",
    "Phase switches and the lift spacing factor are programmed gates, not changes learned by the actor.",
    "A temporal association between weights and motion is not evidence that a weight change caused the motion.",
    "The phase flag is a proximity/alignment transition, not proof of first physical contact.",
    "Recorded success does not establish physical validity; selected episodes retain known contact penetration.",
    "Tray table overlaps in the older audit are fixed-table reference overlaps, not confirmed dynamic table contacts.",
    "Linear panels have independent zero-based y scales; apparent line heights across panels are not comparable.",
)

# Label-to-code mapping reviewed against compute_cost_single at SOURCE_REVISIONS
# and the uploaded cost appendix (shared, ball, box and tray cost sections).
# This validates names/weight symbols, not equivalence of every paper formula.
PAPER_NOTATION_REFERENCE = {
    "sources": [
        {"file": "supp(1).tex", "sections": "Shared Cost Components; Ball Lifting; Box Lifting; Tray Pushing",
         "sha256": "e845fb160aee5f727959a88294a4d345cb9d6202015818c7036456d63e62a636"},
        {"file": "Bimanual_AAMAS_2027-5.pdf", "sections": "1.2-1.5",
         "sha256": "da3558942e80ad8d90860e2e8963584f66b06585e92527e99657b3524eccb02c"},
    ],
    "scope": "Cost-component names and weight-symbol mapping to the recorded policy keys and pinned planner revisions.",
    "exceptions": [
        "Smoothness is absent from the paper cost equations: use its name without inventing a weight symbol.",
        "Box orientation is described without a paper symbol: retain the name without assigning one.",
        "Box z-axis is yz planar alignment (w_yz), not height alone.",
        "Box obj_to_targ also weights contact maintenance; the paper's Object-to-target label is retained.",
        "Ball/Box fixed lift spacing gains are excluded from these policy-weight plots, not removed from the planner.",
    ],
}


def figure_stem(task, *, presentation=False):
    """Keep presentation artifacts distinct from the original figure and CSV."""
    return f"{task}_weights_snapshots_presentation" if presentation else f"ars_{task}_weights_snapshots"


def display_cost_label(label, *, presentation=False):
    label = label.replace(" (not in paper)", "").replace(" (symbol unspecified)", "")
    if presentation:
        # The paper uses text subscripts/superscripts. In particular, eef-obj
        # and obj-targ contain text hyphens, not mathematical minus operators.
        label = label.replace(r"\mathrm{", r"\text{")
    return label


def read_weight_config(notebook):
    """Only literal labels/rules, never imports or executable notebook functions."""
    wanted = {"PAPER_COST_LABELS", "PHASE_WEIGHT_RULES"}
    result = {}
    for cell in json.loads(notebook.read_text())["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "PHASE_WEIGHT_RULES =" not in source:
            continue
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in wanted:
                    result[target.id] = ast.literal_eval(node.value)
    if set(result) != wanted:
        raise ValueError("Notebook must provide literal paper labels and phase rules")
    return result["PAPER_COST_LABELS"], result["PHASE_WEIGHT_RULES"]


def masked_weights(values, phase, names, task_name, labels, rules):
    values, phase = np.asarray(values, float), np.asarray(phase, float)
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError("Weight columns and names do not align")
    if phase.shape != (len(values),) or not np.isin(phase, [0, 1]).all():
        raise ValueError("Invalid planning phase or phase/weight length mismatch")
    if set(names) - labels[task_name].keys():
        raise ValueError("Unknown weight component; inspect its planner before plotting")
    _, approach, transport, unused = rules[task_name]
    active = np.ones(values.shape, bool)
    multiplier = np.ones(values.shape)
    for i, name in enumerate(names):
        if name in approach:
            active[:, i] = phase == 0
        elif name in transport:
            active[:, i] = phase == 1
        elif name in unused:
            active[:, i] = False
        if task_name in ("Ball Lift", "Box Lift") and name == "distance":
            multiplier[:, i] = np.where(phase == 1, 10, 1)
    if not np.all(np.isfinite(values[active]) & (values[active] > 0)):
        raise ValueError("Selected episode contains invalid active log weights")
    logs = np.zeros_like(values)
    logs[active] = np.log(values[active]) + np.log(multiplier[active])
    return logs, active, multiplier


def effective_weights(values, active, multiplier):
    """Actual linear coefficients in the CEM cost sum, not weighted costs."""
    values, active, multiplier = np.asarray(values, float), np.asarray(active, bool), np.asarray(multiplier, float)
    if values.shape != active.shape or values.shape != multiplier.shape:
        raise ValueError("Effective weight inputs must have the same shape")
    result = np.zeros_like(values)
    result[active] = values[active] * multiplier[active]
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("Invalid effective linear weight")
    return result


def plotted_policy_weights(values, active):
    """Mask inactive policy outputs without applying any fixed planner gain."""
    return effective_weights(values, active, np.ones_like(values, dtype=float))


def box_settled_start(qpos, qvel, post, dt):
    """First stable approach window after the saved initial drop/rebound.

    In this frozen Box model the free joint starts at qpos/qvel offset 12.
    Require a downward-to-upward velocity reversal after a >=2 cm fall, then
    >=0.4 s of translation speed <=5 mm/s and height range <=1 mm, after 1 s.
    This is an explicit kinematic settling criterion, NOT a contact certificate.
    No hard-coded floor height or guessed video playback time is used.
    """
    qpos, qvel, post = np.asarray(qpos, float), np.asarray(qvel, float), np.asarray(post, float)
    if (qpos.ndim != 2 or qpos.shape[1] != 19 or qvel.shape != (len(qpos), 18)
            or post.shape != (len(qpos),) or not np.isfinite(dt) or dt <= 0
            or not np.isfinite(qpos).all() or not np.isfinite(qvel).all()
            or not np.isin(post, [0, 1]).all()):
        raise ValueError("Invalid Box settling states")
    z, vz = qpos[:, 14], qvel[:, 14]
    rebounds = np.flatnonzero((vz[:-1] < -.05) & (vz[1:] >= 0)) + 1
    rebounds = [int(k) for k in rebounds if z[0] - np.min(z[:k + 1]) >= .02]
    if not rebounds:
        raise ValueError("No saved initial Box landing/rebound; inspect before choosing a start")
    window = int(np.ceil(.4 / dt)) + 1
    speed = np.linalg.norm(qvel[:, 12:15], axis=1)
    for k in range(rebounds[0], len(z) - window + 1):
        sl = slice(k, k + window)
        if ((k + 1) * dt >= 1 and np.all(post[sl] == 0)
                and np.max(speed[sl]) <= .005 and np.ptp(z[sl]) <= .001):
            return k, dict(
                method="First stable approach window after initial fall and upward velocity reversal",
                initial_rebound_frame=rebounds[0], start_frame=k, start_time_s=(k + 1) * dt,
                minimum_time_s=1., stability_window_s=(window - 1) * dt,
                speed_limit_m_s=.005, height_range_limit_m=.001,
                observed_max_speed_m_s=float(np.max(speed[sl])),
                observed_height_range_m=float(np.ptp(z[sl])),
                box_center_xyz_m=qpos[k, 12:15].tolist(),
                note="Kinematic settling check; original frame also visually reviewed. Not a contact/penetration certificate.")
    raise ValueError("No settled Box approach window; inspect before choosing a start")


def snapshot_times(pre, post, dt, start=0):
    pre, post = np.asarray(pre, float), np.asarray(post, float)
    if (pre.shape != post.shape or pre.ndim != 1 or len(pre) < 4
            or not np.isfinite(dt) or dt <= 0
            or not np.isin(pre, [0, 1]).all() or not np.isin(post, [0, 1]).all()):
        raise ValueError("Invalid pre/post phase arrays")
    if pre[0] != 0 or not np.array_equal(pre[1:], post[:-1]):
        raise ValueError("Pre-step weights and post-step frames are not phase-aligned")
    if np.count_nonzero(np.diff(post)) != 1 or post[-1] != 1:
        raise ValueError("Illustrative episode must have a single approach-to-transport transition")
    approach, transition, end = phase_frame_indices(post)
    if not isinstance(start, (int, np.integer)) or not 0 <= start < transition:
        raise ValueError("Start must be an approach frame before the transition")
    if start:
        approach = (start + transition) // 2
    indices = [int(start), approach, transition, end]
    if any(a >= b for a, b in zip(indices, indices[1:])):
        raise ValueError("Episode needs four distinct start/approach/transition/transport frames")
    edges = np.arange(len(pre) + 1) * dt
    return indices, edges, (np.array(indices) + 1) * dt


def goal_distance(data, task, index):
    if task == "tray":
        # 6117e21 runner:649,713 and evaluator:145: POST-step distance in metres.
        return np.asarray(data["cost_g_tray"][index], float), "saved post-step cost_g_tray (Euclidean metres)"
    qkey = "post_qpos" if task == "ball" else "qpos_post"
    qpos = np.asarray(data[qkey][index], float)
    target = np.asarray(data["target"][index], float)
    # Box XML at 00898ab has 12 robot coordinates then the object's free joint.
    offset = int(data["ball_qpos_idx"].item()) if task == "ball" else 12
    if qpos.ndim != 2 or qpos.shape[1] != 19 or target.shape[0] != qpos.shape[0]:
        raise ValueError("Unexpected saved object/target layout")
    return np.linalg.norm(qpos[:, offset:offset + 3] - target[:, :3], axis=1), f"norm({qkey}[:,{offset}:{offset + 3}] - target[:,:3])"


def load_trace(episode, labels, rules, settle_box=True):
    task, i = episode.task, episode.index
    with np.load(episode.npz, allow_pickle=True) as d:
        if not bool(d["success"][i]):
            raise ValueError("Only recorded-success episodes may be illustrated")
        if task in CHECKPOINT_HASHES:
            key = "policy_checkpoint_sha256" if task == "ball" else "checkpoint_sha256"
            if d[key].item() != CHECKPOINT_HASHES[task]:
                raise ValueError(f"Unexpected ARS checkpoint in {episode.npz}")
            checkpoint_hash = str(d[key].item())
        else:
            if Path(d["checkpoint_path"].item()).name != "ars4124_common_jit.json":
                raise ValueError("Unexpected Tray ARS checkpoint path")
            adapter = json.loads((episode.run_root / "checkpoints/ars4124_common_jit.json").read_text())
            if adapter.get("source_algorithm") != "ars4124":
                raise ValueError("Tray common-JIT adapter is not ARS4124")
            checkpoint_hash = sha256(episode.run_root / "checkpoints/ars4124_common_jit.json")
        key = "cost_weight_keys" if "cost_weight_keys" in d else "cost_weights_keys"
        names = [str(x) for x in d[key]]
        values = np.asarray(d["cost_weights_lin" if task == "tray" else "cost_weights"][i], float)
        phase = np.asarray(d[rules[TASKS[task]][0]][i], float)
        post = np.asarray(d["task_flag"][i], float)
        logs, active, multiplier = masked_weights(values, phase, names, TASKS[task], labels, rules)
        start, start_selection = 0, dict(method="First saved post-step video frame")
        if task == "box" and settle_box:
            start, start_selection = box_settled_start(d["qpos_post"][i], d["qvel_post"][i], post, episode.dt)
        indices, edges, times = snapshot_times(phase, post, episode.dt, start=start)
        effective = effective_weights(values, active, multiplier)
        plotted = plotted_policy_weights(values, active)
        distance, distance_source = goal_distance(d, task, i)
        if len(values) != episode.steps or distance.shape != (episode.steps,) or not np.isfinite(distance).all():
            raise ValueError("Weights, saved states and episode length must align")
        diagnostic_keys = {
            "ball": ("cost_g_pick", "cost_r_pick"),
            "box": ("cost_eef_to_obj", "cost_r"),
            "tray": ("cost_g_pick", "cost_r_pick", "cost_g_tray", "cost_r_tray"),
        }[task]
        snapshot_diagnostics = {
            "note": "Reported evaluator diagnostics at snapshot indices, not per-component predicted CEM costs. "
                    "Derived MJX fields follow the historical evaluator's timing; object goal distance uses saved post-step state. "
                    "Tray ARS grasp diagnostics use the worse hand, not the mean or predicted-horizon sum.",
            "saved_fields": {k: np.asarray(d[k][i], float)[indices].tolist() for k in diagnostic_keys},
        }
        if task != "tray":
            qkey = "post_qpos" if task == "ball" else "qpos_post"
            vkey = "post_qvel" if task == "ball" else "qvel_post"
            qp = np.asarray(d[qkey][i], float)[indices]
            snapshot_diagnostics["post_step_object_xyz_m"] = qp[:, 12:15].tolist()
            snapshot_diagnostics["post_step_joint_velocity_norm_rad_s"] = np.linalg.norm(
                np.asarray(d[vkey][i], float)[indices, :12], axis=1).tolist()
            if task == "box":
                tq = np.asarray(d["target"][i], float)[indices, 3:7]
                snapshot_diagnostics["post_step_box_quaternion_error_norm"] = np.linalg.norm(qp[:, 15:19] - tq, axis=1).tolist()
        ck_key = "policy_path" if task == "ball" else "checkpoint_path"
        return dict(names=names, linear=values, masked_log=logs, effective=effective,
                    plotted_linear=plotted, active=active,
                    multiplier=multiplier, pre_phase=phase, post_phase=post,
                    indices=indices, edges=edges, times=times, goal_distance=distance,
                    plot_start_s=float(times[0]) if task == "box" else 0., start_selection=start_selection,
                    snapshot_diagnostics=snapshot_diagnostics,
                    goal_distance_source=distance_source, checkpoint_path=str(d[ck_key].item()),
                    checkpoint_sha256=checkpoint_hash)


def audit_records(base, task):
    path = base / AUDIT_PATHS[task]
    if not path.is_file():
        return {}, None
    rows = json.loads(path.read_text())["episodes"]
    return {(str((base / r["file"]).resolve()), int(r["episode"])): r
            for r in rows if r["method"] == "ARS"}, path


def selected_audit(base, episode):
    records, path = audit_records(base, episode.task)
    row = records.get((str(episode.npz.resolve()), episode.index))
    if row is None:
        return {"available": False, "note": "No matching local penetration report; physical validity is not certified."}
    result = {"available": True, "source": str(path), "source_sha256": sha256(path),
              "max_depth_mm": row["max_depth_mm"],
              "after_1s_max_depth_mm": row.get("after_1s_max_depth_mm")}
    if episode.task == "tray":
        for name, key in TRAY_EXTRA_AUDITS:
            extra_path = base / "tray_push_penetration_snapshots/archive" / name
            if not extra_path.is_file():
                continue
            matches = [r for r in json.loads(extra_path.read_text())[key]
                       if (base / r["file"]).resolve() == episode.npz.resolve() and r["episode"] == episode.index]
            if len(matches) == 1:
                result[name] = {"source_sha256": sha256(extra_path), **{
                    k: v for k, v in matches[0].items() if k.startswith("max_") or k == "links"}}
    return result


def rank_candidates(episodes, base, task, labels, rules):
    """Transparent illustrative shortlist; NOT a representative-SR estimator."""
    audits, path = audit_records(base, task)
    if path is None:
        raise ValueError(f"--rank needs the existing local audit: {AUDIT_PATHS[task]}")
    rows = []
    for ep in episodes:
        if not ep.success:
            continue
        with np.load(ep.npz, allow_pickle=True) as data:
            pre = np.asarray(data[rules[TASKS[task]][0]][ep.index], float)
            post = np.asarray(data['task_flag'][ep.index], float)
        # A recorded success can switch phase only at its last step. It cannot
        # supply four distinct explanatory snapshots, so exclude it up front.
        if (np.count_nonzero(pre == 0) < 10 or np.count_nonzero(pre == 1) < 10
                or np.count_nonzero(np.diff(post)) != 1):
            continue
        trace = load_trace(ep, labels, rules, settle_box=False)
        phase = trace["pre_phase"]
        count = int(np.count_nonzero(phase == 0))
        if count < 10 or ep.steps - count < 10:
            continue
        row = audits.get((str(ep.npz.resolve()), ep.index))
        if row is None:
            raise ValueError(f"Audit coverage missing for {ep.npz}, index {ep.index}")
        amplitudes = []
        # Measure changes in RAW actor weights only while that component is active.
        # Do not let masking or the hard-coded x10 lift factor win the ranking.
        for name in FOCUS_WEIGHTS[task]:
            j = trace["names"].index(name)
            v = np.log(trace["linear"][trace["active"][:, j], j])
            amplitudes.append(float(np.ptp(np.percentile(v, [10, 90]))))
        depth_key = "robot_" + {"ball": "ball", "box": "box", "tray": "tray"}[task]
        rows.append(dict(episode=ep, amplitudes=amplitudes,
                         depth_mm=row["max_depth_mm"][depth_key],
                         robot_robot_mm=row["max_depth_mm"]["robot_robot"]))
    if not rows:
        raise ValueError("No successful episodes with at least ten steps in each phase")
    median = float(np.median([r["episode"].steps for r in rows]))
    quartile = float(np.quantile([r["depth_mm"] for r in rows], .25))
    shortlist = [r for r in rows if r["depth_mm"] <= quartile and r["robot_robot_mm"] < 1
                 and .6 * median <= r["episode"].steps <= 1.4 * median]
    for r in shortlist:
        r["score"] = float(np.mean(r["amplitudes"]) - .7 * abs(r["episode"].steps / median - 1))
    shortlist.sort(key=lambda r: (-r["score"], r["episode"].batch, r["episode"].seed, r["episode"].index))
    return shortlist, dict(successful_clear_transition=len(rows), median_steps=median,
                          robot_object_lower_quartile_mm=quartile, shortlist_count=len(shortlist))


def render_figure(episode, trace, frames, labels, *, presentation=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import MaxNLocator
    if len(frames) != 4 or len(trace["times"]) != 4:
        raise ValueError("The two-block figure requires four snapshots")
    title = TASKS[episode.task]
    # Exactly one panel per component: no shared axis can conceal small weights.
    ncols = {"ball": 3, "box": 5, "tray": 4}[episode.task]
    nrows = int(np.ceil(len(trace["names"]) / ncols))
    fig = plt.figure(figsize=(18, 5 + 2.6 * nrows), facecolor="white")
    grid = fig.add_gridspec(2, 1, height_ratios=[3, 2.6 * nrows],
                           left=.055, right=.985, bottom=.095, top=.925, hspace=.30)
    photo_grid = grid[0].subgridspec(1, 4, wspace=.065)
    weight_grid = grid[1].subgridspec(nrows, ncols, wspace=.30, hspace=.90)
    image_axes = [fig.add_subplot(photo_grid[0, i]) for i in range(4)]
    edges, plotted = trace["edges"], trace["plotted_linear"]
    start = trace["plot_start_s"]
    transition = float(trace["times"][2])
    transport = "Push" if episode.task == "tray" else "Lift"
    heading = title if presentation else f"{title} | ARS | batch {episode.batch} | seed {episode.seed} | episode {episode.index + 1}"
    fig.suptitle(heading,
                 fontsize=17, y=.972, weight="bold")
    stages = ["Start (settled)" if episode.task == "box" else "Start", "Approach", "Transition", transport]
    for ax, frame, tag, stage, time_s in zip(image_axes, frames, "ABCD", stages, trace["times"]):
        ax.imshow(frame)
        caption = f"{tag}   {stage}" if presentation else f"{tag}   {stage} · {time_s:.1f} s"
        ax.set_title(caption, fontsize=14 if presentation else 12, loc="left", pad=7)
        ax.set_axis_off()

    colors = plt.get_cmap("tab20")
    color_order = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 5]
    for column, name in enumerate(trace["names"]):
        ax = fig.add_subplot(weight_grid[column // ncols, column % ncols])
        label = display_cost_label(labels[title][name], presentation=presentation)
        description, _, symbol = label.partition(" — ")
        active = trace["active"][:, column]
        status = "both phases"
        if not active.any():
            status = "unused in CEM: always zero"
        elif np.array_equal(active, trace["pre_phase"] == 0):
            status = "approach only"
        elif np.array_equal(active, trace["pre_phase"] == 1):
            status = f"{transport.lower()} only"
        subtitle = " · ".join(x for x in [symbol, status] if x)
        ax.set_title(textwrap.fill(description, width=30 if presentation else 38) + "\n" + subtitle,
                     fontsize=12 if presentation else 9.5, pad=33 if presentation else 31)
        ax.stairs(plotted[:, column], edges, color=colors(color_order[column]),
                  lw=1.8, label=label, baseline=None)
        ax.axhline(0, color="#999999", lw=.5, zorder=-1)
        # Banners lie OUTSIDE the axes; snapshot lines stop at the axes boundary.
        for left, right, phase_name, color in [
                (start, transition, "Approach", "#deebf5"),
                (transition, episode.duration, transport, "#fce5c8")]:
            ax.axvspan(left, right, color=color, alpha=.32, zorder=-2)
            ax.add_patch(Rectangle((left, 1.035), right - left, .105,
                                  transform=ax.get_xaxis_transform(), facecolor=color,
                                  edgecolor="none", clip_on=False))
            ax.text((left + right) / 2, 1.0875, phase_name,
                    transform=ax.get_xaxis_transform(), ha="center", va="center", fontsize=10.5 if presentation else 9)
        for tag, time_s in zip("ABCD", trace["times"]):
            ax.axvline(time_s, color="#783c85" if tag == "C" else "#777777",
                       lw=1.25 if tag == "C" else .85, ls="--" if tag == "C" else ":")
            ax.text(time_s, .97, tag, transform=ax.get_xaxis_transform(),
                    ha={"A": "left", "D": "right"}.get(tag, "center"), va="top",
                    fontsize=10 if presentation else 9, weight="bold", color="#783c85" if tag == "C" else "#444444",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=.5))
        visible_peak = float(np.max(plotted[edges[1:] > start, column]))
        peak = visible_peak if visible_peak else 1.
        ax.set_xlim(start, episode.duration)
        ax.set_ylim(-.035 * peak, 1.18 * peak)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.tick_params(labelsize=11 if presentation else 9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.17)
    weights_box = grid[1].get_position(fig)
    fig.text(.010, (weights_box.y0 + weights_box.y1) / 2,
             "Phase-masked policy weight — linear, independent y-scales", rotation=90,
             ha="left", va="center", fontsize=13 if presentation else 12)
    time_label = ("Simulation time (s) · C = recorded phase switch" if presentation else
                  "Simulation time (s) · original timestamps · C = recorded phase switch")
    fig.text(.52, .06, time_label, ha="center", fontsize=12 if presentation else 11)
    phase_note = "Inactive terms = 0; fixed planner gains are excluded from these policy-weight plots."
    start_note = (f"Box drop/rebound omitted before {start:g} s; full data retained in CSV." if episode.task == "box" else
                  f"Start snapshot = first saved frame ({episode.dt:g} s).")
    note = (phase_note + "\nIndependent linear scales; weights are coefficients, not cost contributions." if presentation else
            phase_note + " " + start_note + "\n"
            "Panels show unnormalized policy weights, not cost contributions or causal effects; compare numbers, not heights across panels.")
    fig.text(.055, .025, note, fontsize=10 if presentation else 9, color="#555555")
    return fig


def write_csv(path, trace, dt):
    names = trace["names"]
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step_index", "weight_start_s", "frame_time_s", "planning_phase", "post_phase", "goal_distance_m", "shown_in_plot"]
                        + [f"raw_{n}" for n in names] + [f"active_{n}" for n in names]
                        + [f"multiplier_{n}" for n in names] + [f"masked_ln_{n}" for n in names]
                        + [f"effective_linear_{n}" for n in names]
                        + [f"plotted_policy_linear_{n}" for n in names])
        for k in range(len(trace["linear"])):
            writer.writerow([k, k * dt, (k + 1) * dt, int(trace["pre_phase"][k]), int(trace["post_phase"][k]), trace["goal_distance"][k],
                             int((k + 1) * dt > trace["plot_start_s"])]
                            + trace["linear"][k].tolist() + trace["active"][k].astype(int).tolist()
                            + trace["multiplier"][k].tolist() + trace["masked_log"][k].tolist()
                            + trace["effective"][k].tolist() + trace["plotted_linear"][k].tolist())


def phase_changes(trace):
    """Numbers for describing visible changes without attributing gates to ARS.

    First/last values are the first/last MPC solve in the displayed phase, not
    an interpolation at the post-step snapshot. The last solve's coefficient
    applies until the final photograph. Raw, plotted (masked policy), and actual
    effective CEM values are separate; only the plotted values appear in PNGs.
    """
    visible = trace["edges"][1:] > trace["plot_start_s"]
    boundary = int(np.flatnonzero(trace["pre_phase"] == 1)[0])
    result = {}
    for j, name in enumerate(trace["names"]):
        row = {}
        for phase, label in [(0, "approach"), (1, "transport")]:
            indices = np.flatnonzero(visible & (trace["pre_phase"] == phase))
            if not len(indices):
                raise ValueError("Both phases must be visible")
            first, last = int(indices[0]), int(indices[-1])
            raw = trace["linear"][indices, j]
            effective = trace["effective"][indices, j]
            plotted = trace["plotted_linear"][indices, j]
            active = bool(trace["active"][first, j])
            row[label] = dict(
                active=active, first_step=first, last_step=last,
                first_solve_time_s=float(trace["edges"][first]),
                last_solve_time_s=float(trace["edges"][last]),
                last_interval_end_s=float(trace["edges"][last + 1]),
                raw_first=float(raw[0]), raw_last=float(raw[-1]),
                plotted_first=float(plotted[0]), plotted_last=float(plotted[-1]),
                plotted_min=float(np.min(plotted)), plotted_max=float(np.max(plotted)),
                effective_first=float(effective[0]), effective_last=float(effective[-1]),
                effective_min=float(np.min(effective)), effective_max=float(np.max(effective)),
                within_phase_actor_last_over_first=float(raw[-1] / raw[0]) if active else None)
        row["at_phase_boundary"] = dict(
            raw_before=float(trace["linear"][boundary - 1, j]), raw_after=float(trace["linear"][boundary, j]),
            active_before=bool(trace["active"][boundary - 1, j]), active_after=bool(trace["active"][boundary, j]),
            multiplier_before=float(trace["multiplier"][boundary - 1, j]), multiplier_after=float(trace["multiplier"][boundary, j]),
            plotted_before=float(trace["plotted_linear"][boundary - 1, j]), plotted_after=float(trace["plotted_linear"][boundary, j]),
            effective_before=float(trace["effective"][boundary - 1, j]), effective_after=float(trace["effective"][boundary, j]))
        result[name] = row
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--select", action="append", default=[], metavar="TASK:BATCH:SEED:EPISODE")
    parser.add_argument("--rank", action="store_true", help="Print reproducible shortlist from existing audit files; write nothing")
    parser.add_argument("--dry-run", action="store_true", help="Validate selections and decode frames; write nothing")
    parser.add_argument("--presentation", action="store_true",
                        help="Write separately named figures with larger fonts and no method, batch, seed, episode or snapshot timestamps")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sim-repo", type=Path, default=Path('/home/aks-lab/colcon_ws/src/manipulator_mujoco'))
    args = parser.parse_args(argv)
    if len(set(args.tasks)) != len(args.tasks):
        raise ValueError("Duplicate tasks")
    notebook = args.notebook.resolve()
    base = notebook.parent
    sources, batches, expected = read_sources(notebook)
    labels, rules = read_weight_config(notebook)
    choices = dict(DEFAULT_SELECTIONS)
    overridden = set()
    for item in args.select:
        task, batch, seed, number = item.split(":")
        if task not in args.tasks or task in overridden or int(number) < 1 or int(batch) not in batches:
            raise ValueError(f"Invalid or duplicate selection: {item}")
        choices[task] = int(batch), int(seed), int(number)
        overridden.add(task)
    prepared = []
    for task in args.tasks:
        episodes = []
        for batch in batches:
            files = sorted(base.glob(sources[TASKS[task]]["ARS"][batch]))
            if len(files) != expected:
                raise ValueError(f"Expected {expected} notebook-selected ARS files for {task}/b{batch}, got {len(files)}")
            episodes.extend(load_episodes(task, batch, files))
        if args.rank:
            ranked, info = rank_candidates(episodes, base, task, labels, rules)
            print(TASKS[task], info)
            for r in ranked[:5]:
                e = r["episode"]
                print(f"  {task}:{e.batch}:{e.seed}:{e.index + 1}  {e.duration:.1f}s  "
                      f"score={r['score']:.3f}  robot-object={r['depth_mm']:.2f}mm  "
                      f"within-active log ranges={np.round(r['amplitudes'], 2).tolist()}")
            continue
        batch, seed, number = choices[task]
        matches = [e for e in episodes if (e.batch, e.seed, e.index + 1) == (batch, seed, number) and e.success]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one successful ARS episode for {task}:{batch}:{seed}:{number}")
        ep = matches[0]
        trace = load_trace(ep, labels, rules)
        video = resolve_video(ep)
        frames, _, fps = extract_frames(video, ep, trace["indices"], expected_count=4)
        audit = selected_audit(base, ep)
        print(f"{TASKS[task]}: b{batch}, seed {seed}, episode {number} (NPZ index {number - 1}); "
              f"{ep.steps} steps; snapshots={np.round(trace['times'], 2).tolist()} s\n  {video}")
        prepared.append((ep, trace, video, frames, fps, audit))
    if args.rank or args.dry_run:
        print("No files written.")
        return 0
    output = (args.output_dir or base / "ars_weights_snapshots").resolve()
    targets = [output / f"{figure_stem(ep.task, presentation=args.presentation)}.{ext}"
               for ep, *_ in prepared for ext in ("png", "csv")]
    manifest_path = output / ("sources_presentation.json" if args.presentation else "sources.json")
    targets.append(manifest_path)
    if not args.overwrite and any(p.exists() for p in targets):
        raise ValueError("Output exists; use --overwrite or another --output-dir")
    # Decode and validate every input before writing any artifact.
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(notebook=str(notebook), notebook_sha256=sha256(notebook),
                    generator=str(Path(__file__).resolve()), generator_sha256=sha256(Path(__file__)),
                    candidate_batches=batches, episode_numbering="one-based; NPZ indices zero-based",
                    selection="Pinned, visually reviewed top-ranked illustrative successes. --rank reproduces the shortlist: "
                    "at least 10 steps per phase; lower quartile robot-object penetration; robot-robot <1 mm; "
                    "duration 0.6-1.4 times median; maximize mean active raw-log 10-90% range minus 0.7*relative duration deviation.",
                    masking="PLOTTED: active * raw_policy_weight, with no fixed planner gains. Inactive exactly zero. AUDIT ONLY: effective_linear = active * raw_weight * phase_multiplier; legacy masked logs also retained in CSV.",
                    timing="weight[k] on [k*dt,(k+1)*dt); video frame[k] at (k+1)*dt; no time warping or interpolation",
                    layout="Top: four snapshots A-D. Below: one independent, unnormalized linear panel per phase-masked policy weight (fixed planner gains excluded); external phase banners and A-D time markers.",
                    snapshot_stages=["start (Box: after settling; others: first saved frame)", "approach", "recorded phase transition", "lift/push at successful end"],
                    limitations=list(LIMITATIONS), figures=[])
    if args.presentation:
        manifest["presentation"] = dict(
            omitted_from_figures=["method", "batch", "seed", "episode", "snapshot timestamps"],
            component_font_pt=12, tick_font_pt=11,
            notation_reference=PAPER_NOTATION_REFERENCE,
            math_text="Paper text subscripts/superscripts use \\text, including literal hyphens.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for ep, trace, video, frames, fps, audit in prepared:
        stem = figure_stem(ep.task, presentation=args.presentation)
        figure = render_figure(ep, trace, frames, labels, presentation=args.presentation)
        figure.savefig(output / f"{stem}.png", dpi=180, facecolor="white")
        plt.close(figure)
        write_csv(output / f"{stem}.csv", trace, ep.dt)
        branch, commit, lines = SOURCE_REVISIONS[ep.task]
        source = dict(branch=branch, commit=commit, path=PLANNER_PATH, cost_lines=lines)
        if args.sim_repo.is_dir():
            import hashlib
            blob = subprocess.check_output(["git", "-C", str(args.sim_repo), "show", f"{commit}:{PLANNER_PATH}"])
            source["planner_sha256"] = hashlib.sha256(blob).hexdigest()
        manifest["figures"].append(dict(
            task=TASKS[ep.task], method="ARS", batch=ep.batch, seed=ep.seed,
            episode=ep.index + 1, npz_episode_index=ep.index, recorded_success=True,
            source_pattern=sources[TASKS[ep.task]]["ARS"][ep.batch],
            npz=str(ep.npz), npz_sha256=sha256(ep.npz), checkpoint_path=trace["checkpoint_path"],
            checkpoint_sha256=trace["checkpoint_sha256"],
            video=str(video), video_sha256=sha256(video), video_fps=fps,
            output=str(output / f"{stem}.png"), csv=str(output / f"{stem}.csv"),
            steps=ep.steps, timestep=ep.dt, frame_indices=trace["indices"],
            simulation_times_s=trace["times"].tolist(),
            displayed_time_start_s=trace["plot_start_s"], start_selection=trace["start_selection"],
            video_times_s=[i / fps for i in trace["indices"]],
            pre_phase_key=rules[TASKS[ep.task]][0], planner_source=source,
            paper_labels=labels[TASKS[ep.task]], plotted_weights=trace["names"],
            displayed_labels={name: display_cost_label(labels[TASKS[ep.task]][name], presentation=args.presentation)
                              for name in trace["names"]},
            selection_scoring_weights=list(FOCUS_WEIGHTS[ep.task]),
            visible_phase_weight_changes=phase_changes(trace),
            cost_meanings=COST_MEANINGS[ep.task], goal_distance_source=trace["goal_distance_source"],
            snapshot_goal_distance_m=trace["goal_distance"][trace["indices"]].tolist(),
            snapshot_diagnostics=trace["snapshot_diagnostics"],
            penetration_audit=audit, selection_overridden=ep.task in overridden))
        print(f"Saved {output / (stem + '.png')}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, ImportError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
