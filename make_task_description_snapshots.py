#!/usr/bin/env python3
"""Make a text-free, four-task description row from existing successful videos.

    python make_task_description_snapshots.py
    python make_task_description_snapshots.py --dry-run
    python make_task_description_snapshots.py --transport-fraction 0.75
    python make_task_description_snapshots.py --quad-frame 27 --overwrite

Left to right: Ball Lift, Box Lift, Tray Push, Quadrotor. Bimanual defaults
are the SAME ARS episodes as make_ars_weights_snapshots.py. The default selects
their last recorded lift/push frame. --transport-fraction selects an earlier
instant among steps that both start and end in the transport phase.

The quadrotor default is the existing ARS wall-crossing illustration: batch150,
seed42, zero-based trial0, frame27. Its pixel crop matches quadrotor_ars.json in
the quadrotor repository. This is an illustrative frame near the wall, not a
newly measured exact contact/crossing event. Success is checked in the saved
trial summary and trajectory, and frames on the terminal status card are rejected.

Outputs: one PNG, a PDF at --width inches (default 7.2), and sources.json in
task_description_snapshots/. No text is added to the images. All panels have
equal 4:3 slots; resizing preserves aspect ratio, with white padding if needed.
No simulation, policy execution, notebook execution or source-repository edits.
Recorded success is not a certificate of penetration-free/physically valid motion.
Requires numpy, opencv-python, Pillow, matplotlib and the sibling helper scripts.
Use trusted local bimanual NPZs only: their object arrays require allow_pickle=True.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

from make_ars_video_snapshots import (
    DEFAULT_NOTEBOOK, TASKS, extract_frames, load_episodes, read_sources,
    resolve_video, sha256,
)
from make_ars_weights_snapshots import DEFAULT_SELECTIONS, read_weight_config


DEFAULT_QUAD_REPO = Path('/home/aks-lab/sampling_based_general')
DEFAULT_QUAD_VIDEO = Path(
    'real_demo/data/video/wall_small_batch_five_methods_20261001_135216/'
    'ars_batch150_seed42/quadrotor_150_20_30_50_00_SUCCESS.mp4')
DEFAULT_QUAD_CROP = (400, 180, 560, 420)
QUAD_VIDEO_NAME = re.compile(
    r'(?P<stem>quadrotor_(?P<batch>\d+)_\d+_\d+_\d+_(?P<trial>\d+))'
    r'(?P<track>_track)?_SUCCESS')


def transport_frame(pre, post, fraction):
    """Only choose frames whose generating solve was already in lift/push."""
    pre, post = np.asarray(pre, float), np.asarray(post, float)
    if (pre.ndim != 1 or pre.shape != post.shape or not len(pre)
            or not np.isin(pre, [0, 1]).all() or not np.isin(post, [0, 1]).all()
            or not np.array_equal(pre[1:], post[:-1])):
        raise ValueError('Invalid or misaligned pre/post task phases')
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('--transport-fraction must be between 0 and 1')
    valid = np.flatnonzero((pre == 1) & (post == 1))
    if not len(valid):
        raise ValueError('No recorded lift/push step after the phase transition')
    return int(valid[int(np.rint(fraction * (len(valid) - 1)))])


def bimanual_snapshot(task, notebook, sources, expected, rules, choice, fraction):
    batch, seed, number = choice
    files = sorted(notebook.parent.glob(sources[TASKS[task]]['ARS'][batch]))
    if len(files) != expected:
        raise ValueError(f'Expected {expected} ARS files for {task}/b{batch}, found {len(files)}')
    matches = [e for e in load_episodes(task, batch, files)
               if (e.seed, e.index + 1) == (seed, number) and e.success]
    if len(matches) != 1:
        raise ValueError(f'Expected one successful ARS episode: {task}:{batch}:{seed}:{number}')
    ep = matches[0]
    phase_key = rules[TASKS[task]][0]
    with np.load(ep.npz, allow_pickle=True) as data:
        pre, post = data[phase_key][ep.index], data['task_flag'][ep.index]
        if len(pre) != ep.steps:
            raise ValueError('Episode length and phase record disagree')
        index = transport_frame(pre, post, fraction)
        checkpoint_key = 'policy_path' if task == 'ball' else 'checkpoint_path'
        checkpoint = str(data[checkpoint_key].item())
    video = resolve_video(ep)
    frames, _, fps = extract_frames(video, ep, [index], expected_count=1)
    return frames[0], dict(
        task=TASKS[task], method='ARS', batch=batch, seed=seed, episode_one_based=number,
        npz_episode_index=ep.index, recorded_success=True, phase='push' if task == 'tray' else 'lift',
        npz=str(ep.npz), npz_sha256=sha256(ep.npz), source_pattern=sources[TASKS[task]]['ARS'][batch],
        checkpoint_path_recorded=checkpoint, video=str(video), video_sha256=sha256(video),
        frame_index_zero_based=index, simulation_time_s=(index + 1)*ep.dt,
        video_time_s=index/fps, video_fps=fps, timestep_s=ep.dt,
        phase_pre_key=phase_key, phase_pre=float(pre[index]), phase_post=float(post[index]),
        selection=f'Fraction {fraction:g} of recorded steps with pre_phase=post_phase=1',
        original_size_px=list(frames[0].size), crop_xywh=None)


def crop_image(frame, crop):
    if crop is None:
        return frame
    x, y, width, height = crop
    if (min(x, y) < 0 or min(width, height) <= 0
            or x + width > frame.width or y + height > frame.height):
        raise ValueError(f'Crop {crop} is outside the {frame.width}x{frame.height} frame')
    return frame.crop((x, y, x + width, y + height))


def quadrotor_snapshot(video, index, crop):
    import cv2
    from PIL import Image

    video = video.resolve()
    match = QUAD_VIDEO_NAME.fullmatch(video.stem)
    run = re.fullmatch(r'ars_batch(?P<batch>\d+)_seed(?P<seed>\d+)', video.parent.name)
    if (not match or not run or int(match['batch']) != int(run['batch'])
            or len(video.parents) < 4 or video.parents[2].name != 'video'):
        raise ValueError('Quadrotor input must be a successful ARS project video with its saved summary/trajectory')
    root, session = video.parents[3], video.parent.parent.name
    summary_path = root/'trials'/session/video.parent.name/f"quad_trials_seed{run['seed']}.json"
    trajectory_path = root/'planner/trajectory'/session/video.parent.name/(match['stem']+'.npz')
    summary = json.loads(summary_path.read_text())
    episodes = [r for r in summary['results'] if r['trial'] == int(match['trial'])]
    if (not summary.get('weight_source', '').startswith('ARS policy') or len(episodes) != 1
            or not episodes[0]['success'] or episodes[0].get('crashed', False)):
        raise ValueError('Saved quadrotor summary does not confirm one successful ARS episode')
    ep = episodes[0]
    steps = int(ep['steps'])
    if not 0 <= index < steps:
        raise ValueError(f'Quadrotor frame must be in 0..{steps-1}; terminal status cards are excluded')
    with np.load(trajectory_path, allow_pickle=False) as data:
        times = np.asarray(data['times'], float)
        if not bool(np.asarray(data['success']).item()) or int(data['num_batch'].item()) != int(run['batch']):
            raise ValueError('Quadrotor trajectory success/batch metadata disagrees with the video')
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Invalid saved quadrotor simulation times')
    dt = float(np.median(np.diff(times)))
    if not np.allclose(np.diff(times), dt) or not np.isclose(times[0], dt):
        raise ValueError('Unexpected quadrotor timing convention')
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError(f'Cannot open video: {video}')
        count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
        if count < steps or not math.isfinite(fps) or fps <= 0:
            raise ValueError('Video does not cover the saved episode or has invalid FPS')
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, pixels = cap.read()
        if not ok or abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - (index + 1)) > .5:
            raise ValueError(f'Cannot decode exact quadrotor frame {index}')
        original = Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    frame = crop_image(original, crop)
    return frame, dict(
        task='Quadrotor', method='ARS', batch=int(run['batch']), seed=int(run['seed']),
        trial_zero_based=int(match['trial']), recorded_success=True, phase='flight',
        video=str(video), video_sha256=sha256(video), view='track' if match['track'] else 'wide',
        summary=str(summary_path), summary_sha256=sha256(summary_path),
        trajectory=str(trajectory_path), trajectory_sha256=sha256(trajectory_path),
        checkpoint_path_recorded=summary.get('checkpoint'), frame_index_zero_based=index,
        simulation_time_s=(index+1)*dt, timestep_s=dt, video_time_s=index/fps, video_fps=fps,
        simulation_time_basis='Video frame k is captured after step k+1; inferred from saved fixed timestep',
        usable_video_frames=steps, video_frames=count, trajectory_samples=len(times),
        excluded_terminal_frames=count-steps,
        selection='Explicit frame index; default 27 matches the existing near-wall quadrotor illustration',
        original_size_px=list(original.size), crop_xywh=list(crop) if crop else None)


def make_row(frames, panel_width=640, gap=12):
    """Four equal 4:3 slots, aspect-preserving fit; never stretch or auto-crop."""
    from PIL import Image, ImageOps

    if len(frames) != 4 or panel_width < 4 or gap < 0:
        raise ValueError('Need four frames, panel width >=4 and a nonnegative gap')
    height = round(panel_width*3/4)
    canvas = Image.new('RGB', (4*panel_width+3*gap, height), 'white')
    placements = []
    for i, frame in enumerate(frames):
        fit = ImageOps.contain(frame.convert('RGB'), (panel_width, height), Image.Resampling.LANCZOS)
        x = i*(panel_width+gap)+(panel_width-fit.width)//2
        y = (height-fit.height)//2
        canvas.paste(fit, (x, y))
        placements.append(dict(panel=i+1, paste_xy_px=[x, y], displayed_size_px=list(fit.size)))
    return canvas, placements


def save_row(canvas, output, width):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    resolution = canvas.width/width
    canvas.save(output/'four_tasks_description.png', dpi=(resolution, resolution))
    fig = plt.figure(figsize=(width, width*canvas.height/canvas.width), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(canvas, interpolation='none')
    ax.set_axis_off()
    fig.savefig(output/'four_tasks_description.pdf')
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--notebook', type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument('--select', action='append', default=[], metavar='TASK:BATCH:SEED:EPISODE',
                        help='Override a bimanual episode; EPISODE is one-based')
    parser.add_argument('--transport-fraction', type=float, default=1.)
    parser.add_argument('--quad-repo', type=Path, default=DEFAULT_QUAD_REPO)
    parser.add_argument('--quad-video', type=Path, help='Explicit ARS video with associated project summary/trajectory')
    parser.add_argument('--quad-frame', type=int, default=27, help='Zero-based frame; excludes terminal status cards')
    crops = parser.add_mutually_exclusive_group()
    crops.add_argument('--quad-crop', type=int, nargs=4, default=DEFAULT_QUAD_CROP, metavar=('X','Y','W','H'))
    crops.add_argument('--quad-full-frame', action='store_true', help='Keep the full quadrotor view with aspect-preserving padding')
    parser.add_argument('--width', type=float, default=7.2, help='PDF/PNG print width in inches')
    parser.add_argument('--panel-width', type=int, default=640, help='Raster width of each equal 4:3 panel')
    parser.add_argument('--gap', type=int, default=12, help='White gap in pixels')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Validate/decode every source; write nothing')
    args = parser.parse_args(argv)
    if (not math.isfinite(args.width) or args.width <= 0 or args.panel_width < 4 or args.gap < 0
            or not math.isfinite(args.transport_fraction) or not 0 <= args.transport_fraction <= 1):
        raise ValueError('Invalid figure dimensions, gap or transport fraction')
    notebook = args.notebook.resolve()
    sources, batches, expected = read_sources(notebook)
    _, rules = read_weight_config(notebook)
    choices, overridden = dict(DEFAULT_SELECTIONS), set()
    for item in args.select:
        task, batch, seed, episode = item.split(':')
        if task not in TASKS or task in overridden or int(batch) not in batches or int(episode) < 1:
            raise ValueError(f'Invalid or duplicate selection: {item}')
        choices[task] = int(batch), int(seed), int(episode)
        overridden.add(task)
    frames, records = [], []
    for task in TASKS:
        frame, record = bimanual_snapshot(task, notebook, sources, expected, rules, choices[task], args.transport_fraction)
        frames.append(frame); records.append(record)
    video = args.quad_video or args.quad_repo/DEFAULT_QUAD_VIDEO
    frame, record = quadrotor_snapshot(video, args.quad_frame, None if args.quad_full_frame else args.quad_crop)
    frames.append(frame); records.append(record)
    canvas, placements = make_row(frames, args.panel_width, args.gap)
    for record, placement in zip(records, placements):
        record.update(placement)
        print(f"{record['task']}: {record['video']}\n  frame {record['frame_index_zero_based']}, "
              f"phase={record['phase']}, simulation t={record['simulation_time_s']:.2f}s")
    if args.dry_run:
        print('No files written.')
        return 0
    output = (args.output_dir or notebook.parent/'task_description_snapshots').resolve()
    targets = [output/name for name in ('four_tasks_description.png','four_tasks_description.pdf','sources.json')]
    if not args.overwrite and any(p.exists() for p in targets):
        raise ValueError('Output exists; use --overwrite or a different --output-dir')
    output.mkdir(parents=True, exist_ok=True)
    save_row(canvas, output, args.width)
    manifest = dict(
        generator=str(Path(__file__).resolve()), generator_sha256=sha256(Path(__file__)),
        notebook=str(notebook), notebook_sha256=sha256(notebook),
        layout='One text-free row: Ball Lift | Box Lift | Tray Push | Quadrotor',
        figure_width_inches=args.width, canvas_size_px=list(canvas.size), panel_width_px=args.panel_width,
        gap_px=args.gap, treatment='Explicit quadrotor crop; otherwise full frames; aspect-preserving resize/padding only',
        limitations='Task illustrations only. Recorded success does not certify penetration-free or physically valid motion.',
        frames=records)
    targets[-1].write_text(json.dumps(manifest, indent=2)+'\n')
    for target in targets:
        print(f'Saved: {target}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, ImportError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2)
