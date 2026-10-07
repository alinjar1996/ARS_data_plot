#!/usr/bin/env python3
"""Extract real ARS video frames for the small-batch benchmark (no simulation).

Examples (run in the project Python environment; needs numpy, opencv-python, Pillow):
    python make_ars_video_snapshots.py --dry-run
    python make_ars_video_snapshots.py
    python make_ars_video_snapshots.py --tasks box --batches 250 --columns 1
    python make_ars_video_snapshots.py --tasks ball --batches 50 --select ball:50:0:1

By default, make one text-free, three-frame PNG per TASK (three PNGs total). Select the
recorded-success episode nearest the median executed duration across the requested
batches, considering all episodes, including episode 0. --batches restricts this
candidate pool; it does NOT create extra montages. --select TASK:BATCH:SEED:EPISODE
overrides the choice; EPISODE is ONE-based within its NPZ file and must be successful.
The panels show approach, the recorded pick-to-move transition, and goal-reaching.
Approach is the midpoint of the frames before the first post-step task_flag > 0.5;
transition is that first >0.5 frame; goal-reaching is the final successful frame.
The transition flag is a proximity/alignment proxy, NOT proof of first physical
contact. These videos record POST-step states; frame index i is at (i+1)*dt.
No titles, phase labels, captions or timestamps are drawn inside the figure.

Read ARS source patterns directly from the notebook without executing its code.
NPZ object arrays require allow_pickle=True: use only trusted local result files.
Write sources.json with exact NPZ/video identities, hashes, and frame indices.
Recorded success is not a certification that the trajectory is physically valid.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


TASKS = {"ball": "Ball Lift", "box": "Box Lift", "tray": "Tray Push"}
DEFAULT_NOTEBOOK = Path(__file__).resolve().with_name("benchmark_stat_batch_sm.ipynb")


def config_value(node, env):
    """Interpret only literal notebook configuration; never execute Python code."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [config_value(x, env) for x in node.elts]
    if isinstance(node, ast.Dict):
        return {config_value(k, env): config_value(v, env)
                for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.JoinedStr):
        return "".join(str(config_value(x, env)) for x in node.values)
    if isinstance(node, ast.FormattedValue) and node.conversion == -1 and node.format_spec is None:
        return config_value(node.value, env)
    if isinstance(node, ast.DictComp) and len(node.generators) == 1:
        gen = node.generators[0]
        if isinstance(gen.target, ast.Name) and not gen.ifs and not gen.is_async:
            result = {}
            for value in config_value(gen.iter, env):
                local = {**env, gen.target.id: value}
                result[config_value(node.key, local)] = config_value(node.value, local)
            return result
    raise ValueError(f"Unsupported notebook configuration expression: {ast.dump(node)}")


def read_sources(notebook):
    env = {}
    data = json.loads(notebook.read_text(encoding="utf-8"))
    for cell in data["cells"]:
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") != "code" or "SOURCE_PATTERNS" not in source:
            continue
        for node in ast.parse(source).body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            name = target.id
            if name.endswith("_ROOT") or name in (
                "ALL_BATCHES", "EXPECTED_FILES_PER_BATCH", "SOURCE_PATTERNS"
            ):
                env[name] = config_value(node.value, env)
        if "SOURCE_PATTERNS" in env:
            return env["SOURCE_PATTERNS"], env["ALL_BATCHES"], env["EXPECTED_FILES_PER_BATCH"]
    raise ValueError(f"No supported SOURCE_PATTERNS configuration in {notebook}")


@dataclass(frozen=True)
class Episode:
    task: str
    batch: int
    npz: Path
    index: int
    seed: int
    steps: int
    dt: float
    success: bool
    saved_video: str = ""

    @property
    def duration(self):
        return self.steps * self.dt

    @property
    def run_root(self):
        # All selected notebook patterns are RUN/npz/METHOD_batchN/*.npz.
        if self.npz.parent.parent.name != "npz":
            raise ValueError(f"Unexpected NPZ layout: {self.npz}")
        return self.npz.parent.parent.parent


def scalar(data, key):
    return np.asarray(data[key]).item()


def load_episodes(task, batch, files):
    episodes = []
    for path in files:
        with np.load(path, allow_pickle=True) as data:
            if int(scalar(data, "batch_size")) != batch:
                raise ValueError(f"Batch metadata mismatch: {path}")
            algorithm_key = "policy_algorithm" if task == "ball" else "checkpoint_algorithm"
            expected_algorithm = "ars4124" if task == "tray" else "ars"
            if str(scalar(data, algorithm_key)) != expected_algorithm:
                raise ValueError(f"Not the expected ARS checkpoint: {path}")
            success = np.asarray(data["success"])
            steps = data["step_time"]
            if len(success) != 20 or len(steps) != len(success):
                raise ValueError(f"Expected 20 aligned episode records: {path}")
            if "timestep" in data:
                dt = float(scalar(data, "timestep"))
            elif task == "tray":
                # Historical runner at 6117e21, rl_trainer_batch.py:128.
                dt = 0.1
            else:
                raise ValueError(f"Missing simulation timestep: {path}")
            if not np.isfinite(dt) or dt <= 0:
                raise ValueError(f"Invalid simulation timestep: {path}")
            for index, value in enumerate(success):
                if "episode_seed" in data:
                    seed = int(data["episode_seed"][index])
                elif "episode_seeds" in data:
                    seed = int(data["episode_seeds"][index])
                else:
                    seed = int(scalar(data, "eval_seed"))
                if len(steps[index]) == 0:
                    raise ValueError(f"Empty episode: {path}, index {index}")
                saved = str(data["video_path"][index]) if "video_path" in data else ""
                episodes.append(Episode(task, batch, path, index, seed,
                                        len(steps[index]), dt, bool(value), saved))
    return episodes


def select_episode(episodes, override=None):
    eligible = [e for e in episodes if e.success]
    if override is not None:
        seed, number = override
        eligible = [e for e in eligible if e.seed == seed and e.index == number - 1]
        if len(eligible) != 1:
            raise ValueError(f"Expected one successful episode for seed={seed}, "
                             f"episode={number}; found {len(eligible)}")
        return eligible[0]
    if not eligible:
        raise ValueError("No recorded-success episodes in the requested pool")
    median = float(np.median([e.duration for e in eligible]))
    return min(eligible, key=lambda e: (abs(e.duration - median), e.batch, e.seed, e.index, str(e.npz)))


def resolve_video(episode):
    root = episode.run_root
    if episode.task == "ball":
        # Rebase the saved absolute simulator path into THIS selected data root.
        parts = Path(episode.saved_video).parts
        if "videos" not in parts:
            raise ValueError(f"Missing saved Ball video path: {episode.npz}")
        tail = Path(*parts[parts.index("videos"):])
        expected = Path("videos") / "ars" / f"policy_batch{episode.batch}"
        if tail.parts[:3] != expected.parts:
            raise ValueError(f"Unexpected Ball ARS video path: {tail}")
        matches = [root / tail] if (root / tail).is_file() else []
    else:
        video_dir = root / "videos" / episode.npz.parent.name
        if episode.task == "box":
            # Box NPZ uses reason='na' on success, whereas videos say 'success'.
            suffix = "success" if episode.success else "*"
            name = (f"ars4218_batch{episode.batch}_seed{episode.seed}"
                    f"_ep{episode.index:04d}_{suffix}.mp4")
        else:
            name = f"ep_{episode.index:04d}_seed{episode.seed}.mp4"
        matches = sorted(video_dir.glob(f"*/{name}"))
    if len(matches) != 1:
        raise ValueError(f"Expected ONE matching video for {episode.npz.name}, "
                         f"episode {episode.index + 1}; found {len(matches)}: {matches}")
    path = matches[0].resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Video escapes selected result root: {path}")
    return path


def phase_frame_indices(task_flags):
    """Select three distinct POST-step frames using the recorded task phase."""
    flags = np.asarray(task_flags, dtype=float)
    if flags.ndim != 1 or len(flags) < 3 or not np.isfinite(flags).all():
        raise ValueError("Expected at least three finite post-step task flags")
    move = np.flatnonzero(flags > 0.5)
    if not len(move) or move[0] == 0:
        raise ValueError("Episode has no recorded approach-to-move transition")
    transition = int(move[0])
    goal = len(flags) - 1
    if transition == goal or flags[-1] <= 0.5:
        raise ValueError("Episode has no separate goal-reaching frame after transition")
    return [transition // 2, transition, goal]


def episode_frame_indices(episode):
    with np.load(episode.npz, allow_pickle=True) as data:
        if not episode.success or not bool(data["success"][episode.index]):
            raise ValueError("Goal-reaching panels require a recorded-success episode")
        flags = np.asarray(data["task_flag"][episode.index], dtype=float)
        if flags.shape != (episode.steps,):
            raise ValueError(f"Phase/video step counts do not align: {episode.npz}")
        return phase_frame_indices(flags)


def extract_frames(video, episode, indices):
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video}")
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        # These selected sweeps all rendered one POST-step frame per MPC step.
        if count != episode.steps:
            raise ValueError(f"Video/NPZ length mismatch: {video}: "
                             f"{count} frames versus {episode.steps} steps")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid video frame rate: {video}")
        if len(indices) != 3 or not 0 <= indices[0] < indices[1] < indices[2] < count:
            raise ValueError("Expected three increasing frame indices inside the video")
        frames = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok or abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - (index + 1)) > 0.5:
                raise ValueError(f"Cannot decode exact frame {index}: {video}")
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        return frames, indices, fps
    finally:
        cap.release()


def make_montage(frames, columns=3):
    """Paste unmodified decoded frames; only narrow white gutters are added."""
    from PIL import Image

    if len(frames) != 3 or columns not in (1, 3):
        raise ValueError("A three-frame montage needs one or three columns")
    width, height = frames[0].size
    if any(frame.size != (width, height) for frame in frames):
        raise ValueError("Video frame dimensions changed")
    gutter = 12
    rows = 3 // columns
    canvas = Image.new("RGB", (columns * width + (columns - 1) * gutter,
                              rows * height + (rows - 1) * gutter),
                       "white")
    for panel, frame in enumerate(frames):
        x = (panel % columns) * (width + gutter)
        y = (panel // columns) * (height + gutter)
        canvas.paste(frame, (x, y))  # Original decoded pixels: no crop or rescaling.
    return canvas


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--batches", type=int, nargs="+",
                        help="Candidate batches; still one montage per task. Default: notebook ALL_BATCHES")
    parser.add_argument("--select", action="append", default=[], metavar="TASK:BATCH:SEED:EPISODE",
                        help="Choose an exact episode (1-based); may be repeated")
    parser.add_argument("--columns", type=int, choices=(1, 3), default=3,
                        help="Text-free layout: horizontal (3, default) or vertical (1)")
    parser.add_argument("--output-dir", type=Path, help="Default: ars_video_snapshots beside notebook")
    parser.add_argument("--dry-run", action="store_true", help="Resolve episodes/videos; write nothing")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing montage files/manifest")
    args = parser.parse_args(argv)
    notebook = args.notebook.resolve()
    patterns, available_batches, expected_files = read_sources(notebook)
    batches = args.batches if args.batches is not None else available_batches
    if len(set(batches)) != len(batches) or len(set(args.tasks)) != len(args.tasks):
        raise ValueError("Tasks and batches must not contain duplicates")
    unsupported = set(batches) - set(available_batches)
    if unsupported:
        raise ValueError(f"Batches {sorted(unsupported)} are not selected in the notebook; "
                         f"available batches: {available_batches}. No automatic substitution.")
    overrides = {}
    for spec in args.select:
        task, batch, seed, number = spec.split(":")
        batch, seed, number = int(batch), int(seed), int(number)
        if task not in args.tasks or batch not in batches or number < 1:
            raise ValueError(f"Invalid or unrequested episode selection: {spec}")
        if task in overrides:
            raise ValueError(f"Only one episode per task is allowed: {spec}")
        overrides[task] = batch, seed, number

    selected = []
    for task in args.tasks:
        episodes = []
        for batch in batches:
            pattern = patterns[TASKS[task]]["ARS"][batch]
            files = sorted(notebook.parent.glob(pattern))
            if len(files) != expected_files:
                raise ValueError(f"{task}/b{batch}: expected {expected_files} files, "
                                 f"found {len(files)} for {pattern}")
            episodes.extend(load_episodes(task, batch, files))
        if task in overrides:
            batch, seed, number = overrides[task]
            episode = select_episode([e for e in episodes if e.batch == batch], (seed, number))
        else:
            episode = select_episode(episodes)
        video = resolve_video(episode)
        indices = episode_frame_indices(episode)
        name = f"ars_{TASKS[task].lower().replace(' ', '_')}_b{episode.batch}.png"
        selected.append((episode, video, name, patterns[TASKS[task]]["ARS"][episode.batch], indices))
        print(f"{TASKS[task]} b{episode.batch}: seed {episode.seed}, episode {episode.index + 1} "
              f"(NPZ index {episode.index}), recorded SUCCESS, "
              f"{episode.steps} steps, {episode.duration:.1f} s\n"
              f"  NPZ: {episode.npz}\n  Video: {video}\n"
              f"  Approach / transition / goal: steps {[i + 1 for i in indices]}")
    if args.dry_run:
        print(f"Resolved {len(selected)} montages; no files written.")
        return 0

    output = (args.output_dir or notebook.parent / "ars_video_snapshots").resolve()
    targets = [output / item[2] for item in selected] + [output / "sources.json"]
    if not args.overwrite and any(p.exists() for p in targets):
        raise ValueError("Output already exists; choose a new --output-dir or pass --overwrite")
    manifest = {"notebook": str(notebook), "notebook_sha256": sha256(notebook),
                "selection": "one recorded-success episode per task, nearest median duration across candidate batches unless overridden",
                "candidate_batches": batches,
                "episode_numbering": "one-based; npz_episode_index is zero-based",
                "columns": args.columns, "figure_text": False,
                "panel_order": ["approach", "recorded_pick_to_move_transition", "goal_reaching"],
                "frame_selection": {"phase_key": "task_flag (post-step)",
                                    "approach": "midpoint before first task_flag > 0.5",
                                    "transition": "first task_flag > 0.5",
                                    "goal": "last frame of the recorded-success episode"},
                "note": "Original video frames; phase transition is a proximity/alignment proxy, not confirmed first contact. Recorded success does not imply physical validity.",
                "montages": []}
    rendered = []
    for episode, video, name, pattern, indices in selected:
        frames, indices, fps = extract_frames(video, episode, indices)
        rendered.append((output / name, make_montage(frames, args.columns)))
        manifest["montages"].append({
            "task": TASKS[episode.task], "batch": episode.batch, "method": "ARS",
            "source_pattern": pattern, "npz": str(episode.npz), "npz_sha256": sha256(episode.npz),
            "video": str(video), "video_sha256": sha256(video), "output": str(output / name),
            "seed": episode.seed, "episode": episode.index + 1, "npz_episode_index": episode.index,
            "recorded_success": episode.success, "selection_override": episode.task in overrides,
            "executed_steps": episode.steps, "timestep": episode.dt,
            "timestep_source": "historical Tray runner (6117e21)" if episode.task == "tray" else "NPZ timestep",
            "video_fps": fps, "frame_indices_zero_based": indices,
            "simulated_times_s": [(i + 1) * episode.dt for i in indices],
            "video_times_s": [i / fps for i in indices],
        })
    # Validate/decode every selected video before creating any output files.
    output.mkdir(parents=True, exist_ok=True)
    for path, canvas in rendered:
        canvas.save(path)
        print(f"Saved {path}")
    (output / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved provenance: {output / 'sources.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
