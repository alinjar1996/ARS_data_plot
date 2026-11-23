"""Run the same-seed ball-lift ablation after a particular host process exits."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import traceback

import numpy as np


RUN = Path(__file__).resolve().parent
SOURCE = RUN / "source" / "real_demo"
SCHEDULE = RUN / "ars3950_phase_means.json"
REFERENCE = (RUN.parent / "real_demo/real_demo" /
             "eval_sweep_100_ball_lift_bayes_retrained_20260929_191418/checkpoints/bayesian_weights.json")


def save(name, data):
    target = RUN / name
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(target)


def status(state, **fields):
    entry = dict(time=datetime.now().astimezone().isoformat(), state=state, **fields)
    print(json.dumps(entry), flush=True)
    save("status.json", entry)


def wait_for_process(pid, start_ticks):
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    try:
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists() and stat.read_text().rsplit(")", 1)[1].split()[19] == start_ticks:
            status("queued", predecessor_pid=pid, predecessor="box-lift phase-mean sweep")
            poll = select.poll()
            poll.register(fd, select.POLLIN)
            poll.poll()
    except FileNotFoundError:
        pass  # The predecessor exited while opening its proc entry.
    finally:
        os.close(fd)


def verify(path, batch, seed, report):
    phases = report["policies"]["ars"]["phases"]
    vectors = np.asarray([phases[p]["linear_weights"] for p in ("approach", "lift")])
    fields = {"batch_size": batch, "eval_seed": seed, "complete": True,
              "episodes_completed": 20, "weight_type": "ars_phase_mean", "horizon": 15,
              "max_steps": 300, "timeout_secs": 0, "fold_step_keys": True,
              "maxiter_cem": 2, "maxiter_projection": 5,
              "phase_schedule_sha256": hashlib.sha256(SCHEDULE.read_bytes()).hexdigest()}
    with np.load(path, allow_pickle=True) as data:
        for key, expected in fields.items():
            if data[key].item() != expected:
                raise ValueError(f"{path}: unexpected {key}")
        if data["cost_weight_keys"].tolist() != report["cost_weight_keys"]:
            raise ValueError(f"{path}: cost keys differ")
        success = data["success"]
        if success.shape != (20,) or not np.array_equal(success, data["reason"] == "success"):
            raise ValueError(f"{path}: inconsistent episode outcomes")
        pre, post, weights, times = (data[k] for k in ("task_id", "task_flag", "cost_weights", "step_time"))
        solves = transitions = 0
        for i in range(20):
            phase = np.asarray(pre[i], dtype=int)
            after = np.asarray(post[i])
            used = np.asarray(weights[i])
            if used.dtype == object:
                # NumPy may pack equal-length episodes as a rectangular object array.
                used = np.asarray(used.tolist(), dtype=float)
            if (not len(phase) or len(phase) != len(times[i]) or phase.shape != after.shape
                    or not np.isin(phase, (0, 1)).all() or phase[0] != 0
                    or not np.array_equal(phase[1:], after[:-1])):
                raise ValueError(f"{path}: inconsistent phase in episode {i + 1}")
            # The interface can cast the saved float64 mean to planner float32.
            expected = vectors[phase]
            if not (np.array_equal(used, expected) or np.array_equal(used, expected.astype(np.float32))):
                raise ValueError(f"{path}: wrong planner inputs in episode {i + 1}")
            solves += len(phase)
            transitions += int(np.count_nonzero(phase[1:] != phase[:-1]))
    return dict(seed=seed, episodes=len(success), successes=int(success.sum()),
                verified_solves=solves, verified_phase_transitions=transitions, trace=str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pid", type=int)
    parser.add_argument("start_ticks")
    args = parser.parse_args()
    wait_for_process(args.pid, args.start_ticks)
    status("fitting")
    temp = SCHEDULE.with_suffix(".tmp")
    subprocess.run([sys.executable, str(RUN.parent / "compute_ball_lift_phase_means.py"),
                    "--policy", "ars", "--output", str(temp)], check=True)
    temp.replace(SCHEDULE)
    report = json.loads(SCHEDULE.read_text())
    results = {}
    for batch in (50, 150, 250):
        blocks = []
        for seed in range(5):
            out = RUN / "npz" / f"batch{batch}" / f"seed{seed}"
            status("evaluating", batch=batch, seed=seed)
            completed = [p.with_suffix("").with_suffix(".npz") for p in out.glob("*.summary.json")
                         if json.loads(p.read_text())["complete"]]
            if not completed:
                cmd = [sys.executable, "-u", str(SOURCE / "real_demo/jax_evaluate.py"),
                       "--weight_type", "ars_phase_mean", "--phase_means", str(SCHEDULE),
                       "--bayesian_weights_path", str(REFERENCE), "--model_path",
                       str(SOURCE / "ur5e_hande_mjx/scene.xml"), "--batch_size", str(batch),
                       "--seed", str(seed), "--n_eval", "20", "--num_steps", "15",
                       "--maxiter_cem", "2", "--max_steps", "300", "--timeout", "0",
                       "--fold_step_keys", "--save_video", "--video_stride", "1",
                       "--video_start", "1", "--video_dir", str(RUN / "videos"), "--out_dir", str(out)]
                with (RUN / f"batch{batch}_seed{seed}.log").open("a") as log:
                    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
                completed = [p.with_suffix("").with_suffix(".npz") for p in out.glob("*.summary.json")
                             if json.loads(p.read_text())["complete"]]
            if len(completed) != 1:
                raise ValueError(f"{out}: expected exactly one completed block")
            blocks.append(verify(completed[0], batch, seed, report))
            results[batch] = {"episodes": sum(b["episodes"] for b in blocks),
                              "successes": sum(b["successes"] for b in blocks), "blocks": blocks}
            save("results.json", results)
    status("complete", results=str(RUN / "results.json"))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        status("failed", error=traceback.format_exc())
        raise
