#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NOTEBOOK="${1:-$SCRIPT_DIR/benchmark_stat_batch_sm.ipynb}"

if [[ ! -f "$NOTEBOOK" ]]; then
    echo "ERROR: notebook not found: $NOTEBOOK" >&2
    exit 2
fi

python3 - "$NOTEBOOK" <<'PY'
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np


notebook_path = Path(sys.argv[1]).resolve()
base = notebook_path.parent
notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

# Execute only the import and input-definition cells. This uses the notebook's
# SOURCE_PATTERNS directly, so the checker cannot drift from its configuration.
namespace = {}
old_cwd = Path.cwd()
try:
    import os

    os.chdir(base)
    with contextlib.redirect_stdout(io.StringIO()):
        for cell in notebook["cells"]:
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            exec(compile(source, str(notebook_path), "exec"), namespace)
            if "DOC_FILES" in namespace:
                break
finally:
    os.chdir(old_cwd)

required_names = ("DOC_FILES", "METHOD_TOKENS", "ALL_BATCHES", "EXPECTED_FILES_PER_BATCH")
missing_names = [name for name in required_names if name not in namespace]
if missing_names:
    raise RuntimeError(f"Notebook setup did not define: {', '.join(missing_names)}")

expected_files = int(namespace["EXPECTED_FILES_PER_BATCH"])
expected_episodes = 20
errors = []
group_rows = []
total_files = 0

for task, task_cfg in namespace["DOC_FILES"].items():
    for method in namespace["METHOD_TOKENS"]:
        for batch in namespace["ALL_BATCHES"]:
            files = task_cfg["candidates"][method].get(batch, [])
            total_files += len(files)
            group_success = 0
            task_time_eligible = 0
            comp_time_eligible = 0

            if len(files) != expected_files:
                errors.append(
                    f"{task} / {method} / batch {batch}: "
                    f"expected {expected_files} files, found {len(files)}"
                )

            for relative_name in files:
                path = base / relative_name
                try:
                    with np.load(path, allow_pickle=True) as data:
                        missing_keys = [
                            key for key in ("success", "total_time", "step_time")
                            if key not in data.files
                        ]
                        if missing_keys:
                            errors.append(f"{relative_name}: missing keys {missing_keys}")
                            continue

                        success = np.asarray(data["success"], dtype=float)
                        total_time = np.asarray(data["total_time"], dtype=float)
                        step_time = data["step_time"]

                        lengths = (len(success), len(total_time), len(step_time))
                        if lengths != (expected_episodes,) * 3:
                            errors.append(
                                f"{relative_name}: success/total_time/step_time "
                                f"lengths are {lengths}, expected "
                                f"{(expected_episodes,) * 3}"
                            )

                        if not np.all(np.isin(success, (0.0, 1.0))):
                            errors.append(f"{relative_name}: success contains values outside 0/1")
                        if not np.all(np.isfinite(total_time)):
                            errors.append(f"{relative_name}: total_time contains non-finite values")

                        group_success += int(np.sum(success == 1))
                        # Notebook task-time rule: exclude saved episode 0, then
                        # retain successful episodes only.
                        task_time_eligible += int(np.sum(success[1:] == 1))

                        # Notebook computation-time rule: include every episode,
                        # but exclude its first step and invalid/nonpositive values.
                        for episode_index, episode in enumerate(step_time):
                            values = np.asarray(episode, dtype=float)
                            valid = values[1:]
                            valid = valid[np.isfinite(valid) & (valid > 0)]
                            if valid.size:
                                comp_time_eligible += 1
                            else:
                                errors.append(
                                    f"{relative_name}: episode {episode_index} has no "
                                    "usable computation-time values after filtering"
                                )
                except Exception as exc:
                    errors.append(f"{relative_name}: could not validate file: {exc}")

            group_rows.append(
                (task, method, batch, len(files), group_success,
                 task_time_eligible, comp_time_eligible)
            )

print("Benchmark data audit")
print(f"Notebook: {notebook_path}")
print()
print("Task / Method / Batch                      Files  Success n  Task-time n  Comp-time episodes")
print("-" * 96)
for task, method, batch, files, successes, task_n, comp_n in group_rows:
    label = f"{task} / {method} / {batch}"
    print(f"{label:<43} {files:>5}  {files * expected_episodes:>9}  {task_n:>11}  {comp_n:>18}")

print()
print("Metric rules used by benchmark_stat_batch_sm.ipynb:")
print("  Success rate     : all 20 episodes/file = 100 observations/group")
print("  Task time        : episodes 1-19 and successful only; count varies by group")
print("  Computation time : all 20 episodes/file, first MPC step removed from each episode")
print("  Group aggregation: per-file means, then mean across the 5 files")
print()

if errors:
    print(f"FAIL: {len(errors)} validation issue(s) found:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    raise SystemExit(1)

expected_groups = len(namespace["DOC_FILES"]) * len(namespace["METHOD_TOKENS"]) * len(namespace["ALL_BATCHES"])
print(
    f"PASS: {expected_groups} groups, {total_files} files, and "
    f"{total_files * expected_episodes} aligned episode records validated."
)
PY
