#!/usr/bin/env bash
# Inference only: Bayesian5001, with the previous matched box-lift settings.
# Every measured episode is rendered AND has real per-step timings.
# Run with the project environment activated, or set PYTHON=/path/to/python3.
# --dry-run checks checkpoint architectures and prints commands without running.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
SCRIPT="$PROJECT_ROOT/real_demo/real_demo/jax_evaluate.py"
MODEL_PATH="$PROJECT_ROOT/real_demo/ur5e_hande_mjx/scene.xml"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/real_demo/real_demo/eval_box_lift_bayesian5001_b50_150_250_$(date +%Y%m%d_%H%M%S)}"
BAYESIAN_WEIGHTS="$PROJECT_ROOT/real_demo/real_demo/bayesian_box_lift_matched_20260925_183328_398207/bayesian_5001.json"
N_EVAL_PER_BLOCK="${N_EVAL_PER_BLOCK:-20}"
# Preserve the original protocol: repeat each base seed for the whole block.
SEED_STRIDE=0
NUM_STEPS="${NUM_STEPS:-15}"
MAXITER_CEM="${MAXITER_CEM:-3}"
MAX_STEPS="${MAX_STEPS:-300}"
# Use the historical ARS schedule for ALL methods. fold_in is also supported,
# but must likewise be shared; algorithm-specific schedules are disallowed.
CEM_KEY_MODE="${CEM_KEY_MODE:-episode}"
SEEDS=(0 5 4 10 15)
read -r -a BATCH_SIZES <<< "${BATCH_SIZES:-50 150 250}"
# METHODS=(ars4218 ppo5005best ppo5005besteval)
# METHODS=(handtuned)
# METHODS=(handtuned bayesian ppo5005besteval ars4218)
METHODS=(bayesian5001)
declare -A CHECKPOINTS=(
    [ars4218]="$PROJECT_ROOT/ars_v2_linear_policy_4218.json"
    [ppo5005best]="$PROJECT_ROOT/real_demo/real_demo/ppo_linear_policy_5005.json"
    [ppo5005besteval]="$PROJECT_ROOT/real_demo/real_demo/ppo_linear_policy_5005_best_eval.json"
)

DRY_RUN=0
case "${1:-}" in
    --dry-run) DRY_RUN=1 ;;
    "") ;;
    *) echo "Usage: bash run_eval_box_lift_matched.sh [--dry-run]" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
    echo "Only --dry-run is accepted." >&2
    exit 2
fi
case "$CEM_KEY_MODE" in
    episode|fold_in) ;;
    *) echo "CEM_KEY_MODE must be episode or fold_in for all methods." >&2; exit 2 ;;
esac
for value in "$N_EVAL_PER_BLOCK" "$NUM_STEPS" "$MAXITER_CEM" "$MAX_STEPS" "${BATCH_SIZES[@]}"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "Episode count and CEM settings must be positive integers." >&2
        exit 2
    fi
done

# Pure-stdlib preflight: no GPU, training, or checkpoint writes.
"$PYTHON" - "$PROJECT_ROOT" "$N_EVAL_PER_BLOCK" "$SEED_STRIDE" "$BAYESIAN_WEIGHTS" "${METHODS[@]}" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
n, stride = map(int, sys.argv[2:4])
selected = sys.argv[5:]
if 'bayesian5001' in selected:
    raw = Path(sys.argv[4]).read_bytes()
    weights = json.loads(raw)
    keys = {'collision', 'theta', 'z-axis', 'velocity', 'distance', 'orientation',
            'eef_to_obj', 'obj_to_targ', 'object_orientation'}
    if set(weights) != keys or any(not math.isfinite(v) or v < 0 for v in weights.values()):
        raise SystemExit('Expected nine finite, nonnegative Bayesian5001 box weights.')
    print(f'bayesian5001: fixed cost weights; smoothness=0.01; SHA256={hashlib.sha256(raw).hexdigest()}')
seeds = [base + i * stride for base in (0, 5, 4, 10, 15) for i in range(n)]
if min(seeds) < 0 or max(seeds) >= 2**32:
    raise SystemExit('Episode seeds must fit in uint32.')
for name, path, kind in [
    ('ars4218', root/'ars_v2_linear_policy_4218.json', None),
    ('ppo5005best', root/'real_demo/real_demo/ppo_linear_policy_5005.json', 'best'),
    ('ppo5005besteval', root/'real_demo/real_demo/ppo_linear_policy_5005_best_eval.json', 'best_eval'),
]:
    if name not in selected:
        continue
    raw = path.read_bytes()
    ck = json.loads(raw)
    obs = len(ck['mu'])
    act = int(ck.get('act_dim', len(ck['params']) // (obs + 1)))
    if (ck.get('policy_type', 'linear'), obs, act, len(ck['params']), len(ck['var'])) != ('linear', 53, 10, 540, 53):
        raise SystemExit(f'{name}: expected a linear 53->10 actor, 540 parameters.')
    if kind and (ck.get('algorithm') != 'ppo' or ck.get('checkpoint_type') != kind):
        raise SystemExit(f'{name}: wrong algorithm/checkpoint type.')
    print(f'{name}: linear 53->10, 540 parameters; SHA256={hashlib.sha256(raw).hexdigest()}')
print(f'{len(seeds)} episodes per method/batch: bases 0, 5, 4, 10, 15, each repeated {n} times (seed stride 0).')
if any(name in selected for name in ('ars4218', 'ppo5005best', 'ppo5005besteval')):
    print('PPO uses bounded tanh residuals; ARS uses clipped residuals.')
PY

echo "Inference methods: ${METHODS[*]}"
echo "Batches: ${BATCH_SIZES[*]}; horizon=$NUM_STEPS; CEM iterations=$MAXITER_CEM; projection iterations=5"
echo "CEM key mode=$CEM_KEY_MODE; max steps=$MAX_STEPS; initial cube yaw=0; timestep=0.1 s"
echo "Every measured episode: --video --video_stride 1 --timed. One unreported warm-up per process."
echo "Results: $RESULTS_DIR"

if (( ! DRY_RUN )); then
    if [[ -e "$RESULTS_DIR" ]]; then
        echo "Refusing to mix with existing results: $RESULTS_DIR. Set a new RESULTS_DIR." >&2
        exit 2
    fi
    # Fail before starting the sweep if the selected environment lacks imports.
    "$PYTHON" "$SCRIPT" --help >/dev/null
    mkdir -p "$RESULTS_DIR/logs" "$RESULTS_DIR/npz" "$RESULTS_DIR/videos"
    # Keep the exact evaluator/runner and launch settings alongside this run.
    cp "$SCRIPT" "$RESULTS_DIR/jax_evaluate.py"
    cp "$PROJECT_ROOT/real_demo/real_demo/rl_trainer_batch.py" "$RESULTS_DIR/rl_trainer_batch.py"
    cp "${BASH_SOURCE[0]}" "$RESULTS_DIR/run_eval_box_lift_matched.sh"
    cp "$BAYESIAN_WEIGHTS" "$RESULTS_DIR/bayesian_5001.json"
    BAYESIAN_WEIGHTS="$RESULTS_DIR/bayesian_5001.json"
fi

for method in "${METHODS[@]}"; do
    if [[ "$method" == "bayesian5001" ]]; then
        method_args=(--weights_json "$BAYESIAN_WEIGHTS")
    elif [[ "$method" == "handtuned" || "$method" == "bayesian" ]]; then
        method_args=(--weights "$method")
    else
        method_args=(--load "${CHECKPOINTS[$method]}")
    fi
    for batch in "${BATCH_SIZES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            label="${method}_batch${batch}"
            tag="${label}_seed${seed}"
            npz_dir="$RESULTS_DIR/npz/$label"
            command=("$PYTHON" -u "$SCRIPT" "${method_args[@]}"
                --model_path "$MODEL_PATH"
                --num_batch "$batch" --num_steps "$NUM_STEPS"
                --maxiter_cem "$MAXITER_CEM" --max_steps "$MAX_STEPS"
                --init_cube_rot_deg 0
                --n_eval "$N_EVAL_PER_BLOCK" --seed "$seed" --seed_stride "$SEED_STRIDE"
                --cem_key_mode "$CEM_KEY_MODE"
                --timed --warmup_episodes 1 --video --video_stride 1
                --video_dir "$RESULTS_DIR/videos/$label" --video_tag "$tag"
                --save_npz "$npz_dir/eval_${label}_n${N_EVAL_PER_BLOCK}_seed${seed}.npz")
            if (( DRY_RUN )); then
                printf '%q ' "${command[@]}"
                printf '\n'
            else
                mkdir -p "$npz_dir"
                # pipefail stops the sweep immediately on an evaluation failure.
                "${command[@]}" 2>&1 | tee "$RESULTS_DIR/logs/$tag.log"
            fi
        done
    done
done
