#!/usr/bin/env bash
# Queued follow-up for the active ball-lift sweep.
# Each fraction has its own archive-compatible npz/handtuned_batch{N}/ tree.
set -euo pipefail

script_dir="/home/aks-lab/colcon_ws_36_hw/src/manipulator_mujoco/real_demo/real_demo"
repo_root=$(cd -- "$script_dir/../.." && pwd)
python_bin=${PYTHON_BIN:-/home/aks-lab/manipulator_env/bin/python}
scene="$repo_root/real_demo/ur5e_hande_mjx/scene.xml"
archive="$script_dir/eval_sweep_100_ball_lift_bayes_retrained_20260929_191418"
reference="$archive/checkpoints/bayesian_weights.json"
eval_script="$script_dir/jax_evaluate.py"
export PYTHONPATH="$repo_root/real_demo${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cuda
export MUJOCO_GL="${MUJOCO_GL:-egl}"

fractions=(0.001)
batches=(50 150 250)
seeds=(0 1 2 3 4)
common_args=(
    --weight_type handtuned --bayesian_weights_path "$reference" --model_path "$scene"
    --n_eval 20 --num_steps 15 --maxiter_cem 2 --max_steps 300 --timeout 0
    --fold_step_keys --save_video --video_start 1 --video_stride 1
)

sweep=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
echo "Waiting for original sweep launcher PID $1 to exit"
tail --pid="$1" -f /dev/null
echo "Starting ESS 0.001 in $sweep/ess_0.001"
sha256sum "$reference" "$scene" "$(dirname "$scene")"/{objects,ur5e_dual_arm_robotiq_hande0}.xml \
    "$eval_script" "$script_dir"/{ARS,rl_trainer_batch,evaluation_output}.py \
    "$repo_root/real_demo/sampling_based_planner"/{mpc_planner,mjx_planner}.py \
    > "$sweep/ess_0.001/artifact_hashes.sha256"

for fraction in "${fractions[@]}"; do
    result_dir="$sweep/ess_$fraction"
    mkdir -p "$result_dir/logs"
    for batch in "${batches[@]}"; do
        for seed in "${seeds[@]}"; do
            echo "ESS fraction $fraction | batch $batch | seed $seed"
            cmd=(
                "$python_bin" -u "$eval_script" "${common_args[@]}"
                --temperature_ess_fraction "$fraction" --batch_size "$batch" --seed "$seed"
                --out_dir "$result_dir/npz/handtuned_batch$batch"
                --video_dir "$result_dir/videos/handtuned"
            )
            printf '%q ' "${cmd[@]}" >> "$result_dir/run_commands.sh"
            printf '\n' >> "$result_dir/run_commands.sh"
            "${cmd[@]}" 2>&1 | tee "$result_dir/logs/eval_handtuned_b${batch}_ino${seed}.log"
        done
    done
    "$python_bin" "$script_dir/analyze_ars_ppo_eval.py" "$result_dir"
done
echo "Finished ESS 0.001 in $sweep/ess_0.001"
