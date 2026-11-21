#!/usr/bin/env bash
# Matched hand-tuned box-lift inference, one fraction, batch and seed at a time.
set -euo pipefail

dry_run=0
if [[ $# == 1 && $1 == --dry-run ]]; then
    dry_run=1
elif (( $# )); then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi

root=/home/aks-lab/manipulator_mujoco_comp/manipulator_mujoco
python=${PYTHON:-python3}
evaluator="$root/real_demo/real_demo/jax_evaluate.py"
model_dir="$root/real_demo/ur5e_hande_mjx"
results=${RESULTS_DIR:-"$root/real_demo/real_demo/eval_box_lift_handtuned_ess_$(date +%Y%m%d_%H%M%S)_$$"}
seeds=(0 5 4 10 15)
episodes=20

echo "Sweep output: $results"
if (( ! dry_run )); then
    "$python" "$evaluator" --help >/dev/null
    mkdir -p -- "$(dirname -- "$results")"
    # An existing run may hold completed episodes. Always use a new directory.
    mkdir -- "$results"
    mkdir -- "$results/logs"
    git -C "$root" rev-parse HEAD > "$results/git_commit.txt"
    sha256sum "${BASH_SOURCE[0]}" "$evaluator" \
        "$root/real_demo/real_demo/"{ARS,rl_trainer_batch}.py \
        "$root/real_demo/sampling_based_planner/"{mjx_planner,mpc_planner}.py \
        "$model_dir/"{scene,objects,ur5e_dual_arm_robotiq_hande0}.xml \
        > "$results/input_sha256.txt"
fi

for fraction in 0.001; do
    for batch in 50 150 250; do
        label="handtuned_ess${fraction}_batch${batch}"
        for seed in "${seeds[@]}"; do
            tag="${label}_seed${seed}"
            command=("$python" -u "$evaluator"
                --model_path "$model_dir/scene.xml"
                --weights handtuned --temperature_ess_fraction "$fraction"
                --num_batch "$batch" --num_steps 15 --maxiter_cem 3 --max_steps 300
                --init_cube_rot_deg 0 --n_eval "$episodes" --seed "$seed" --seed_stride 0
                --cem_key_mode episode --timed --warmup_episodes 1 --video --video_stride 1
                --video_dir "$results/videos/$label" --video_tag "$tag"
                --save_npz "$results/npz/$label/eval_${label}_n${episodes}_seed${seed}.npz"
                --episode_npz_dir "$results/episodes/$label/$tag")
            if (( dry_run )); then
                printf '%q ' "${command[@]}"
                printf '\n'
            else
                echo "Running $tag"
                { printf '%q ' "${command[@]}"; printf '\n'; } >> "$results/commands.sh"
                "${command[@]}" 2>&1 | tee "$results/logs/$tag.log"
            fi
        done
    done
done
