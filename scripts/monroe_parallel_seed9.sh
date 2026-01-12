#!/usr/bin/env bash
set -euo pipefail

cd /home/huangyihe/PycharmProjects/DUASVS

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate DUASVS

# Seed9 baseline hyperparams (as used in your seed9 NOwatchfrac baseline)
SEED=9
DEVICE=${DEVICE:-cuda} # you only have 1 GPU; parallel runs will share it and run slower.
TOTAL_STEPS=${TOTAL_STEPS:-400000}

COMMON_ARGS=(
  --mode official
  --seed "$SEED"
  --device "$DEVICE"
  --total_steps "$TOTAL_STEPS"
  --rollout_steps 1024
  --max_rebuf_s_per_video 10
  --rebuf_penalty 4.0
  --switch_penalty 1.0
  --lambda_waste 0.5
  --r_min_mbps 0.2
  --lr 2e-4
  --ent_coef 0.025
  --clip_ratio 0.1
  --batch_size 256
  --train_iters 10
  --target_kl 0
  --no_qoe_scale_by_watch_frac
  --bitrates_mbps "0.35,0.7,1.2,2.5,5.0"
  --bitrate_labels "180p,360p,480p,720p,1080p"
  --prefetch_thresholds "1,2,4,6,8,12,18,24,30,36"
  # Keep eval cadence ~every 4096 env steps (like official): 4096/1024 = 4 updates
  --eval_every_updates 4
)

run_one () {
  local country="$1"   # italy/norway/spain/sweden
  local eval_n="$2"    # 48 or 179

  local traces_csv="data/monroe_${country}.csv"
  local train_ids="splits/monroe_${country}_train_ids.txt"
  local val_ids="splits/monroe_${country}_val_ids.txt"

  local run_name="monroe_${country}_seed9_r1024_b256_it10_train${TOTAL_STEPS}_NOwatchfrac"

  echo "[Launch] ${run_name}"
  nohup python train_duasvs_ppo.py \
    "${COMMON_ARGS[@]}" \
    --traces_csv "${traces_csv}" \
    --trace_ids_file "${train_ids}" \
    --eval_ids_file "${val_ids}" \
    --eval_n_traces "${eval_n}" \
    --run_name "${run_name}" \
    > "checkpoints/${run_name}.log" 2>&1 &
}

run_one italy 48
run_one norway 48
run_one spain 48
run_one sweden 179

echo
echo "Launched 4 jobs."
echo "Logs: checkpoints/monroe_*_seed9_train${TOTAL_STEPS}_NOwatchfrac.log"
echo "To monitor: tail -f checkpoints/<run_name>.log"


