#!/usr/bin/env bash
set -euo pipefail

cd /home/huangyihe/PycharmProjects/DUASVS
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate DUASVS

# Evaluate each country's trained best checkpoint on its own TEST traces.
# Note: run_name must match the training run_name.

DEVICE=${DEVICE:-cuda}
MAX_VIDEOS=${MAX_VIDEOS:-200}

COMMON_ENV=(
  --device "$DEVICE"
  --seed 0
  --max_videos "$MAX_VIDEOS"
  --r_min_mbps 0.2
  --rebuf_penalty 4.0
  --switch_penalty 1.0
  --lambda_waste 0.5
  --no_qoe_scale_by_watch_frac
  --bitrates_mbps "0.35,0.7,1.2,2.5,5.0"
  --bitrate_labels "180p,360p,480p,720p,1080p"
  --prefetch_thresholds "1,2,4,6,8,12,18,24,30,36"
)

eval_one () {
  local country="$1"
  local run_name="monroe_${country}_seed9_r1024_b256_it10_train400000_NOwatchfrac"
  local ckpt="checkpoints/${run_name}_best.pt"
  local traces_csv="data/monroe_${country}.csv"
  local ids_file="splits/monroe_${country}_test_ids.txt"
  local out_name="${run_name}_TEST"

  if [ ! -f "$ckpt" ]; then
    echo "SKIP: missing ckpt $ckpt (training not finished?)" >&2
    return 0
  fi
  echo "[Eval] ${out_name}"
  python evaluate_duasvs.py \
    --ckpt "$ckpt" \
    "${COMMON_ENV[@]}" \
    --eval "${country}_test,${traces_csv},${ids_file}" \
    --out_dir eval_outputs \
    --run_name "${out_name}"
}

eval_one italy
eval_one norway
eval_one spain
eval_one sweden


