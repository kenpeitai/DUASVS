#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Offline evaluation for FCCMobile using the exact same model + eval logic as train_duasvs_ppo.py.

This avoids config drift between training and evaluation.

Example:
  python eval_fccmobile_ckpt.py \
    --ckpt checkpoints/<run>_best.pt \
    --ids_file splits/fccmobile_traces_1hz_w120_s30_test_ids.txt \
    --traces_csv data/fccmobile_traces_1hz_w120_s30.csv \
    --n_traces 100 \
    --select_mode shuffle \
    --select_seed 0 \
    --eval_seeds 2026,2027,2028 \
    --out_name fccmobile_seed83_test_eval_100traces
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from envs.duasvs_env import DUASVSEnv
from train_duasvs_ppo import ActorCritic, eval_on_traces


def _parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip() != ""]


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip() != ""]


def _load_ids(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _select_ids(ids_all: List[str], *, n_traces: int, select_mode: str, select_seed: int) -> List[str]:
    if n_traces <= 0 or n_traces >= len(ids_all):
        return list(ids_all)
    if str(select_mode).lower() == "head":
        return list(ids_all[:n_traces])
    if str(select_mode).lower() == "shuffle":
        rng = np.random.default_rng(int(select_seed))
        idx = np.arange(len(ids_all))
        rng.shuffle(idx)
        sel = [ids_all[int(i)] for i in idx[:n_traces]]
        return sel
    raise ValueError(f"Unknown select_mode: {select_mode}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--out_name", type=str, required=True)

    p.add_argument("--ids_file", type=str, required=True)
    p.add_argument("--traces_csv", type=str, required=True)
    p.add_argument("--n_traces", type=int, default=100)
    p.add_argument("--select_mode", type=str, default="head", choices=["head", "shuffle"])
    p.add_argument("--select_seed", type=int, default=0)

    p.add_argument("--eval_seeds", type=str, default="2026", help="Comma-separated eval seeds, e.g. '2026,2027,2028'")
    p.add_argument("--max_videos", type=int, default=200)

    # --- env config (match training defaults) ---
    p.add_argument("--events_csv", type=str, default="data/big_matrix_valid_video.csv")
    p.add_argument("--behavior_mode", type=str, default="user_sessions")
    p.add_argument("--video_index_csv", type=str, default="data/video_index.csv")
    p.add_argument("--min_events", type=int, default=5)
    p.add_argument("--max_events", type=int, default=200)
    p.add_argument("--k_hist", type=int, default=8)

    p.add_argument("--qoe_alpha", type=float, default=1.0)
    p.add_argument("--r_min_mbps", type=float, default=0.35)
    p.add_argument("--rebuf_penalty", type=float, default=1.0)
    p.add_argument("--rebuf_penalty_mode", type=str, default="linear", choices=["linear", "cap", "log"])
    p.add_argument("--max_rebuf_s_per_video", type=float, default=0.0)
    p.add_argument("--switch_penalty", type=float, default=0.1)
    p.add_argument("--switch_penalty_mode", type=str, default="log", choices=["log", "abs"])
    p.add_argument("--lambda_waste", type=float, default=0.3)
    p.add_argument("--waste_saturate_k", type=float, default=1.0)

    p.add_argument("--bitrates_mbps", type=str, default="0.35,0.6,0.9,1.2,1.8,2.5,4.0")
    p.add_argument("--bitrate_labels", type=str, default="180p,270p,360p,480p,720p,1080p,1440p")
    p.add_argument("--prefetch_thresholds", type=str, default="1,3,6,10,16,30")

    p.add_argument("--trace_id_col", type=str, default="trace_id")
    p.add_argument("--trace_time_col", type=str, default="t")
    p.add_argument("--trace_bw_col", type=str, default="throughput_kbps")
    p.add_argument("--trace_bw_multiplier", type=float, default=1.0)
    p.add_argument("--trace_fill_mode", type=str, default="ffill")
    p.add_argument("--trace_exhaust_mode", type=str, default="drop_video", choices=["truncate", "next", "drop_video"])
    p.add_argument("--enable_future_prefetch", action="store_true")
    p.add_argument("--strict_no_prefetch_discard", action="store_true")

    args = p.parse_args()

    ids_all = _load_ids(str(args.ids_file))
    trace_ids = _select_ids(
        ids_all,
        n_traces=int(args.n_traces),
        select_mode=str(args.select_mode),
        select_seed=int(args.select_seed),
    )
    eval_seeds = [int(x) for x in _parse_csv_list(str(args.eval_seeds))]
    if not eval_seeds:
        raise ValueError("--eval_seeds must be non-empty")

    device = torch.device(str(args.device))

    bitrates = _parse_float_list(str(args.bitrates_mbps))
    labels = _parse_csv_list(str(args.bitrate_labels))
    thresholds = _parse_float_list(str(args.prefetch_thresholds))
    if labels and len(labels) != len(bitrates):
        raise ValueError("--bitrate_labels length must match --bitrates_mbps length")

    # Probe env dims (reuse the same env across seeds for speed)
    probe = DUASVSEnv(
        {
            "seed": int(eval_seeds[0]),
            "events_csv": str(args.events_csv),
            "behavior_mode": str(args.behavior_mode),
            "video_index_csv": str(args.video_index_csv),
            "min_events": int(args.min_events),
            "max_events": int(args.max_events),
            "traces_csv": str(args.traces_csv),
            "allowed_trace_ids": [str(trace_ids[0])] if trace_ids else None,
            "trace_id_col": str(args.trace_id_col),
            "trace_time_col": str(args.trace_time_col),
            "trace_bw_col": str(args.trace_bw_col),
            "trace_bw_multiplier": float(args.trace_bw_multiplier),
            "trace_fill_mode": str(args.trace_fill_mode),
            "k_hist": int(args.k_hist),
            "qoe_alpha": float(args.qoe_alpha),
            "r_min_mbps": float(args.r_min_mbps),
            "rebuf_penalty": float(args.rebuf_penalty),
            "rebuf_penalty_mode": str(args.rebuf_penalty_mode),
            "max_rebuf_s_per_video": float(args.max_rebuf_s_per_video),
            "switch_penalty": float(args.switch_penalty),
            "switch_penalty_mode": str(args.switch_penalty_mode),
            "lambda_waste": float(args.lambda_waste),
            "waste_saturate_k": float(args.waste_saturate_k),
            "prefetch_thresholds_s": thresholds,
            "video_bitrates_mbps": bitrates,
            "video_bitrate_labels": labels,
            "trace_exhaust_mode": str(args.trace_exhaust_mode),
            "allow_future_prefetch": bool(args.enable_future_prefetch),
            "strict_no_prefetch_discard": bool(args.strict_no_prefetch_discard),
        }
    )
    obs_dim = int(probe.observation_space.shape[0])
    act_dim = int(probe.action_space.n)

    model = ActorCritic(obs_dim, act_dim).to(device)
    ck = torch.load(str(args.ckpt), map_location=device)
    if isinstance(ck, dict) and "model" in ck:
        model.load_state_dict(ck["model"])
    else:
        model.load_state_dict(ck)
    model.eval()

    env_cfg: Dict = {
        "seed": int(eval_seeds[0]),
        "events_csv": str(args.events_csv),
        "behavior_mode": str(args.behavior_mode),
        "video_index_csv": str(args.video_index_csv),
        "min_events": int(args.min_events),
        "max_events": int(args.max_events),
        "traces_csv": str(args.traces_csv),
        "allowed_trace_ids": [str(x) for x in trace_ids],
        "trace_id_col": str(args.trace_id_col),
        "trace_time_col": str(args.trace_time_col),
        "trace_bw_col": str(args.trace_bw_col),
        "trace_bw_multiplier": float(args.trace_bw_multiplier),
        "trace_fill_mode": str(args.trace_fill_mode),
        "k_hist": int(args.k_hist),
        "qoe_alpha": float(args.qoe_alpha),
        "r_min_mbps": float(args.r_min_mbps),
        "rebuf_penalty": float(args.rebuf_penalty),
        "rebuf_penalty_mode": str(args.rebuf_penalty_mode),
        "max_rebuf_s_per_video": float(args.max_rebuf_s_per_video),
        "switch_penalty": float(args.switch_penalty),
        "switch_penalty_mode": str(args.switch_penalty_mode),
        "lambda_waste": float(args.lambda_waste),
        "waste_saturate_k": float(args.waste_saturate_k),
        "prefetch_thresholds_s": thresholds,
        "video_bitrates_mbps": bitrates,
        "video_bitrate_labels": labels,
        "trace_exhaust_mode": str(args.trace_exhaust_mode),
        "allow_future_prefetch": bool(args.enable_future_prefetch),
        "strict_no_prefetch_discard": bool(args.strict_no_prefetch_discard),
    }

    # Reuse a single env across all eval_seeds for speed (same CSV parsing cost).
    eval_env = DUASVSEnv({**env_cfg, "seed": int(eval_seeds[0])})

    out_csv = os.path.join(str(args.out_dir), f"{args.out_name}.csv")
    f = open(out_csv, "w", newline="")
    w = csv.DictWriter(
        f,
        fieldnames=[
            "ckpt",
            "ids_file",
            "traces_csv",
            "n_traces",
            "select_mode",
            "select_seed",
            "eval_seed",
            "return_per_step",
            "rebuf_s_per_step",
            "waste_mbit_per_step",
            "waste_raw_mean",
            "waste_raw_p50",
            "waste_raw_p90",
            "waste_ratio_mean",
            "waste_ratio_p50",
            "waste_ratio_p90",
            "waste_frac_mean",
            "waste_frac_p50",
            "waste_frac_p90",
            "waste_ratio_sat_frac",
        ],
    )
    w.writeheader()

    for es in eval_seeds:
        ev = eval_on_traces(
            model=model,
            device=device,
            env_cfg=env_cfg,
            trace_ids=trace_ids,
            eval_seed=int(es),
            max_videos=int(args.max_videos),
            reuse_env=eval_env,
        )
        row = {
            "ckpt": str(args.ckpt),
            "ids_file": str(args.ids_file),
            "traces_csv": str(args.traces_csv),
            "n_traces": int(len(trace_ids)),
            "select_mode": str(args.select_mode),
            "select_seed": int(args.select_seed),
            "eval_seed": int(es),
            "return_per_step": ev.get("return_per_step"),
            "rebuf_s_per_step": ev.get("rebuf_s_per_step"),
            "waste_mbit_per_step": ev.get("waste_mbit_per_step"),
            "waste_raw_mean": ev.get("waste_raw_mean"),
            "waste_raw_p50": ev.get("waste_raw_p50"),
            "waste_raw_p90": ev.get("waste_raw_p90"),
            "waste_ratio_mean": ev.get("waste_ratio_mean"),
            "waste_ratio_p50": ev.get("waste_ratio_p50"),
            "waste_ratio_p90": ev.get("waste_ratio_p90"),
            "waste_frac_mean": ev.get("waste_frac_mean"),
            "waste_frac_p50": ev.get("waste_frac_p50"),
            "waste_frac_p90": ev.get("waste_frac_p90"),
            "waste_ratio_sat_frac": ev.get("waste_ratio_sat_frac"),
        }
        w.writerow(row)
        f.flush()
        print(
            f"[test-eval] eval_seed={es} return/step={float(ev.get('return_per_step', float('nan'))):.4f} "
            f"rebuf/step={float(ev.get('rebuf_s_per_step', float('nan'))):.4f} "
            f"waste_mbit/step={float(ev.get('waste_mbit_per_step', float('nan'))):.4f}"
        )

    f.close()
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()

