#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate a trained checkpoint on multiple trace domains using the SAME eval function as training:
  - `train_duasvs_ppo.eval_on_traces`

This avoids reward/env config drift compared to older standalone evaluators.

Example:
  python eval_multi_domain_ckpt.py \
    --ckpt checkpoints/<run>_best.pt \
    --eval_seed 2026 \
    --out_name seed83_best_inD300_ood300 \
    --eval "fccmobile_test,data/fccmobile_traces_1hz_w120_s30.csv,splits/fccmobile_traces_1hz_w120_s30_test_ids.txt,300,shuffle,0" \
    --eval "fcc_test,data/fcc_httpgetmt_trace_1s_kbps.csv,splits/fcc_test_trace_ids.txt,300,shuffle,0" \
    --eval "monroe_sweden,data/monroe_sweden.csv,splits/monroe_sweden_test_ids.txt,300,shuffle,0"
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

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
        return [ids_all[int(i)] for i in idx[:n_traces]]
    raise ValueError(f"Unknown select_mode: {select_mode}")


@dataclass
class EvalSpec:
    domain: str
    traces_csv: str
    ids_file: str
    n_traces: int
    select_mode: str
    select_seed: int


def _parse_eval_spec(s: str) -> EvalSpec:
    # "domain,traces_csv,ids_file,n_traces,select_mode,select_seed"
    parts = s.split(",")
    if len(parts) != 6:
        raise ValueError(f"--eval must have 6 comma-separated fields, got {len(parts)}: {s}")
    domain, traces_csv, ids_file, n_traces, select_mode, select_seed = parts
    return EvalSpec(
        domain=str(domain),
        traces_csv=str(traces_csv),
        ids_file=str(ids_file),
        n_traces=int(n_traces),
        select_mode=str(select_mode),
        select_seed=int(select_seed),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--out_name", type=str, required=True)

    p.add_argument("--eval_seed", type=int, default=2026)
    p.add_argument("--max_videos", type=int, default=200)
    p.add_argument("--eval", action="append", required=True, help="domain,traces_csv,ids_file,n_traces,select_mode,select_seed")

    # --- env config (match our training defaults) ---
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

    device = torch.device(str(args.device))
    eval_seed = int(args.eval_seed)

    bitrates = _parse_float_list(str(args.bitrates_mbps))
    labels = _parse_csv_list(str(args.bitrate_labels))
    thresholds = _parse_float_list(str(args.prefetch_thresholds))
    if labels and len(labels) != len(bitrates):
        raise ValueError("--bitrate_labels length must match --bitrates_mbps length")

    specs = [_parse_eval_spec(x) for x in (args.eval or [])]

    # Probe dims using first spec + first id
    ids0 = _load_ids(specs[0].ids_file)
    if not ids0:
        raise ValueError(f"Empty ids file: {specs[0].ids_file}")
    probe = DUASVSEnv(
        {
            "seed": int(eval_seed),
            "events_csv": str(args.events_csv),
            "behavior_mode": str(args.behavior_mode),
            "video_index_csv": str(args.video_index_csv),
            "min_events": int(args.min_events),
            "max_events": int(args.max_events),
            "traces_csv": str(specs[0].traces_csv),
            "allowed_trace_ids": [str(ids0[0])],
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

    out_csv = os.path.join(str(args.out_dir), f"{args.out_name}.csv")
    os.makedirs(str(args.out_dir), exist_ok=True)
    f = open(out_csv, "w", newline="")
    w = csv.DictWriter(
        f,
        fieldnames=[
            "domain",
            "ckpt",
            "eval_seed",
            "ids_file",
            "traces_csv",
            "n_traces_selected",
            "select_mode",
            "select_seed",
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

    for spec in specs:
        ids_all = _load_ids(spec.ids_file)
        trace_ids = _select_ids(
            ids_all,
            n_traces=int(spec.n_traces),
            select_mode=str(spec.select_mode),
            select_seed=int(spec.select_seed),
        )
        env_cfg: Dict = {
            "seed": int(eval_seed),
            "events_csv": str(args.events_csv),
            "behavior_mode": str(args.behavior_mode),
            "video_index_csv": str(args.video_index_csv),
            "min_events": int(args.min_events),
            "max_events": int(args.max_events),
            "traces_csv": str(spec.traces_csv),
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

        # Reuse one env per domain for speed
        env = DUASVSEnv({**env_cfg, "seed": int(eval_seed)})
        ev = eval_on_traces(
            model=model,
            device=device,
            env_cfg=env_cfg,
            trace_ids=trace_ids,
            eval_seed=int(eval_seed),
            max_videos=int(args.max_videos),
            reuse_env=env,
        )
        row = {
            "domain": str(spec.domain),
            "ckpt": str(args.ckpt),
            "eval_seed": int(eval_seed),
            "ids_file": str(spec.ids_file),
            "traces_csv": str(spec.traces_csv),
            "n_traces_selected": int(len(trace_ids)),
            "select_mode": str(spec.select_mode),
            "select_seed": int(spec.select_seed),
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
            f"[multi-eval] domain={spec.domain} n={len(trace_ids)} "
            f"return/step={float(ev.get('return_per_step', float('nan'))):.4f}"
        )

    f.close()
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()

