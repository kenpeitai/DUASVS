#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate a constant-action (fixed policy) baseline table on the same eval traces as training.

For each discrete action a in {0..A-1}, run deterministic-ish evaluation:
- fixed trace ids (use a selected list file if provided)
- reproducible session sampling per (eval_seed, trace_id)
- fixed action selection (always choose action a)

Outputs a CSV with per-step metrics, plus the action->(bitrate, threshold) decoding.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure repo root is on sys.path so `envs/` can be imported when running as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from envs.duasvs_env import DUASVSEnv


def _read_ids(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        xs = [line.strip() for line in f if line.strip()]
    # de-dup preserving order
    out: List[str] = []
    seen = set()
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(str(x))
    return out


def _decode_action(a: int, n_prefetch: int) -> Tuple[int, int]:
    bi = int(a // n_prefetch)
    ti = int(a % n_prefetch)
    return bi, ti


def eval_constant_action(
    *,
    env: DUASVSEnv,
    trace_ids: List[str],
    eval_seed: int,
    max_videos: int,
    action: int,
) -> Dict[str, float]:
    ep_returns: List[float] = []
    ep_lens: List[int] = []
    ep_rebuf: List[float] = []
    ep_waste_s: List[float] = []
    ep_dl_mbit: List[float] = []
    ep_waste_mbit: List[float] = []

    for tid in trace_ids:
        session_seed = (hash((int(eval_seed), str(tid))) & 0xFFFFFFFF)
        obs, _ = env.reset(seed=int(eval_seed), options={"first_trace_id": str(tid), "session_seed": int(session_seed)})
        ep_ret = 0.0
        ep_len = 0
        last_info: Dict = {}
        terminated = False
        truncated = False
        while not (terminated or truncated) and ep_len < int(max_videos):
            obs, r, terminated, truncated, last_info = env.step(int(action))
            ep_ret += float(r)
            ep_len += 1
        ep_returns.append(float(ep_ret))
        ep_lens.append(int(ep_len))
        if isinstance(last_info, dict):
            ep_rebuf.append(float(last_info.get("sum_rebuf_s", np.nan)))
            ep_waste_s.append(float(last_info.get("sum_waste_s", np.nan)))
            ep_dl_mbit.append(float(last_info.get("sum_downloaded_mbit", np.nan)))
            ep_waste_mbit.append(float(last_info.get("sum_waste_mbit", np.nan)))

    rps = float(np.mean([ep_returns[i] / max(1, ep_lens[i]) for i in range(len(ep_returns))]))
    rebuf_ps = float(np.mean([ep_rebuf[i] / max(1, ep_lens[i]) for i in range(len(ep_rebuf))])) if ep_rebuf else float("nan")
    waste_s_ps = float(np.mean([ep_waste_s[i] / max(1, ep_lens[i]) for i in range(len(ep_waste_s))])) if ep_waste_s else float("nan")
    dl_mbit_ps = float(np.mean([ep_dl_mbit[i] / max(1, ep_lens[i]) for i in range(len(ep_dl_mbit))])) if ep_dl_mbit else float("nan")
    waste_mbit_ps = float(np.mean([ep_waste_mbit[i] / max(1, ep_lens[i]) for i in range(len(ep_waste_mbit))])) if ep_waste_mbit else float("nan")

    return {
        "n_eps": float(len(ep_returns)),
        "ep_return_mean": float(np.mean(ep_returns)) if ep_returns else float("nan"),
        "ep_len_mean": float(np.mean(ep_lens)) if ep_lens else 0.0,
        "return_per_step": rps,
        "rebuf_s_per_step": rebuf_ps,
        "waste_s_per_step": waste_s_ps,
        "downloaded_mbit_per_step": dl_mbit_ps,
        "waste_mbit_per_step": waste_mbit_ps,
    }


def main() -> None:
    p = argparse.ArgumentParser()

    # eval traces
    p.add_argument("--eval_trace_ids_selected", type=str, required=True, help="Path to *_eval_trace_ids_selected.txt")
    p.add_argument("--eval_seed", type=int, default=1000)
    p.add_argument("--max_videos", type=int, default=200)

    # env config (defaults match current training defaults in this repo)
    p.add_argument("--events_csv", type=str, default="data/big_matrix_valid_video.csv")
    p.add_argument("--behavior_mode", type=str, default="user_sessions", choices=["user_sessions", "video_index"])
    p.add_argument("--video_index_csv", type=str, default="data/video_index.csv")
    p.add_argument("--min_events", type=int, default=5)
    p.add_argument("--max_events", type=int, default=200)
    p.add_argument("--no_weight_by_views", action="store_true")

    p.add_argument("--traces_csv", type=str, default="data/fccmobile_traces_1hz_w120_s30.csv")
    p.add_argument("--trace_id_col", type=str, default="trace_id")
    p.add_argument("--trace_time_col", type=str, default="t")
    p.add_argument("--trace_bw_col", type=str, default="throughput_kbps")
    p.add_argument("--trace_bw_multiplier", type=float, default=1.0)
    p.add_argument("--trace_fill_mode", type=str, default="ffill", choices=["ffill", "zero", "hold"])
    p.add_argument("--trace_exhaust_mode", type=str, default="drop_video", choices=["truncate", "next", "drop_video"])

    # reward/action space
    p.add_argument("--k_hist", type=int, default=8)
    p.add_argument("--r_min_mbps", type=float, default=0.35)
    p.add_argument("--rebuf_penalty", type=float, default=1.0)
    p.add_argument("--rebuf_penalty_mode", type=str, default="linear", choices=["linear", "cap", "log"])
    p.add_argument("--rebuf_cap_s", type=float, default=12.0)
    p.add_argument("--rebuf_log_scale_s", type=float, default=2.0)
    p.add_argument("--switch_penalty", type=float, default=0.1)
    p.add_argument("--switch_penalty_mode", type=str, default="log", choices=["log", "abs"])
    p.add_argument("--qoe_alpha", type=float, default=1.0)
    p.add_argument("--no_qoe_scale_by_watch_frac", action="store_true")
    p.add_argument("--lambda_waste", type=float, default=0.3)
    p.add_argument("--waste_saturate_k", type=float, default=1.0)
    p.add_argument("--max_rebuf_s_per_video", type=float, default=0.0)
    p.add_argument("--enable_future_prefetch", action="store_true")

    p.add_argument("--bitrates_mbps", type=str, default="0.35,0.6,0.9,1.2,1.8,2.5,4.0")
    p.add_argument("--bitrate_labels", type=str, default="180p,270p,360p,480p,720p,1080p,1440p")
    p.add_argument("--prefetch_thresholds", type=str, default="1,3,6,10,16,30")

    # output
    p.add_argument("--out_csv", type=str, required=True)

    args = p.parse_args()

    trace_ids = _read_ids(str(args.eval_trace_ids_selected))
    if not trace_ids:
        raise RuntimeError("Empty eval trace id list.")

    thresholds = [float(x.strip()) for x in str(args.prefetch_thresholds).split(",") if x.strip() != ""]
    bitrates = [float(x.strip()) for x in str(args.bitrates_mbps).split(",") if x.strip() != ""]
    labels = [x.strip() for x in str(args.bitrate_labels).split(",") if x.strip() != ""]
    if labels and len(labels) != len(bitrates):
        raise ValueError("--bitrate_labels length must match --bitrates_mbps length (or be empty).")
    if not labels:
        labels = [str(i) for i in range(len(bitrates))]

    env = DUASVSEnv({
        "seed": int(args.eval_seed),
        "events_csv": str(args.events_csv),
        "behavior_mode": str(args.behavior_mode),
        "video_index_csv": str(args.video_index_csv),
        "min_events": int(args.min_events),
        "max_events": int(args.max_events),
        "weight_by_views": (not bool(args.no_weight_by_views)),
        "traces_csv": str(args.traces_csv),
        "trace_id_col": str(args.trace_id_col),
        "trace_time_col": str(args.trace_time_col),
        "trace_bw_col": str(args.trace_bw_col),
        "trace_bw_multiplier": float(args.trace_bw_multiplier),
        "trace_fill_mode": str(args.trace_fill_mode),
        "trace_exhaust_mode": str(args.trace_exhaust_mode),
        "allowed_trace_ids": [str(x) for x in trace_ids],
        "k_hist": int(args.k_hist),
        "r_min_mbps": float(args.r_min_mbps),
        "rebuf_penalty": float(args.rebuf_penalty),
        "rebuf_penalty_mode": str(args.rebuf_penalty_mode),
        "rebuf_cap_s": float(args.rebuf_cap_s),
        "rebuf_log_scale_s": float(args.rebuf_log_scale_s),
        "switch_penalty": float(args.switch_penalty),
        "switch_penalty_mode": str(args.switch_penalty_mode),
        "qoe_alpha": float(args.qoe_alpha),
        "qoe_scale_by_watch_frac": (not bool(args.no_qoe_scale_by_watch_frac)),
        "lambda_waste": float(args.lambda_waste),
        "waste_saturate_k": float(args.waste_saturate_k),
        "allow_future_prefetch": bool(args.enable_future_prefetch),
        "max_rebuf_s_per_video": (1e18 if float(args.max_rebuf_s_per_video) <= 0 else float(args.max_rebuf_s_per_video)),
        "prefetch_thresholds_s": thresholds,
        "video_bitrates_mbps": bitrates,
        "video_bitrate_labels": labels,
    })

    act_dim = int(env.action_space.n)
    P = int(len(thresholds))

    rows: List[Dict[str, float]] = []
    for a in range(act_dim):
        bi, ti = _decode_action(int(a), int(P))
        s = eval_constant_action(
            env=env,
            trace_ids=trace_ids,
            eval_seed=int(args.eval_seed),
            max_videos=int(args.max_videos),
            action=int(a),
        )
        rows.append({
            "action": int(a),
            "bitrate_idx": int(bi),
            "prefetch_idx": int(ti),
            "bitrate_mbps": float(bitrates[bi]) if 0 <= bi < len(bitrates) else float("nan"),
            "bitrate_label": str(labels[bi]) if 0 <= bi < len(labels) else str(bi),
            "prefetch_s": float(thresholds[ti]) if 0 <= ti < len(thresholds) else float("nan"),
            **s,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["return_per_step"], ascending=False).reset_index(drop=True)

    out_csv = str(args.out_csv)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)

    # Print top-10 for quick look
    show = df.head(10)[[
        "action", "bitrate_label", "bitrate_mbps", "prefetch_s",
        "return_per_step", "rebuf_s_per_step", "waste_mbit_per_step", "downloaded_mbit_per_step"
    ]]
    print(show.to_string(index=False))
    print(f"\nWrote: {out_csv}")


if __name__ == "__main__":
    main()

