#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate a trained DUASVS PPO checkpoint on multiple network domains (In-D + OOD).

Example:
  python evaluate_duasvs.py --ckpt checkpoints/duasvs_ppo.pt \\
    --eval fcc,data/fcc_httpgetmt_trace_1s_kbps.csv,splits/fcc_test_trace_ids.txt \\
    --eval monroe_sweden,data/monroe_sweden.csv,splits/monroe_sweden_ids.txt
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from envs.duasvs_env import DUASVSEnv


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.pi = nn.Linear(128, act_dim)
        self.v = nn.Linear(128, 1)

    def forward(self, obs):
        x = self.net(obs)
        return self.pi(x), self.v(x).squeeze(-1)


@torch.no_grad()
def act_greedy(model: ActorCritic, obs: np.ndarray, device: torch.device) -> int:
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    logits, _ = model.forward(obs_t)
    return int(torch.argmax(logits, dim=-1).item())


@dataclass
class EpisodeRow:
    domain: str
    trace_id: str
    ep_len: int
    ep_return: float
    sum_rebuf_s: float
    sum_waste_s: float
    sum_downloaded_mbit: float
    sum_waste_mbit: float
    sum_qoe: float
    wall_time_s: float


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--events_csv", type=str, default="data/big_matrix_valid_video.csv")
    p.add_argument("--behavior_mode", type=str, default="user_sessions", choices=["user_sessions", "video_index"])
    p.add_argument("--video_index_csv", type=str, default="data/video_index.csv")
    p.add_argument("--min_events", type=int, default=5)
    p.add_argument("--max_events", type=int, default=200)
    p.add_argument("--no_weight_by_views", action="store_true", help="(video_index mode) disable n_views weighting.")
    p.add_argument("--k_hist", type=int, default=8)
    p.add_argument("--lambda_waste", type=float, default=1.0)
    p.add_argument("--r_min_mbps", type=float, default=0.35, help="QoE log utility reference bitrate (Mbps).")
    p.add_argument("--rebuf_penalty", type=float, default=2.66, help="QoE rebuffering penalty coefficient (paper-style).")
    p.add_argument("--switch_penalty", type=float, default=1.0, help="QoE bitrate switching penalty coefficient.")
    p.add_argument(
        "--no_qoe_scale_by_watch_frac",
        action="store_true",
        help="Disable scaling bitrate utility by watched fraction (played_s/watch_s). "
             "If set, any non-zero playback grants full bitrate utility.",
    )
    p.add_argument(
        "--bitrates_mbps",
        type=str,
        default="0.35,0.7,1.2,2.5,5.0",
        help="Comma-separated bitrate ladder in Mbps. Default matches [180p,360p,480p,720p,1080p].",
    )
    p.add_argument(
        "--bitrate_labels",
        type=str,
        default="180p,360p,480p,720p,1080p",
        help="Comma-separated labels aligned with --bitrates_mbps.",
    )
    p.add_argument(
        "--prefetch_thresholds",
        type=str,
        default="1,2,4,6,8,12,18,24,30,36",
        help="Comma-separated prefetch thresholds in seconds for joint action space (bitrate x thresholds).",
    )

    p.add_argument("--max_videos", type=int, default=200)
    p.add_argument("--eval", action="append", required=True, help='Repeatable: "domain,traces_csv,ids_file"')

    p.add_argument("--out_dir", type=str, default="eval_outputs")
    p.add_argument("--run_name", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    thresholds = [float(x.strip()) for x in str(args.prefetch_thresholds).split(",") if x.strip() != ""]
    if not thresholds:
        raise ValueError("--prefetch_thresholds must be non-empty, e.g. '2,5,10,20,40'")

    bitrates = [float(x.strip()) for x in str(args.bitrates_mbps).split(",") if x.strip() != ""]
    if not bitrates:
        raise ValueError("--bitrates_mbps must be non-empty, e.g. '0.7,2.5,5.0,8.0'")
    labels = [x.strip() for x in str(args.bitrate_labels).split(",") if x.strip() != ""]
    if labels and len(labels) != len(bitrates):
        raise ValueError("--bitrate_labels length must match --bitrates_mbps length (or be empty).")
    if not labels:
        labels = [str(i) for i in range(len(bitrates))]

    # Probe env to infer dims
    domain0, csv0, ids0 = args.eval[0].split(",", 2)
    with open(ids0, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    probe = DUASVSEnv({
        "seed": int(args.seed),
        "events_csv": args.events_csv,
        "behavior_mode": str(args.behavior_mode),
        "video_index_csv": str(args.video_index_csv),
        "min_events": int(args.min_events),
        "max_events": int(args.max_events),
        "weight_by_views": (not bool(args.no_weight_by_views)),
        "traces_csv": csv0,
        "allowed_trace_ids": [ids[0]] if ids else None,
        "k_hist": int(args.k_hist),
        "r_min_mbps": float(args.r_min_mbps),
        "rebuf_penalty": float(args.rebuf_penalty),
        "switch_penalty": float(args.switch_penalty),
        "qoe_scale_by_watch_frac": (not bool(args.no_qoe_scale_by_watch_frac)),
        "lambda_waste": float(args.lambda_waste),
        "prefetch_thresholds_s": thresholds,
        "video_bitrates_mbps": bitrates,
        "video_bitrate_labels": labels,
    })
    obs_dim = int(probe.observation_space.shape[0])
    act_dim = int(probe.action_space.n)

    model = ActorCritic(obs_dim, act_dim).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    # Accept both legacy state_dict and new checkpoint dict {"model": ...}
    if isinstance(sd, dict) and "model" in sd:
        model.load_state_dict(sd["model"])
    else:
        model.load_state_dict(sd)
    model.eval()

    # Stream episode rows to disk so long evaluations (e.g., all Monroe traces) are observable
    # and don't lose all progress if interrupted.
    ep_path = os.path.join(out_dir, "episodes.csv")
    ep_f = open(ep_path, "w", newline="")
    ep_w = csv.DictWriter(ep_f, fieldnames=list(asdict(EpisodeRow("", "", 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)).keys()))
    ep_w.writeheader()
    ep_f.flush()

    # Running aggregates per domain for summary without storing all rows in memory.
    agg: Dict[str, Dict[str, float]] = {}
    agg_n: Dict[str, int] = {}

    def _agg_add(domain: str, r: EpisodeRow):
        if domain not in agg:
            agg[domain] = {
                "ep_return_sum": 0.0,
                "sum_rebuf_s_sum": 0.0,
                "sum_waste_s_sum": 0.0,
                "sum_downloaded_mbit_sum": 0.0,
                "sum_waste_mbit_sum": 0.0,
                "sum_qoe_sum": 0.0,
                "ep_len_sum": 0.0,
            }
            agg_n[domain] = 0
        agg_n[domain] += 1
        agg[domain]["ep_return_sum"] += float(r.ep_return)
        agg[domain]["sum_rebuf_s_sum"] += float(r.sum_rebuf_s)
        agg[domain]["sum_waste_s_sum"] += float(r.sum_waste_s)
        agg[domain]["sum_downloaded_mbit_sum"] += float(r.sum_downloaded_mbit)
        agg[domain]["sum_waste_mbit_sum"] += float(r.sum_waste_mbit)
        agg[domain]["sum_qoe_sum"] += float(r.sum_qoe)
        agg[domain]["ep_len_sum"] += float(r.ep_len)

    for spec in args.eval:
        domain, traces_csv, ids_file = spec.split(",", 2)
        with open(ids_file, "r", encoding="utf-8") as f:
            trace_ids = [line.strip() for line in f if line.strip()]
        print(f"[Eval] domain={domain} n_traces={len(trace_ids)}")

        # IMPORTANT: Reuse a single env per domain to avoid repeatedly loading large datasets
        # (events CSV, traces CSV) which can cause huge memory/time overhead.
        env: Optional[DUASVSEnv] = None
        try:
            env = DUASVSEnv({
                "seed": int(args.seed),
                "events_csv": args.events_csv,
                "behavior_mode": str(args.behavior_mode),
                "video_index_csv": str(args.video_index_csv),
                "min_events": int(args.min_events),
                "max_events": int(args.max_events),
                "weight_by_views": (not bool(args.no_weight_by_views)),
                "traces_csv": traces_csv,
                "allowed_trace_ids": trace_ids,  # restrict to this domain's ids
                "k_hist": int(args.k_hist),
                "r_min_mbps": float(args.r_min_mbps),
                "rebuf_penalty": float(args.rebuf_penalty),
                "switch_penalty": float(args.switch_penalty),
                "qoe_scale_by_watch_frac": (not bool(args.no_qoe_scale_by_watch_frac)),
                "lambda_waste": float(args.lambda_waste),
                "prefetch_thresholds_s": thresholds,
                "video_bitrates_mbps": bitrates,
                "video_bitrate_labels": labels,
            })
        except Exception as e:
            print(f"[Eval] ERROR: failed to create env for domain={domain}: {e}")
            raise

        for i, tid in enumerate(trace_ids, start=1):
            t0 = time.time()
            obs, info = env.reset(options={"first_trace_id": str(tid)})
            ep_ret = 0.0
            ep_len = 0
            last_info: Dict[str, Any] = dict(info) if isinstance(info, dict) else {}
            terminated = False
            truncated = False
            while not (terminated or truncated) and ep_len < int(args.max_videos):
                a = act_greedy(model, obs, device)
                obs, r, terminated, truncated, last_info = env.step(a)
                ep_ret += float(r)
                ep_len += 1
            t1 = time.time()
            row = EpisodeRow(
                domain=str(domain),
                trace_id=str(last_info.get("trace_id", tid)),
                ep_len=int(ep_len),
                ep_return=float(ep_ret),
                sum_rebuf_s=float(last_info.get("sum_rebuf_s", np.nan)),
                sum_waste_s=float(last_info.get("sum_waste_s", np.nan)),
                sum_downloaded_mbit=float(last_info.get("sum_downloaded_mbit", np.nan)),
                sum_waste_mbit=float(last_info.get("sum_waste_mbit", np.nan)),
                sum_qoe=float(last_info.get("sum_qoe", np.nan)),
                wall_time_s=float(t1 - t0),
            )
            ep_w.writerow(asdict(row))
            # flush periodically to make progress observable
            if i % 10 == 0:
                ep_f.flush()
            _agg_add(str(domain), row)
            if i % 10 == 0 or i == len(trace_ids):
                print(f"  {domain}: {i}/{len(trace_ids)} ep_return={ep_ret:.3f} len={ep_len}")

        # ensure domain flush
        ep_f.flush()

    ep_f.close()

    # summarize
    sum_rows = []
    for domain in sorted(agg.keys()):
        n = int(agg_n.get(domain, 0))
        if n <= 0:
            continue
        g = agg[domain]
        ep_return_mean = float(g["ep_return_sum"] / n)
        ep_len_mean = float(g["ep_len_sum"] / n)
        return_per_step_mean = float(ep_return_mean / max(1e-9, ep_len_mean))
        qoe_log_per_step_mean = float((float(g["sum_qoe_sum"]) / n) / max(1e-9, ep_len_mean))
        sum_rows.append({
            "domain": domain,
            "n_eps": n,
            "ep_return_mean": ep_return_mean,
            "sum_rebuf_s_mean": float(g["sum_rebuf_s_sum"] / n),
            "sum_waste_s_mean": float(g["sum_waste_s_sum"] / n),
            "sum_downloaded_mbit_mean": float(g["sum_downloaded_mbit_sum"] / n),
            "sum_waste_mbit_mean": float(g["sum_waste_mbit_sum"] / n),
            "sum_qoe_mean": float(g["sum_qoe_sum"] / n),
            "ep_len_mean": ep_len_mean,
            "return_per_step_mean": return_per_step_mean,
            "qoe_log_per_step_mean": qoe_log_per_step_mean,
        })
    df_sum = pd.DataFrame(sum_rows).sort_values("domain")
    sum_path = os.path.join(out_dir, "summary.csv")
    df_sum.to_csv(sum_path, index=False)

    print("Saved:")
    print(f"  {ep_path}")
    print(f"  {sum_path}")
    print(df_sum.to_string(index=False))


if __name__ == "__main__":
    main()


