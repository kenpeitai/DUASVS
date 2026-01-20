#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot In-D action CDFs (bitrate and prefetch-threshold separately) for a checkpoint.

We evaluate greedily on a selected set of FCCMobile traces and collect per-step actions.
Then decode the joint action:
  action = bitrate_idx * P + prefetch_idx

Outputs:
  - <out_name>_bitrate_cdf.png
  - <out_name>_threshold_cdf.png
  - <out_name>_bitrate_cdf.csv
  - <out_name>_threshold_cdf.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt

from envs.duasvs_env import DUASVSEnv
from train_duasvs_ppo import ActorCritic


@torch.no_grad()
def act_greedy(model: ActorCritic, obs: np.ndarray, device: torch.device) -> int:
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    logits, _ = model.forward(obs_t)
    return int(torch.argmax(logits, dim=-1).item())


@torch.no_grad()
def act_sample(model: ActorCritic, obs: np.ndarray, device: torch.device, *, gen: torch.Generator) -> int:
    """
    Sample an action from the policy distribution (Categorical over logits).
    Uses an explicit torch.Generator for reproducibility.
    """
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    logits, _ = model.forward(obs_t)
    # sample on CPU for generator compatibility and to avoid device-specific RNG differences
    probs = torch.softmax(logits.detach().cpu(), dim=-1).squeeze(0)
    a = torch.multinomial(probs, num_samples=1, replacement=True, generator=gen).item()
    return int(a)


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


def _write_cdf_csv(path: str, xs: List[float], pmf: np.ndarray, cdf: np.ndarray):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "pmf", "cdf"])
        for i, x in enumerate(xs):
            w.writerow([float(x), float(pmf[i]), float(cdf[i])])


def _plot_cdf(out_png: str, *, xs: List[float], pmf: np.ndarray, cdf: np.ndarray, x_label: str, title: str):
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.step(xs, cdf, where="post", color="C0", linewidth=2.0, label="CDF")
    ax.scatter(xs, cdf, color="C0", s=18, alpha=0.9)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Cumulative probability")
    ax.grid(True, alpha=0.25)
    ax.set_title(title)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--out_name", type=str, required=True)

    # In-D selection
    p.add_argument("--ids_file", type=str, required=True)
    p.add_argument("--traces_csv", type=str, required=True)
    p.add_argument("--n_traces", type=int, default=300)
    p.add_argument("--select_mode", type=str, default="shuffle", choices=["head", "shuffle"])
    p.add_argument("--select_seed", type=int, default=0)
    p.add_argument("--eval_seed", type=int, default=2026)
    p.add_argument("--max_videos", type=int, default=200)
    p.add_argument("--action_mode", type=str, default="greedy", choices=["greedy", "sample"])

    # env config (match training defaults used in our recent runs)
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

    os.makedirs(str(args.out_dir), exist_ok=True)
    device = torch.device(str(args.device))

    bitrates = _parse_float_list(str(args.bitrates_mbps))
    thresholds = _parse_float_list(str(args.prefetch_thresholds))
    labels = _parse_csv_list(str(args.bitrate_labels))
    if labels and len(labels) != len(bitrates):
        raise ValueError("--bitrate_labels length must match --bitrates_mbps length")

    ids_all = _load_ids(str(args.ids_file))
    trace_ids = _select_ids(
        ids_all,
        n_traces=int(args.n_traces),
        select_mode=str(args.select_mode),
        select_seed=int(args.select_seed),
    )
    if not trace_ids:
        raise ValueError("No trace ids selected")

    # Env config
    env_cfg: Dict = {
        "seed": int(args.eval_seed),
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

    env = DUASVSEnv({**env_cfg, "seed": int(args.eval_seed)})
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.n)

    model = ActorCritic(obs_dim, act_dim).to(device)
    ck = torch.load(str(args.ckpt), map_location=device)
    if isinstance(ck, dict) and "model" in ck:
        model.load_state_dict(ck["model"])
    else:
        model.load_state_dict(ck)
    model.eval()

    H = int(len(bitrates))
    P = int(len(thresholds))
    bitrate_counts = np.zeros(H, dtype=np.int64)
    thr_counts = np.zeros(P, dtype=np.int64)
    total_steps = 0

    for tid in trace_ids:
        # Deterministic session sampling & action sampling per (eval_seed, trace_id)
        session_seed = (hash((int(args.eval_seed), str(tid))) & 0xFFFFFFFF)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(session_seed))
        obs, _ = env.reset(seed=int(args.eval_seed), options={"first_trace_id": str(tid), "session_seed": int(session_seed)})
        terminated = False
        truncated = False
        ep_len = 0
        while not (terminated or truncated) and ep_len < int(args.max_videos):
            if str(args.action_mode) == "sample":
                a = act_sample(model, obs, device, gen=gen)
            else:
                a = act_greedy(model, obs, device)
            bidx = int(a // P)
            pidx = int(a % P)
            if 0 <= bidx < H:
                bitrate_counts[bidx] += 1
            if 0 <= pidx < P:
                thr_counts[pidx] += 1
            total_steps += 1
            obs, _, terminated, truncated, _ = env.step(int(a))
            ep_len += 1

    if total_steps <= 0:
        raise RuntimeError("No steps collected (unexpected).")

    bitrate_pmf = bitrate_counts.astype(np.float64) / float(total_steps)
    thr_pmf = thr_counts.astype(np.float64) / float(total_steps)
    bitrate_cdf = np.cumsum(bitrate_pmf)
    thr_cdf = np.cumsum(thr_pmf)

    out_b_csv = os.path.join(str(args.out_dir), f"{args.out_name}_bitrate_cdf.csv")
    out_t_csv = os.path.join(str(args.out_dir), f"{args.out_name}_threshold_cdf.csv")
    _write_cdf_csv(out_b_csv, bitrates, bitrate_pmf, bitrate_cdf)
    _write_cdf_csv(out_t_csv, thresholds, thr_pmf, thr_cdf)

    out_b_png = os.path.join(str(args.out_dir), f"{args.out_name}_bitrate_cdf.png")
    out_t_png = os.path.join(str(args.out_dir), f"{args.out_name}_threshold_cdf.png")

    title_base = f"In-D action CDF (n_traces={len(trace_ids)}, steps={total_steps}, eval_seed={int(args.eval_seed)})"
    _plot_cdf(out_b_png, xs=bitrates, pmf=bitrate_pmf, cdf=bitrate_cdf, x_label="bitrate (Mbps)", title=title_base + " - bitrate")
    _plot_cdf(out_t_png, xs=thresholds, pmf=thr_pmf, cdf=thr_cdf, x_label="prefetch threshold (s)", title=title_base + " - threshold")

    print("Saved:", out_b_png)
    print("Saved:", out_t_png)
    print("Saved:", out_b_csv)
    print("Saved:", out_t_csv)


if __name__ == "__main__":
    main()

