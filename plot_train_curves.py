#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot episode-level training curves from checkpoints/<run>_train.csv.

Typical usage:
  python plot_train_curves.py --run_name <run>
  python plot_train_curves.py --run_name <run> --exclude_drop_video_zeros

Also supports concatenating multiple runs (e.g., seed71-75) and marking boundaries:
  python plot_train_curves.py --concat_runs runA,runB,runC --exclude_drop_video_zeros
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


def _rolling_mean_std(y: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    s = pd.Series(y)
    w = int(max(1, window))
    m = s.rolling(window=w, min_periods=max(1, w // 2)).mean()
    sd = s.rolling(window=w, min_periods=max(1, w // 2)).std()
    return m.to_numpy(), sd.to_numpy()


def _is_drop_video_zero_row(df: pd.DataFrame) -> pd.Series:
    # Strict signature for the "drop_video rollback + truncated episode at first step" case:
    # ep_len == 1 AND (ep_return, sum_rebuf_s, sum_waste_s, sum_downloaded_mbit, sum_waste_mbit) all zero.
    # Note: we use a small tolerance for float formatting.
    tol = 1e-12
    cols = ["ep_return", "sum_rebuf_s", "sum_waste_s", "sum_downloaded_mbit", "sum_waste_mbit"]
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"train.csv missing column: {c}")
    if "ep_len" not in df.columns:
        raise KeyError("train.csv missing column: ep_len")
    return (df["ep_len"].astype(int) == 1) & (df[cols].abs().max(axis=1) <= tol)


@dataclass
class TrainSeries:
    run_name: str
    df: pd.DataFrame
    excluded_cnt: int
    total_cnt: int


def _load_train_series(checkpoints_dir: str, run_name: str, *, exclude_drop_video_zeros: bool) -> TrainSeries:
    train_csv = os.path.join(checkpoints_dir, f"{run_name}_train.csv")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(train_csv)
    df = pd.read_csv(train_csv)
    if df.empty:
        raise ValueError(f"Empty train csv: {train_csv}")

    excluded_cnt = 0
    total_cnt = int(len(df))
    if bool(exclude_drop_video_zeros):
        mask = _is_drop_video_zero_row(df)
        excluded_cnt = int(mask.sum())
        df = df.loc[~mask].reset_index(drop=True)

    return TrainSeries(run_name=run_name, df=df, excluded_cnt=excluded_cnt, total_cnt=total_cnt)


def plot_single(
    series: TrainSeries,
    out_png: str,
    out_svg: Optional[str],
    *,
    smooth_window: int,
    y_mode: str,
    x_mode: str,
):
    df = series.df
    if str(x_mode) == "global_steps":
        x = df["global_steps"].to_numpy()
        x_label = "global_steps"
        force_x0 = False
    elif str(x_mode) == "global_steps_start0":
        # Keep real global_steps (no shifting), but force axis to start at 0 for "from init" view.
        x = df["global_steps"].to_numpy()
        x_label = "global_steps (axis starts at 0)"
        force_x0 = True
    elif str(x_mode) == "global_steps_from0":
        x0 = float(df["global_steps"].iloc[0])
        x = (df["global_steps"].to_numpy().astype(float) - x0)
        x_label = "global_steps (from 0)"
        force_x0 = False
    elif str(x_mode) == "episode_index":
        x = np.arange(len(df), dtype=np.int64)
        x_label = "episode_index"
        force_x0 = False
    else:
        raise ValueError(f"Unknown x_mode: {x_mode}")

    if str(y_mode) == "ep_return":
        y = df["ep_return"].to_numpy()
        y_label = "ep_return"
    elif str(y_mode) == "return_per_step":
        y = (df["ep_return"].to_numpy() / np.maximum(1.0, df["ep_len"].to_numpy().astype(float)))
        y_label = "ep_return / ep_len"
    else:
        raise ValueError(f"Unknown y_mode: {y_mode}")

    m, sd = _rolling_mean_std(y, int(smooth_window))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, label=f"{y_label} (raw)", color="C0", linewidth=0.9, alpha=0.45)
    ax.plot(x, m, label=f"{y_label} (roll{int(max(1, smooth_window))})", color="C0", linewidth=2.0)
    ax.fill_between(x, m - sd, m + sd, color="C0", alpha=0.15, linewidth=0)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    if bool(force_x0):
        ax.set_xlim(left=0.0)

    title = series.run_name
    if series.excluded_cnt > 0:
        frac = series.excluded_cnt / max(1, series.total_cnt)
        title += f" (excluded drop_video zeros: {series.excluded_cnt}/{series.total_cnt}={frac:.2%})"
    ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    if out_svg:
        fig.savefig(out_svg)
    plt.close(fig)


def plot_concat(
    series_list: List[TrainSeries],
    out_png: str,
    out_svg: Optional[str],
    *,
    smooth_window: int,
    y_mode: str,
):
    # Concatenate by episode index; also maintain boundaries.
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    boundaries: List[int] = []
    labels: List[str] = []

    ep_cursor = 0
    for s in series_list:
        df = s.df
        n = int(len(df))
        x_ep = np.arange(ep_cursor, ep_cursor + n, dtype=np.int64)
        if str(y_mode) == "ep_return":
            y = df["ep_return"].to_numpy()
            y_label = "ep_return"
        elif str(y_mode) == "return_per_step":
            y = (df["ep_return"].to_numpy() / np.maximum(1.0, df["ep_len"].to_numpy().astype(float)))
            y_label = "ep_return / ep_len"
        else:
            raise ValueError(f"Unknown y_mode: {y_mode}")

        xs.append(x_ep)
        ys.append(y)
        ep_cursor += n
        boundaries.append(ep_cursor)
        labels.append(s.run_name)

    x = np.concatenate(xs) if xs else np.array([], dtype=np.int64)
    y_all = np.concatenate(ys) if ys else np.array([], dtype=np.float64)
    if len(x) == 0:
        raise ValueError("No data to plot for concat runs")

    m, sd = _rolling_mean_std(y_all, int(smooth_window))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, y_all, label=f"{y_label} (raw)", color="C0", linewidth=0.8, alpha=0.35)
    ax.plot(x, m, label=f"{y_label} (roll{int(max(1, smooth_window))})", color="C0", linewidth=2.0)
    ax.fill_between(x, m - sd, m + sd, color="C0", alpha=0.15, linewidth=0)

    # boundaries
    for b in boundaries[:-1]:
        ax.axvline(b, color="k", linewidth=0.8, alpha=0.25)

    title = "concat: " + ",".join(labels)
    excl = sum(s.excluded_cnt for s in series_list)
    tot = sum(s.total_cnt for s in series_list)
    if excl > 0:
        title += f" (excluded drop_video zeros: {excl}/{tot}={excl/max(1,tot):.2%})"
    ax.set_title(title)

    ax.set_xlabel("episode_index (concatenated)")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    if out_svg:
        fig.savefig(out_svg)
    plt.close(fig)


def _short_concat_name(run_names: List[str]) -> str:
    """
    Build a filesystem-friendly short name for concatenated runs.
    Prefer extracting seed numbers like 'seed71' from run names.
    """
    seeds: List[int] = []
    for rn in run_names:
        m = re.search(r"seed(\d+)", rn)
        if m:
            try:
                seeds.append(int(m.group(1)))
            except Exception:
                pass
    if seeds:
        seeds_sorted = sorted(set(seeds))
        if len(seeds_sorted) == 1:
            return f"seed{seeds_sorted[0]}"
        return f"seed{seeds_sorted[0]}-{seeds_sorted[-1]}"
    return f"concat_{len(run_names)}runs"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    p.add_argument("--run_name", type=str, default=None, help="Single run name (without _train.csv suffix).")
    p.add_argument("--concat_runs", type=str, default=None, help="Comma-separated run names to concatenate.")
    p.add_argument(
        "--exclude_drop_video_zeros",
        action="store_true",
        help="Exclude episodes with ep_len==1 and all-zero accumulators (drop_video rollback signature).",
    )
    p.add_argument("--smooth_window", type=int, default=200, help="Rolling window (in episodes) for smoothing.")
    p.add_argument("--y_mode", type=str, default="return_per_step", choices=["return_per_step", "ep_return"])
    p.add_argument(
        "--x_mode",
        type=str,
        default="global_steps",
        choices=["global_steps", "global_steps_start0", "global_steps_from0", "episode_index"],
        help="X-axis mode for single-run plots.",
    )
    p.add_argument("--out_png", type=str, default=None)
    p.add_argument("--out_svg", type=str, default=None)
    args = p.parse_args()

    if (args.run_name is None) == (args.concat_runs is None):
        raise ValueError("Specify exactly one of --run_name or --concat_runs")

    if args.run_name is not None:
        s = _load_train_series(
            str(args.checkpoints_dir),
            str(args.run_name),
            exclude_drop_video_zeros=bool(args.exclude_drop_video_zeros),
        )
        out_png = args.out_png
        if not out_png:
            suffix = "train_reward_excl0" if bool(args.exclude_drop_video_zeros) else "train_reward"
            out_png = os.path.join(str(args.checkpoints_dir), f"{s.run_name}_{suffix}.png")
        out_svg = args.out_svg
        if out_svg is None:
            suffix = "train_reward_excl0" if bool(args.exclude_drop_video_zeros) else "train_reward"
            out_svg = os.path.join(str(args.checkpoints_dir), f"{s.run_name}_{suffix}.svg")

        print(
            f"[plot_train] {s.run_name}: episodes={len(s.df)} (excluded={s.excluded_cnt}/{s.total_cnt}) -> {out_png}"
        )
        plot_single(
            s,
            out_png,
            out_svg,
            smooth_window=int(args.smooth_window),
            y_mode=str(args.y_mode),
            x_mode=str(args.x_mode),
        )
        return

    run_names = [x.strip() for x in str(args.concat_runs).split(",") if x.strip()]
    if not run_names:
        raise ValueError("Empty --concat_runs")
    series_list = [
        _load_train_series(
            str(args.checkpoints_dir),
            rn,
            exclude_drop_video_zeros=bool(args.exclude_drop_video_zeros),
        )
        for rn in run_names
    ]
    out_png = args.out_png
    if not out_png:
        suffix = "ep_return_concat_excl0" if bool(args.exclude_drop_video_zeros) else "ep_return_concat"
        short = _short_concat_name(run_names)
        out_png = os.path.join(str(args.checkpoints_dir), f"{short}_{suffix}.png")
    out_svg = args.out_svg
    if out_svg is None:
        suffix = "ep_return_concat_excl0" if bool(args.exclude_drop_video_zeros) else "ep_return_concat"
        short = _short_concat_name(run_names)
        out_svg = os.path.join(str(args.checkpoints_dir), f"{short}_{suffix}.svg")

    excl = sum(s.excluded_cnt for s in series_list)
    tot = sum(s.total_cnt for s in series_list)
    kept = sum(len(s.df) for s in series_list)
    print(f"[plot_train] concat runs={run_names}: kept_eps={kept} (excluded={excl}/{tot}) -> {out_png}")
    plot_concat(series_list, out_png, out_svg, smooth_window=int(args.smooth_window), y_mode=str(args.y_mode))


if __name__ == "__main__":
    main()

