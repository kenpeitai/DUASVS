#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stitch two runs' logs (train/eval csv) into a single continuous timeline on global_steps.

Use-case: when a run was resumed from a checkpoint and its CSV starts at a large global_steps,
we can "fill the missing early part" by stitching another run that started from 0.

Stitch rule (default):
  boundary = min(late.global_steps)
  keep early rows with global_steps < boundary
  keep late  rows with global_steps >= boundary

Optionally exclude drop_video rollback episodes (all-zero signature).
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _rolling_mean_std(y: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    s = pd.Series(y)
    w = int(max(1, window))
    m = s.rolling(window=w, min_periods=max(1, w // 2)).mean()
    sd = s.rolling(window=w, min_periods=max(1, w // 2)).std()
    return m.to_numpy(), sd.to_numpy()


def _is_drop_video_zero_row(df: pd.DataFrame) -> pd.Series:
    tol = 1e-12
    cols = ["ep_return", "sum_rebuf_s", "sum_waste_s", "sum_downloaded_mbit", "sum_waste_mbit"]
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"train.csv missing column: {c}")
    if "ep_len" not in df.columns:
        raise KeyError("train.csv missing column: ep_len")
    return (df["ep_len"].astype(int) == 1) & (df[cols].abs().max(axis=1) <= tol)


def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty csv: {path}")
    if "global_steps" not in df.columns:
        raise KeyError(f"Missing global_steps in {path}")
    return df


def stitch_by_global_steps(
    df_early: pd.DataFrame,
    df_late: pd.DataFrame,
    *,
    boundary: Optional[float],
) -> Tuple[pd.DataFrame, float]:
    if boundary is None:
        boundary = float(df_late["global_steps"].min())
    b = float(boundary)
    early_keep = df_early.loc[df_early["global_steps"].astype(float) < b]
    late_keep = df_late.loc[df_late["global_steps"].astype(float) >= b]
    out = pd.concat([early_keep, late_keep], axis=0, ignore_index=True)
    out = out.sort_values("global_steps", kind="stable").reset_index(drop=True)
    return out, b


def plot_stitched_train(
    df: pd.DataFrame,
    *,
    boundary: float,
    out_png: str,
    out_svg: Optional[str],
    smooth_window: int,
    y_mode: str,
):
    if str(y_mode) == "return_per_step":
        y = df["ep_return"].to_numpy() / np.maximum(1.0, df["ep_len"].to_numpy().astype(float))
        y_label = "ep_return / ep_len"
    elif str(y_mode) == "ep_return":
        y = df["ep_return"].to_numpy()
        y_label = "ep_return"
    else:
        raise ValueError(f"Unknown y_mode: {y_mode}")

    x = df["global_steps"].to_numpy().astype(float)
    m, sd = _rolling_mean_std(y, int(smooth_window))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, label=f"{y_label} (raw)", color="C0", linewidth=0.9, alpha=0.35)
    ax.plot(x, m, label=f"{y_label} (roll{int(max(1, smooth_window))})", color="C0", linewidth=2.0)
    ax.fill_between(x, m - sd, m + sd, color="C0", alpha=0.15, linewidth=0)
    ax.axvline(float(boundary), color="k", linewidth=1.0, alpha=0.35, label=f"stitch@{int(boundary)}")
    ax.set_xlim(left=0.0)
    ax.set_xlabel("global_steps (axis starts at 0)")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    if out_svg:
        fig.savefig(out_svg)
    plt.close(fig)


def plot_stitched_eval(
    df: pd.DataFrame,
    *,
    boundary: float,
    out_png: str,
    out_svg: Optional[str],
    smooth_window: int,
):
    x = df["global_steps"].to_numpy().astype(float)
    y_ret = df["return_per_step"].to_numpy().astype(float)
    m_ret, sd_ret = _rolling_mean_std(y_ret, int(smooth_window))
    best_so_far = pd.Series(y_ret).cummax().to_numpy()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(x, y_ret, label="return_per_step (raw)", color="C0", linewidth=1.0, alpha=0.5)
    ax1.plot(x, m_ret, label=f"return_per_step (roll{int(max(1, smooth_window))})", color="C0", linewidth=2.0)
    ax1.fill_between(x, m_ret - sd_ret, m_ret + sd_ret, color="C0", alpha=0.15, linewidth=0)
    ax1.plot(x, best_so_far, label="return_per_step (best-so-far)", color="C4", linewidth=1.5, alpha=0.9)
    ax1.axvline(float(boundary), color="k", linewidth=1.0, alpha=0.35, label=f"stitch@{int(boundary)}")
    ax1.set_xlim(left=0.0)
    ax1.set_xlabel("global_steps (axis starts at 0)")
    ax1.set_ylabel("return_per_step", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.grid(True, alpha=0.25)

    # Optional secondary axis if columns exist
    if "rebuf_s_per_step" in df.columns and "waste_mbit_per_step" in df.columns:
        ax2 = ax1.twinx()
        ax2.plot(x, df["rebuf_s_per_step"].to_numpy(), label="rebuf_s_per_step", color="C1", linewidth=1.2, alpha=0.9)
        ax2.plot(x, df["waste_mbit_per_step"].to_numpy(), label="waste_mbit_per_step", color="C2", linewidth=1.2, alpha=0.9)
        ax2.set_ylabel("rebuf_s_per_step / waste_mbit_per_step")
        lines = ax1.get_lines() + ax2.get_lines()
    else:
        lines = ax1.get_lines()

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    if out_svg:
        fig.savefig(out_svg)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    p.add_argument("--early_run", type=str, required=True)
    p.add_argument("--late_run", type=str, required=True)
    p.add_argument("--out_name", type=str, required=True)
    p.add_argument("--boundary", type=float, default=None, help="Stitch boundary in global_steps; default=min(late.global_steps).")
    p.add_argument("--exclude_drop_video_zeros", action="store_true")
    p.add_argument("--train_smooth_window", type=int, default=200)
    p.add_argument("--train_y_mode", type=str, default="return_per_step", choices=["return_per_step", "ep_return"])
    p.add_argument("--eval_smooth_window", type=int, default=10)
    args = p.parse_args()

    ck = str(args.checkpoints_dir)
    early_train = os.path.join(ck, f"{args.early_run}_train.csv")
    late_train = os.path.join(ck, f"{args.late_run}_train.csv")
    early_eval = os.path.join(ck, f"{args.early_run}_eval.csv")
    late_eval = os.path.join(ck, f"{args.late_run}_eval.csv")

    df_early_train = _read_csv(early_train)
    df_late_train = _read_csv(late_train)
    if bool(args.exclude_drop_video_zeros):
        m1 = _is_drop_video_zero_row(df_early_train)
        m2 = _is_drop_video_zero_row(df_late_train)
        df_early_train = df_early_train.loc[~m1].reset_index(drop=True)
        df_late_train = df_late_train.loc[~m2].reset_index(drop=True)

    stitched_train, b = stitch_by_global_steps(df_early_train, df_late_train, boundary=args.boundary)

    out_train_csv = os.path.join(ck, f"{args.out_name}_train_stitched.csv")
    stitched_train.to_csv(out_train_csv, index=False)

    out_train_png = os.path.join(ck, f"{args.out_name}_train_stitched.png")
    out_train_svg = os.path.join(ck, f"{args.out_name}_train_stitched.svg")
    plot_stitched_train(
        stitched_train,
        boundary=b,
        out_png=out_train_png,
        out_svg=out_train_svg,
        smooth_window=int(args.train_smooth_window),
        y_mode=str(args.train_y_mode),
    )

    # eval (if both exist)
    if os.path.exists(early_eval) and os.path.exists(late_eval):
        df_early_eval = _read_csv(early_eval)
        df_late_eval = _read_csv(late_eval)
        stitched_eval, b2 = stitch_by_global_steps(df_early_eval, df_late_eval, boundary=b)
        out_eval_csv = os.path.join(ck, f"{args.out_name}_eval_stitched.csv")
        stitched_eval.to_csv(out_eval_csv, index=False)

        out_eval_png = os.path.join(ck, f"{args.out_name}_eval_stitched.png")
        out_eval_svg = os.path.join(ck, f"{args.out_name}_eval_stitched.svg")
        plot_stitched_eval(
            stitched_eval,
            boundary=b2,
            out_png=out_eval_png,
            out_svg=out_eval_svg,
            smooth_window=int(args.eval_smooth_window),
        )
    else:
        out_eval_csv = None
        out_eval_png = None

    print("[stitch] boundary =", int(b))
    print("[stitch] wrote:", out_train_csv)
    print("[stitch] wrote:", out_train_png)
    if out_eval_csv:
        print("[stitch] wrote:", out_eval_csv)
    if out_eval_png:
        print("[stitch] wrote:", out_eval_png)


if __name__ == "__main__":
    main()

