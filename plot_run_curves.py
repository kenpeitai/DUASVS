#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot training curves for a DUASVS PPO run (with convergence-friendly smoothing).

Inputs:
  checkpoints/<run_name>_{eval,loss}.csv

Outputs:
  checkpoints/<run_name>_{eval,loss}.png
  checkpoints/<run_name>_{eval,loss}.svg

Eval plot includes:
  - raw return_per_step
  - rolling mean + ±1 std band (to visualize convergence)
  - best-so-far curve
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt


def _rolling_mean_std(y, window: int):
    s = pd.Series(y)
    m = s.rolling(window=window, min_periods=max(1, window // 2)).mean()
    sd = s.rolling(window=window, min_periods=max(1, window // 2)).std()
    return m.to_numpy(), sd.to_numpy()


def plot_eval(eval_csv: str, out_png: str, out_svg: str, *, smooth_window: int):
    df = pd.read_csv(eval_csv)
    if df.empty:
        raise ValueError(f"Empty eval csv: {eval_csv}")

    x = df["global_steps"].to_numpy()
    y_ret = df["return_per_step"].to_numpy()
    y_rebuf = df["rebuf_s_per_step"].to_numpy()
    y_waste = df["waste_mbit_per_step"].to_numpy()

    # smoothing & best-so-far
    m_ret, sd_ret = _rolling_mean_std(y_ret, int(max(1, smooth_window)))
    best_so_far = pd.Series(y_ret).cummax().to_numpy()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(x, y_ret, label="return_per_step (raw)", color="C0", linewidth=1.0, alpha=0.5)
    ax1.plot(x, m_ret, label=f"return_per_step (roll{int(max(1, smooth_window))})", color="C0", linewidth=2.0)
    ax1.fill_between(x, m_ret - sd_ret, m_ret + sd_ret, color="C0", alpha=0.15, linewidth=0)
    ax1.plot(x, best_so_far, label="return_per_step (best-so-far)", color="C4", linewidth=1.5, alpha=0.9)
    ax1.set_xlabel("global_steps")
    ax1.set_ylabel("return_per_step", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, y_rebuf, label="rebuf_s_per_step", color="C1", linewidth=1.2, alpha=0.9)
    ax2.plot(x, y_waste, label="waste_mbit_per_step", color="C2", linewidth=1.2, alpha=0.9)
    ax2.set_ylabel("rebuf_s_per_step / waste_mbit_per_step")

    # combined legend
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    fig.savefig(out_svg)
    plt.close(fig)


def plot_loss(loss_csv: str, out_png: str, out_svg: str, *, smooth_window: int):
    df = pd.read_csv(loss_csv)
    if df.empty:
        raise ValueError(f"Empty loss csv: {loss_csv}")

    x = df["global_steps"].to_numpy()
    y_ent = df["entropy"].to_numpy()
    y_pi = df["policy_loss"].to_numpy()
    y_v = df["value_loss"].to_numpy()

    m_ent, _ = _rolling_mean_std(y_ent, int(max(1, smooth_window)))

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(x, y_ent, label="entropy (raw)", color="C3", linewidth=1.0, alpha=0.5)
    ax1.plot(x, m_ent, label=f"entropy (roll{int(max(1, smooth_window))})", color="C3", linewidth=2.0)
    ax1.set_xlabel("global_steps")
    ax1.set_ylabel("entropy", color="C3")
    ax1.tick_params(axis="y", labelcolor="C3")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, y_pi, label="policy_loss", color="C0", linewidth=1.0, alpha=0.8)
    ax2.plot(x, y_v, label="value_loss", color="C1", linewidth=1.0, alpha=0.8)
    ax2.set_ylabel("policy_loss / value_loss")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    fig.savefig(out_svg)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    p.add_argument("--smooth_window", type=int, default=10, help="Rolling window (in eval/loss points) for smoothing.")
    args = p.parse_args()

    eval_csv = os.path.join(args.checkpoints_dir, f"{args.run_name}_eval.csv")
    loss_csv = os.path.join(args.checkpoints_dir, f"{args.run_name}_loss.csv")
    out_eval = os.path.join(args.checkpoints_dir, f"{args.run_name}_eval.png")
    out_loss = os.path.join(args.checkpoints_dir, f"{args.run_name}_loss.png")
    out_eval_svg = os.path.join(args.checkpoints_dir, f"{args.run_name}_eval.svg")
    out_loss_svg = os.path.join(args.checkpoints_dir, f"{args.run_name}_loss.svg")

    if not os.path.exists(eval_csv):
        raise FileNotFoundError(eval_csv)
    if not os.path.exists(loss_csv):
        raise FileNotFoundError(loss_csv)

    plot_eval(eval_csv, out_eval, out_eval_svg, smooth_window=int(args.smooth_window))
    plot_loss(loss_csv, out_loss, out_loss_svg, smooth_window=int(args.smooth_window))

    print("Saved:")
    print(" ", out_eval)
    print(" ", out_loss)
    print(" ", out_eval_svg)
    print(" ", out_loss_svg)


if __name__ == "__main__":
    main()


