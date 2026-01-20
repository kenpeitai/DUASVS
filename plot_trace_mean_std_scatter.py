#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot per-trace throughput mean/std scatter for multiple domains (e.g., FCCMobile train vs Monroe).

We compute, for each trace_id:
  mean_mbps = mean(throughput_kbps / 1000)
  std_mbps  = std(throughput_kbps / 1000)

FCCMobile CSV can be large, so we stream with chunks + online aggregation to keep memory low.

Example:
  python plot_trace_mean_std_scatter.py \
    --fcc_csv data/fccmobile_traces_1hz_w120_s30.csv \
    --fcc_ids_file splits/fccmobile_traces_1hz_w120_s30_train_ids.txt \
    --monroe_csvs data/monroe_sweden.csv,data/monroe_norway.csv,data/monroe_spain.csv,data/monroe_italy.csv \
    --monroe_ids_files splits/monroe_sweden_ids.txt,splits/monroe_norway_ids.txt,splits/monroe_spain_ids.txt,splits/monroe_italy_ids.txt \
    --max_points_fcc 3000 \
    --max_points_monroe 3000 \
    --seed 0 \
    --out checkpoints/fccmobile_train_vs_monroe_mean_std.png
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip() != ""]


def _load_ids(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


@dataclass
class Agg:
    n: int = 0
    s: float = 0.0
    s2: float = 0.0

    def add(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        self.n += int(x.size)
        self.s += float(x.sum())
        self.s2 += float(np.square(x).sum())

    def mean_std(self) -> Tuple[float, float]:
        if self.n <= 0:
            return float("nan"), float("nan")
        mean = self.s / float(self.n)
        ex2 = self.s2 / float(self.n)
        var = max(0.0, ex2 - mean * mean)
        std = float(np.sqrt(var))
        return float(mean), float(std)


def compute_mean_std_stream(
    *,
    traces_csv: str,
    ids: Optional[Sequence[str]],
    trace_id_col: str = "trace_id",
    bw_col: str = "throughput_kbps",
    bw_multiplier: float = 1.0,
    chunksize: int = 2_000_000,
) -> pd.DataFrame:
    """
    Stream a large trace CSV and compute per-trace mean/std of throughput in Mbps.
    If ids is provided, only those trace_ids are aggregated.
    """
    ids_set = set(map(str, ids)) if ids is not None else None
    agg: Dict[str, Agg] = {}

    usecols = [trace_id_col, bw_col]
    for chunk in pd.read_csv(traces_csv, usecols=usecols, chunksize=int(chunksize)):
        if trace_id_col not in chunk.columns or bw_col not in chunk.columns:
            raise KeyError(f"CSV missing columns: need {trace_id_col} and {bw_col}, got {chunk.columns.tolist()}")

        # Normalize types and filter ids if needed
        tid = chunk[trace_id_col].astype(str)
        if ids_set is not None:
            m = tid.isin(ids_set)
            if not bool(m.any()):
                continue
            chunk = chunk.loc[m]
            tid = chunk[trace_id_col].astype(str)

        bw_mbps = (chunk[bw_col].astype(np.float64) * float(bw_multiplier)) / 1000.0
        # group within chunk to reduce per-row python overhead
        g = pd.DataFrame({"trace_id": tid.values, "bw_mbps": bw_mbps.values}).groupby("trace_id")["bw_mbps"]
        for trace_id, series in g:
            a = agg.get(trace_id)
            if a is None:
                a = Agg()
                agg[trace_id] = a
            a.add(series.to_numpy(dtype=np.float64))

    rows = []
    for trace_id, a in agg.items():
        mean, std = a.mean_std()
        rows.append({"trace_id": trace_id, "mean_mbps": mean, "std_mbps": std})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("trace_id").reset_index(drop=True)
    return df


def _maybe_sample(df: pd.DataFrame, *, max_points: int, seed: int) -> pd.DataFrame:
    if int(max_points) <= 0 or df.empty or len(df) <= int(max_points):
        return df
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(df), size=int(max_points), replace=False)
    return df.iloc[np.sort(idx)].reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fcc_csv", type=str, default="data/fccmobile_traces_1hz_w120_s30.csv")
    p.add_argument("--fcc_ids_file", type=str, default="splits/fccmobile_traces_1hz_w120_s30_train_ids.txt")

    p.add_argument("--monroe_csvs", type=str, default="data/monroe_sweden.csv,data/monroe_norway.csv,data/monroe_spain.csv,data/monroe_italy.csv")
    p.add_argument("--monroe_ids_files", type=str, default="splits/monroe_sweden_ids.txt,splits/monroe_norway_ids.txt,splits/monroe_spain_ids.txt,splits/monroe_italy_ids.txt")

    p.add_argument("--trace_id_col", type=str, default="trace_id")
    p.add_argument("--bw_col", type=str, default="throughput_kbps")
    p.add_argument("--bw_multiplier", type=float, default=1.0)
    p.add_argument("--chunksize", type=int, default=2_000_000)

    p.add_argument("--max_points_fcc", type=int, default=3000)
    p.add_argument("--max_points_monroe", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--title", type=str, default="Trace throughput mean/std (train vs Monroe)")
    p.add_argument("--out", type=str, default="checkpoints/fccmobile_train_vs_monroe_mean_std.png")
    args = p.parse_args()

    fcc_ids = _load_ids(str(args.fcc_ids_file)) if str(args.fcc_ids_file) else None
    monroe_csvs = _parse_csv_list(str(args.monroe_csvs))
    monroe_ids_files = _parse_csv_list(str(args.monroe_ids_files)) if str(args.monroe_ids_files) else []
    if monroe_ids_files and (len(monroe_ids_files) != len(monroe_csvs)):
        raise ValueError("--monroe_ids_files length must match --monroe_csvs length (or be empty)")

    # compute FCC train
    df_fcc = compute_mean_std_stream(
        traces_csv=str(args.fcc_csv),
        ids=fcc_ids,
        trace_id_col=str(args.trace_id_col),
        bw_col=str(args.bw_col),
        bw_multiplier=float(args.bw_multiplier),
        chunksize=int(args.chunksize),
    )
    df_fcc = _maybe_sample(df_fcc, max_points=int(args.max_points_fcc), seed=int(args.seed))

    # compute Monroe (per-country)
    monroe_dfs: List[Tuple[str, pd.DataFrame]] = []
    for i, csv_path in enumerate(monroe_csvs):
        ids = None
        if monroe_ids_files:
            ids = _load_ids(monroe_ids_files[i])
        # label from filename: monroe_sweden.csv -> sweden
        base = os.path.basename(str(csv_path))
        label = base.replace("monroe_", "").replace(".csv", "")
        df = compute_mean_std_stream(
            traces_csv=str(csv_path),
            ids=ids,
            trace_id_col=str(args.trace_id_col),
            bw_col=str(args.bw_col),
            bw_multiplier=float(args.bw_multiplier),
            chunksize=min(int(args.chunksize), 500_000),  # monroe is smaller, smaller chunk is fine
        )
        df = _maybe_sample(df, max_points=int(args.max_points_monroe), seed=int(args.seed) + 100 + i)
        monroe_dfs.append((label, df))

    # Plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(str(args.out)) or ".", exist_ok=True)
    plt.figure(figsize=(8, 6))

    if not df_fcc.empty:
        plt.scatter(
            df_fcc["mean_mbps"].to_numpy(),
            df_fcc["std_mbps"].to_numpy(),
            s=10,
            alpha=0.25,
            label=f"FCCMobile train (n={len(df_fcc)})",
            c="#1f77b4",
            edgecolors="none",
        )

    # distinct colors for monroe countries
    palette = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#17becf"]
    for j, (label, df) in enumerate(monroe_dfs):
        if df.empty:
            continue
        plt.scatter(
            df["mean_mbps"].to_numpy(),
            df["std_mbps"].to_numpy(),
            s=14,
            alpha=0.35,
            label=f"Monroe {label} (n={len(df)})",
            c=palette[j % len(palette)],
            edgecolors="none",
        )

    plt.xlabel("mean throughput (Mbps)")
    plt.ylabel("std throughput (Mbps)")
    plt.title(str(args.title))
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(str(args.out), dpi=200)
    print("Saved:", str(args.out))


if __name__ == "__main__":
    main()

