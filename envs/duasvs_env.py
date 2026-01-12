"""
Per-video short-form streaming simulator for OOD evaluation (as described in 复现说明.md).

Design goals:
- Decision granularity: per-video (choose bitrate for the next video).
- User behavior comes from KuaiRec-style logs: (user_id, video_id, video_duration, play_duration).
- Network dynamics comes from independent throughput traces (FCC/MONROE/etc).
- OOD is induced by swapping the trace distribution at test time.

This environment is a per-video simulator and is independent from any older chunk/slot-based implementations.

State (minimal, stable):
  - last K throughputs (Mbps, normalized/clipped)
  - buffer_seconds (seconds)
  - current_video_length (seconds)
  - prev_bitrate_index (normalized scalar)

Action:
  - bitrate_index in {0..H-1}

Step dynamics (simple continuous-time approximation):
  - Agent picks bitrate r for current video i (size = bitrate * video_length).
  - Download happens against the trace throughput timeline; playback consumes buffer at 1x.
  - Rebuffer occurs if buffer hits 0 while downloading.
  - User watches play_time seconds; if play_time < video_length, the remaining downloaded seconds are wasted.
  - Episode ends when the user's session ends OR the trace ends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    import gym  # type: ignore
    from gym import spaces  # type: ignore


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))


@dataclass
class VideoEvent:
    user_id: str
    video_id: str
    video_len_s: float
    watch_s: float


class KuaiRecSessionSampler:
    """
    Build per-user time-ordered sessions from KuaiRec-style logs.
    Expected columns:
      - user_id
      - video_id
      - video_duration  (ms)
      - play_duration   (ms)
      - timestamp or time-like ordering column
    """

    def __init__(
        self,
        *,
        events_csv: str,
        user_col: str = "user_id",
        video_col: str = "video_id",
        video_dur_col: str = "video_duration",
        play_dur_col: str = "play_duration",
        time_col: str = "timestamp",
        seed: int = 0,
        min_events: int = 5,
        max_events: int = 200,
    ):
        self.events_csv = str(events_csv)
        self.user_col = str(user_col)
        self.video_col = str(video_col)
        self.video_dur_col = str(video_dur_col)
        self.play_dur_col = str(play_dur_col)
        self.time_col = str(time_col)
        self.min_events = int(min_events)
        self.max_events = int(max_events)
        self._rng = np.random.default_rng(int(seed))

        df = pd.read_csv(self.events_csv)
        required = {self.user_col, self.video_col, self.video_dur_col, self.play_dur_col}
        missing = [c for c in sorted(required) if c not in df.columns]
        if missing:
            raise ValueError(f"KuaiRec events CSV missing required columns: {missing}. Got columns={list(df.columns)}")
        if self.time_col not in df.columns:
            # Fall back to file order if no explicit timestamp
            df[self.time_col] = np.arange(len(df), dtype=np.int64)

        # Normalize types and derive seconds
        df[self.user_col] = df[self.user_col].astype(str)
        df[self.video_col] = df[self.video_col].astype(str)
        df[self.video_dur_col] = df[self.video_dur_col].astype(float)
        df[self.play_dur_col] = df[self.play_dur_col].astype(float)
        df[self.time_col] = df[self.time_col].astype(float)

        # Pre-split by user for fast sampling
        self._by_user: Dict[str, pd.DataFrame] = {}
        for uid, g in df.groupby(self.user_col, sort=False):
            gg = g.sort_values(self.time_col, kind="mergesort")
            if len(gg) >= self.min_events:
                self._by_user[str(uid)] = gg
        if not self._by_user:
            raise ValueError("No users with enough events to form sessions.")
        self._user_ids = list(self._by_user.keys())

    def sample_session(self) -> List[VideoEvent]:
        uid = str(self._rng.choice(self._user_ids))
        g = self._by_user[uid]

        n = int(len(g))
        max_len = int(min(self.max_events, n))
        min_len = int(min(self.min_events, max_len))
        L = int(self._rng.integers(min_len, max_len + 1))
        if n == L:
            start = 0
        else:
            start = int(self._rng.integers(0, n - L + 1))
        seg = g.iloc[start : start + L]

        out: List[VideoEvent] = []
        for _, row in seg.iterrows():
            video_len_s = float(max(1.0, row[self.video_dur_col] / 1000.0))
            watch_s = float(max(0.0, row[self.play_dur_col] / 1000.0))
            watch_s = float(min(watch_s, video_len_s))
            out.append(
                VideoEvent(
                    user_id=str(uid),
                    video_id=str(row[self.video_col]),
                    video_len_s=video_len_s,
                    watch_s=watch_s,
                )
            )
        return out

    def reseed(self, seed: int) -> None:
        self._rng = np.random.default_rng(int(seed))


class VideoIndexSessionSampler:
    """
    Session generator using global (all-users) per-video statistics from `video_index.csv`.

    Expected columns (see data/video_index.csv in this repo):
      - video_id
      - video_duration_ms
      - watch_ratio_mean
      - watch_ratio_p10 (optional; fallback to mean)
      - watch_ratio_p90 (optional; fallback to mean)
      - n_views (optional; used for weighted sampling)

    This mode removes per-user heterogeneity and samples a synthetic session by drawing videos
    from the global distribution and sampling watch_ratio from (mean, p10, p90).
    """

    def __init__(
        self,
        *,
        index_csv: str,
        seed: int = 0,
        min_events: int = 5,
        max_events: int = 200,
        weight_by_views: bool = True,
    ):
        self.index_csv = str(index_csv)
        self.min_events = int(min_events)
        self.max_events = int(max_events)
        self.weight_by_views = bool(weight_by_views)
        self._rng = np.random.default_rng(int(seed))

        df = pd.read_csv(self.index_csv)
        required = {"video_id", "video_duration_ms", "watch_ratio_mean"}
        missing = [c for c in sorted(required) if c not in df.columns]
        if missing:
            raise ValueError(f"video_index CSV missing required columns: {missing}. Got columns={list(df.columns)}")

        if "watch_ratio_p10" not in df.columns:
            df["watch_ratio_p10"] = df["watch_ratio_mean"]
        if "watch_ratio_p90" not in df.columns:
            df["watch_ratio_p90"] = df["watch_ratio_mean"]

        df["video_id"] = df["video_id"].astype(str)
        df["video_duration_ms"] = df["video_duration_ms"].astype(float)
        df["watch_ratio_mean"] = df["watch_ratio_mean"].astype(float)
        df["watch_ratio_p10"] = df["watch_ratio_p10"].astype(float)
        df["watch_ratio_p90"] = df["watch_ratio_p90"].astype(float)

        w = None
        if self.weight_by_views and "n_views" in df.columns:
            vv = df["n_views"].astype(float).to_numpy()
            vv = np.where(np.isfinite(vv) & (vv > 0), vv, 0.0)
            s = float(vv.sum())
            if s > 0:
                w = (vv / s).astype(np.float64)
        self._df = df.reset_index(drop=True)
        self._w = w

    def _sample_watch_ratio(self, mean: float, p10: float, p90: float) -> float:
        mean = float(np.clip(mean, 0.0, 1.0))
        p10 = float(np.clip(p10, 0.0, 1.0))
        p90 = float(np.clip(p90, 0.0, 1.0))
        # approximate std from percentiles under Normal assumption (p90-p10 ≈ 2.56*std)
        std = float(max(1e-6, (p90 - p10) / 2.56))
        r = float(self._rng.normal(mean, std))
        return float(np.clip(r, 0.0, 1.0))

    def sample_session(self) -> List[VideoEvent]:
        n = int(len(self._df))
        if n <= 0:
            raise RuntimeError("video_index is empty.")

        L = int(self._rng.integers(int(self.min_events), int(self.max_events) + 1))
        out: List[VideoEvent] = []
        for _ in range(L):
            idx = int(self._rng.choice(n, p=self._w))
            row = self._df.iloc[idx]
            video_len_s = float(max(1.0, float(row["video_duration_ms"]) / 1000.0))
            ratio = self._sample_watch_ratio(
                mean=float(row["watch_ratio_mean"]),
                p10=float(row["watch_ratio_p10"]),
                p90=float(row["watch_ratio_p90"]),
            )
            watch_s = float(min(video_len_s, max(0.0, video_len_s * ratio)))
            out.append(VideoEvent(
                user_id="global",
                video_id=str(row["video_id"]),
                video_len_s=video_len_s,
                watch_s=watch_s,
            ))
        return out

    def reseed(self, seed: int) -> None:
        self._rng = np.random.default_rng(int(seed))


class TracePool:
    """
    Load throughput traces from a CSV into 1Hz arrays.
    Expected schema (default in this repo for FCC/MONROE converted exports):
      - trace_id
      - t
      - throughput_kbps
    """

    def __init__(
        self,
        *,
        traces_csv: str,
        id_col: str = "trace_id",
        time_col: str = "t",
        bw_col: str = "throughput_kbps",
        bw_multiplier: float = 1.0,
        allowed_trace_ids: Optional[List[str]] = None,
    ):
        df = pd.read_csv(str(traces_csv))
        df[id_col] = df[id_col].astype(str)
        if allowed_trace_ids is not None:
            allowed = set(map(str, allowed_trace_ids))
            df = df[df[id_col].isin(allowed)]
        if df.empty:
            raise ValueError("No trace rows after filtering. Check traces_csv and allowed_trace_ids.")

        traces: Dict[str, np.ndarray] = {}
        for tid, g in df.groupby(id_col):
            t_sec = np.floor(g[time_col].astype(float).to_numpy()).astype(int)
            bw_kbps = g[bw_col].astype(float).to_numpy() * float(bw_multiplier)
            max_sec = int(t_sec.max())
            bw_1hz = np.zeros(max_sec + 1, dtype=np.float32)
            counts = np.zeros(max_sec + 1, dtype=np.int32)
            for s, v in zip(t_sec, bw_kbps):
                if s < 0:
                    continue
                bw_1hz[s] += float(v)
                counts[s] += 1
            prev = float(bw_kbps[0])
            for s in range(max_sec + 1):
                if counts[s] > 0:
                    bw_1hz[s] /= counts[s]
                    prev = float(bw_1hz[s])
                else:
                    bw_1hz[s] = prev
            bw_1hz = np.clip(bw_1hz, 1e-3, None)
            traces[str(tid)] = bw_1hz
        if not traces:
            raise ValueError("No traces loaded.")
        self.traces = traces
        self.trace_ids = sorted(traces.keys())


class DUASVSEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, config: Dict):
        super().__init__()

        # --- behavior ---
        self.behavior_mode: str = str(config.get("behavior_mode", "user_sessions")).strip().lower()
        if self.behavior_mode not in {"user_sessions", "video_index"}:
            raise ValueError("behavior_mode must be one of: user_sessions, video_index")

        self.events_csv = str(config.get("events_csv", "data/big_matrix_valid_video.csv"))
        self.video_index_csv = str(config.get("video_index_csv", "data/video_index.csv"))
        min_events = int(config.get("min_events", 5))
        max_events = int(config.get("max_events", 200))
        if self.behavior_mode == "video_index":
            self.session_sampler = VideoIndexSessionSampler(
                index_csv=self.video_index_csv,
                seed=int(config.get("seed", 0)),
                min_events=min_events,
                max_events=max_events,
                weight_by_views=bool(config.get("weight_by_views", True)),
            )
        else:
            self.session_sampler = KuaiRecSessionSampler(
                events_csv=self.events_csv,
                user_col=str(config.get("user_col", "user_id")),
                video_col=str(config.get("video_col", "video_id")),
                video_dur_col=str(config.get("video_dur_col", "video_duration")),
                play_dur_col=str(config.get("play_dur_col", "play_duration")),
                time_col=str(config.get("event_time_col", "timestamp")),
                seed=int(config.get("seed", 0)),
                min_events=min_events,
                max_events=max_events,
            )

        # --- traces ---
        self.trace_pool = TracePool(
            traces_csv=str(config.get("traces_csv", "data/fcc_httpgetmt_trace_1s_kbps.csv")),
            id_col=str(config.get("trace_id_col", "trace_id")),
            time_col=str(config.get("trace_time_col", "t")),
            bw_col=str(config.get("trace_bw_col", "throughput_kbps")),
            bw_multiplier=float(config.get("trace_bw_multiplier", 1.0)),
            allowed_trace_ids=config.get("allowed_trace_ids", None),
        )

        # --- bitrate ladder ---
        # Default ladder mapped to common short-video resolutions (CBR proxy, Mbps):
        #   180p, 360p, 480p, 720p, 1080p
        # (Avoids overly high 2k bitrate which is uncommon for short-form feeds.)
        # Added 180p as a low-bitrate action to improve robustness under poor throughput.
        self.video_bitrates_mbps: List[float] = list(map(float, config.get("video_bitrates_mbps", [0.35, 0.7, 1.2, 2.5, 5.0])))
        if len(self.video_bitrates_mbps) <= 0:
            raise ValueError("video_bitrates_mbps must be non-empty.")
        self.video_bitrate_labels: List[str] = list(map(str, config.get("video_bitrate_labels", ["180p", "360p", "480p", "720p", "1080p"])))
        if len(self.video_bitrate_labels) != len(self.video_bitrates_mbps):
            raise ValueError("video_bitrate_labels length must match video_bitrates_mbps length.")

        # --- Joint action: (bitrate, prefetch_threshold) ---
        # DUASVS semantics: prefetch_threshold (seconds) caps the allowed *unplayed buffer* ahead of playback.
        # Default thresholds calibrated to KuaiRec watch-time stats (seconds), with a slightly larger action range:
        # covers short watches (~1-4s), typical (~6-12s), and tail (~18s).
        # Extended action range with longer thresholds for ablations: 24, 30, 36.
        self.prefetch_thresholds_s: List[float] = list(map(float, config.get("prefetch_thresholds_s", [1, 2, 4, 6, 8, 12, 18, 24, 30, 36])))
        if len(self.prefetch_thresholds_s) <= 0:
            raise ValueError("prefetch_thresholds_s must be non-empty.")
        if any(x < 0 for x in self.prefetch_thresholds_s):
            raise ValueError("prefetch_thresholds_s must be non-negative seconds.")

        # --- state params ---
        self.k_hist: int = int(config.get("k_hist", 8))
        self.bw_ref_mbps: float = float(config.get("bw_ref_mbps", 5.0))
        self.bw_clip: float = float(config.get("bw_clip", 3.0))
        self.b_max: float = float(config.get("buffer_max_s", 60.0))
        self.L_max: float = float(config.get("video_len_max_s", 60.0))

        # --- simulation numerics / outage semantics ---
        # Preserve the original throughput sequence, but define an outage threshold:
        # if throughput < thr_outage_mbps => treat as outage, downloaded increment = 0 for that second.
        self.thr_outage_mbps: float = float(config.get("thr_outage_mbps", 0.02))
        # Numerical protection for divisions/logs (Mbps units).
        self.eps_mbps: float = float(config.get("eps_mbps", 0.01))

        # --- user patience (short-video realism) ---
        # If rebuffering accumulates beyond this cap (seconds) within a single video,
        # the user is assumed to swipe away (stop watching), preventing unbounded stall time.
        self.max_rebuf_s_per_video: float = float(config.get("max_rebuf_s_per_video", 8.0))

        # --- reward params (doc-style) ---
        self.r_min_mbps: float = float(config.get("r_min_mbps", 0.2))
        self.rebuf_penalty: float = float(config.get("rebuf_penalty", 2.66))
        # Bitrate utility scaling:
        # If True (default), scale bitrate utility by watched fraction (played_s / watch_s) for short-video realism.
        # If False, any non-zero playback grants the full bitrate utility (paper may implicitly do this).
        self.qoe_scale_by_watch_frac: bool = bool(config.get("qoe_scale_by_watch_frac", True))
        # Rebuffer penalty shaping:
        # - linear: penalty * rebuf_s  (paper-style)
        # - cap:    penalty * min(rebuf_s, rebuf_cap_s)   (bounds extreme stalls for learnability)
        # - log:    penalty * (log1p(rebuf_s / rebuf_log_scale_s) * rebuf_log_scale_s)
        #          (≈linear for small rebuf, grows sublinearly for large rebuf)
        self.rebuf_penalty_mode: str = str(config.get("rebuf_penalty_mode", "linear")).lower()
        self.rebuf_cap_s: float = float(config.get("rebuf_cap_s", 12.0))
        self.rebuf_log_scale_s: float = float(config.get("rebuf_log_scale_s", 2.0))
        self.switch_penalty: float = float(config.get("switch_penalty", 1.0))
        self.lambda_waste: float = float(config.get("lambda_waste", 0.5))

        # RNG
        self._rng = np.random.default_rng(int(config.get("seed", 0)))

        # spaces
        # Joint discrete action encoding:
        #   action = bitrate_idx * P + prefetch_idx
        self._n_bitrates = int(len(self.video_bitrates_mbps))
        self._n_prefetch = int(len(self.prefetch_thresholds_s))
        self.action_space = spaces.Discrete(self._n_bitrates * self._n_prefetch)

        # --- strict DUASVS state alignment ---
        # Paper states:
        #   State0: past K measured throughputs
        #   State1: current buffer occupancy b_i
        #   State2: video size at each bitrate level {s_{i,0}..s_{i,H-1}}
        #   State3: video duration L_i
        #   State4: last requested bitrate r_{i-1}
        #
        # We use a CBR proxy: size_mbit(h) = bitrate_mbps(h) * L_i (seconds)  (Mbit)
        # Normalize sizes by a conservative upper bound to map into [0,1].
        self.size_ref_mbit: float = float(
            config.get("size_ref_mbit", max(self.video_bitrates_mbps) * max(1e-9, float(self.L_max)))
        )

        obs_dim = self.k_hist + 1 + self._n_bitrates + 1 + 1  # bw_hist + buffer + sizes(H) + video_len + prev_bitrate
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        # episode state
        self.trace_id: Optional[str] = None
        self.trace_bw_mbps: Optional[np.ndarray] = None
        self.t: int = 0  # seconds index into trace
        self.buffer_s: float = 0.0
        self.prev_bitrate_idx: int = 0
        self.session: List[VideoEvent] = []
        self.i: int = 0  # video index in session
        # Cross-video prefetch state (DUASVS semantics):
        # prefetched_unplayed_s[j] = seconds of video j already downloaded (unplayed) and available when we reach it.
        # prefetched_bitrate_idx[j] = bitrate idx used to download that prefetched content (single idx per video;
        # if a later step tries to prefetch the same video with a different bitrate, we discard old prefetch as waste).
        self.prefetched_unplayed_s: List[float] = []
        self.prefetched_bitrate_idx: List[int] = []

        # metrics
        self.sum_rebuf_s: float = 0.0
        self.sum_waste_s: float = 0.0
        self.sum_qoe: float = 0.0
        # derived data metrics (for reporting "data saving")
        self.sum_downloaded_mbit: float = 0.0
        self.sum_waste_mbit: float = 0.0
        # last step action decoded (for logging)
        self.last_prefetch_s: float = float(self.prefetch_thresholds_s[0])

    def _decode_action(self, action: int) -> Tuple[int, int]:
        a = int(action)
        if a < 0 or a >= int(self.action_space.n):
            raise ValueError(f"Invalid action={a}")
        bitrate_idx = int(a // self._n_prefetch)
        prefetch_idx = int(a % self._n_prefetch)
        return bitrate_idx, prefetch_idx

    def _pick_trace(self, first_trace_id: Optional[str] = None):
        if first_trace_id is not None:
            tid = str(first_trace_id)
            if tid not in self.trace_pool.traces:
                raise ValueError(f"Trace id {tid} not found in trace pool.")
        else:
            tid = str(self._rng.choice(self.trace_pool.trace_ids))
        self.trace_id = tid
        bw_kbps = self.trace_pool.traces[tid]
        self.trace_bw_mbps = (bw_kbps.astype(np.float32) / 1000.0)
        self.t = 0

    def _get_bw_hist_norm(self) -> np.ndarray:
        assert self.trace_bw_mbps is not None
        # last K seconds ending at current t (pad with first value)
        idx = max(0, min(self.t, len(self.trace_bw_mbps) - 1))
        cur = float(self.trace_bw_mbps[idx])
        hist = [cur] * self.k_hist
        for k in range(1, self.k_hist + 1):
            j = idx - k
            if j >= 0:
                hist[-k] = float(self.trace_bw_mbps[j])
        hist = np.asarray(hist, dtype=np.float32)
        norm = np.clip(hist / max(1e-9, self.bw_ref_mbps), 0.0, self.bw_clip) / max(1e-9, self.bw_clip)
        return norm.astype(np.float32)

    def _obs(self) -> np.ndarray:
        bw_hist = self._get_bw_hist_norm()
        b_norm = np.array([_clip(self.buffer_s / max(1e-9, self.b_max), 0.0, 1.0)], dtype=np.float32)
        cur_L = float(self.session[self.i].video_len_s) if (0 <= self.i < len(self.session)) else 0.0
        L_norm = np.array([_clip(cur_L / max(1e-9, self.L_max), 0.0, 1.0)], dtype=np.float32)
        # Paper State2: size at each bitrate level for the current video.
        # CBR proxy: size_mbit(h) = bitrate_mbps(h) * L_i (seconds)
        s_ref = float(max(1e-9, self.size_ref_mbit))
        sizes = (np.asarray(self.video_bitrates_mbps, dtype=np.float32) * np.float32(max(0.0, cur_L))) / np.float32(s_ref)
        sizes = np.clip(sizes, 0.0, 1.0).astype(np.float32)
        # Normalize prev bitrate index by (H-1), NOT by joint action dim.
        prev_norm = np.array([float(self.prev_bitrate_idx) / float(max(1, self._n_bitrates - 1))], dtype=np.float32)
        # Order matches DUASVS paper: [throughput_hist, buffer, sizes(H), duration, last_bitrate]
        return np.concatenate([bw_hist, b_norm, sizes, L_norm, prev_norm], axis=0).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))

        # Optional: make session sampling deterministic for eval curves.
        # If provided, reseed the session sampler RNG so each eval uses the same behavior distribution.
        if isinstance(options, dict) and options.get("session_seed") is not None:
            ss = int(options.get("session_seed"))
            if hasattr(self.session_sampler, "reseed"):
                try:
                    self.session_sampler.reseed(ss)  # type: ignore[attr-defined]
                except Exception:
                    pass

        first_tid = None
        if isinstance(options, dict) and options.get("first_trace_id") is not None:
            first_tid = str(options.get("first_trace_id"))

        self._pick_trace(first_trace_id=first_tid)
        self.session = self.session_sampler.sample_session()
        self.i = 0
        self.buffer_s = 0.0
        self.prev_bitrate_idx = 0
        self.prefetched_unplayed_s = [0.0 for _ in range(len(self.session))]
        self.prefetched_bitrate_idx = [-1 for _ in range(len(self.session))]

        self.sum_rebuf_s = 0.0
        self.sum_waste_s = 0.0
        self.sum_qoe = 0.0
        self.sum_downloaded_mbit = 0.0
        self.sum_waste_mbit = 0.0

        obs = self._obs()
        info = {"trace_id": self.trace_id, "trace_len": int(len(self.trace_bw_mbps) if self.trace_bw_mbps is not None else 0)}
        return obs, info

    def _buffer_ahead_s(self, cur_idx: int, cur_buf_s: float) -> float:
        """Total unplayed buffer ahead of playback progress (seconds), session-level."""
        if not self.prefetched_unplayed_s:
            return float(max(0.0, cur_buf_s))
        # current video buffer + prefetched future buffers
        fut = 0.0
        for j in range(int(cur_idx) + 1, len(self.prefetched_unplayed_s)):
            fut += float(self.prefetched_unplayed_s[j])
        return float(max(0.0, cur_buf_s) + fut)

    def _discard_prefetch_as_waste(self, vid_idx: int) -> Tuple[float, float]:
        """Discard prefetched content for a specific video (due to bitrate mismatch). Returns (waste_s, waste_mbit)."""
        if vid_idx < 0 or vid_idx >= len(self.prefetched_unplayed_s):
            return 0.0, 0.0
        s = float(self.prefetched_unplayed_s[vid_idx])
        if s <= 0.0:
            self.prefetched_unplayed_s[vid_idx] = 0.0
            self.prefetched_bitrate_idx[vid_idx] = -1
            return 0.0, 0.0
        bidx = int(self.prefetched_bitrate_idx[vid_idx])
        # If bitrate idx is unknown, conservatively convert using the lowest ladder bitrate.
        mbps = float(self.video_bitrates_mbps[bidx]) if (0 <= bidx < len(self.video_bitrates_mbps)) else float(self.video_bitrates_mbps[0])
        mbit = float(mbps * s)
        self.prefetched_unplayed_s[vid_idx] = 0.0
        self.prefetched_bitrate_idx[vid_idx] = -1
        return float(s), float(mbit)

    def _simulate_video_with_threshold(
        self,
        *,
        video_len_s: float,
        watch_s: float,
        bitrate_mbps: float,
        prefetch_s: float,
    ) -> Tuple[float, float, float, float]:
        """
        Simulate download/playback for ONE video under a buffer-threshold prefetch policy.

        Policy:
          - If buffer (downloaded - played) < prefetch_s and not fully downloaded => download this 1s slot.
          - Else => pause download this 1s slot.

        Returns:
          (download_time_s, rebuffer_s, downloaded_s, played_s)
        """
        assert self.trace_bw_mbps is not None

        # played is per-video counter (user watches at most watch_s seconds of this video).
        played_s = 0.0
        dl_time_s = 0.0
        rebuf_s = 0.0
        swiped_due_to_rebuf = False

        # Current-video unplayed buffer starts from cross-video prefetch.
        # (Future-video prefetch is stored in self.prefetched_unplayed_s[j>i].)
        buf_cur = float(max(0.0, self.prefetched_unplayed_s[self.i] if (0 <= self.i < len(self.prefetched_unplayed_s)) else 0.0))

        downloaded_cur_new = 0.0
        downloaded_future_new = 0.0

        # Simulate in 1s slots until user stops watching or trace ends (1Hz)
        while played_s < watch_s - 1e-9 and self.t < len(self.trace_bw_mbps):
            # 1) Decide whether to download this slot.
            #    IMPORTANT (DUASVS semantics, option A):
            #    - Always prioritize keeping CURRENT video's buffer under the threshold to avoid deadlocks.
            #    - Only when CURRENT video is fully downloaded do we use remaining capacity to prefetch FUTURE videos
            #      up to the same session-level threshold.
            downloaded_cur_total = float(played_s + buf_cur)
            cur_fully_downloaded = bool(downloaded_cur_total >= video_len_s - 1e-9)
            do_download_cur = bool((not cur_fully_downloaded) and (buf_cur < prefetch_s - 1e-9))
            total_buf = self._buffer_ahead_s(self.i, buf_cur)
            do_download_future = bool(cur_fully_downloaded and (total_buf < prefetch_s - 1e-9))
            do_download = bool(do_download_cur or do_download_future)

            if do_download:
                bw = float(self.trace_bw_mbps[self.t])  # Mbps
                # Outage handling: preserve original throughput series, but treat very low throughput as outage.
                # If bw < thr_outage_mbps => no progress this second.
                bw_eff = 0.0 if (bw < float(self.thr_outage_mbps)) else bw
                delta_s_raw = float(bw_eff / max(float(self.eps_mbps), float(bitrate_mbps)))  # seconds of video content in this 1s slot
                # IMPORTANT: Do not exceed threshold within a single 1s slot.
                # Cap depends on whether we are filling CURRENT buffer or prefetching FUTURE.
                if do_download_cur:
                    cap_s = float(max(0.0, float(prefetch_s) - float(buf_cur)))
                else:
                    cap_s = float(max(0.0, float(prefetch_s) - float(total_buf)))
                delta_s = float(min(delta_s_raw, cap_s))
                if delta_s > 0.0:
                    # First, download into CURRENT video until it's fully downloaded.
                    rem_cur = float(max(0.0, video_len_s - downloaded_cur_total))
                    add_cur = float(min(rem_cur, delta_s))
                    if add_cur > 0.0:
                        buf_cur += add_cur
                        downloaded_cur_new += add_cur
                        delta_s -= add_cur

                    # Then, if still have capacity, prefetch FUTURE videos sequentially (A: use current bitrate).
                    j = int(self.i) + 1
                    while delta_s > 1e-12 and j < len(self.session):
                        if self.prefetched_unplayed_s[j] > 0.0 and self.prefetched_bitrate_idx[j] != int(self.prev_bitrate_idx) and self.prefetched_bitrate_idx[j] != -1:
                            # Discard old prefetch and replace with current bitrate's prefetch.
                            ws, wm = self._discard_prefetch_as_waste(j)
                            self.sum_waste_s += float(ws)
                            self.sum_waste_mbit += float(wm)

                        if self.prefetched_bitrate_idx[j] == -1:
                            self.prefetched_bitrate_idx[j] = int(self.prev_bitrate_idx)

                        rem_j = float(max(0.0, float(self.session[j].video_len_s) - float(self.prefetched_unplayed_s[j])))
                        if rem_j <= 0.0:
                            j += 1
                            continue
                        add_j = float(min(rem_j, delta_s))
                        self.prefetched_unplayed_s[j] = float(self.prefetched_unplayed_s[j] + add_j)
                        downloaded_future_new += add_j
                        delta_s -= add_j
                        if self.prefetched_unplayed_s[j] >= float(self.session[j].video_len_s) - 1e-9:
                            j += 1

                dl_time_s += 1.0

            # 2) Playback for 1 second (or stall)
            if buf_cur >= 1.0 - 1e-9:
                buf_cur = float(buf_cur - 1.0)
                played_s += 1.0
            else:
                rebuf_s += 1.0
                # User patience: if stall is too long, user swipes away.
                if float(rebuf_s) >= float(self.max_rebuf_s_per_video) - 1e-9:
                    swiped_due_to_rebuf = True
                    self.t += 1
                    break

            self.t += 1

        # Remaining unplayed buffer of CURRENT video is wasted when user scrolls away.
        waste_cur_s = float(max(0.0, buf_cur))
        # Clear current video's prefetch buffer (it can't be used after scrolling away).
        if 0 <= self.i < len(self.prefetched_unplayed_s):
            self.prefetched_unplayed_s[self.i] = 0.0
            self.prefetched_bitrate_idx[self.i] = -1
        # Session-level buffer for observation will be recomputed in step().
        return (
            float(dl_time_s),
            float(rebuf_s),
            float(played_s + waste_cur_s),
            float(played_s),
            float(waste_cur_s),
            float(downloaded_cur_new),
            float(downloaded_future_new),
            bool(swiped_due_to_rebuf),
        )

    def step(self, action: int):
        if not (0 <= self.i < len(self.session)):
            raise RuntimeError("Episode is done; call reset().")
        assert self.trace_bw_mbps is not None

        bitrate_idx, prefetch_idx = self._decode_action(int(action))
        prefetch_s = float(self.prefetch_thresholds_s[prefetch_idx])
        self.last_prefetch_s = prefetch_s

        ev = self.session[self.i]
        bitrate_mbps = float(self.video_bitrates_mbps[bitrate_idx])

        watch_s = float(ev.watch_s)
        video_len_s = float(ev.video_len_s)
        # If current video already has prefetched content at a different bitrate, discard it as waste.
        step_discard_waste_s = 0.0
        step_discard_waste_mbit = 0.0
        if 0 <= self.i < len(self.prefetched_unplayed_s) and self.prefetched_unplayed_s[self.i] > 0.0:
            if self.prefetched_bitrate_idx[self.i] != -1 and int(self.prefetched_bitrate_idx[self.i]) != int(bitrate_idx):
                ws, wm = self._discard_prefetch_as_waste(self.i)
                step_discard_waste_s += float(ws)
                step_discard_waste_mbit += float(wm)
                self.sum_waste_s += float(ws)
                self.sum_waste_mbit += float(wm)
        # Switching penalty needs the previous video's bitrate; prefetch (A) uses the current decision bitrate.
        prev_idx_for_switch = int(self.prev_bitrate_idx)
        self.prev_bitrate_idx = int(bitrate_idx)

        dl_time_s, rebuf_s, downloaded_s, played_s, waste_cur_s, dl_cur_new_s, dl_future_new_s, swiped_due_to_rebuf = self._simulate_video_with_threshold(
            video_len_s=video_len_s,
            watch_s=watch_s,
            bitrate_mbps=bitrate_mbps,
            prefetch_s=prefetch_s,
        )
        self.sum_rebuf_s += float(rebuf_s)

        # Waste is downloaded-but-unwatched seconds of THIS video when user scrolls away, plus any discarded prefetch.
        waste_s = float(max(0.0, waste_cur_s + step_discard_waste_s))
        self.sum_waste_s += float(max(0.0, waste_cur_s))
        # Data volume proxy under CBR assumption:
        #   video_size_mbit = bitrate_mbps * seconds
        downloaded_mbit = float(bitrate_mbps * float(dl_cur_new_s + dl_future_new_s))
        waste_mbit = float(bitrate_mbps * float(waste_cur_s)) + float(step_discard_waste_mbit)
        self.sum_downloaded_mbit += float(downloaded_mbit)
        self.sum_waste_mbit += float(waste_mbit)

        # QoE (Pensieve-style, per video)
        r_min = max(float(self.eps_mbps), float(self.r_min_mbps))
        # Bitrate utility: optionally scale by how much the user actually watched (short-video realism).
        if bool(self.qoe_scale_by_watch_frac):
            watched_frac = float(played_s / max(float(self.eps_mbps), float(watch_s)))
            watched_frac = float(np.clip(watched_frac, 0.0, 1.0))
        else:
            # If the user watched anything (played_s > 0), grant full utility.
            watched_frac = 1.0 if float(played_s) > 0.0 else 0.0
        qoe_bitrate = float(np.log(max(float(self.eps_mbps), float(bitrate_mbps)) / r_min) * float(watched_frac))
        # Switching penalty is between PREVIOUS video bitrate and CURRENT video bitrate.
        prev_bitrate = float(self.video_bitrates_mbps[prev_idx_for_switch]) if (0 <= prev_idx_for_switch < len(self.video_bitrates_mbps)) else float(self.video_bitrates_mbps[0])
        qoe_switch = abs(
            np.log(max(float(self.eps_mbps), float(prev_bitrate)) / r_min)
            - np.log(max(float(self.eps_mbps), float(bitrate_mbps)) / r_min)
        )
        # Rebuffer penalty (optionally shaped for learnability under infinite patience).
        mode = str(self.rebuf_penalty_mode).lower()
        if mode == "cap":
            rebuf_term = float(min(float(rebuf_s), float(max(0.0, self.rebuf_cap_s))))
        elif mode == "log":
            s = float(max(1e-9, float(self.rebuf_log_scale_s)))
            rebuf_term = float(np.log1p(float(rebuf_s) / s) * s)
        else:
            rebuf_term = float(rebuf_s)

        qoe = float(qoe_bitrate - self.switch_penalty * qoe_switch - self.rebuf_penalty * rebuf_term)
        self.sum_qoe += qoe

        # Penalize wasted DATA volume (Mbit), not wasted seconds.
        reward = float(qoe - self.lambda_waste * waste_mbit)

        self.i += 1

        terminated = (self.i >= len(self.session))
        truncated = bool(self.t >= len(self.trace_bw_mbps))

        # Update session-level buffer state (seconds ahead for remaining videos).
        if not (terminated or truncated):
            cur_idx = int(self.i)
            fut = 0.0
            for j in range(cur_idx, len(self.prefetched_unplayed_s)):
                fut += float(self.prefetched_unplayed_s[j])
            self.buffer_s = float(max(0.0, fut))
        else:
            # If episode ends, any remaining prefetched content is wasted.
            rem = 0.0
            for j in range(int(self.i), len(self.prefetched_unplayed_s)):
                rem += float(self.prefetched_unplayed_s[j])
            if rem > 0.0:
                self.sum_waste_s += float(rem)
                # Approximate conversion: treat remaining prefetched as at current bitrate.
                self.sum_waste_mbit += float(bitrate_mbps * rem)
            self.buffer_s = 0.0

        obs = self._obs() if not (terminated or truncated) else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {
            "trace_id": self.trace_id,
            "t": int(self.t),
            "video_idx": int(self.i),
            "video_len_s": float(ev.video_len_s),
            "watch_s": float(ev.watch_s),
            "bitrate_idx": int(bitrate_idx),
            "prefetch_s": float(prefetch_s),
            "bitrate_mbps": float(bitrate_mbps),
            "bitrate_label": str(self.video_bitrate_labels[bitrate_idx]) if 0 <= int(bitrate_idx) < len(self.video_bitrate_labels) else str(bitrate_idx),
            "download_time_s": float(dl_time_s),
            "rebuf_s": float(rebuf_s),
            "waste_s": float(waste_s),
            "downloaded_s": float(downloaded_s),
            "played_s": float(played_s),
            "watched_s": float(played_s),
            "swiped_due_to_rebuf": bool(swiped_due_to_rebuf),
            "downloaded_mbit": float(downloaded_mbit),
            "waste_mbit": float(waste_mbit),
            "sum_rebuf_s": float(self.sum_rebuf_s),
            "sum_waste_s": float(self.sum_waste_s),
            "sum_qoe": float(self.sum_qoe),
            "sum_downloaded_mbit": float(self.sum_downloaded_mbit),
            "sum_waste_mbit": float(self.sum_waste_mbit),
        }
        return obs, reward, bool(terminated), bool(truncated), info   


