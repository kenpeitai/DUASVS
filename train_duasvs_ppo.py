#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal PPO training entrypoint for DUASVSEnv (per-video simulator).

This is intentionally simple and self-contained (no stable-baselines dependency),
so it can run in a clean conda env with just torch/numpy/pandas/gymnasium.
"""

from __future__ import annotations

import argparse
import os
import csv
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from envs.duasvs_env import DUASVSEnv


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, *, logits_clip: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.pi = nn.Linear(128, act_dim)
        self.v = nn.Linear(128, 1)
        self.logits_clip = float(logits_clip)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(obs)
        logits = self.pi(x)
        # Optional numerics guard: extremely large logits can create inf/nan in downstream ops.
        # (torch's categorical is usually stable, but this helps when policies become very peaky.)
        if float(self.logits_clip) > 0.0:
            logits = torch.clamp(logits, -float(self.logits_clip), float(self.logits_clip))
        return logits, self.v(x).squeeze(-1)

    def act(self, obs: torch.Tensor):
        logits, v = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        return a, logp, v


@dataclass
class Rollout:
    obs: np.ndarray
    acts: np.ndarray
    rews: np.ndarray
    dones: np.ndarray
    vals: np.ndarray
    logps: np.ndarray
    last_val: float


@dataclass
class EpisodeLog:
    global_steps: int
    update: int
    trace_id: str
    ep_len: int
    ep_return: float
    sum_rebuf_s: float
    sum_waste_s: float
    sum_downloaded_mbit: float
    sum_waste_mbit: float


def compute_gae(rews: np.ndarray, dones: np.ndarray, vals: np.ndarray, last_val: float, gamma: float, lam: float):
    """
    vals is V(s_t) for each step t in rollout.
    last_val is V(s_{T}) bootstrap.
    """
    T = len(rews)
    adv = np.zeros(T, dtype=np.float32)
    ret = np.zeros(T, dtype=np.float32)
    gae = 0.0
    next_val = float(last_val)
    next_nonterminal = 1.0
    for t in reversed(range(T)):
        next_nonterminal = 1.0 - float(dones[t])
        delta = float(rews[t]) + gamma * next_val * next_nonterminal - float(vals[t])
        gae = delta + gamma * lam * next_nonterminal * gae
        adv[t] = gae
        next_val = float(vals[t])
    ret = adv + vals.astype(np.float32)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv.astype(np.float32), ret.astype(np.float32)


def ppo_update(
    model: ActorCritic,
    opt: optim.Optimizer,
    obs: torch.Tensor,
    acts: torch.Tensor,
    logp_old: torch.Tensor,
    adv: torch.Tensor,
    ret: torch.Tensor,
    *,
    clip_ratio: float,
    vf_coef: float,
    ent_coef: float,
    uniform_kl_coef: float,
    target_kl: Optional[float],
    train_iters: int,
    batch_size: int,
    debug_numerics: bool = False,
):
    n = obs.shape[0]
    stats_sum = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "total_loss": 0.0,
        "approx_kl": 0.0,
        "clip_frac": 0.0,
    }
    stats_cnt = 0
    for _ in range(int(train_iters)):
        idx = torch.randperm(n, device=obs.device)
        for start in range(0, n, int(batch_size)):
            mb = idx[start : start + int(batch_size)]
            logits, v = model.forward(obs[mb])
            if bool(debug_numerics):
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Non-finite policy logits detected.")
                if not torch.isfinite(v).all():
                    raise FloatingPointError("Non-finite value predictions detected.")
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(acts[mb])
            ratio = torch.exp(logp - logp_old[mb])
            obj1 = ratio * adv[mb]
            obj2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv[mb]
            pi_loss = -torch.min(obj1, obj2).mean()
            vf_loss = (ret[mb] - v).pow(2).mean()
            ent = dist.entropy().mean()
            loss = pi_loss + vf_coef * vf_loss - ent_coef * ent
            if float(uniform_kl_coef) > 0.0:
                # Penalize KL(pi || uniform) to encourage action spread.
                uniform_dist = torch.distributions.Categorical(logits=torch.zeros_like(logits))
                kl_uniform = torch.distributions.kl_divergence(dist, uniform_dist).mean()
                loss = loss + float(uniform_kl_coef) * kl_uniform
            if bool(debug_numerics) and (not torch.isfinite(loss)):
                raise FloatingPointError("Non-finite PPO loss detected.")

            # PPO diagnostics (SB3-style)
            # approx_kl estimates KL(old||new) for samples drawn from old policy.
            approx_kl = (logp_old[mb] - logp).mean()
            clip_frac = (torch.abs(ratio - 1.0) > float(clip_ratio)).float().mean()

            stats_sum["policy_loss"] += float(pi_loss.detach().cpu().item())
            stats_sum["value_loss"] += float(vf_loss.detach().cpu().item())
            stats_sum["entropy"] += float(ent.detach().cpu().item())
            stats_sum["total_loss"] += float(loss.detach().cpu().item())
            stats_sum["approx_kl"] += float(approx_kl.detach().cpu().item())
            stats_sum["clip_frac"] += float(clip_frac.detach().cpu().item())
            stats_cnt += 1

            opt.zero_grad()
            loss.backward()
            if bool(debug_numerics):
                for p in model.parameters():
                    if p.grad is None:
                        continue
                    if not torch.isfinite(p.grad).all():
                        raise FloatingPointError("Non-finite gradients detected.")
            opt.step()

            # Target-KL early stop to reduce policy drift / oscillation.
            if (target_kl is not None) and np.isfinite(float(target_kl)) and float(target_kl) > 0.0:
                if float(approx_kl.detach().cpu().item()) > float(target_kl):
                    break
        if (target_kl is not None) and np.isfinite(float(target_kl)) and float(target_kl) > 0.0:
            # If we broke due to KL, stop further epochs too.
            # (Heuristic: if last minibatch exceeded target_kl, stats already captured.)
            if stats_cnt > 0 and (stats_sum["approx_kl"] / stats_cnt) > float(target_kl):
                break
    if stats_cnt <= 0:
        return {
            "policy_loss": float("nan"),
            "value_loss": float("nan"),
            "entropy": float("nan"),
            "total_loss": float("nan"),
            "approx_kl": float("nan"),
            "clip_frac": float("nan"),
        }
    return {k: v / stats_cnt for k, v in stats_sum.items()}


@torch.no_grad()
def act_greedy(model: ActorCritic, obs: np.ndarray, device: torch.device) -> int:
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    logits, _ = model.forward(obs_t)
    return int(torch.argmax(logits, dim=-1).item())


@torch.no_grad()
def eval_on_traces(
    *,
    model: ActorCritic,
    device: torch.device,
    env_cfg: Optional[Dict],
    trace_ids: List[str],
    eval_seed: int,
    max_videos: int,
    reuse_env: Optional[DUASVSEnv] = None,
) -> Dict[str, float]:
    """
    Deterministic-ish eval:
    - greedy action selection
    - fixed trace ids
    - reproducible simulator RNG via eval_seed
    Returns mean per-episode and per-step metrics.
    """
    if not trace_ids:
        return {"n_eps": 0.0}

    ep_returns: List[float] = []
    ep_lens: List[int] = []
    ep_rebuf: List[float] = []
    ep_waste_s: List[float] = []
    ep_dl_mbit: List[float] = []
    ep_waste_mbit: List[float] = []
    # Per-step diagnostic distributions (across all eval steps).
    waste_raw_steps: List[float] = []
    waste_ratio_steps: List[float] = []

    # Key perf optimization:
    # Reuse a single env across all trace_ids to avoid repeatedly parsing large FCC CSVs.
    if reuse_env is not None:
        env_i = reuse_env
    else:
        if env_cfg is None:
            raise ValueError("env_cfg is required when reuse_env is not provided.")
        env_i = DUASVSEnv({**env_cfg, "allowed_trace_ids": [str(x) for x in trace_ids], "seed": int(eval_seed)})

    for tid in trace_ids:
        # Deterministic session sampling per (eval_seed, trace_id) to stabilize eval curves.
        session_seed = (hash((int(eval_seed), str(tid))) & 0xFFFFFFFF)
        obs, _ = env_i.reset(seed=int(eval_seed), options={"first_trace_id": str(tid), "session_seed": int(session_seed)})
        ep_ret = 0.0
        ep_len = 0
        last_info: Dict = {}
        terminated = False
        truncated = False
        while not (terminated or truncated) and ep_len < int(max_videos):
            a = act_greedy(model, obs, device)
            obs, r, terminated, truncated, last_info = env_i.step(a)
            ep_ret += float(r)
            ep_len += 1
            if isinstance(last_info, dict):
                wr = last_info.get("waste_raw", None)
                wz = last_info.get("waste_ratio", None)
                if wr is not None and np.isfinite(float(wr)):
                    waste_raw_steps.append(float(wr))
                if wz is not None and np.isfinite(float(wz)):
                    waste_ratio_steps.append(float(wz))
        ep_returns.append(float(ep_ret))
        ep_lens.append(int(ep_len))
        if isinstance(last_info, dict):
            ep_rebuf.append(float(last_info.get("sum_rebuf_s", np.nan)))
            ep_waste_s.append(float(last_info.get("sum_waste_s", np.nan)))
            ep_dl_mbit.append(float(last_info.get("sum_downloaded_mbit", np.nan)))
            ep_waste_mbit.append(float(last_info.get("sum_waste_mbit", np.nan)))

    n = max(1, len(ep_returns))
    mean_len = float(np.mean(ep_lens)) if ep_lens else 0.0
    # per-step (normalized by each episode length)
    rps = float(np.mean([ep_returns[i] / max(1, ep_lens[i]) for i in range(len(ep_returns))]))
    rebuf_ps = float(np.mean([ep_rebuf[i] / max(1, ep_lens[i]) for i in range(len(ep_rebuf))])) if ep_rebuf else float("nan")
    waste_s_ps = float(np.mean([ep_waste_s[i] / max(1, ep_lens[i]) for i in range(len(ep_waste_s))])) if ep_waste_s else float("nan")
    dl_mbit_ps = float(np.mean([ep_dl_mbit[i] / max(1, ep_lens[i]) for i in range(len(ep_dl_mbit))])) if ep_dl_mbit else float("nan")
    waste_mbit_ps = float(np.mean([ep_waste_mbit[i] / max(1, ep_lens[i]) for i in range(len(ep_waste_mbit))])) if ep_waste_mbit else float("nan")

    def _stats(xs: List[float]) -> Dict[str, float]:
        if not xs:
            return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan")}
        a = np.asarray(xs, dtype=float)
        return {
            "mean": float(np.mean(a)),
            "p50": float(np.quantile(a, 0.50)),
            "p90": float(np.quantile(a, 0.90)),
        }

    raw_s = _stats(waste_raw_steps)
    ratio_s = _stats(waste_ratio_steps)

    return {
        "n_eps": float(len(ep_returns)),
        "ep_return_mean": float(np.mean(ep_returns)),
        "ep_len_mean": mean_len,
        "return_per_step": rps,
        "rebuf_s_per_step": rebuf_ps,
        "waste_s_per_step": waste_s_ps,
        "downloaded_mbit_per_step": dl_mbit_ps,
        "waste_mbit_per_step": waste_mbit_ps,
        # Waste shaping diagnostics (per-step distribution on eval set)
        "waste_raw_mean": raw_s["mean"],
        "waste_raw_p50": raw_s["p50"],
        "waste_raw_p90": raw_s["p90"],
        "waste_ratio_mean": ratio_s["mean"],
        "waste_ratio_p50": ratio_s["p50"],
        "waste_ratio_p90": ratio_s["p90"],
    }


def rollout_steps(
    env: DUASVSEnv,
    model: ActorCritic,
    device: torch.device,
    steps: int,
    *,
    global_steps_before: int,
    update_idx: int,
    action_mode: str = "sample",
) -> Tuple[Rollout, Dict[str, float], List[EpisodeLog]]:
    obs_list: List[np.ndarray] = []
    act_list: List[int] = []
    rew_list: List[float] = []
    done_list: List[float] = []
    val_list: List[float] = []
    logp_list: List[float] = []

    ep_ret = 0.0
    ep_len = 0
    ep_count = 0
    sum_rebuf = 0.0
    sum_waste = 0.0
    ep_logs: List[EpisodeLog] = []

    action_mode_n = str(action_mode).strip().lower()
    if action_mode_n not in {"sample", "greedy"}:
        raise ValueError(f"rollout action_mode must be 'sample' or 'greedy'. Got: {action_mode}")

    obs, _ = env.reset()
    for _ in range(int(steps)):
        # Optional strict checks (enabled via env.debug_checks)
        if bool(getattr(env, "debug_checks", False)):
            if not np.isfinite(obs).all():
                raise FloatingPointError("Non-finite observation detected before action selection.")
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, v = model.forward(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            if action_mode_n == "greedy":
                a = torch.argmax(logits, dim=-1)
            else:
                a = dist.sample()
            logp = dist.log_prob(a)
        a_i = int(a.item())

        next_obs, rew, terminated, truncated, info = env.step(a_i)
        done = bool(terminated or truncated)
        if bool(getattr(env, "debug_checks", False)):
            if not np.isfinite(float(rew)):
                raise FloatingPointError(f"Non-finite reward detected: {rew}")
            if not np.isfinite(next_obs).all():
                raise FloatingPointError("Non-finite observation detected after env.step().")
            if not np.isfinite(float(v.item())):
                raise FloatingPointError("Non-finite value prediction detected.")
            if not np.isfinite(float(logp.item())):
                raise FloatingPointError("Non-finite log-prob detected.")

        obs_list.append(obs.astype(np.float32))
        act_list.append(a_i)
        rew_list.append(float(rew))
        done_list.append(float(done))
        val_list.append(float(v.item()))
        logp_list.append(float(logp.item()))

        ep_ret += float(rew)
        ep_len += 1
        if isinstance(info, dict):
            sum_rebuf = float(info.get("sum_rebuf_s", sum_rebuf))
            sum_waste = float(info.get("sum_waste_s", sum_waste))

        obs = next_obs
        if done:
            ep_count += 1
            if isinstance(info, dict):
                ep_logs.append(EpisodeLog(
                    global_steps=int(global_steps_before + len(obs_list)),
                    update=int(update_idx),
                    trace_id=str(info.get("trace_id", "unknown")),
                    ep_len=int(ep_len),
                    ep_return=float(ep_ret),
                    sum_rebuf_s=float(info.get("sum_rebuf_s", np.nan)),
                    sum_waste_s=float(info.get("sum_waste_s", np.nan)),
                    sum_downloaded_mbit=float(info.get("sum_downloaded_mbit", np.nan)),
                    sum_waste_mbit=float(info.get("sum_waste_mbit", np.nan)),
                ))
            obs, _ = env.reset()
            ep_ret = 0.0
            ep_len = 0

    # bootstrap
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        _, last_v = model.forward(obs_t)
    ro = Rollout(
        obs=np.stack(obs_list, axis=0),
        acts=np.asarray(act_list, dtype=np.int64),
        rews=np.asarray(rew_list, dtype=np.float32),
        dones=np.asarray(done_list, dtype=np.float32),
        vals=np.asarray(val_list, dtype=np.float32),
        logps=np.asarray(logp_list, dtype=np.float32),
        last_val=float(last_v.item()),
    )
    stats = {"episodes_in_rollout": float(ep_count), "sum_rebuf_s": float(sum_rebuf), "sum_waste_s": float(sum_waste)}
    return ro, stats, ep_logs


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        type=str,
        default="sanity",
        choices=["sanity", "official"],
        help="Training preset. official: rollout_steps=4096 and eval every 4096 steps on 50 FCC-val traces (unless overridden).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Note: defaults here are "sanity". official mode overrides them unless explicitly set.
    p.add_argument("--total_steps", type=int, default=50_000)
    p.add_argument("--rollout_steps", type=int, default=2048)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--train_iters", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--clip_ratio", type=float, default=0.2)
    p.add_argument(
        "--target_kl",
        type=float,
        default=0.02,
        help="Optional target KL for early-stopping PPO updates (approx_kl > target_kl). "
             "Set <=0 to disable.",
    )
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument(
        "--ent_coef_end",
        type=float,
        default=None,
        help="If set with --ent_anneal_steps, linearly anneal ent_coef to this value.",
    )
    p.add_argument(
        "--ent_coef_min",
        type=float,
        default=None,
        help="If set, ent_coef will not go below this value after annealing.",
    )
    p.add_argument(
        "--ent_anneal_steps",
        type=int,
        default=0,
        help="Linear anneal steps for ent_coef (0 to disable).",
    )
    p.add_argument(
        "--uniform_kl_coef",
        type=float,
        default=0.0,
        help="Coefficient for KL(pi || uniform) regularization. Set >0 to enable.",
    )
    p.add_argument("--logits_clip", type=float, default=0.0, help="Clamp policy logits to [-logits_clip, +logits_clip]. Set <=0 to disable.")
    p.add_argument(
        "--rollout_action_mode",
        type=str,
        default="sample",
        choices=["sample", "greedy"],
        help="Action selection for rollout collection.",
    )
    p.add_argument("--debug_numerics", action="store_true", help="Enable strict finite checks for env obs/reward and PPO tensors (raises on NaN/inf).")

    # env/data
    p.add_argument("--events_csv", type=str, default="data/big_matrix_valid_video.csv")
    p.add_argument("--behavior_mode", type=str, default="user_sessions", choices=["user_sessions", "video_index"])
    p.add_argument("--video_index_csv", type=str, default="data/video_index.csv")
    p.add_argument("--min_events", type=int, default=5)
    p.add_argument("--max_events", type=int, default=200)
    p.add_argument("--weight_by_views", action="store_true", help="(video_index mode) sample videos weighted by n_views.")
    p.add_argument("--no_weight_by_views", action="store_true", help="(video_index mode) disable n_views weighting.")
    p.add_argument("--traces_csv", type=str, default="data/fcc_httpgetmt_trace_1s_kbps.csv")
    p.add_argument("--trace_id_col", type=str, default="trace_id")
    p.add_argument("--trace_time_col", type=str, default="t")
    p.add_argument("--trace_bw_col", type=str, default="throughput_kbps")
    p.add_argument("--trace_bw_multiplier", type=float, default=1.0)
    p.add_argument("--trace_fill_mode", type=str, default="ffill", choices=["ffill", "zero", "hold"])
    p.add_argument(
        "--trace_exhaust_mode",
        type=str,
        default="drop_video",
        choices=["truncate", "next", "drop_video"],
        help="What to do when current trace is exhausted during simulation: truncate episode, switch to next trace, or drop current video and truncate.",
    )
    p.add_argument("--trace_ids_file", type=str, default="splits/fcc_train_trace_ids.txt")
    p.add_argument("--k_hist", type=int, default=8)
    p.add_argument("--lambda_waste", type=float, default=0.5)
    p.add_argument(
        "--waste_saturate_k",
        type=float,
        default=1.0,
        help="Waste penalty soft-saturation strength k in waste_ratio = 1 - exp(-k * raw), raw=waste_mbit/max(downloaded_mbit,eps).",
    )
    p.add_argument("--r_min_mbps", type=float, default=0.35, help="QoE log utility reference bitrate (Mbps).")
    p.add_argument("--rebuf_penalty", type=float, default=1.0, help="QoE rebuffering penalty coefficient (paper-style).")
    p.add_argument("--switch_penalty", type=float, default=0.1, help="QoE bitrate switching penalty coefficient.")
    p.add_argument("--qoe_alpha", type=float, default=1.0, help="QoE bitrate log utility scaling alpha (paper: 1).")
    p.add_argument(
        "--switch_penalty_mode",
        type=str,
        default="log",
        choices=["log", "abs"],
        help="Switch penalty mode: 'log' uses abs(log(prev)-log(cur)); 'abs' uses abs(prev_bitrate-cur_bitrate) (paper).",
    )
    p.add_argument(
        "--no_qoe_scale_by_watch_frac",
        action="store_true",
        help="Disable scaling bitrate utility by watched fraction (played_s/watch_s). "
             "If set, any non-zero playback grants full bitrate utility (paper may implicitly do this).",
    )
    p.add_argument(
        "--max_rebuf_s_per_video",
        type=float,
        default=0.0,
        help="User patience cap (seconds) within a single video. Set <=0 for 'infinite patience' (effectively disables swipe due to rebuffer).",
    )
    p.add_argument(
        "--rebuf_penalty_mode",
        type=str,
        default="linear",
        choices=["linear", "cap", "log"],
        help="Rebuffer penalty shaping in env reward. "
             "'linear' matches paper-style; 'cap' or 'log' can improve learnability under infinite patience.",
    )
    p.add_argument("--rebuf_cap_s", type=float, default=12.0, help="Used when --rebuf_penalty_mode=cap.")
    p.add_argument("--rebuf_log_scale_s", type=float, default=2.0, help="Used when --rebuf_penalty_mode=log.")
    p.add_argument(
        "--bitrates_mbps",
        type=str,
        default="0.35,0.6,0.9,1.2,1.8,2.5,4.0",
        help="Comma-separated bitrate ladder in Mbps. Default matches [180p,270p,360p,480p,720p,1080p,1440p].",
    )
    p.add_argument(
        "--bitrate_labels",
        type=str,
        default="180p,270p,360p,480p,720p,1080p,1440p",
        help="Comma-separated labels aligned with --bitrates_mbps.",
    )
    p.add_argument(
        "--prefetch_thresholds",
        type=str,
        default="1,3,6,10,16,30",
        help="Comma-separated prefetch thresholds in seconds for joint action space (bitrate x thresholds).",
    )
    p.add_argument(
        "--enable_future_prefetch",
        action="store_true",
        help="Enable cross-video prefetch: after CURRENT video is fully downloaded, use remaining capacity to prefetch FUTURE videos up to the same prefetch threshold. Default is disabled.",
    )

    # periodic in-distribution eval (FCC val)
    p.add_argument("--eval_every_updates", type=int, default=5)
    p.add_argument("--eval_ids_file", type=str, default="splits/fcc_val_trace_ids.txt")
    p.add_argument("--eval_n_traces", type=int, default=20)
    p.add_argument("--eval_max_videos", type=int, default=200)
    p.add_argument(
        "--eval_select_mode",
        type=str,
        default="head",
        choices=["head", "shuffle"],
        help="How to pick eval_n_traces from eval_ids_file. "
             "'head' takes the first N lines; 'shuffle' selects N without replacement using a deterministic RNG once at startup.",
    )
    p.add_argument(
        "--eval_select_seed",
        type=int,
        default=None,
        help="Seed for deterministic eval trace selection when eval_select_mode=shuffle. Default: seed+99991.",
    )

    # convergence detector (FCC-val)
    p.add_argument("--converge_check", action="store_true", help="Enable convergence detection based on FCC-val metric stability.")
    p.add_argument("--converge_metric", type=str, default="return_per_step", choices=["return_per_step", "rebuf_s_per_step", "waste_mbit_per_step"])
    p.add_argument("--converge_mode", type=str, default="max", choices=["max", "min"], help="Whether higher (max) or lower (min) is better.")
    p.add_argument("--converge_window", type=int, default=5, help="Window size (in eval points) for rolling mean/std.")
    p.add_argument("--converge_mean_delta_max", type=float, default=0.01, help="|mean(t)-mean(t-1)| <= this counts as stable.")
    p.add_argument("--converge_std_max", type=float, default=0.05, help="std(last window) <= this counts as stable.")
    p.add_argument("--converge_patience", type=int, default=3, help="How many consecutive stable windows to declare converged.")
    p.add_argument("--stop_on_converged", action="store_true", help="Stop training early once convergence is declared.")

    # early stopping (based on periodic FCC-val eval)
    p.add_argument("--early_stop", action="store_true", help="Enable early stopping based on FCC-val eval metric.")
    p.add_argument(
        "--early_stop_metric",
        type=str,
        default="return_per_step",
        choices=[
            "return_per_step",
            "ep_return_mean",
            "waste_mbit_per_step",
            "rebuf_s_per_step",
        ],
        help="Which eval metric to monitor for early stopping.",
    )
    p.add_argument(
        "--early_stop_mode",
        type=str,
        default="max",
        choices=["max", "min"],
        help="Whether larger (max) or smaller (min) metric is better.",
    )
    p.add_argument("--early_stop_patience", type=int, default=10, help="Stop if no improvement for this many evals.")
    p.add_argument("--early_stop_min_delta", type=float, default=1e-3, help="Minimum improvement to reset patience.")
    p.add_argument("--early_stop_min_updates", type=int, default=5, help="Do not early-stop before this many updates.")

    # outputs
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--run_name", type=str, default="duasvs_ppo")
    # resume / continue training
    p.add_argument("--init_ckpt", type=str, default=None, help="Optional path to a model state_dict (.pt) to initialize from.")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from a full checkpoint dict (model+optimizer+global_steps+update). "
             "When enabled, --init_ckpt should point to such a dict; otherwise only model weights are loaded.",
    )
    p.add_argument(
        "--start_global_steps",
        type=int,
        default=0,
        help="If resuming, set the starting global_steps (e.g., 200000) so logs/eval continue from that step.",
    )
    args = p.parse_args()

    set_seed(int(args.seed))
    device = torch.device(args.device)

    # --- mode presets ---
    # User requirement: official training validates every 4096 steps using 50 FCC-val traces.
    # In this script, one "update" corresponds to one rollout of rollout_steps.
    if str(args.mode).lower() == "official":
        # Use 4096-step rollouts by default; keep user override if they explicitly changed it.
        if int(args.rollout_steps) == 2048:
            args.rollout_steps = 4096
        # Eval every update => every rollout_steps env steps.
        if int(args.eval_every_updates) == 5:
            args.eval_every_updates = 1
        # Default val traces to 50 in official mode (unless user already changed it from sanity default 20).
        if int(args.eval_n_traces) == 20:
            args.eval_n_traces = 50

    allowed_trace_ids = None
    if args.trace_ids_file and os.path.exists(args.trace_ids_file):
        with open(args.trace_ids_file, "r", encoding="utf-8") as f:
            allowed_trace_ids = [line.strip() for line in f if line.strip()]

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

    env = DUASVSEnv({
        "seed": int(args.seed),
        "events_csv": args.events_csv,
        "behavior_mode": str(args.behavior_mode),
        "video_index_csv": str(args.video_index_csv),
        "min_events": int(args.min_events),
        "max_events": int(args.max_events),
        "weight_by_views": (not bool(args.no_weight_by_views)),
        "traces_csv": args.traces_csv,
        "trace_id_col": str(args.trace_id_col),
        "trace_time_col": str(args.trace_time_col),
        "trace_bw_col": str(args.trace_bw_col),
        "trace_bw_multiplier": float(args.trace_bw_multiplier),
        "trace_fill_mode": str(args.trace_fill_mode),
        "trace_exhaust_mode": str(args.trace_exhaust_mode),
        "allowed_trace_ids": allowed_trace_ids,
        "k_hist": int(args.k_hist),
        "r_min_mbps": float(args.r_min_mbps),
        "rebuf_penalty": float(args.rebuf_penalty),
        "switch_penalty": float(args.switch_penalty),
        "qoe_alpha": float(args.qoe_alpha),
        "switch_penalty_mode": str(args.switch_penalty_mode),
        "qoe_scale_by_watch_frac": (not bool(args.no_qoe_scale_by_watch_frac)),
        "lambda_waste": float(args.lambda_waste),
        "waste_saturate_k": float(args.waste_saturate_k),
        "allow_future_prefetch": bool(args.enable_future_prefetch),
        "max_rebuf_s_per_video": (1e18 if float(args.max_rebuf_s_per_video) <= 0 else float(args.max_rebuf_s_per_video)),
        "rebuf_penalty_mode": str(args.rebuf_penalty_mode),
        "rebuf_cap_s": float(args.rebuf_cap_s),
        "rebuf_log_scale_s": float(args.rebuf_log_scale_s),
        "prefetch_thresholds_s": thresholds,
        "video_bitrates_mbps": bitrates,
        "video_bitrate_labels": labels,
        "debug_checks": bool(args.debug_numerics),
    })

    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.n)
    model = ActorCritic(obs_dim, act_dim, logits_clip=float(args.logits_clip)).to(device)
    opt = optim.Adam(model.parameters(), lr=float(args.lr))

    os.makedirs(args.out_dir, exist_ok=True)
    # User request: "train file" logs per-episode.
    log_path = os.path.join(args.out_dir, f"{args.run_name}_train.csv")
    # Keep per-update summary separately for convergence/throughput diagnostics.
    updates_path = os.path.join(args.out_dir, f"{args.run_name}_updates.csv")
    eval_path = os.path.join(args.out_dir, f"{args.run_name}_eval.csv")
    loss_path = os.path.join(args.out_dir, f"{args.run_name}_loss.csv")
    action_hist_path = os.path.join(args.out_dir, f"{args.run_name}_action_hist.csv")
    ckpt_path = os.path.join(args.out_dir, f"{args.run_name}.pt")
    best_ckpt_path = os.path.join(args.out_dir, f"{args.run_name}_best.pt")
    latest_ckpt_path = os.path.join(args.out_dir, f"{args.run_name}_latest.pt")

    # training loop (update-based)
    global_steps = int(args.start_global_steps)
    if global_steps < 0:
        global_steps = 0

    # optional init/resume from checkpoint
    if args.init_ckpt:
        ck: Any = torch.load(str(args.init_ckpt), map_location=device)
        if isinstance(ck, dict) and "model" in ck:
            model.load_state_dict(ck["model"])
            if bool(args.resume) and "optimizer" in ck:
                try:
                    opt.load_state_dict(ck["optimizer"])
                except Exception as e:
                    print(f"[Resume] WARNING: failed to load optimizer state: {e}")
            if bool(args.resume):
                global_steps = int(ck.get("global_steps", global_steps))
                update = int(ck.get("update", 0))
            print(f"[Init] Loaded checkpoint dict from {args.init_ckpt} (resume={bool(args.resume)})")
        else:
            # backward compatible: plain state_dict
            model.load_state_dict(ck)
            print(f"[Init] Loaded model weights (state_dict) from {args.init_ckpt}")

    # keep update counter consistent with global_steps when possible (unless resumed update is provided)
    if "update" not in locals() or int(update) <= 0:
        if int(args.rollout_steps) > 0:
            update = int(global_steps // int(args.rollout_steps))
        else:
            update = 0

    t_train0 = time.time()

    # early stop tracking
    best_metric: Optional[float] = None
    best_update: int = 0
    bad_evals: int = 0

    # convergence tracking
    conv_vals: List[float] = []
    conv_stable_cnt: int = 0

    # prepare fixed eval trace ids (FCC val by default)
    eval_ids: Optional[List[str]] = None
    if args.eval_ids_file and os.path.exists(args.eval_ids_file):
        with open(args.eval_ids_file, "r", encoding="utf-8") as f:
            eval_ids_all = [line.strip() for line in f if line.strip()]

        # de-dup while preserving order
        seen = set()
        eval_ids_all_uniq: List[str] = []
        for x in eval_ids_all:
            if x not in seen:
                seen.add(x)
                eval_ids_all_uniq.append(str(x))
        eval_ids_all = eval_ids_all_uniq

        n_req = int(args.eval_n_traces)
        if n_req > 0 and len(eval_ids_all) > 0:
            n = int(min(n_req, len(eval_ids_all)))
            if str(args.eval_select_mode).lower() == "shuffle":
                sel_seed = int(args.eval_select_seed) if args.eval_select_seed is not None else (int(args.seed) + 99991)
                rng = np.random.default_rng(int(sel_seed))
                perm = rng.permutation(len(eval_ids_all))[:n]
                eval_ids = [eval_ids_all[int(i)] for i in perm]
            else:
                eval_ids = eval_ids_all[:n]
        else:
            eval_ids = None

    # Persist the selected eval ids for reproducibility / debugging.
    if eval_ids:
        try:
            os.makedirs(args.out_dir, exist_ok=True)
            sel_path = os.path.join(args.out_dir, f"{args.run_name}_eval_trace_ids_selected.txt")
            with open(sel_path, "w", encoding="utf-8") as f:
                for tid in eval_ids:
                    f.write(f"{tid}\n")
        except Exception as e:
            print(f"[EvalIds] WARNING: failed to write selected eval ids file: {e}")

    # Build a persistent eval environment to avoid re-reading FCC CSV every eval call.
    eval_env: Optional[DUASVSEnv] = None
    if eval_ids:
        eval_env = DUASVSEnv({
            "seed": int(args.seed),
            "events_csv": args.events_csv,
            "behavior_mode": str(args.behavior_mode),
            "video_index_csv": str(args.video_index_csv),
            "min_events": int(args.min_events),
            "max_events": int(args.max_events),
            "weight_by_views": (not bool(args.no_weight_by_views)),
            "traces_csv": args.traces_csv,
            "trace_id_col": str(args.trace_id_col),
            "trace_time_col": str(args.trace_time_col),
            "trace_bw_col": str(args.trace_bw_col),
            "trace_bw_multiplier": float(args.trace_bw_multiplier),
            "trace_fill_mode": str(args.trace_fill_mode),
            "trace_exhaust_mode": str(args.trace_exhaust_mode),
            "allowed_trace_ids": [str(x) for x in eval_ids],
            "k_hist": int(args.k_hist),
            "r_min_mbps": float(args.r_min_mbps),
            "rebuf_penalty": float(args.rebuf_penalty),
            "switch_penalty": float(args.switch_penalty),
            "qoe_alpha": float(args.qoe_alpha),
            "switch_penalty_mode": str(args.switch_penalty_mode),
            "qoe_scale_by_watch_frac": (not bool(args.no_qoe_scale_by_watch_frac)),
            "lambda_waste": float(args.lambda_waste),
            "waste_saturate_k": float(args.waste_saturate_k),
            "allow_future_prefetch": bool(args.enable_future_prefetch),
            "max_rebuf_s_per_video": (1e18 if float(args.max_rebuf_s_per_video) <= 0 else float(args.max_rebuf_s_per_video)),
            "rebuf_penalty_mode": str(args.rebuf_penalty_mode),
            "rebuf_cap_s": float(args.rebuf_cap_s),
            "rebuf_log_scale_s": float(args.rebuf_log_scale_s),
            "prefetch_thresholds_s": thresholds,
            "video_bitrates_mbps": bitrates,
            "video_bitrate_labels": labels,
            "debug_checks": bool(args.debug_numerics),
        })

    # IMPORTANT: avoid mixing multiple fresh runs into the same CSVs.
    # If --resume is NOT set, we overwrite existing logs for this run_name.
    file_mode = "a" if bool(args.resume) else "w"

    # Per-episode training log
    with open(log_path, file_mode, newline="") as ep_f:
        ep_w = csv.writer(ep_f)
        if ep_f.tell() == 0:
            ep_w.writerow([
                "global_steps", "update", "trace_id",
                "ep_len", "ep_return",
                "sum_rebuf_s", "sum_waste_s",
                "sum_downloaded_mbit", "sum_waste_mbit",
            ])

        # Per-update summary
        upd_f = open(updates_path, file_mode, newline="")
        upd_w = csv.writer(upd_f)
        if upd_f.tell() == 0:
            upd_w.writerow([
                "global_steps", "update", "episodes_in_rollout",
                "sum_rebuf_s", "sum_waste_s",
                "act_dim", "prefetch_thresholds_s",
                "update_wall_time_s", "steps_per_sec", "elapsed_s",
                "best_metric", "bad_evals",
            ])

        # eval csv
        eval_f = open(eval_path, file_mode, newline="")
        eval_w = csv.writer(eval_f)
        if eval_f.tell() == 0:
            eval_w.writerow([
                "global_steps", "update", "eval_seed", "n_traces",
                "ep_return_mean", "ep_len_mean",
                "return_per_step", "rebuf_s_per_step", "waste_s_per_step",
                "downloaded_mbit_per_step", "waste_mbit_per_step",
                "waste_raw_mean", "waste_raw_p50", "waste_raw_p90",
                "waste_ratio_mean", "waste_ratio_p50", "waste_ratio_p90",
                "early_stop_metric", "early_stop_best", "early_stop_bad_evals",
            ])

        # loss csv
        loss_f = open(loss_path, file_mode, newline="")
        loss_w = csv.writer(loss_f)
        if loss_f.tell() == 0:
            loss_w.writerow(["global_steps", "update", "policy_loss", "value_loss", "entropy", "total_loss", "approx_kl", "clip_frac"])

        # action histogram csv (to detect policy collapse)
        act_f = open(action_hist_path, file_mode, newline="")
        act_w = csv.writer(act_f)
        if act_f.tell() == 0:
            act_dim_i = int(env.action_space.n)
            P = int(len(thresholds))
            H = int(len(bitrates))
            act_w.writerow(
                ["global_steps", "update", "entropy", "top_action", "top_action_frac"]
                + [f"act_frac_{i}" for i in range(act_dim_i)]
                + [f"bitrate_frac_{i}" for i in range(H)]
                + [f"thr_frac_{i}" for i in range(P)]
            )

        while global_steps < int(args.total_steps):
            t_up0 = time.time()
            ro, stats, ep_logs = rollout_steps(
                env,
                model,
                device,
                int(args.rollout_steps),
                global_steps_before=int(global_steps),
                update_idx=int(update + 1),
                action_mode=str(args.rollout_action_mode),
            )
            global_steps += int(args.rollout_steps)
            update += 1

            adv_np, ret_np = compute_gae(ro.rews, ro.dones, ro.vals, ro.last_val, float(args.gamma), float(args.lam))

            obs_t = torch.as_tensor(ro.obs, dtype=torch.float32, device=device)
            acts_t = torch.as_tensor(ro.acts, dtype=torch.int64, device=device)
            logp_old_t = torch.as_tensor(ro.logps, dtype=torch.float32, device=device)
            adv_t = torch.as_tensor(adv_np, dtype=torch.float32, device=device)
            ret_t = torch.as_tensor(ret_np, dtype=torch.float32, device=device)

            ent_coef_now = float(args.ent_coef)
            if args.ent_coef_end is not None and int(getattr(args, "ent_anneal_steps", 0)) > 0:
                end = float(args.ent_coef_end)
                steps = float(max(1, int(args.ent_anneal_steps)))
                frac = float(np.clip(float(global_steps) / steps, 0.0, 1.0))
                ent_coef_now = float(ent_coef_now + frac * (end - ent_coef_now))
            if args.ent_coef_min is not None:
                ent_coef_now = float(max(ent_coef_now, float(args.ent_coef_min)))

            loss_stats = ppo_update(
                model,
                opt,
                obs_t,
                acts_t,
                logp_old_t,
                adv_t,
                ret_t,
                clip_ratio=float(args.clip_ratio),
                vf_coef=float(args.vf_coef),
                ent_coef=float(ent_coef_now),
                uniform_kl_coef=float(args.uniform_kl_coef),
                target_kl=(None if float(args.target_kl) <= 0 else float(args.target_kl)),
                train_iters=int(args.train_iters),
                batch_size=int(args.batch_size),
                debug_numerics=bool(args.debug_numerics),
            )

            t_up1 = time.time()
            update_wall = float(t_up1 - t_up0)
            sps = float(int(args.rollout_steps) / max(1e-9, update_wall))
            elapsed = float(t_up1 - t_train0)

            # write per-episode logs produced during this rollout
            for r in ep_logs:
                ep_w.writerow([
                    r.global_steps,
                    r.update,
                    r.trace_id,
                    r.ep_len,
                    r.ep_return,
                    r.sum_rebuf_s,
                    r.sum_waste_s,
                    r.sum_downloaded_mbit,
                    r.sum_waste_mbit,
                ])
            ep_f.flush()

            # write per-update summary
            upd_w.writerow([
                global_steps,
                update,
                stats["episodes_in_rollout"],
                stats["sum_rebuf_s"],
                stats["sum_waste_s"],
                int(env.action_space.n),
                "|".join(str(x) for x in thresholds),
                update_wall,
                sps,
                elapsed,
                best_metric,
                bad_evals,
            ])
            upd_f.flush()
            loss_w.writerow([
                global_steps,
                update,
                loss_stats.get("policy_loss"),
                loss_stats.get("value_loss"),
                loss_stats.get("entropy"),
                loss_stats.get("total_loss"),
                loss_stats.get("approx_kl"),
                loss_stats.get("clip_frac"),
            ])
            loss_f.flush()
            print(f"[Update {update}] steps={global_steps} eps_in_rollout={stats['episodes_in_rollout']:.0f} "
                  f"sum_rebuf_s={stats['sum_rebuf_s']:.1f} sum_waste_s={stats['sum_waste_s']:.1f}")

            # action distribution logging (rollout actions are sampled from the policy)
            try:
                act_dim_i = int(env.action_space.n)
                P = int(len(thresholds))
                H = int(len(bitrates))
                counts = np.bincount(ro.acts.astype(np.int64), minlength=act_dim_i).astype(np.float64)
                denom = float(max(1.0, counts.sum()))
                act_frac = (counts / denom).astype(np.float64)
                top_a = int(np.argmax(act_frac)) if act_frac.size > 0 else 0
                top_frac = float(act_frac[top_a]) if act_frac.size > 0 else 0.0
                bitrate_frac = np.zeros(H, dtype=np.float64)
                thr_frac = np.zeros(P, dtype=np.float64)
                if act_frac.size > 0 and P > 0:
                    for a_i, p_i in enumerate(act_frac):
                        if p_i <= 0:
                            continue
                        bi = int(a_i // P)
                        ti = int(a_i % P)
                        if 0 <= bi < H:
                            bitrate_frac[bi] += float(p_i)
                        if 0 <= ti < P:
                            thr_frac[ti] += float(p_i)
                act_w.writerow(
                    [global_steps, update, float(loss_stats.get("entropy", float("nan"))), top_a, top_frac]
                    + [float(x) for x in act_frac.tolist()]
                    + [float(x) for x in bitrate_frac.tolist()]
                    + [float(x) for x in thr_frac.tolist()]
                )
                act_f.flush()
            except Exception as e:
                print(f"[ActionHist] WARNING: failed to write action histogram: {e}")

            # periodic deterministic eval on fixed FCC val traces
            if eval_ids and int(args.eval_every_updates) > 0 and (update % int(args.eval_every_updates) == 0):
                eval_seed = int(1000 + update)
                ev = eval_on_traces(
                    model=model,
                    device=device,
                    env_cfg=None,
                    trace_ids=eval_ids,
                    eval_seed=eval_seed,
                    max_videos=int(args.eval_max_videos),
                    reuse_env=eval_env,
                )

                # Track best checkpoint on FCC-val (always useful; independent of early_stop).
                metric_name = str(args.early_stop_metric)
                metric_val = ev.get(metric_name)
                improved = False
                if (metric_val is not None) and np.isfinite(float(metric_val)):
                    mv = float(metric_val)
                    if best_metric is None:
                        improved = True
                    else:
                        # Use the same improvement rule as early_stop (min_delta + mode),
                        # but apply it regardless of whether early_stop is enabled.
                        if str(args.early_stop_mode) == "max":
                            improved = (mv > float(best_metric) + float(args.early_stop_min_delta))
                        else:
                            improved = (mv < float(best_metric) - float(args.early_stop_min_delta))
                    if improved:
                        best_metric = mv
                        best_update = int(update)
                        bad_evals = 0
                        torch.save(
                            {
                                "model": model.state_dict(),
                                "optimizer": opt.state_dict(),
                                "global_steps": int(global_steps),
                                "update": int(update),
                                "best_metric": best_metric,
                            },
                            best_ckpt_path,
                        )
                    else:
                        bad_evals += 1

                eval_w.writerow([
                    global_steps,
                    update,
                    eval_seed,
                    len(eval_ids),
                    ev.get("ep_return_mean"),
                    ev.get("ep_len_mean"),
                    ev.get("return_per_step"),
                    ev.get("rebuf_s_per_step"),
                    ev.get("waste_s_per_step"),
                    ev.get("downloaded_mbit_per_step"),
                    ev.get("waste_mbit_per_step"),
                    ev.get("waste_raw_mean"),
                    ev.get("waste_raw_p50"),
                    ev.get("waste_raw_p90"),
                    ev.get("waste_ratio_mean"),
                    ev.get("waste_ratio_p50"),
                    ev.get("waste_ratio_p90"),
                    metric_name,
                    best_metric,
                    bad_evals,
                ])
                eval_f.flush()
                print(
                    f"[Eval@FCC] update={update} step={global_steps} "
                    f"return/step={ev.get('return_per_step'):.4f} "
                    f"rebuf_s/step={ev.get('rebuf_s_per_step'):.4f} "
                    f"waste_mbit/step={ev.get('waste_mbit_per_step'):.4f} "
                    f"raw(p50/p90)={ev.get('waste_raw_p50'):.3f}/{ev.get('waste_raw_p90'):.3f} "
                    f"wz(p50/p90)={ev.get('waste_ratio_p50'):.3f}/{ev.get('waste_ratio_p90'):.3f}"
                )

                # Convergence detection: stability of rolling mean/std on FCC-val.
                if bool(args.converge_check):
                    mname = str(args.converge_metric)
                    mv = ev.get(mname)
                    if mv is not None and np.isfinite(float(mv)):
                        conv_vals.append(float(mv))
                        w = int(max(2, args.converge_window))
                        if len(conv_vals) >= 2 * w:
                            prev = np.asarray(conv_vals[-2*w:-w], dtype=float)
                            cur = np.asarray(conv_vals[-w:], dtype=float)
                            prev_mean = float(prev.mean())
                            cur_mean = float(cur.mean())
                            cur_std = float(cur.std())
                            mean_delta = float(abs(cur_mean - prev_mean))

                            stable = (mean_delta <= float(args.converge_mean_delta_max)) and (cur_std <= float(args.converge_std_max))
                            if stable:
                                conv_stable_cnt += 1
                            else:
                                conv_stable_cnt = 0

                            if conv_stable_cnt >= int(args.converge_patience):
                                print(
                                    f"[Converged] metric={mname} window={w} "
                                    f"prev_mean={prev_mean:.4f} cur_mean={cur_mean:.4f} cur_std={cur_std:.4f} "
                                    f"mean_delta={mean_delta:.4f} stable_windows={conv_stable_cnt}"
                                )
                                if bool(args.stop_on_converged):
                                    break

                if args.early_stop and update >= int(args.early_stop_min_updates):
                    if bad_evals >= int(args.early_stop_patience):
                        print(
                            f"[EarlyStop] Triggered at update={update} step={global_steps}. "
                            f"metric={metric_name} best={best_metric} best_update={best_update} "
                            f"bad_evals={bad_evals} patience={int(args.early_stop_patience)}"
                        )
                        break

            # Always save a "latest" checkpoint for true resume (model + optimizer + counters)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "global_steps": int(global_steps),
                    "update": int(update),
                    "best_metric": best_metric,
                },
                latest_ckpt_path,
            )

        eval_f.close()
        loss_f.close()
        upd_f.close()
        act_f.close()

    # Final checkpoint as a dict (supports --resume).
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "global_steps": int(global_steps),
            "update": int(update),
            "best_metric": best_metric,
        },
        ckpt_path,
    )
    print(f"Saved: {ckpt_path}")
    if os.path.exists(best_ckpt_path):
        print(f"Best : {best_ckpt_path}")
    if os.path.exists(latest_ckpt_path):
        print(f"Latest: {latest_ckpt_path}")
    print(f"Train (per-episode): {log_path}")
    print(f"Updates (per-4096): {updates_path}")
    print(f"Eval : {eval_path}")
    print(f"Loss : {loss_path}")
    print(f"ActionHist: {action_hist_path}")


if __name__ == "__main__":
    main()


