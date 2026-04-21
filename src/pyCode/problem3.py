#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
problem3.py — 2026 MathorCup C 题 · 问题三
6 个月痰湿体质个性化干预方案 — v2 统一 NMB 架构

在原严格按题意的加性动力学之上，引入频次序列的月间渐进性约束，
避免最优解出现“五月恒定 + 一月异常”的剧烈跳变处方。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGURE_DIR, TABLE_DIR, ensure_output_dirs, find_sample_data_path


mpl.rcParams["font.sans-serif"] = [
    "Noto Sans CJK JP",
    "Noto Sans CJK SC",
    "WenQuanYi Zen Hei",
    "SimHei",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


# ========================================================================
# Part 0 · 参数常量
# ========================================================================
ACT_COST = {1: 3, 2: 5, 3: 8}
ACT_MIN = {1: 10, 2: 20, 3: 30}
TCM_COST = {1: 30, 2: 80, 3: 130}

RULE_K = 0.03
RULE_F = 0.01
F_ANCHOR = 5
MONTHS = 6
WEEKS_PER_MONTH = 4
BUDGET_SOFT = 2000
BUDGET_HARD = 2400

LAMBDA_LIST = [30, 50, 100, 200]
LAMBDA_BASE = 30
LAMBDA_GRID = np.linspace(0, 260, 53)

F_EFFECTIVE_SET = [5, 6, 7, 8, 9, 10]

TAU_GRID = [1.0, 1.5, 2.0, 2.5, math.inf]
TAU_BASE = 1.5


# ========================================================================
# Part 1 · 动力学
# ========================================================================
def tcm_tier(score: float) -> int:
    if score >= 62:
        return 3
    if score >= 59:
        return 2
    return 1


def r_exercise(
    k: int,
    f: int,
    rule_k: float = RULE_K,
    rule_f: float = RULE_F,
) -> float:
    if f < F_ANCHOR:
        return 0.0
    return rule_k * k + rule_f * (f - F_ANCHOR)


def step_month(
    k: int,
    f: int,
    score_prev: float,
    **dyn_kw,
) -> Tuple[float, int, float, float, float]:
    tier = tcm_tier(score_prev)
    rate = r_exercise(k, f, **dyn_kw)
    score_new = score_prev * (1 - rate)
    act_cost = WEEKS_PER_MONTH * f * ACT_COST[k]
    tcm_cost = TCM_COST[tier]
    return score_new, tier, rate, act_cost, tcm_cost


def tv_l2_rms(f_seq: Iterable[int]) -> float:
    arr = np.asarray(list(f_seq), dtype=float)
    if arr.size < 2:
        return 0.0
    diff = np.diff(arr)
    return float(np.sqrt(np.mean(diff**2)))


# ========================================================================
# Part 2 · 月度 Pareto 前沿推进
# ========================================================================
@dataclass
class Plan:
    k_seq: Tuple[int, ...]
    f_seq: Tuple[int, ...]
    S_traj: Tuple[float, ...]
    L_seq: Tuple[int, ...]
    r_seq: Tuple[float, ...]
    cost_month: Tuple[float, ...]
    C_total: float
    S6: float
    S0: float

    @property
    def E(self) -> float:
        return self.S0 - self.S6

    @property
    def rms_df(self) -> float:
        return tv_l2_rms(self.f_seq)


def enumerate_pareto(
    S0: float,
    K_i: List[int],
    f_values: List[int] = F_EFFECTIVE_SET,
    budget: float = BUDGET_HARD,
    tau_max: float = math.inf,
    **dyn_kw,
) -> List[Plan]:
    frontier: List[tuple] = [(S0, 0.0, (), (), (S0,), (), (), ())]
    for t in range(MONTHS):
        new_states = []
        for (S_t, C_t, ks, fs, Straj, Ls, rs, cms) in frontier:
            for k in K_i:
                for f in f_values:
                    if fs:
                        partial = fs + (f,)
                        partial_rms = float(
                            np.sqrt(np.sum(np.diff(np.asarray(partial, dtype=float)) ** 2) / (MONTHS - 1))
                        )
                        if partial_rms > tau_max + 1e-9:
                            continue
                    S_new, L, r, c_act, c_tcm = step_month(k, f, S_t, **dyn_kw)
                    dC = c_act + c_tcm
                    C_new = C_t + dC
                    if C_new > budget:
                        continue
                    new_states.append(
                        (
                            S_new,
                            C_new,
                            ks + (k,),
                            fs + (f,),
                            Straj + (S_new,),
                            Ls + (L,),
                            rs + (r,),
                            cms + (dC,),
                        )
                    )
        if t < MONTHS - 1:
            frontier = _pareto_prune(new_states)
        else:
            frontier = _pareto_prune_global(new_states)

    plans = []
    for (S_t, C_t, ks, fs, Straj, Ls, rs, cms) in frontier:
        if tv_l2_rms(fs) > tau_max + 1e-9:
            continue
        plans.append(
            Plan(
                k_seq=ks,
                f_seq=fs,
                S_traj=Straj,
                L_seq=Ls,
                r_seq=rs,
                cost_month=cms,
                C_total=C_t,
                S6=S_t,
                S0=S0,
            )
        )
    plans.sort(key=lambda p: (p.E, p.C_total))
    return plans


def _pareto_prune(states: List[tuple]) -> List[tuple]:
    if not states:
        return []
    from collections import defaultdict

    groups = defaultdict(list)
    for s in states:
        groups[tcm_tier(s[0])].append(s)

    kept = []
    for items in groups.values():
        items = sorted(items, key=lambda s: (s[1], s[0]))
        min_s = float("inf")
        for s in items:
            if s[0] < min_s - 1e-9:
                kept.append(s)
                min_s = s[0]
    return kept


def _pareto_prune_global(states: List[tuple]) -> List[tuple]:
    if not states:
        return []
    states = sorted(states, key=lambda s: (s[1], s[0]))
    kept = []
    min_s = float("inf")
    for s in states:
        if s[0] < min_s - 1e-9:
            kept.append(s)
            min_s = s[0]
    return kept


# ========================================================================
# Part 3 · Extended Dominance
# ========================================================================
def extended_dominance(plans: List[Plan]) -> List[Plan]:
    P = sorted(plans, key=lambda p: (p.E, p.C_total))
    dedup: Dict[float, Plan] = {}
    for p in P:
        if p.E not in dedup or p.C_total < dedup[p.E].C_total:
            dedup[p.E] = p
    P = sorted(dedup.values(), key=lambda p: p.E)
    changed = True
    while changed and len(P) >= 3:
        changed = False
        for j in range(1, len(P) - 1):
            dE1 = P[j].E - P[j - 1].E
            dE2 = P[j + 1].E - P[j].E
            if dE1 <= 1e-9 or dE2 <= 1e-9:
                continue
            icer_prev = (P[j].C_total - P[j - 1].C_total) / dE1
            icer_next = (P[j + 1].C_total - P[j].C_total) / dE2
            if icer_prev > icer_next + 1e-9:
                P.pop(j)
                changed = True
                break
    return P


# ========================================================================
# Part 4 · 选点
# ========================================================================
def nmb_argmax(plans: List[Plan], lam: float) -> Tuple[int, Plan, float]:
    nmbs = [lam * p.E - p.C_total for p in plans]
    idx = int(np.argmax(nmbs))
    return idx, plans[idx], nmbs[idx]


def icer_frontier(plans: List[Plan]) -> List[float]:
    P = sorted(plans, key=lambda p: p.E)
    icers = []
    for i in range(1, len(P)):
        dE = P[i].E - P[i - 1].E
        dC = P[i].C_total - P[i - 1].C_total
        icers.append(dC / dE if dE > 1e-9 else np.inf)
    return icers


def topsis_entropy(plans: List[Plan]) -> Tuple[int, Plan]:
    X = np.array([[p.S6, p.C_total] for p in plans], dtype=float)
    norm = np.sqrt((X**2).sum(axis=0))
    norm[norm == 0] = 1
    R = X / norm
    P = R / R.sum(axis=0)
    P = np.clip(P, 1e-12, 1.0)
    E = -1 / np.log(len(R)) * (P * np.log(P)).sum(axis=0)
    w = (1 - E) / (1 - E).sum()
    V = R * w
    A_plus = V.min(axis=0)
    A_minus = V.max(axis=0)
    d_plus = np.sqrt(((V - A_plus) ** 2).sum(axis=1))
    d_minus = np.sqrt(((V - A_minus) ** 2).sum(axis=1))
    c_rel = d_minus / (d_plus + d_minus + 1e-12)
    idx = int(np.argmax(c_rel))
    return idx, plans[idx]


def kneedle(plans: List[Plan]) -> Tuple[int, Plan]:
    if len(plans) <= 2:
        return 0, plans[0]
    P = sorted(plans, key=lambda p: p.C_total)
    x = np.array([p.C_total for p in P])
    y = np.array([p.S6 for p in P])
    x_n = (x - x.min()) / (np.ptp(x) + 1e-12)
    y_n = (y - y.min()) / (np.ptp(y) + 1e-12)
    x0, y0 = x_n[0], y_n[0]
    x1, y1 = x_n[-1], y_n[-1]
    d = np.abs((x1 - x0) * (y0 - y_n) - (x0 - x_n) * (y1 - y0))
    idx = int(np.argmax(d))
    return idx, P[idx]


# ========================================================================
# Part 5 · 数据读取与可行集
# ========================================================================
FEATURE_RENAME = {
    "样本ID": "id",
    "样本 ID": "id",
    "ID": "id",
    "年龄组": "age",
    "年龄": "age",
    "性别": "gender",
    "体质标签": "type",
    "中医体质": "type",
    "痰湿质": "S0",
    "痰湿积分": "S0",
    "痰湿质积分": "S0",
    "ADL总分": "adl",
    "ADL 总分": "adl",
    "IADL总分": "iadl",
    "IADL 总分": "iadl",
    "活动量表总分（ADL总分+IADL总分）": "act_score",
    "活动量表总分": "act_score",
    "活动量表": "act_score",
}


def load_patients(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"不支持的文件格式：{ext}")
    df = df.rename(columns={c: FEATURE_RENAME.get(c.strip(), c) for c in df.columns})
    if "S0" not in df.columns:
        for c in df.columns:
            if str(c) == "痰湿质" or ("痰湿" in str(c) and "积分" in str(c)):
                df = df.rename(columns={c: "S0"})
                break
    if "act_score" not in df.columns and "adl" in df.columns and "iadl" in df.columns:
        df["act_score"] = df["adl"] + df["iadl"]
    return df


def resolve_output_dirs(out_root: Optional[str] = None) -> Tuple[Path, Path]:
    if out_root:
        root = Path(out_root)
        fig_dir = root / "figures"
        table_dir = root / "tables"
        fig_dir.mkdir(parents=True, exist_ok=True)
        table_dir.mkdir(parents=True, exist_ok=True)
        return fig_dir, table_dir
    ensure_output_dirs()
    return FIGURE_DIR, TABLE_DIR


def feasible_K(age_group: int, act_score: float) -> List[int]:
    K_age = {1: [1, 2, 3], 2: [1, 2, 3], 3: [1, 2], 4: [1, 2], 5: [1]}
    if act_score < 40:
        K_score = [1]
    elif act_score < 60:
        K_score = [1, 2]
    else:
        K_score = [1, 2, 3]
    return sorted(set(K_age[int(age_group)]) & set(K_score))


# ========================================================================
# Part 6 · 推荐对象
# ========================================================================
@dataclass
class Recommendation:
    pid: int
    S0: float
    K_i: List[int]
    age_group: int
    pareto_full: List[Plan]
    frontier_ed: List[Plan]
    rec_nmb: Plan
    nmb_values: Dict[float, float]
    rec_by_lam: Dict[float, Plan]
    rec_topsis: Plan
    rec_knee: Plan
    icer_series: List[float]


def recommend_patient(
    pid: int,
    S0: float,
    age: int,
    act: float,
    tau_max: float = math.inf,
    lam_list: List[float] = LAMBDA_LIST,
    lam_base: float = LAMBDA_BASE,
    **dyn_kw,
) -> Recommendation:
    K_i = feasible_K(age, act)
    if not K_i:
        K_i = [1]
    plans_all = enumerate_pareto(S0, K_i, tau_max=tau_max, **dyn_kw)
    plans_ed = extended_dominance(plans_all)
    rec_by_lam = {lam: nmb_argmax(plans_ed, lam)[1] for lam in lam_list}
    rec_nmb = rec_by_lam[lam_base]
    nmb_values = {lam: nmb_argmax(plans_ed, lam)[2] for lam in lam_list}
    _, rec_topsis = topsis_entropy(plans_ed)
    _, rec_knee = kneedle(plans_ed)
    icers = icer_frontier(plans_ed)
    return Recommendation(
        pid=pid,
        S0=S0,
        K_i=K_i,
        age_group=age,
        pareto_full=plans_all,
        frontier_ed=plans_ed,
        rec_nmb=rec_nmb,
        nmb_values=nmb_values,
        rec_by_lam=rec_by_lam,
        rec_topsis=rec_topsis,
        rec_knee=rec_knee,
        icer_series=icers,
    )


# ========================================================================
# Part 7 · 表格
# ========================================================================
def tabulate_prescription(plan: Plan) -> pd.DataFrame:
    rows = []
    for t in range(MONTHS):
        k, f, L = plan.k_seq[t], plan.f_seq[t], plan.L_seq[t]
        rate = plan.r_seq[t]
        S_start = plan.S_traj[t]
        S_end = plan.S_traj[t + 1]
        c_act = WEEKS_PER_MONTH * f * ACT_COST[k]
        c_tcm = TCM_COST[L]
        rows.append(
            {
                "月": t + 1,
                "k": k,
                "f": f,
                "L": L,
                "月初S": round(S_start, 2),
                "r(%)": round(rate * 100, 2),
                "月末S": round(S_end, 2),
                "活动费": c_act,
                "中医费": c_tcm,
                "小计": c_act + c_tcm,
            }
        )
    total_row = pd.DataFrame(
        [
            {
                "月": "合计",
                "k": "",
                "f": "",
                "L": "",
                "月初S": "",
                "r(%)": "",
                "月末S": round(plan.S6, 2),
                "活动费": int(sum(plan.cost_month) - sum(TCM_COST[L] for L in plan.L_seq)),
                "中医费": int(sum(TCM_COST[L] for L in plan.L_seq)),
                "小计": round(plan.C_total, 2),
            }
        ]
    )
    return pd.concat([pd.DataFrame(rows), total_row], ignore_index=True)


def expand_prescription_v2(plan: Plan, S0: float) -> pd.DataFrame:
    rows = []
    S = S0
    partial_f = []
    for month, (k, f) in enumerate(zip(plan.k_seq, plan.f_seq), start=1):
        partial_f.append(f)
        L = tcm_tier(S)
        r = r_exercise(k, f)
        S_end = S * (1 - r)
        act_cost = WEEKS_PER_MONTH * f * ACT_COST[k]
        tcm_cost = TCM_COST[L]
        rows.append(
            {
                "月": month,
                "k": k,
                "f": f,
                "L": L,
                "月初S": round(S, 2),
                "r(%)": round(r * 100, 2),
                "月末S": round(S_end, 2),
                "活动费": act_cost,
                "中医费": tcm_cost,
                "小计": act_cost + tcm_cost,
                "RMS(Δf)": round(tv_l2_rms(partial_f), 2),
            }
        )
        S = S_end
    return pd.DataFrame(rows)


# ========================================================================
# Part 8 · 可视化
# ========================================================================
def plot_pareto_three(recs: List[Recommendation], out: str, lam: float = LAMBDA_BASE) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, r in zip(axes, recs):
        C = [p.C_total for p in r.pareto_full]
        S = [p.S6 for p in r.pareto_full]
        Ce = [p.C_total for p in r.frontier_ed]
        Se = [p.S6 for p in r.frontier_ed]
        ax.scatter(C, S, s=18, color="#a0a0a0", alpha=0.4, label="原始 Pareto")
        ax.plot(Ce, Se, "-o", color="steelblue", ms=5, label="ED 后效率前沿")
        ax.scatter(
            r.rec_nmb.C_total,
            r.rec_nmb.S6,
            marker="*",
            s=260,
            c="red",
            zorder=10,
            edgecolor="k",
            linewidths=0.8,
            label=f"NMB 推荐 (λ={int(lam)})",
        )
        ax.scatter(
            r.rec_knee.C_total,
            r.rec_knee.S6,
            marker="^",
            s=95,
            facecolors="none",
            edgecolors="darkorange",
            linewidths=1.6,
            label="Kneedle 对照",
        )
        ax.scatter(
            r.rec_topsis.C_total,
            r.rec_topsis.S6,
            marker="s",
            s=75,
            facecolors="none",
            edgecolors="green",
            linewidths=1.6,
            label="TOPSIS 对照",
        )
        ax.axvline(BUDGET_SOFT, ls="--", color="gray", alpha=0.6)
        ax.set_xlabel("总成本 $C_{\\mathrm{total}}$ (元)")
        ax.set_ylabel("6 月末积分 $S_6$")
        ax.set_title(f"样本 {r.pid}  $S_0={r.S0:.0f}$, $K_i=\\{{ {','.join(map(str, r.K_i))} \\}}$")
        ax.grid(alpha=0.3)
        if r.pid == recs[0].pid:
            ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()


def plot_trajectory_three(recs: List[Recommendation], out: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, r in zip(axes, recs):
        traj = list(r.rec_nmb.S_traj)
        ax.plot(range(len(traj)), traj, "-o", color="crimson", lw=2, ms=7)
        for t, (k, f, L) in enumerate(zip(r.rec_nmb.k_seq, r.rec_nmb.f_seq, r.rec_nmb.L_seq)):
            ax.annotate(
                f"$L{L}$\n$({k},{f})$",
                xy=(t + 1, traj[t + 1]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
            )
        ax.axhline(62, ls="--", color="crimson", alpha=0.5, label="$L=3$ 阈值")
        ax.axhline(59, ls="--", color="darkorange", alpha=0.5, label="$L=2$ 阈值")
        ax.set_xlabel("月份")
        ax.set_ylabel("痰湿积分 $S_t$")
        drop = r.rec_nmb.E / r.S0 * 100 if r.S0 > 0 else 0
        ax.set_title(
            f"样本 {r.pid} — NMB 推荐 (总成本 {r.rec_nmb.C_total:.0f} 元, "
            f"$S_6={r.rec_nmb.S6:.2f}$, 降 {drop:.1f}%)"
        )
        ax.grid(alpha=0.3)
        if r.pid == recs[0].pid:
            ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()


def plot_lambda_sweep(recs: List[Recommendation], out: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    lam_grid = np.arange(0, 260, 2)
    for ax, r in zip(axes, recs):
        Cs, S6s = [], []
        for lam in lam_grid:
            _, p, _ = nmb_argmax(r.frontier_ed, lam)
            Cs.append(p.C_total)
            S6s.append(p.S6)
        Ce = [p.C_total for p in r.frontier_ed]
        Se = [p.S6 for p in r.frontier_ed]
        ax.plot(Ce, Se, "-o", color="lightblue", ms=4, lw=1, alpha=0.6, label="ED 效率前沿", zorder=2)
        sc = ax.scatter(Cs, S6s, c=lam_grid, cmap="plasma", s=35, zorder=5)
        for lam, mk, col in [(30, "*", "red"), (50, "D", "orange"), (100, "s", "green"), (200, "^", "purple")]:
            _, p, _ = nmb_argmax(r.frontier_ed, lam)
            ax.scatter(
                p.C_total,
                p.S6,
                marker=mk,
                s=160,
                c=col,
                edgecolor="k",
                linewidths=0.8,
                zorder=10,
                label=f"λ={lam}",
            )
        ax.set_xlabel("总成本 $C_{\\mathrm{total}}$ (元)")
        ax.set_ylabel("6 月末积分 $S_6$")
        ax.set_title(f"样本 {r.pid}  $S_0={r.S0:.0f}$, $|K|={len(r.K_i)}$")
        ax.grid(alpha=0.3)
        if r.pid == recs[0].pid:
            ax.legend(fontsize=7, loc="upper right")
        fig.colorbar(sc, ax=ax, label="λ (元/分)", shrink=0.8)
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()


def plot_icer_ladder(recs: List[Recommendation], out: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, r in zip(axes, recs):
        if len(r.frontier_ed) < 2:
            ax.text(0.5, 0.5, "前沿点数不足", transform=ax.transAxes, ha="center")
            ax.set_title(f"样本 {r.pid}")
            continue
        P = sorted(r.frontier_ed, key=lambda p: p.E)
        Es = [p.E for p in P]
        icers = [0] + r.icer_series
        for j in range(len(icers) - 1):
            ax.plot([Es[j], Es[j + 1]], [icers[j + 1], icers[j + 1]], "-", color="steelblue", lw=2)
            if j < len(icers) - 2:
                ax.plot([Es[j + 1], Es[j + 1]], [icers[j + 1], icers[j + 2]], ":", color="steelblue", lw=1, alpha=0.7)
        for lam, col, lbl in [(30, "red", "λ=30"), (50, "orange", "λ=50"), (100, "green", "λ=100"), (200, "purple", "λ=200")]:
            ax.axhline(lam, ls="--", color=col, alpha=0.6, lw=1, label=lbl)
        ax.set_xlabel("效果 $E = S_0 - S_6$ (积分下降)")
        ax.set_ylabel("相邻 ICER $\\Delta C / \\Delta E$ (元/分)")
        ax.set_title(f"样本 {r.pid}: ED 后 ICER 阶梯")
        ax.grid(alpha=0.3)
        if r.pid == recs[0].pid:
            ax.legend(fontsize=8, loc="upper left")
        ax.set_ylim(0, max(260, max(icers) * 1.05) if icers else 260)
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()


def plot_batch_summary(df_batch: pd.DataFrame, out: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    (ax1, ax2, ax3), (ax4, ax5, ax6) = axes

    sc = ax1.scatter(df_batch["S0"], df_batch["S6_nmb"], c=df_batch["C_nmb"], cmap="viridis", s=28)
    ax1.plot(
        [df_batch["S0"].min(), df_batch["S0"].max()],
        [df_batch["S0"].min(), df_batch["S0"].max()],
        "--",
        color="gray",
        alpha=0.5,
    )
    ax1.set_xlabel("$S_0$")
    ax1.set_ylabel("$S_6$ (NMB@λ=30)")
    ax1.set_title("(a) $S_0$-$S_6$ 散点（色标=总成本）")
    fig.colorbar(sc, ax=ax1, label="$C_\\mathrm{total}$")
    ax1.grid(alpha=0.3)

    ax2.hist(df_batch["C_nmb"], bins=40, color="steelblue", alpha=0.8)
    ax2.axvline(BUDGET_SOFT, ls="--", color="red", label="预算建议 2000")
    ax2.set_xlabel("总成本 (元)")
    ax2.set_ylabel("人数")
    ax2.set_title("(b) 总成本分布")
    ax2.legend()
    ax2.grid(alpha=0.3)

    ax3.hist(df_batch["drop_pct"], bins=40, color="salmon", alpha=0.8)
    ax3.set_xlabel("降幅 (%)")
    ax3.set_ylabel("人数")
    ax3.set_title("(c) 降幅分布")
    ax3.grid(alpha=0.3)

    groups = [df_batch[df_batch["K_size"] == k]["drop_pct"].values for k in [1, 2, 3]]
    ax4.boxplot(groups, tick_labels=["$|K|=1$", "$|K|=2$", "$|K|=3$"])
    ax4.set_ylabel("降幅 (%)")
    ax4.set_title("(d) 按可行集 $|K_i|$ 分组降幅")
    ax4.grid(alpha=0.3)

    age_grps = [df_batch[df_batch["age"] == a]["C_nmb"].values for a in [1, 2, 3, 4, 5]]
    ax5.boxplot(age_grps, tick_labels=["40-49", "50-59", "60-69", "70-79", "80-89"])
    ax5.set_ylabel("总成本 (元)")
    ax5.set_title("(e) 按年龄组总成本")
    ax5.grid(alpha=0.3)

    cer_by_tier = df_batch.groupby("S0_tier")["CER"].mean()
    ax6.bar(["<=58", "59-61", ">=62"], [cer_by_tier.get(t, 0) for t in ["低", "中", "高"]], color=["#7fbf7b", "#ffcc66", "#d73027"])
    ax6.set_ylabel("平均 CER (元/分)")
    ax6.set_title("(f) 按 $S_0$ 档位的 CER")
    ax6.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()


def plot_frontier_comparison(records: List[dict], out: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, rec in zip(axes, records):
        ed_free = rec["unconstrained"].frontier_ed
        ed_base = rec["baseline"].frontier_ed
        best_free = rec["unconstrained"].rec_nmb
        best_base = rec["baseline"].rec_nmb
        C_free = np.array([p.C_total for p in ed_free])
        S_free = np.array([p.S6 for p in ed_free])
        C_base = np.array([p.C_total for p in ed_base])
        S_base = np.array([p.S6 for p in ed_base])
        ax.plot(C_free, S_free, "o-", color="gray", alpha=0.5, label="无约束 ED")
        ax.plot(C_base, S_base, "s-", color="tab:blue", alpha=0.85, label="基准约束 ED")
        ax.scatter([best_free.C_total], [best_free.S6], marker="*", s=250, c="gray", edgecolor="black", label=f"无约束 NMB@30 ({best_free.C_total:.0f}, {best_free.S6:.2f})")
        ax.scatter([best_base.C_total], [best_base.S6], marker="*", s=250, c="tab:red", edgecolor="black", label=f"基准 NMB@30 ({best_base.C_total:.0f}, {best_base.S6:.2f})")
        ax.axvline(2000, ls="--", color="red", alpha=0.5)
        ax.set_xlabel("总成本 $C_{total}$ (元)")
        ax.set_ylabel("终末积分 $S_6$")
        ax.set_title(f"样本 {rec['sid']}  $S_0$={rec['S0']:.0f},  $K$={rec['K']}")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)
    plt.suptitle("无平滑约束 vs 基准平滑约束的效率前沿与 NMB@λ=30 推荐对比", fontsize=12)
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()


def plot_trajectory_comparison(records: List[dict], out: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, rec in zip(axes, records):
        free = rec["unconstrained"].rec_nmb
        base = rec["baseline"].rec_nmb
        traj_free = list(free.S_traj)
        traj_base = list(base.S_traj)
        t = np.arange(0, MONTHS + 1)
        ax.plot(t, traj_free, "o--", color="gray", alpha=0.7, label=f"无约束: f={free.f_seq}")
        ax.plot(t, traj_base, "s-", color="tab:red", lw=2, label=f"基准: f={base.f_seq}")
        ax.axhline(62, ls=":", color="red", alpha=0.5, label="$L=3$ 阈")
        ax.axhline(59, ls=":", color="orange", alpha=0.5, label="$L=2$ 阈")
        ax.set_xlabel("月份")
        ax.set_ylabel("痰湿积分 $S$")
        ax.set_title(f"样本 {rec['sid']} 轨迹对比")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)
    plt.suptitle("NMB@λ=30 推荐方案下的 6 月积分轨迹: 无约束 vs 基准约束", fontsize=12)
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()


def plot_3d_pareto(records: List[dict], out: str) -> None:
    fig = plt.figure(figsize=(15, 4.5))
    for i, rec in enumerate(records, start=1):
        pts = np.array(
            [[p.C_total, p.S6, p.rms_df] for p in rec["unconstrained"].pareto_full + rec["baseline"].pareto_full]
        )
        pts = np.unique(np.round(pts, 10), axis=0)
        ax = fig.add_subplot(1, 3, i, projection="3d")
        sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 2], cmap="viridis", s=18, alpha=0.7)
        best_base = rec["baseline"].rec_nmb
        best_free = rec["unconstrained"].rec_nmb
        ax.scatter([best_base.C_total], [best_base.S6], [best_base.rms_df], marker="*", s=200, c="red", edgecolor="black", label=f"基准 ({best_base.C_total:.0f},{best_base.S6:.2f})")
        ax.scatter([best_free.C_total], [best_free.S6], [best_free.rms_df], marker="P", s=170, c="gray", edgecolor="black", label=f"无约束 ({best_free.C_total:.0f},{best_free.S6:.2f})")
        ax.set_xlabel("成本 $C$")
        ax.set_ylabel("$S_6$")
        ax.set_zlabel("RMS(Δf)")
        ax.set_title(f"样本 {rec['sid']}   $S_0$={rec['S0']:.0f}, K={rec['K']}")
        ax.legend(fontsize=7, loc="upper left")
        plt.colorbar(sc, ax=ax, shrink=0.6, label="RMS(Δf)")
    plt.suptitle("三目标 Pareto 散点图: 降分效果 vs 成本 vs 月间频次波动", fontsize=12)
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()


def plot_batch_comparison(df: pd.DataFrame, out: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].hist(df["unconstrained_rms"], bins=20, color="gray", alpha=0.7, label="无约束")
    axes[0, 0].hist(df["baseline_rms"], bins=20, color="tab:red", alpha=0.7, label="基准")
    axes[0, 0].axvline(1.5, ls="--", color="red")
    axes[0, 0].set_xlabel("RMS(Δf)")
    axes[0, 0].set_ylabel("人数")
    axes[0, 0].set_title("频次跳变强度分布: 无约束 vs 基准")
    axes[0, 0].legend()

    axes[0, 1].scatter(df["unconstrained_C"], df["baseline_C"], alpha=0.4, s=20)
    lim = [df["unconstrained_C"].min() - 50, df["unconstrained_C"].max() + 50]
    axes[0, 1].plot(lim, lim, "r--", alpha=0.5, label="y=x")
    axes[0, 1].set_xlabel("无约束成本 (元)")
    axes[0, 1].set_ylabel("基准成本 (元)")
    axes[0, 1].set_title("每位患者总成本: 无约束 vs 基准")
    axes[0, 1].legend()

    axes[1, 0].scatter(df["unconstrained_S6"], df["baseline_S6"], alpha=0.4, s=20)
    lim = [df["unconstrained_S6"].min() - 1, df["unconstrained_S6"].max() + 1]
    axes[1, 0].plot(lim, lim, "r--", alpha=0.5, label="y=x")
    axes[1, 0].set_xlabel("无约束 $S_6$")
    axes[1, 0].set_ylabel("基准 $S_6$")
    axes[1, 0].set_title("每位患者终末积分: 无约束 vs 基准")
    axes[1, 0].legend()

    axes[1, 1].hist(df["delta_S6"], bins=25, color="steelblue", alpha=0.8)
    axes[1, 1].axvline(0, color="red", ls="--")
    axes[1, 1].set_xlabel("ΔS6 = 基准 - 无约束 (积分)")
    axes[1, 1].set_ylabel("人数")
    axes[1, 1].set_title(f"基准相对无约束的临床代价分布\n(均值 {df['delta_S6'].mean():+.3f})")
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()


# ========================================================================
# Part 9 · 敏感性分析
# ========================================================================
def sensitivity_lambda(recs: List[Recommendation], lam_grid: np.ndarray = LAMBDA_GRID) -> pd.DataFrame:
    rows = []
    for r in recs:
        for lam in lam_grid:
            _, p, nmb = nmb_argmax(r.frontier_ed, lam)
            rows.append(
                {
                    "样本": r.pid,
                    "λ": lam,
                    "S6": round(p.S6, 2),
                    "C_total": round(p.C_total, 0),
                    "E": round(p.E, 2),
                    "NMB": round(nmb, 1),
                    "k_seq": str(p.k_seq),
                    "f_seq": str(p.f_seq),
                }
            )
    return pd.DataFrame(rows)


def sensitivity_rule_coefs(recs: List[Recommendation], factors=(0.8, 1.0, 1.2), tau_max: float = math.inf) -> pd.DataFrame:
    rows = []
    for r in recs:
        for s_k in factors:
            for s_f in factors:
                rk = RULE_K * s_k
                rf = RULE_F * s_f
                plans = enumerate_pareto(r.S0, r.K_i, rule_k=rk, rule_f=rf, tau_max=tau_max)
                plans = extended_dominance(plans)
                _, p, _ = nmb_argmax(plans, LAMBDA_BASE)
                rows.append(
                    {
                        "样本": r.pid,
                        "rule_B 缩放": s_k,
                        "rule_C 缩放": s_f,
                        "S6": round(p.S6, 2),
                        "C_total": round(p.C_total, 0),
                        "k_seq": str(p.k_seq),
                        "f_seq": str(p.f_seq),
                    }
                )
    return pd.DataFrame(rows)


def sensitivity_tau(samples: List[dict]) -> Tuple[Dict[int, dict], pd.DataFrame]:
    results = {}
    rows = []
    for sample in samples:
        sid = sample["pid"]
        S0 = sample["S0"]
        age = sample["age"]
        act = sample["act"]
        K = feasible_K(age, act)
        tau_sweep = {}
        for tau in TAU_GRID:
            rec = recommend_patient(sid, S0, age, act, tau_max=tau)
            tau_sweep[tau] = {
                "n_front": len(rec.pareto_full),
                "n_ed": len(rec.frontier_ed),
                "S6": rec.rec_nmb.S6,
                "C": rec.rec_nmb.C_total,
                "kseq": rec.rec_nmb.k_seq,
                "fseq": rec.rec_nmb.f_seq,
                "tv_rms": rec.rec_nmb.rms_df,
            }
            rows.append(
                {
                    "样本": sid,
                    "tau": "∞" if math.isinf(tau) else f"{tau:.1f}",
                    "n_Pareto": len(rec.pareto_full),
                    "n_ED": len(rec.frontier_ed),
                    "S6": f"{rec.rec_nmb.S6:.2f}",
                    "总成本": int(round(rec.rec_nmb.C_total)),
                    "降幅": f"{100 * rec.rec_nmb.E / S0:.1f}%",
                    "RMS": f"{rec.rec_nmb.rms_df:.2f}",
                    "f序列": str(rec.rec_nmb.f_seq),
                }
            )
        results[sid] = {
            "S0": S0,
            "K": K,
            "age": age,
            "score": act,
            "tau_sweep": tau_sweep,
        }
    return results, pd.DataFrame(rows)


# ========================================================================
# Part 10 · 批量
# ========================================================================
def load_target_patients(data_path: str) -> pd.DataFrame:
    df = load_patients(data_path)
    if "type" in df.columns:
        return df[df["type"] == 5].reset_index(drop=True).copy()
    return df.copy()


def _write_batch_group_tables(df_out: pd.DataFrame, table_dir: Path) -> None:
    by_k = df_out.groupby("K_size").agg(
        人数=("id", "size"),
        平均S6=("S6_nmb", "mean"),
        平均总成本=("C_nmb", "mean"),
        平均降幅=("drop_pct", "mean"),
    ).round(2).reset_index()
    by_k.to_csv(table_dir / "Q3_batch_by_K.csv", index=False, encoding="utf-8-sig")

    by_s0 = (
        df_out.groupby("S0_tier")
        .agg(
            人数=("id", "size"),
            平均S6=("S6_nmb", "mean"),
            平均总成本=("C_nmb", "mean"),
            平均降幅=("drop_pct", "mean"),
            平均CER=("CER", "mean"),
        )
        .reindex(["低", "中", "高"])
        .round(2)
        .reset_index()
    )
    by_s0.to_csv(table_dir / "Q3_batch_by_S0_tier.csv", index=False, encoding="utf-8-sig")

    by_age = df_out.groupby("age").agg(
        人数=("id", "size"),
        平均S6=("S6_nmb", "mean"),
        平均总成本=("C_nmb", "mean"),
        平均降幅=("drop_pct", "mean"),
        平均CER=("CER", "mean"),
    ).round(2).reset_index()
    by_age.to_csv(table_dir / "Q3_batch_by_age.csv", index=False, encoding="utf-8-sig")

    df_cost = df_out.copy()
    df_cost["cost_bin"] = pd.cut(df_cost["C_nmb"], bins=[-np.inf, 500, 1000, np.inf], labels=["低档", "中档", "高档"])
    by_cost = df_cost.groupby("cost_bin", observed=True).agg(
        人数=("id", "size"),
        平均S0=("S0", "mean"),
        平均S6=("S6_nmb", "mean"),
        平均成本=("C_nmb", "mean"),
        平均降幅=("drop_pct", "mean"),
        平均CER=("CER", "mean"),
    ).round(2).reset_index()
    by_cost.to_csv(table_dir / "Q3_batch_by_cost.csv", index=False, encoding="utf-8-sig")


def _write_batch_overall_table(df_out: pd.DataFrame, table_dir: Path, tau_max: float) -> None:
    rows = [
        {"metric": "n_patients", "value": int(len(df_out))},
        {"metric": "tau_max", "value": "inf" if math.isinf(tau_max) else round(float(tau_max), 4)},
        {"metric": "mean_S6", "value": round(float(df_out["S6_nmb"].mean()), 6)},
        {"metric": "mean_total_cost", "value": round(float(df_out["C_nmb"].mean()), 6)},
        {"metric": "mean_drop_pct", "value": round(float(df_out["drop_pct"].mean()), 6)},
        {"metric": "mean_rms_df", "value": round(float(df_out["rms_df"].mean()), 6)},
        {"metric": "share_rms_le_tau", "value": round(float((df_out["rms_df"] <= tau_max + 1e-9).mean()), 6)},
        {"metric": "share_cost_le_budget_soft", "value": round(float((df_out["C_nmb"] <= BUDGET_SOFT + 1e-9).mean()), 6)},
    ]
    pd.DataFrame(rows).to_csv(table_dir / "Q3_batch_overall.csv", index=False, encoding="utf-8-sig")


def run_batch(data_path: str, fig_dir: Path, table_dir: Path, tau_max: float = TAU_BASE) -> pd.DataFrame:
    df_p = load_target_patients(data_path)
    print(f"共 {len(df_p)} 名痰湿体质患者进入批量优化")
    rows = []
    for idx, row in df_p.iterrows():
        rec = recommend_patient(
            int(row.get("id", idx + 1)),
            float(row["S0"]),
            int(row["age"]),
            float(row.get("act_score", row.get("adl", 0) + row.get("iadl", 0))),
            tau_max=tau_max,
        )
        rows.append(
            {
                "id": rec.pid,
                "age": rec.age_group,
                "S0": rec.S0,
                "K_size": len(rec.K_i),
                "S0_tier": ("低" if rec.S0 <= 58 else "中" if rec.S0 < 62 else "高"),
                "S6_nmb": rec.rec_nmb.S6,
                "C_nmb": rec.rec_nmb.C_total,
                "E_nmb": rec.rec_nmb.E,
                "drop_pct": rec.rec_nmb.E / rec.S0 * 100 if rec.S0 > 0 else 0,
                "rms_df": rec.rec_nmb.rms_df,
                "CER": (rec.rec_nmb.C_total / rec.rec_nmb.E) if rec.rec_nmb.E > 1e-6 else np.inf,
                "k_seq": str(rec.rec_nmb.k_seq),
                "f_seq": str(rec.rec_nmb.f_seq),
                "agree_nmb_topsis": int(rec.rec_nmb.k_seq == rec.rec_topsis.k_seq and rec.rec_nmb.f_seq == rec.rec_topsis.f_seq),
                "agree_nmb_knee": int(rec.rec_nmb.k_seq == rec.rec_knee.k_seq and rec.rec_nmb.f_seq == rec.rec_knee.f_seq),
            }
        )
    df_out = pd.DataFrame(rows)
    df_out.to_csv(table_dir / "Q3_batch_results.csv", index=False, encoding="utf-8-sig")
    _write_batch_group_tables(df_out, table_dir)
    _write_batch_overall_table(df_out, table_dir, tau_max)
    print("\n按 |K_i| 分组：")
    print(df_out.groupby("K_size")[["S6_nmb", "C_nmb", "drop_pct"]].mean().round(2))
    print("\n按 S0 档位分组：")
    print(df_out.groupby("S0_tier")[["S6_nmb", "C_nmb", "drop_pct", "CER"]].mean().round(2))
    print("\n按年龄组分组：")
    print(df_out.groupby("age")[["S6_nmb", "C_nmb", "drop_pct"]].mean().round(2))
    print(f"\nNMB vs TOPSIS 一致率: {df_out['agree_nmb_topsis'].mean() * 100:.1f}%")
    print(f"NMB vs Kneedle 一致率: {df_out['agree_nmb_knee'].mean() * 100:.1f}%")
    plot_batch_summary(df_out, str(fig_dir / "Q3_summary.png"))
    return df_out


def run_comparison_bundle(data_path: str, fig_dir: Path, table_dir: Path) -> None:
    df_p = load_target_patients(data_path)

    samples = []
    for sid in [1, 2, 3]:
        row = df_p[df_p["id"] == sid].iloc[0]
        samples.append(
            {
                "pid": sid,
                "S0": float(row["S0"]),
                "age": int(row["age"]),
                "act": float(row.get("act_score", row.get("adl", 0) + row.get("iadl", 0))),
            }
        )

    tau_results, tau_df = sensitivity_tau(samples)
    with open(table_dir / "Q3_sample123_tau_sweep.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                str(k): {
                    kk: vv if kk != "tau_sweep" else {
                        ("inf" if math.isinf(tau) else str(tau)): {
                            inner_k: (list(inner_v) if isinstance(inner_v, tuple) else inner_v)
                            for inner_k, inner_v in vals.items()
                        }
                        for tau, vals in v["tau_sweep"].items()
                    }
                    for kk, vv in v.items()
                }
                for k, v in tau_results.items()
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    tau_df.to_csv(table_dir / "Q3_tau_sensitivity.csv", index=False, encoding="utf-8-sig")

    records = []
    sample_rows = []
    for sample in samples:
        free = recommend_patient(sample["pid"], sample["S0"], sample["age"], sample["act"], tau_max=math.inf)
        base = recommend_patient(sample["pid"], sample["S0"], sample["age"], sample["act"], tau_max=TAU_BASE)
        records.append({"sid": sample["pid"], "S0": sample["S0"], "K": feasible_K(sample["age"], sample["act"]), "unconstrained": free, "baseline": base})
        tabulate_prescription(base.rec_nmb).to_csv(table_dir / f"sample{sample['pid']}_prescription.csv", index=False, encoding="utf-8-sig")
        note = {1: "平滑约束非紧", 2: "后两月平滑回落", 3: "逐步升频后保持"}[sample["pid"]]
        sample_rows.append(
            {
                "sample": sample["pid"],
                "S0": sample["S0"],
                "K": str(feasible_K(sample["age"], sample["act"])),
                "k_seq": str(base.rec_nmb.k_seq),
                "f_seq": str(base.rec_nmb.f_seq),
                "S6": round(base.rec_nmb.S6, 2),
                "C_total": round(base.rec_nmb.C_total, 2),
                "rms_df": round(base.rec_nmb.rms_df, 2),
                "note": note,
            }
        )
    pd.DataFrame(sample_rows).to_csv(table_dir / "Q3_sample_recommendations.csv", index=False, encoding="utf-8-sig")

    plot_frontier_comparison(records, str(fig_dir / "Q3_frontier_comparison.png"))
    plot_trajectory_comparison(records, str(fig_dir / "Q3_trajectory_comparison.png"))
    plot_3d_pareto(records, str(fig_dir / "Q3_3d_pareto.png"))

    rows = []
    for _, row in df_p.iterrows():
        sid = int(row["id"])
        S0 = float(row["S0"])
        age = int(row["age"])
        act = float(row.get("act_score", row.get("adl", 0) + row.get("iadl", 0)))
        free = recommend_patient(sid, S0, age, act, tau_max=math.inf)
        base = recommend_patient(sid, S0, age, act, tau_max=TAU_BASE)
        rows.append(
            {
                "id": sid,
                "S0": S0,
                "K": str(feasible_K(age, act)),
                "unconstrained_S6": free.rec_nmb.S6,
                "unconstrained_C": free.rec_nmb.C_total,
                "unconstrained_rms": free.rec_nmb.rms_df,
                "baseline_S6": base.rec_nmb.S6,
                "baseline_C": base.rec_nmb.C_total,
                "baseline_rms": base.rec_nmb.rms_df,
                "delta_S6": base.rec_nmb.S6 - free.rec_nmb.S6,
                "delta_C": base.rec_nmb.C_total - free.rec_nmb.C_total,
                "unconstrained_fseq": str(free.rec_nmb.f_seq),
                "baseline_fseq": str(base.rec_nmb.f_seq),
            }
        )
    batch_cmp = pd.DataFrame(rows)
    batch_cmp.to_csv(table_dir / "Q3_compare_unconstrained_vs_baseline.csv", index=False, encoding="utf-8-sig")
    plot_batch_comparison(batch_cmp, str(fig_dir / "Q3_batch_comparison.png"))


# ========================================================================
# Part 11 · 主程序
# ========================================================================
DEMO_SAMPLES = [
    dict(pid=1, gender="女", age=2, adl=20, iadl=18, S0=64, act=38),
    dict(pid=2, gender="男", age=1, adl=24, iadl=16, S0=58, act=40),
    dict(pid=3, gender="女", age=1, adl=27, iadl=36, S0=59, act=63),
]


def run_demo(fig_dir: Path, table_dir: Path, tau_max: float = TAU_BASE) -> List[Recommendation]:
    recs: List[Recommendation] = []
    for sample in DEMO_SAMPLES:
        print(f"\n=== 样本 {sample['pid']}  S0={sample['S0']}, K_i={feasible_K(sample['age'], sample['act'])} ===")
        rec = recommend_patient(sample["pid"], sample["S0"], sample["age"], sample["act"], tau_max=tau_max)
        print(f"原始 Pareto {len(rec.pareto_full)} 点；ED 后 {len(rec.frontier_ed)} 点")
        print(f"ICER 序列: {[round(x, 2) for x in rec.icer_series]}")
        for lam in LAMBDA_LIST:
            p = rec.rec_by_lam[lam]
            drop = p.E / rec.S0 * 100
            print(f"[λ={lam:3d}]  S6={p.S6:6.2f}, C={p.C_total:5.0f}, 降={drop:5.1f}%, k={p.k_seq}, f={p.f_seq}")
        print(f"TOPSIS  → S6={rec.rec_topsis.S6:.2f}, C={rec.rec_topsis.C_total:.0f}")
        print(f"Kneedle → S6={rec.rec_knee.S6:.2f}, C={rec.rec_knee.C_total:.0f}")
        tab = tabulate_prescription(rec.rec_nmb)
        tab.to_csv(table_dir / f"sample{sample['pid']}_prescription.csv", index=False, encoding="utf-8-sig")
        print("NMB 推荐处方:")
        print(tab.to_string(index=False))
        recs.append(rec)

    plot_pareto_three(recs, str(fig_dir / "Q3_pareto.png"))
    plot_trajectory_three(recs, str(fig_dir / "Q3_trajectory.png"))
    plot_lambda_sweep(recs, str(fig_dir / "Q3_lambda_sweep.png"))
    plot_icer_ladder(recs, str(fig_dir / "Q3_icer_ladder.png"))

    df_lam = sensitivity_lambda(recs, np.arange(0, 305, 5))
    df_lam.to_csv(table_dir / "Q3_lambda_sweep.csv", index=False, encoding="utf-8-sig")
    df_rule = sensitivity_rule_coefs(recs, tau_max=tau_max)
    df_rule.to_csv(table_dir / "Q3_rule_sens.csv", index=False, encoding="utf-8-sig")
    return recs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Problem 3 solver with baseline smoothness constraint and comparison outputs")
    parser.add_argument("--data", default=None, help="附件 1 路径（xlsx/csv），省略则自动使用 src/Data 中的附件 1")
    parser.add_argument("--out", default=None, help="输出根目录；省略则写入 src/outputs/figures 与 src/outputs/tables")
    parser.add_argument("--mode", choices=["demo", "batch", "all"], default="all")
    args = parser.parse_args()

    fig_dir, table_dir = resolve_output_dirs(args.out)
    data_path = args.data
    if data_path is None and args.mode in ("batch", "all"):
        data_path = str(find_sample_data_path())

    if args.mode in ("demo", "all"):
        print("=" * 70)
        print(f"  Part A · 三位样本基准情景（τ_max={TAU_BASE}）")
        print("=" * 70)
        recs_demo = run_demo(fig_dir, table_dir, tau_max=TAU_BASE)

    if args.mode in ("batch", "all") and data_path:
        print("\n" + "=" * 70)
        print(f"  Part B · 附件 1 · 278 人基准批量（τ_max={TAU_BASE}）")
        print("=" * 70)
        df_batch = run_batch(data_path, fig_dir, table_dir, tau_max=TAU_BASE)
        run_comparison_bundle(data_path, fig_dir, table_dir)

    print(f"\n完成。图写入 {fig_dir}；表写入 {table_dir}")
