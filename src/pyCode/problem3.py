#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
problem3.py — 2026 MathorCup C 题 · 问题三
6 个月痰湿体质个性化干预方案 — 统一 NMB 架构（严格按题意修订）

========================================================================
动力学严格对应题目附表 3 文字（无任何外部假设）
========================================================================
  运动项月降分率：
      f < 5     →  r_t = 0               （积分基本稳定）
      f ≥ 5     →  r_t = 0.03·k + 0.01·(f − 5)
                     其中 0.03·k 来自规则 B："每提升一级强度每月降 3%"
                         0.01·(f−5) 来自规则 C："每加 1 次每月多降 1%"

  中医调理档位（纯按月初积分判定、仅产生固定月成本、不贡献任何降分）：
      L_t = g(S_{t-1}) ∈ {1,2,3}  由 S_{t-1} 对应附表 2 区间决定
      月成本 c_tcm(L) ∈ {30, 80, 130} 元/月
      —— 题目未赋予中医额外降分率，模型严格忠于题意，不做外部推测

  状态方程：  S_{t+1} = S_t · (1 − r_t)
  月总成本：  cost_t = 4·f_t·c_act(k_t) + c_tcm(L_t)
             其中 c_act ∈ {3, 5, 8} 元/次  (附表 3)

========================================================================
核心方法论
========================================================================
  [1] 月度 Pareto 前沿推进 ε-约束法             生成完整非被支配前沿
  [2] Extended Dominance 预筛选 (Cantor 1994)    剔除被扩展支配的非效率点
  [3] NMB argmax 统一选点  (Stinnett-Mullahy 1998 Medical Decision Making)
      NMB_j(λ) = λ · (S_0 − S_6^j) − C_total^j
      j* = argmax_j NMB_j(λ)       ← 单一公式替代原三条 if-else 规则
  [4] λ 四档扫描  (Neumann, Cohen & Weinstein 2014 NEJM)
      λ ∈ {30, 50, 100, 200} 元/分   基准 30
  [5] TOPSIS-熵权 + Kneedle 交叉对照            独立几何方法做稳健性验证
  [6] 规则系数 ±20% 敏感性分析                  取代伪中医降分率敏感性

作者: MC2606970
"""

from __future__ import annotations
import os, json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC",
                                    "WenQuanYi Zen Hei", "SimHei", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

# ========================================================================
# Part 0 · 参数常量（全部来自题目附表 2、3，无任何外部推测）
# ========================================================================
ACT_COST   = {1: 3, 2: 5, 3: 8}              # 元/次（附表 3）
ACT_MIN    = {1: 10, 2: 20, 3: 30}           # 分钟/次（附表 3）
TCM_COST   = {1: 30, 2: 80, 3: 130}          # 元/月（附表 2）
# 注意：TCM_RATE 已移除，题目附表 2 未赋予中医任何降分率
RULE_K     = 0.03                            # 规则 B：每升一级月降 3%
RULE_F     = 0.01                            # 规则 C：每多 1 次月多降 1%
F_ANCHOR   = 5                               # 规则 A：f<5 时 r=0 的临界点
MONTHS     = 6
WEEKS_PER_MONTH = 4
BUDGET_SOFT     = 2000                       # 建议预算上限（元）
BUDGET_HARD     = 2400                       # Pareto 构造的硬截断（略宽）

# WTP 四档 + 基准（Neumann 2014 NEJM 三档法 + 中国社区卫生保守基准）
LAMBDA_LIST = [30, 50, 100, 200]
LAMBDA_BASE = 30
LAMBDA_GRID = np.linspace(0, 260, 53)        # 扫描网格

# 有效频次：题目规则 A 明示 f<5 时积分基本稳定（即运动干预无效）
# 既然 f<5 无效却仍需支付活动费 4·f·c_act，属于劣策略，
# 故决策域严格限制为 f ∈ {5, 6, 7, 8, 9, 10} —— 只保留有效频次
F_EFFECTIVE_SET = [5, 6, 7, 8, 9, 10]

# ========================================================================
# Part 1 · 动力学（严格按题目附表 3 文字）
# ========================================================================
def tcm_tier(S: float) -> int:
    """中医档位 L_t = g(S_{t-1})，按月初积分判定（附表 2）。"""
    if S >= 62:  return 3
    if S >= 59:  return 2
    return 1

def r_exercise(k: int, f: int,
               rule_k: float = RULE_K,
               rule_f: float = RULE_F) -> float:
    """
    运动项月降分率（加法模型，完全对应题目附表 3 文字）：
        f ≥ 5: r = 0.03·k + 0.01·(f − 5)     # 规则 B + 规则 C 相加
    决策域已限制 f ∈ {5,...,10}，所以 f<5 的情形不会被调用。
    """
    if f < F_ANCHOR:
        return 0.0                                   # 规则 A: 基本稳定
    return rule_k * k + rule_f * (f - F_ANCHOR)

def step_month(k: int, f: int, S_prev: float,
               **dyn_kw) -> Tuple[float, int, float, float, float]:
    """
    单月状态更新：
      S_new   = S_prev × (1 − r_exercise)
      L       = tcm_tier(S_prev)
      act_cost = 4·f·c_act(k)
      tcm_cost = c_tcm(L)       ← 中医为纯固定成本，不贡献降分
    """
    L = tcm_tier(S_prev)
    r = r_exercise(k, f, **dyn_kw)
    S_new = S_prev * (1 - r)
    act_cost = WEEKS_PER_MONTH * f * ACT_COST[k]
    tcm_cost = TCM_COST[L]
    return S_new, L, r, act_cost, tcm_cost


# ========================================================================
# Part 2 · 月度 Pareto 前沿推进（ε-约束法 + 跨档位安全分组去劣）
# ========================================================================
@dataclass
class Plan:
    """一个完整 6 月方案的状态-决策-成本完整元组"""
    k_seq:  Tuple[int, ...]
    f_seq:  Tuple[int, ...]
    S_traj: Tuple[float, ...]
    L_seq:  Tuple[int, ...]
    r_seq:  Tuple[float, ...]
    cost_month: Tuple[float, ...]
    C_total:  float
    S6:       float
    S0:       float

    @property
    def E(self) -> float:
        return self.S0 - self.S6


def enumerate_pareto(S0: float, K_i: List[int],
                     f_values: List[int] = F_EFFECTIVE_SET,
                     budget: float = BUDGET_HARD,
                     **dyn_kw) -> List[Plan]:
    """
    逐月 Pareto 前沿推进。只枚举 f ∈ F_EFFECTIVE_SET
    （因 f∈{2,3,4} 被 f=1 严格支配，同 r=0 但成本递增）。
    """
    frontier: List[tuple] = [(
        S0, 0.0, (), (), (S0,), (), (), ()
    )]
    for t in range(MONTHS):
        new_states = []
        for (S_t, C_t, ks, fs, Straj, Ls, rs, cms) in frontier:
            for k in K_i:
                for f in f_values:
                    S_new, L, r, c_act, c_tcm = step_month(k, f, S_t, **dyn_kw)
                    dC    = c_act + c_tcm
                    C_new = C_t + dC
                    if C_new > budget:
                        continue
                    new_states.append((
                        S_new, C_new,
                        ks + (k,), fs + (f,),
                        Straj + (S_new,),
                        Ls + (L,), rs + (r,),
                        cms + (dC,)
                    ))
        # 中间月按档位分组做 Pareto 去劣（保证跨档位候选不丢失）
        if t < MONTHS - 1:
            frontier = _pareto_prune(new_states)
        else:
            # 最终月末：做一次"全局跨档位严格 Pareto 去劣"
            # 因为终点后不再有后续动力学，跨档位严格支配判据直接有效
            frontier = _pareto_prune_global(new_states)

    plans = [Plan(
        k_seq=ks, f_seq=fs, S_traj=Straj, L_seq=Ls, r_seq=rs,
        cost_month=cms, C_total=C_t, S6=S_t, S0=S0
    ) for (S_t, C_t, ks, fs, Straj, Ls, rs, cms) in frontier]
    plans.sort(key=lambda p: (p.E, p.C_total))
    return plans


def _pareto_prune(states: List[tuple]) -> List[tuple]:
    """
    档位安全的月度 Pareto 去劣：因中医成本 c_tcm(L) 依赖月初档位 L = g(S_{t-1})，
    跨档位的状态在下月成本结构不同，故先按"下月档位"L_next = g(S_t) 分组，
    组内做 (S_t, C_t) 二维 Pareto 去劣，跨组互不支配。
    """
    if not states:
        return []
    from collections import defaultdict
    groups = defaultdict(list)
    for s in states:
        L_next = tcm_tier(s[0])
        groups[L_next].append(s)
    kept = []
    for L, items in groups.items():
        items = sorted(items, key=lambda s: (s[1], s[0]))
        min_S = float("inf")
        for s in items:
            if s[0] < min_S - 1e-9:
                kept.append(s)
                min_S = s[0]
    return kept


def _pareto_prune_global(states: List[tuple]) -> List[tuple]:
    """
    全局严格 Pareto 去劣（不分档位）：用于最终月末，
    此时后续无动力学，跨档位严格支配判据直接有效。
    """
    if not states:
        return []
    states = sorted(states, key=lambda s: (s[1], s[0]))  # 按 C 升序
    kept = []
    min_S = float("inf")
    for s in states:
        if s[0] < min_S - 1e-9:
            kept.append(s)
            min_S = s[0]
    return kept


# ========================================================================
# Part 3 · Extended Dominance 预筛选 (Cantor 1994)
# ========================================================================
def extended_dominance(plans: List[Plan]) -> List[Plan]:
    """剔除被 (A,C) 凸组合支配的凹陷点，保证 ICER 序列严格单调递增。"""
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
            dE1 = P[j].E - P[j-1].E
            dE2 = P[j+1].E - P[j].E
            if dE1 <= 1e-9 or dE2 <= 1e-9:
                continue
            icer_prev = (P[j].C_total   - P[j-1].C_total) / dE1
            icer_next = (P[j+1].C_total - P[j].C_total)   / dE2
            if icer_prev > icer_next + 1e-9:
                P.pop(j)
                changed = True
                break
    return P


# ========================================================================
# Part 4 · NMB argmax 统一选点
# ========================================================================
def nmb_argmax(plans: List[Plan], lam: float) -> Tuple[int, Plan, float]:
    """NMB_j(λ) = λ·(S_0 − S_6^j) − C_total^j"""
    nmbs = [lam * p.E - p.C_total for p in plans]
    j    = int(np.argmax(nmbs))
    return j, plans[j], nmbs[j]

def icer_frontier(plans: List[Plan]) -> List[float]:
    P = sorted(plans, key=lambda p: p.E)
    icers = []
    for i in range(1, len(P)):
        dE = P[i].E - P[i-1].E
        dC = P[i].C_total - P[i-1].C_total
        icers.append(dC / dE if dE > 1e-9 else np.inf)
    return icers


# ========================================================================
# Part 5 · 对照组 — TOPSIS 熵权 + Kneedle
# ========================================================================
def topsis_entropy(plans: List[Plan]) -> Tuple[int, Plan]:
    X = np.array([[p.S6, p.C_total] for p in plans], dtype=float)
    norm = np.sqrt((X ** 2).sum(axis=0)); norm[norm == 0] = 1
    R = X / norm
    P = R / R.sum(axis=0); P = np.clip(P, 1e-12, 1.0)
    E = -1 / np.log(len(R)) * (P * np.log(P)).sum(axis=0)
    w = (1 - E) / (1 - E).sum()
    V = R * w
    A_plus  = V.min(axis=0); A_minus = V.max(axis=0)
    d_plus  = np.sqrt(((V - A_plus ) ** 2).sum(axis=1))
    d_minus = np.sqrt(((V - A_minus) ** 2).sum(axis=1))
    C_rel   = d_minus / (d_plus + d_minus + 1e-12)
    j = int(np.argmax(C_rel))
    return j, plans[j]

def kneedle(plans: List[Plan]) -> Tuple[int, Plan]:
    if len(plans) <= 2:
        return 0, plans[0]
    P = sorted(plans, key=lambda p: p.C_total)
    x = np.array([p.C_total for p in P])
    y = np.array([p.S6      for p in P])
    x_n = (x - x.min()) / (np.ptp(x) + 1e-12)
    y_n = (y - y.min()) / (np.ptp(y) + 1e-12)
    x0, y0 = x_n[0],  y_n[0]
    x1, y1 = x_n[-1], y_n[-1]
    d = np.abs((x1 - x0) * (y0 - y_n) - (x0 - x_n) * (y1 - y0))
    j = int(np.argmax(d))
    return j, P[j]


# ========================================================================
# Part 6 · 数据读取与可行集
# ========================================================================
FEATURE_RENAME = {
    "样本ID": "id", "样本 ID": "id", "ID": "id",
    "年龄组": "age", "年龄": "age",
    "性别": "gender",
    "体质标签": "type", "中医体质": "type",
    "痰湿积分": "S0", "痰湿质积分": "S0",
    "ADL总分": "adl", "ADL 总分": "adl",
    "IADL总分": "iadl", "IADL 总分": "iadl",
    "活动量表总分": "act_score", "活动量表": "act_score",
}

def load_patients(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if   ext in (".xlsx", ".xls"): df = pd.read_excel(path)
    elif ext == ".csv":            df = pd.read_csv(path)
    else: raise ValueError(f"不支持的文件格式：{ext}")
    df = df.rename(columns={c: FEATURE_RENAME.get(c.strip(), c) for c in df.columns})
    if "S0" not in df.columns:
        for c in df.columns:
            if "痰湿" in str(c) and "积分" in str(c):
                df = df.rename(columns={c: "S0"}); break
    if "act_score" not in df.columns:
        if "adl" in df.columns and "iadl" in df.columns:
            df["act_score"] = df["adl"] + df["iadl"]
    return df

def feasible_K(age_group: int, act_score: float) -> List[int]:
    K_age   = {1: [1,2,3], 2: [1,2,3], 3: [1,2], 4: [1,2], 5: [1]}
    if   act_score < 40:  K_score = [1]
    elif act_score < 60:  K_score = [1, 2]
    else:                 K_score = [1, 2, 3]
    return sorted(set(K_age[int(age_group)]) & set(K_score))


# ========================================================================
# Part 7 · 单患者推荐
# ========================================================================
@dataclass
class Recommendation:
    pid:         int
    S0:          float
    K_i:         List[int]
    age_group:   int
    pareto_full: List[Plan]
    frontier_ed: List[Plan]
    rec_nmb:     Plan
    nmb_values:  Dict[float, float]
    rec_by_lam:  Dict[float, Plan]
    rec_topsis:  Plan
    rec_knee:    Plan
    icer_series: List[float]


def recommend_patient(pid: int, S0: float, age: int, act: float,
                      lam_list: List[float] = LAMBDA_LIST,
                      lam_base: float = LAMBDA_BASE) -> Recommendation:
    K_i = feasible_K(age, act)
    if not K_i: K_i = [1]
    plans_all = enumerate_pareto(S0, K_i)
    plans_ed  = extended_dominance(plans_all)
    rec_by_lam = {lam: nmb_argmax(plans_ed, lam)[1] for lam in lam_list}
    rec_nmb    = rec_by_lam[lam_base]
    nmb_values = {lam: nmb_argmax(plans_ed, lam)[2] for lam in lam_list}
    _, rec_topsis = topsis_entropy(plans_ed)
    _, rec_knee   = kneedle(plans_ed)
    icers = icer_frontier(plans_ed)
    return Recommendation(
        pid=pid, S0=S0, K_i=K_i, age_group=age,
        pareto_full=plans_all, frontier_ed=plans_ed,
        rec_nmb=rec_nmb, nmb_values=nmb_values,
        rec_by_lam=rec_by_lam,
        rec_topsis=rec_topsis, rec_knee=rec_knee,
        icer_series=icers,
    )


# ========================================================================
# Part 8 · 可视化
# ========================================================================
def plot_pareto_three(recs: List[Recommendation], out: str,
                      lam: float = LAMBDA_BASE):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, r in zip(axes, recs):
        C  = [p.C_total for p in r.pareto_full]
        S  = [p.S6      for p in r.pareto_full]
        Ce = [p.C_total for p in r.frontier_ed]
        Se = [p.S6      for p in r.frontier_ed]
        ax.scatter(C, S, s=18, color="#a0a0a0", alpha=0.4, label="原始 Pareto")
        ax.plot(Ce, Se, "-o", color="steelblue", ms=5, label="ED 后效率前沿")
        ax.scatter(r.rec_nmb.C_total, r.rec_nmb.S6,
                   marker="*", s=260, c="red", zorder=10,
                   edgecolor="k", linewidths=0.8,
                   label=f"NMB 推荐 (λ={int(lam)})")
        ax.scatter(r.rec_knee.C_total, r.rec_knee.S6,
                   marker="^", s=95, facecolors="none", edgecolors="darkorange",
                   linewidths=1.6, label="Kneedle 对照")
        ax.scatter(r.rec_topsis.C_total, r.rec_topsis.S6,
                   marker="s", s=75, facecolors="none", edgecolors="green",
                   linewidths=1.6, label="TOPSIS 对照")
        ax.axvline(BUDGET_SOFT, ls="--", color="gray", alpha=0.6)
        ax.set_xlabel("总成本 $C_{\\mathrm{total}}$ (元)")
        ax.set_ylabel("6 月末积分 $S_6$")
        ax.set_title(f"样本 {r.pid}  $S_0={r.S0:.0f}$, $K_i=\\{{ {','.join(map(str,r.K_i))} \\}}$")
        ax.grid(alpha=0.3)
        if r.pid == recs[0].pid:
            ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches="tight"); plt.close()


def plot_trajectory_three(recs: List[Recommendation], out: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, r in zip(axes, recs):
        traj = list(r.rec_nmb.S_traj)
        ax.plot(range(len(traj)), traj, "-o", color="crimson", lw=2, ms=7)
        for t, (k, f, L) in enumerate(
                zip(r.rec_nmb.k_seq, r.rec_nmb.f_seq, r.rec_nmb.L_seq)):
            ax.annotate(f"$L{L}$\n$({k},{f})$",
                        xy=(t+1, traj[t+1]), xytext=(6, 6),
                        textcoords="offset points", fontsize=8)
        ax.axhline(62, ls="--", color="crimson", alpha=0.5, label="$L=3$ 阈值")
        ax.axhline(59, ls="--", color="darkorange", alpha=0.5, label="$L=2$ 阈值")
        ax.set_xlabel("月份"); ax.set_ylabel("痰湿积分 $S_t$")
        drop = r.rec_nmb.E / r.S0 * 100 if r.S0 > 0 else 0
        ax.set_title(f"样本 {r.pid} — NMB 推荐 (总成本 {r.rec_nmb.C_total:.0f} 元, "
                     f"$S_6={r.rec_nmb.S6:.2f}$, 降 {drop:.1f}%)")
        ax.grid(alpha=0.3)
        if r.pid == recs[0].pid:
            ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches="tight"); plt.close()


def plot_lambda_sweep(recs: List[Recommendation], out: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    lam_grid = np.arange(0, 260, 2)
    for ax, r in zip(axes, recs):
        Cs, S6s = [], []
        for lam in lam_grid:
            _, p, _ = nmb_argmax(r.frontier_ed, lam)
            Cs.append(p.C_total); S6s.append(p.S6)
        Ce = [p.C_total for p in r.frontier_ed]
        Se = [p.S6      for p in r.frontier_ed]
        ax.plot(Ce, Se, "-o", color="lightblue", ms=4, lw=1, alpha=0.6,
                label="ED 效率前沿", zorder=2)
        sc = ax.scatter(Cs, S6s, c=lam_grid, cmap="plasma", s=35, zorder=5)
        for lam, mk, col in [(30, "*", "red"), (50, "D", "orange"),
                             (100, "s", "green"), (200, "^", "purple")]:
            _, p, _ = nmb_argmax(r.frontier_ed, lam)
            ax.scatter(p.C_total, p.S6, marker=mk, s=160, c=col,
                       edgecolor="k", linewidths=0.8, zorder=10,
                       label=f"λ={lam}")
        ax.set_xlabel("总成本 $C_{\\mathrm{total}}$ (元)")
        ax.set_ylabel("6 月末积分 $S_6$")
        ax.set_title(f"样本 {r.pid}  $S_0={r.S0:.0f}$, $|K|={len(r.K_i)}$")
        ax.grid(alpha=0.3)
        if r.pid == recs[0].pid:
            ax.legend(fontsize=7, loc="upper right")
        fig.colorbar(sc, ax=ax, label="λ (元/分)", shrink=0.8)
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches="tight"); plt.close()


def plot_icer_ladder(recs: List[Recommendation], out: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, r in zip(axes, recs):
        if len(r.frontier_ed) < 2:
            ax.text(0.5, 0.5, "前沿点数不足",
                    transform=ax.transAxes, ha="center")
            ax.set_title(f"样本 {r.pid}"); continue
        P = sorted(r.frontier_ed, key=lambda p: p.E)
        Es = [p.E for p in P]
        icers = [0] + r.icer_series
        for j in range(len(icers) - 1):
            ax.plot([Es[j], Es[j+1]], [icers[j+1], icers[j+1]],
                    "-", color="steelblue", lw=2)
            if j < len(icers) - 2:
                ax.plot([Es[j+1], Es[j+1]], [icers[j+1], icers[j+2]],
                        ":", color="steelblue", lw=1, alpha=0.7)
        for lam, col, lbl in [(30, "red", "λ=30"), (50, "orange", "λ=50"),
                              (100, "green", "λ=100"), (200, "purple", "λ=200")]:
            ax.axhline(lam, ls="--", color=col, alpha=0.6, lw=1, label=lbl)
        ax.set_xlabel("效果 $E = S_0 - S_6$ (积分下降)")
        ax.set_ylabel("相邻 ICER $\\Delta C / \\Delta E$ (元/分)")
        ax.set_title(f"样本 {r.pid}: ED 后 ICER 阶梯")
        ax.grid(alpha=0.3)
        if r.pid == recs[0].pid:
            ax.legend(fontsize=8, loc="upper left")
        ax.set_ylim(0, max(260, max(icers) * 1.05) if icers else 260)
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches="tight"); plt.close()


# ========================================================================
# Part 9 · 处方表
# ========================================================================
def tabulate_prescription(rec: Recommendation,
                          which: str = "nmb") -> pd.DataFrame:
    p = {"nmb": rec.rec_nmb, "topsis": rec.rec_topsis,
         "knee": rec.rec_knee}[which]
    rows = []
    for t in range(MONTHS):
        k, f, L = p.k_seq[t], p.f_seq[t], p.L_seq[t]
        r = p.r_seq[t]
        S_start = p.S_traj[t]
        S_end   = p.S_traj[t+1]
        c_act = WEEKS_PER_MONTH * f * ACT_COST[k]
        c_tcm = TCM_COST[L]
        rows.append({
            "月": t+1, "k": k, "f": f, "L": L,
            "月初S": round(S_start, 2),
            "r(%)":  round(r * 100, 2),
            "月末S": round(S_end, 2),
            "活动费": c_act, "中医费": c_tcm,
            "小计":   c_act + c_tcm
        })
    df = pd.DataFrame(rows)
    total_row = pd.DataFrame([{
        "月": "合计", "k": "", "f": "", "L": "",
        "月初S": "", "r(%)": "",
        "月末S": round(p.S6, 2),
        "活动费": df["活动费"].sum(),
        "中医费": df["中医费"].sum(),
        "小计":   df["小计"].sum(),
    }])
    return pd.concat([df, total_row], ignore_index=True)


# ========================================================================
# Part 10 · 敏感性分析
# ========================================================================
def sensitivity_lambda(recs: List[Recommendation],
                       lam_grid: np.ndarray = LAMBDA_GRID) -> pd.DataFrame:
    rows = []
    for r in recs:
        for lam in lam_grid:
            _, p, nmb = nmb_argmax(r.frontier_ed, lam)
            rows.append({
                "样本": r.pid, "λ": lam,
                "S6": round(p.S6, 2),
                "C_total": round(p.C_total, 0),
                "E": round(p.E, 2),
                "NMB": round(nmb, 1),
                "k_seq": str(p.k_seq), "f_seq": str(p.f_seq),
            })
    return pd.DataFrame(rows)

def sensitivity_rule_coefs(recs: List[Recommendation],
                           factors=(0.8, 1.0, 1.2)) -> pd.DataFrame:
    """规则系数 3% / 1% 的 ±20% 扰动敏感性分析"""
    rows = []
    for r in recs:
        for s_k in factors:
            for s_f in factors:
                rk = RULE_K * s_k
                rf = RULE_F * s_f
                plans = enumerate_pareto(r.S0, r.K_i, rule_k=rk, rule_f=rf)
                plans = extended_dominance(plans)
                _, p, _ = nmb_argmax(plans, LAMBDA_BASE)
                rows.append({
                    "样本": r.pid,
                    "rule_B 缩放": s_k, "rule_C 缩放": s_f,
                    "S6": round(p.S6, 2),
                    "C_total": round(p.C_total, 0),
                    "k_seq": str(p.k_seq), "f_seq": str(p.f_seq),
                })
    return pd.DataFrame(rows)


# ========================================================================
# Part 11 · 主程序
# ========================================================================
DEMO_SAMPLES = [
    dict(pid=1, gender="女", age=2, adl=20, iadl=18, S0=64, act=38),
    dict(pid=2, gender="男", age=1, adl=24, iadl=16, S0=58, act=40),
    dict(pid=3, gender="女", age=1, adl=27, iadl=36, S0=59, act=63),
]


def run_demo(out_dir: str = "outputs"):
    os.makedirs(out_dir, exist_ok=True)
    recs: List[Recommendation] = []
    for s in DEMO_SAMPLES:
        print(f"\n=== 样本 {s['pid']}  S0={s['S0']}, K_i={feasible_K(s['age'],s['act'])} ===")
        r = recommend_patient(s["pid"], s["S0"], s["age"], s["act"])
        print(f"原始 Pareto {len(r.pareto_full)} 点；ED 后 {len(r.frontier_ed)} 点")
        print(f"ICER 序列: {[round(x,2) for x in r.icer_series]}")
        for lam in [30, 50, 100, 200]:
            p = r.rec_by_lam[lam]
            drop = p.E/r.S0*100
            print(f"[λ={lam:3d}]  S6={p.S6:6.2f}, C={p.C_total:5.0f}, 降={drop:5.1f}%, "
                  f"k={p.k_seq}, f={p.f_seq}")
        print(f"TOPSIS  → S6={r.rec_topsis.S6:.2f}, C={r.rec_topsis.C_total:.0f}")
        print(f"Kneedle → S6={r.rec_knee.S6:.2f}, C={r.rec_knee.C_total:.0f}")
        tab = tabulate_prescription(r, "nmb")
        tab.to_csv(f"{out_dir}/sample{s['pid']}_prescription.csv",
                   index=False, encoding="utf-8-sig")
        print("NMB 推荐处方:")
        print(tab.to_string(index=False))
        recs.append(r)
    plot_pareto_three    (recs, f"{out_dir}/Q3_pareto.png")
    plot_trajectory_three(recs, f"{out_dir}/Q3_trajectory.png")
    plot_lambda_sweep    (recs, f"{out_dir}/Q3_lambda_sweep.png")
    plot_icer_ladder     (recs, f"{out_dir}/Q3_icer_ladder.png")
    df_lam = sensitivity_lambda(recs, np.arange(0, 305, 5))
    df_lam.to_csv(f"{out_dir}/Q3_lambda_sweep.csv",
                  index=False, encoding="utf-8-sig")
    df_rule = sensitivity_rule_coefs(recs)
    df_rule.to_csv(f"{out_dir}/Q3_rule_sens.csv",
                   index=False, encoding="utf-8-sig")
    return recs


def plot_batch_summary(df_batch: pd.DataFrame, out: str):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    (ax1, ax2, ax3), (ax4, ax5, ax6) = axes
    sc = ax1.scatter(df_batch["S0"], df_batch["S6_nmb"],
                     c=df_batch["C_nmb"], cmap="viridis", s=28)
    ax1.plot([df_batch["S0"].min(), df_batch["S0"].max()],
             [df_batch["S0"].min(), df_batch["S0"].max()],
             "--", color="gray", alpha=0.5)
    ax1.set_xlabel("$S_0$"); ax1.set_ylabel("$S_6$ (NMB@λ=30)")
    ax1.set_title("(a) $S_0$-$S_6$ 散点（色标=总成本）")
    fig.colorbar(sc, ax=ax1, label="$C_\\mathrm{total}$"); ax1.grid(alpha=0.3)
    ax2.hist(df_batch["C_nmb"], bins=40, color="steelblue", alpha=0.8)
    ax2.axvline(BUDGET_SOFT, ls="--", color="red", label="预算建议 2000")
    ax2.set_xlabel("总成本 (元)"); ax2.set_ylabel("人数")
    ax2.set_title("(b) 总成本分布"); ax2.legend(); ax2.grid(alpha=0.3)
    ax3.hist(df_batch["drop_pct"], bins=40, color="salmon", alpha=0.8)
    ax3.set_xlabel("降幅 (%)"); ax3.set_ylabel("人数")
    ax3.set_title("(c) 降幅分布"); ax3.grid(alpha=0.3)
    groups = [df_batch[df_batch["K_size"] == k]["drop_pct"].values for k in [1,2,3]]
    ax4.boxplot(groups, tick_labels=["$|K|=1$","$|K|=2$","$|K|=3$"])
    ax4.set_ylabel("降幅 (%)"); ax4.set_title("(d) 按可行集 $|K_i|$ 分组降幅")
    ax4.grid(alpha=0.3)
    age_grps = [df_batch[df_batch["age"] == a]["C_nmb"].values for a in [1,2,3,4,5]]
    ax5.boxplot(age_grps, tick_labels=["40-49","50-59","60-69","70-79","80-89"])
    ax5.set_ylabel("总成本 (元)"); ax5.set_title("(e) 按年龄组总成本")
    ax5.grid(alpha=0.3)
    cnt = df_batch.groupby("S0_tier").size()
    ax6.bar(["$\\le 58$","59-61","$\\ge 62$"],
            [cnt.get(t, 0) for t in ["低","中","高"]],
            color=["#7fbf7b", "#ffcc66", "#d73027"])
    ax6.set_ylabel("人数"); ax6.set_title("(f) 按 $S_0$ 档位分组人数")
    ax6.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches="tight"); plt.close()


def run_batch(data_path: str, out_dir: str = "outputs"):
    os.makedirs(out_dir, exist_ok=True)
    df = load_patients(data_path)
    if "type" in df.columns:
        df_p = df[df["type"] == 5].reset_index(drop=True).copy()
    else:
        df_p = df.copy()
    print(f"共 {len(df_p)} 名痰湿体质患者进入批量优化")
    rows = []
    for idx, row in df_p.iterrows():
        r = recommend_patient(int(row.get("id", idx + 1)),
                              float(row["S0"]),
                              int(row["age"]),
                              float(row.get("act_score",
                                    row.get("adl", 0) + row.get("iadl", 0))))
        rows.append({
            "id": r.pid, "age": r.age_group, "S0": r.S0,
            "K_size": len(r.K_i),
            "S0_tier": ("低" if r.S0 <= 58 else "中" if r.S0 < 62 else "高"),
            "S6_nmb": r.rec_nmb.S6, "C_nmb": r.rec_nmb.C_total,
            "E_nmb": r.rec_nmb.E,
            "drop_pct": r.rec_nmb.E / r.S0 * 100 if r.S0 > 0 else 0,
            "CER": (r.rec_nmb.C_total / r.rec_nmb.E) if r.rec_nmb.E > 1e-6 else np.inf,
            "agree_nmb_topsis": int(r.rec_nmb.k_seq == r.rec_topsis.k_seq
                                    and r.rec_nmb.f_seq == r.rec_topsis.f_seq),
            "agree_nmb_knee":   int(r.rec_nmb.k_seq == r.rec_knee.k_seq
                                    and r.rec_nmb.f_seq == r.rec_knee.f_seq),
        })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(f"{out_dir}/Q3_batch_results.csv",
                  index=False, encoding="utf-8-sig")
    print("\n按 |K_i| 分组：")
    print(df_out.groupby("K_size")[["S6_nmb", "C_nmb", "drop_pct"]].mean().round(2))
    print("\n按 S0 档位分组：")
    print(df_out.groupby("S0_tier")[["S6_nmb", "C_nmb", "drop_pct", "CER"]].mean().round(2))
    print("\n按年龄组分组：")
    print(df_out.groupby("age")[["S6_nmb", "C_nmb", "drop_pct"]].mean().round(2))
    print(f"\nNMB vs TOPSIS 一致率: {df_out['agree_nmb_topsis'].mean()*100:.1f}%")
    print(f"NMB vs Kneedle 一致率: {df_out['agree_nmb_knee'].mean()*100:.1f}%")
    plot_batch_summary(df_out, f"{out_dir}/Q3_summary.png")
    return df_out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Problem 3 NMB solver (严格按题意修订)")
    parser.add_argument("--data", default=None,
                        help="附件 1 路径（xlsx/csv），省略则运行 demo")
    parser.add_argument("--out", default="outputs", help="输出目录")
    parser.add_argument("--mode", choices=["demo", "batch", "all"], default="all")
    args = parser.parse_args()

    if args.mode in ("demo", "all"):
        print("="*70)
        print("  Part A · 三位样本 demo  (严格按题意: f<5 无效 + 中医无降分)")
        print("="*70)
        recs_demo = run_demo(args.out)

    if args.mode in ("batch", "all") and args.data:
        print("\n" + "="*70)
        print("  Part B · 附件 1 · 278 人批量")
        print("="*70)
        df_batch = run_batch(args.data, args.out)

    print(f"\n完成。结果写入 {args.out}/")
