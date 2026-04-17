"""
问题3：痰湿体质患者 6 个月个性化干预方案优化
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import configure_plotting, load_data, save_csv, save_figure


C_L = {1: 30, 2: 80, 3: 130}
C_S = {1: 3, 2: 5, 3: 8}
N_WK = 4


def next_phlegm(p_prev: float, strength: int, freq: int) -> float:
    if freq < 5:
        return p_prev
    decay = strength * 0.03 + max(0, freq - 5) * 0.01
    return max(p_prev * (1 - decay), 0)


def level_from_phlegm(score: float) -> int:
    if score >= 62:
        return 3
    if score >= 59:
        return 2
    return 1


def feasible_s(age_group: int, act: float) -> list[int]:
    if age_group in [1, 2]:
        strengths = {1, 2, 3}
    elif age_group in [3, 4]:
        strengths = {1, 2}
    else:
        strengths = {1}
    if act < 40:
        strengths &= {1}
    elif act < 60:
        strengths &= {1, 2}
    return sorted(strengths)


def optimize_patient(
    p0: float,
    age_group: int,
    act: float,
    budget: float = 2000,
    alpha: float = 1.0,
    beta: float = 0.01,
    gamma: float = 0.5,
    delta: float = 0.3,
    max_states: int = 2000,
) -> tuple[float, float | None, float | None, list[tuple[int, int, int, float, float]] | None]:
    strengths = feasible_s(age_group, act)
    states: dict[tuple[float, float], tuple[float, list[tuple[int, int, int, float, float]]]] = {
        (round(p0, 2), 0.0): (0.0, [])
    }
    best_final: tuple[float, float | None, float | None, list[tuple[int, int, int, float, float]] | None] = (
        float("inf"),
        None,
        None,
        None,
    )
    for month in range(1, 7):
        new_states: dict[tuple[float, float], tuple[float, list[tuple[int, int, int, float, float]]]] = {}
        for (p_prev, cost), (penalty, path) in states.items():
            level = level_from_phlegm(p_prev)
            level_cost = C_L[level]
            for strength in strengths:
                for freq in range(1, 11):
                    monthly_cost = level_cost + C_S[strength] * freq * N_WK
                    new_cost = cost + monthly_cost
                    if new_cost > budget:
                        continue
                    p_new = next_phlegm(p_prev, strength, freq)
                    delta_penalty = beta * monthly_cost + gamma * max(0, freq - 5) + delta * (3 - strength)
                    new_penalty = penalty + delta_penalty
                    new_path = path + [(level, strength, freq, round(p_new, 2), monthly_cost)]
                    key = (round(p_new, 2), round(new_cost, 1))
                    if key not in new_states or new_states[key][0] > new_penalty:
                        new_states[key] = (new_penalty, new_path)
        if month == 6:
            for (p6, total_cost), (penalty, path) in new_states.items():
                obj = alpha * p6 + penalty
                if obj < best_final[0]:
                    best_final = (obj, total_cost, p6, path)
        if len(new_states) > max_states:
            sorted_states = sorted(
                new_states.items(),
                key=lambda item: item[0][0] + 0.02 * item[0][1] + item[1][0],
            )[:max_states]
            new_states = dict(sorted_states)
        states = new_states
    return best_final


def main() -> None:
    configure_plotting()
    df = load_data()
    dft = df[df["体质标签"] == 5].copy()
    if dft.empty:
        raise ValueError("数据中未找到痰湿体质（体质标签=5）样本，无法执行问题三优化。")
    print(f"痰湿体质共 {len(dft)} 人")

    samples_to_show = dft[dft["样本ID"].isin([1, 2, 3])].copy()
    if len(samples_to_show) < 3:
        samples_to_show = dft.head(3).copy()
    print("\n前3位患者信息:")
    print(samples_to_show[["样本ID", "体质标签", "痰湿质", "活动量表总分（ADL总分+IADL总分）", "年龄组", "性别"]])

    results: list[dict[str, object]] = []
    for _, row in samples_to_show.iterrows():
        p0 = float(row["痰湿质"])
        age = int(row["年龄组"])
        act = float(row["活动量表总分（ADL总分+IADL总分）"])
        print(f"\n=========== 样本 {int(row['样本ID'])} ===========")
        print(f"初始痰湿积分={p0}, 年龄组={age}, 活动评分={act}")
        obj, cost, p6, path = optimize_patient(p0, age, act)
        if path is None or cost is None or p6 is None:
            continue
        print(f"最优目标值={obj:.3f}, 总成本={cost:.1f}元, 6月末痰湿积分={p6:.2f} (下降{(p0 - p6) / p0 * 100:.1f}%)")
        print("月份 | 调理级 | 强度s | 周频次f | 月末P | 月成本")
        for month, (level, strength, freq, p_val, monthly_cost) in enumerate(path, start=1):
            print(f"  {month}   |   {level}    |   {strength}   |    {freq}    | {p_val:6.2f} | {monthly_cost:.1f}")
        results.append(
            {
                "样本ID": int(row["样本ID"]),
                "P0": p0,
                "age": age,
                "act": act,
                "total_cost": float(cost),
                "P6": float(p6),
                "path": path,
            }
        )

    demo_rows: list[dict[str, float | int]] = []
    for result in results:
        for month, (level, strength, freq, p_val, monthly_cost) in enumerate(result["path"], start=1):
            demo_rows.append(
                {
                    "样本ID": result["样本ID"],
                    "月份": month,
                    "调理分级": level,
                    "活动强度": strength,
                    "周训练次数": freq,
                    "月末痰湿积分": p_val,
                    "本月成本(元)": monthly_cost,
                }
            )
    save_csv(pd.DataFrame(demo_rows), "Q3_demo_3patients.csv", index=False)

    traj_fig, traj_axes = plt.subplots(1, len(results), figsize=(5 * max(len(results), 1), 4), sharey=True)
    if len(results) == 1:
        traj_axes = [traj_axes]
    for ax, result in zip(traj_axes, results):
        months = [0] + list(range(1, 7))
        scores = [result["P0"]] + [step[3] for step in result["path"]]
        ax.plot(months, scores, marker="o", color="#4C72B0", linewidth=2)
        ax.set_xticks(months)
        ax.set_xlabel("Month")
        ax.set_title(f"Sample {result['样本ID']}")
        ax.grid(alpha=0.3)
    if results:
        traj_axes[0].set_ylabel("Phlegm Score")
    save_figure(traj_fig, "Q3_demo_trajectories.png")

    print("\n对全部痰湿体质患者求解...")
    all_results: list[dict[str, float | int]] = []
    for _, row in dft.iterrows():
        p0 = float(row["痰湿质"])
        age = int(row["年龄组"])
        act = float(row["活动量表总分（ADL总分+IADL总分）"])
        _, cost, p6, path = optimize_patient(p0, age, act, max_states=1000)
        if path is None or cost is None or p6 is None:
            continue
        avg_s = float(np.mean([step[1] for step in path]))
        avg_f = float(np.mean([step[2] for step in path]))
        max_l = int(max(step[0] for step in path))
        all_results.append(
            {
                "样本ID": int(row["样本ID"]),
                "P0": p0,
                "age": age,
                "act": act,
                "BMI": float(row["BMI"]),
                "性别": int(row["性别"]),
                "total_cost": float(cost),
                "P6": float(p6),
                "reduction_pct": float((p0 - p6) / p0 * 100),
                "avg_s": avg_s,
                "avg_f": avg_f,
                "max_L": max_l,
            }
        )
    all_df = pd.DataFrame(all_results)
    print(all_df[["P0", "total_cost", "P6", "reduction_pct", "avg_s", "avg_f"]].describe().round(2))
    save_csv(all_df, "Q3_all_phlegm.csv", index=False)

    print("\n==== 按初始痰湿积分分组 ====")
    all_df["P0_bin"] = pd.cut(all_df["P0"], [-1, 60, 62, 100], labels=["<60", "60-62", "≥62"])
    print(all_df.groupby("P0_bin", observed=True)[["total_cost", "P6", "reduction_pct", "avg_s", "avg_f"]].mean().round(2))

    print("\n==== 按年龄组分组 ====")
    print(all_df.groupby("age")[["total_cost", "P6", "reduction_pct", "avg_s", "avg_f"]].mean().round(2))

    print("\n==== 按活动能力评分分组 ====")
    all_df["act_bin"] = pd.cut(all_df["act"], [-1, 40, 60, 100], labels=["<40", "40-60", "≥60"])
    print(all_df.groupby("act_bin", observed=True)[["total_cost", "P6", "reduction_pct", "avg_s", "avg_f"]].mean().round(2))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    ax = axes[0, 0]
    scatter = ax.scatter(all_df["P0"], all_df["P6"], alpha=0.5, c=all_df["total_cost"], cmap="viridis")
    ax.plot([50, 70], [50, 70], "k--", alpha=0.4, label="y=x")
    ax.set_xlabel("Initial Phlegm Score P0")
    ax.set_ylabel("6-month P6")
    ax.set_title("P0 vs P6 (color=cost)")
    ax.legend()
    plt.colorbar(scatter, ax=ax, label="Cost (RMB)")

    ax = axes[0, 1]
    ax.hist(all_df["total_cost"], bins=30, color="steelblue", edgecolor="k")
    ax.axvline(2000, color="r", linestyle="--", label="Budget=2000")
    ax.set_xlabel("Total Cost (RMB)")
    ax.set_ylabel("# Patients")
    ax.set_title("Total Cost Distribution")
    ax.legend()

    ax = axes[0, 2]
    ax.hist(all_df["reduction_pct"], bins=30, color="forestgreen", edgecolor="k")
    ax.set_xlabel("Phlegm Score Reduction (%)")
    ax.set_ylabel("# Patients")
    ax.set_title("6-Month Reduction Distribution")

    ax = axes[1, 0]
    age_summary = all_df.groupby("age")[["avg_s", "avg_f"]].mean()
    age_x = age_summary.index
    ax.plot(age_x, age_summary["avg_s"], "o-", label="Avg s", color="crimson")
    ax.plot(age_x, age_summary["avg_f"], "s-", label="Avg f", color="navy")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Avg value")
    ax.set_title("Strategy by Age Group")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.scatter(all_df["P0"], all_df["reduction_pct"], alpha=0.5, color="purple")
    ax.set_xlabel("Initial P0")
    ax.set_ylabel("Reduction (%)")
    ax.set_title("P0 vs Reduction %")
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    ax.scatter(all_df["act"], all_df["avg_s"], alpha=0.5, color="teal")
    ax.axvline(40, color="k", linestyle="--", alpha=0.4)
    ax.axvline(60, color="k", linestyle="--", alpha=0.4)
    ax.set_xlabel("Activity Score")
    ax.set_ylabel("Avg s")
    ax.set_title("Activity vs Optimal s")
    ax.grid(alpha=0.3)

    save_figure(fig, "Q3_summary.png")
    print("\n[OK] Q3 结果保存完成")


if __name__ == "__main__":
    main()
