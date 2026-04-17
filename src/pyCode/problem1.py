"""
问题1：关键指标筛选与九种体质贡献度分析
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import configure_plotting, load_data, save_csv, save_figure, set_random_seed


warnings.filterwarnings("ignore")

TIZHI = ["平和质", "气虚质", "阳虚质", "阴虚质", "痰湿质", "湿热质", "血瘀质", "气郁质", "特禀质"]
BLOOD = [
    "HDL-C（高密度脂蛋白）",
    "LDL-C（低密度脂蛋白）",
    "TG（甘油三酯）",
    "TC（总胆固醇）",
    "空腹血糖",
    "血尿酸",
    "BMI",
]
ACT = ["ADL总分", "IADL总分", "活动量表总分（ADL总分+IADL总分）"]
BASE = ["年龄组", "性别", "吸烟史", "饮酒史"]


def zscore_rank(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / (series.std() + 1e-9)


def main() -> None:
    configure_plotting()
    set_random_seed()
    df = load_data()

    y = df["高血脂症二分类标签"].to_numpy()
    tz = df["痰湿质"].to_numpy()
    features_for_phlegm = BLOOD + ACT + BASE

    rank_phlegm: dict[str, dict[str, float]] = {}
    tz_bin = pd.cut(tz, bins=[-np.inf, 20, 40, 60, np.inf], labels=[0, 1, 2, 3]).astype(int)
    for feature in features_for_phlegm:
        x = df[feature].to_numpy()
        r_sp, p_sp = stats.spearmanr(x, tz)
        mi = mutual_info_classif(x.reshape(-1, 1), tz_bin, random_state=42)[0]
        rank_phlegm[feature] = {
            "spearman_r": float(r_sp),
            "spearman_p": float(p_sp),
            "MI_with_phlegm": float(mi),
        }

    df_phlegm = pd.DataFrame(rank_phlegm).T
    df_phlegm["|spearman|"] = df_phlegm["spearman_r"].abs()
    df_phlegm["综合得分_痰湿表征"] = zscore_rank(df_phlegm["|spearman|"]) + zscore_rank(df_phlegm["MI_with_phlegm"])
    df_phlegm_sorted = df_phlegm.sort_values("综合得分_痰湿表征", ascending=False)

    rank_risk: dict[str, dict[str, float]] = {}
    x_all = df[features_for_phlegm].to_numpy()
    x_scaled = StandardScaler().fit_transform(x_all)
    for idx, feature in enumerate(features_for_phlegm):
        model = LogisticRegression(max_iter=500)
        model.fit(x_scaled[:, idx : idx + 1], y)
        coef = float(model.coef_[0][0])
        r_pb, _ = stats.pointbiserialr(y, df[feature].to_numpy())
        mi = mutual_info_classif(df[[feature]].to_numpy(), y, random_state=42)[0]
        f_stat, p_value = f_classif(df[[feature]].to_numpy(), y)
        rank_risk[feature] = {
            "logit_beta": coef,
            "point_biserial_r": float(r_pb),
            "MI_with_risk": float(mi),
            "F_stat": float(f_stat[0]),
            "p_value": float(p_value[0]),
        }

    df_risk = pd.DataFrame(rank_risk).T
    df_risk["|beta|"] = df_risk["logit_beta"].abs()

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf.fit(df[features_for_phlegm].to_numpy(), y)
    df_risk["RF_importance"] = pd.Series(rf.feature_importances_, index=features_for_phlegm)
    df_risk["综合得分_发病风险"] = (
        zscore_rank(df_risk["|beta|"])
        + zscore_rank(df_risk["MI_with_risk"])
        + zscore_rank(df_risk["RF_importance"])
        + zscore_rank(df_risk["F_stat"])
    )
    df_risk_sorted = df_risk.sort_values("综合得分_发病风险", ascending=False)

    contrib: dict[str, dict[str, float]] = {}
    mean_risk_all = float(y.mean())
    for constitution in TIZHI:
        x = df[constitution].to_numpy()
        model = LogisticRegression(max_iter=500)
        model.fit(StandardScaler().fit_transform(x.reshape(-1, 1)), y)
        beta = float(model.coef_[0][0])
        high = df[df[constitution] >= 60]["高血脂症二分类标签"]
        low = df[df[constitution] <= 20]["高血脂症二分类标签"]
        rr = (high.mean() + 1e-6) / (low.mean() + 1e-6) if len(high) > 0 and len(low) > 0 else np.nan
        r_pb, _ = stats.pointbiserialr(y, x)
        mi = mutual_info_classif(x.reshape(-1, 1), y, random_state=42)[0]
        contrib[constitution] = {
            "单变量beta": beta,
            "发病率比RR(高/低)": float(rr),
            "点二列r": float(r_pb),
            "互信息": float(mi),
        }

    x9 = StandardScaler().fit_transform(df[TIZHI].to_numpy())
    lr9 = LogisticRegression(max_iter=1000)
    lr9.fit(x9, y)
    for idx, constitution in enumerate(TIZHI):
        contrib[constitution]["多元beta"] = float(lr9.coef_[0][idx])

    rf9 = RandomForestClassifier(
        n_estimators=500,
        max_depth=5,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf9.fit(df[TIZHI].to_numpy(), y)
    for idx, constitution in enumerate(TIZHI):
        contrib[constitution]["RF重要度"] = float(rf9.feature_importances_[idx])

    df_contrib = pd.DataFrame(contrib).T
    df_contrib["综合贡献度"] = (
        zscore_rank(df_contrib["单变量beta"])
        + zscore_rank(df_contrib["多元beta"])
        + zscore_rank(df_contrib["互信息"])
        + zscore_rank(df_contrib["RF重要度"])
    )
    df_contrib_sorted = df_contrib.sort_values("综合贡献度", ascending=False)

    tizhi_stats: list[dict[str, float | int | str]] = []
    for idx, constitution in enumerate(TIZHI, start=1):
        subset = df[df["体质标签"] == idx]
        tizhi_stats.append(
            {
                "体质编号": idx,
                "体质名": constitution,
                "样本量": int(len(subset)),
                "发病率": float(subset["高血脂症二分类标签"].mean()),
                "平均痰湿积分": float(subset["痰湿质"].mean()),
            }
        )
    df_stat = pd.DataFrame(tizhi_stats)

    print("==== 表征痰湿严重程度的关键指标排序（前10）====")
    print(df_phlegm_sorted[["spearman_r", "MI_with_phlegm", "综合得分_痰湿表征"]].round(4))
    print("\n==== 预警高血脂发病风险的关键指标排序 ====")
    print(
        df_risk_sorted[
            ["logit_beta", "point_biserial_r", "MI_with_risk", "RF_importance", "F_stat", "综合得分_发病风险"]
        ].round(4)
    )
    print("\n==== 九种体质对发病风险的贡献度 ====")
    print(df_contrib_sorted.round(4))
    print("\n==== 各体质人群的高血脂发病率与平均痰湿积分 ====")
    print(df_stat.round(3))

    save_csv(df_phlegm_sorted, "Q1_rank_phlegm.csv")
    save_csv(df_risk_sorted, "Q1_rank_risk.csv")
    save_csv(df_contrib_sorted, "Q1_tizhi_contribution.csv")
    save_csv(df_stat, "Q1_tizhi_stats.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    top_phlegm = df_phlegm_sorted.head(8)
    ax.barh(range(len(top_phlegm))[::-1], top_phlegm["综合得分_痰湿表征"], color="steelblue")
    ax.set_yticks(range(len(top_phlegm))[::-1])
    ax.set_yticklabels([label[:15] for label in top_phlegm.index])
    ax.set_xlabel("Composite Score (phlegm-damp representation)")
    ax.set_title("Top Indicators for Phlegm-damp Severity")

    ax = axes[0, 1]
    top_risk = df_risk_sorted.head(10)
    ax.barh(range(len(top_risk))[::-1], top_risk["综合得分_发病风险"], color="firebrick")
    ax.set_yticks(range(len(top_risk))[::-1])
    ax.set_yticklabels([label[:15] for label in top_risk.index])
    ax.set_xlabel("Composite Score (hyperlipidemia risk)")
    ax.set_title("Top Indicators for Hyperlipidemia Risk")

    ax = axes[1, 0]
    constitution_colors = ["lightgray"] * 9
    constitution_colors[TIZHI.index("痰湿质")] = "darkorange"
    ax.bar(
        range(9),
        df_contrib_sorted["综合贡献度"].to_numpy(),
        color=[constitution_colors[TIZHI.index(name)] for name in df_contrib_sorted.index],
    )
    ax.set_xticks(range(9))
    ax.set_xticklabels([name[:3] for name in df_contrib_sorted.index], rotation=30)
    ax.set_ylabel("Composite contribution score")
    ax.set_title("Contribution of 9 TCM Constitutions to Hyperlipidemia")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 1]
    ax.bar(
        range(9),
        df_stat["发病率"],
        color=["#FF6666" if value == 5 else "#6688CC" for value in df_stat["体质编号"]],
    )
    ax.set_xticks(range(9))
    ax.set_xticklabels([name[:3] for name in TIZHI], rotation=30)
    ax.set_ylabel("Hyperlipidemia prevalence")
    ax.set_title("Hyperlipidemia Rate by Constitution Type")
    ax.axhline(mean_risk_all, color="k", linestyle="--", alpha=0.5, label=f"overall={mean_risk_all:.2f}")
    ax.legend()

    save_figure(fig, "Q1_summary.png")
    print("\n[OK] Figures saved: Q1_summary.png")


if __name__ == "__main__":
    main()
