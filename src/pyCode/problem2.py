# -*- coding: utf-8 -*-
"""
Problem 2: three-level hyperlipidemia risk warning model (v3).

This version follows the current paper logic:
- hard-rule layer keeps only R1 (blood-lipid abnormal items >= 2);
- R2/R3 are documented as rejected candidates, not executed as hard rules;
- score layer uses S_clin in [0, 7], S_prog in [0, 11], S_total in [0, 18];
- final score cutoffs are {2, 6}.
"""

from __future__ import annotations

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

from common import configure_plotting, figure_path, load_data, set_random_seed, table_path


configure_plotting()
set_random_seed()

RNG = np.random.default_rng(42)
LOW_CUT = 2
HIGH_CUT = 6


def count_abnormal(df: pd.DataFrame) -> pd.Series:
    """Count lipid abnormalities using the 2016 Chinese guideline thresholds."""
    return (
        (df["TC（总胆固醇）"] > 6.2).astype(int)
        + (df["TG（甘油三酯）"] > 1.7).astype(int)
        + (df["LDL-C（低密度脂蛋白）"] > 3.1).astype(int)
        + (df["HDL-C（高密度脂蛋白）"] < 1.04).astype(int)
    )


def clinical_score(df: pd.DataFrame) -> pd.Series:
    """Clinical severity score S_clin in [0, 7]."""
    return (
        (df["TC（总胆固醇）"] > 6.2).astype(int)
        + (df["TC（总胆固醇）"] > 7.2).astype(int)
        + (df["TG（甘油三酯）"] > 1.7).astype(int)
        + (df["TG（甘油三酯）"] > 2.3).astype(int)
        + (df["LDL-C（低密度脂蛋白）"] > 3.1).astype(int)
        + (df["LDL-C（低密度脂蛋白）"] > 4.1).astype(int)
        + (df["HDL-C（高密度脂蛋白）"] < 1.04).astype(int)
    )


def progression_score(df: pd.DataFrame) -> pd.Series:
    """Progression risk score S_prog in [0, 11]."""
    phlegm = df["痰湿质"]
    bmi = df["BMI"]
    activity = df["活动量表总分（ADL总分+IADL总分）"]
    age_group = df["年龄组"].astype(int)

    phlegm_score = np.select(
        [phlegm >= 80, phlegm >= 60, phlegm >= 40],
        [3, 2, 1],
        default=0,
    )
    bmi_score = np.select(
        [bmi >= 28, bmi >= 24],
        [2, 1],
        default=0,
    )
    activity_score = np.select(
        [activity < 40, activity < 60],
        [2, 1],
        default=0,
    )
    age_score = age_group.map({1: 0, 2: 0, 3: 1, 4: 2, 5: 2}).fillna(0).astype(int)

    return (
        pd.Series(phlegm_score, index=df.index)
        + pd.Series(bmi_score, index=df.index)
        + pd.Series(activity_score, index=df.index)
        + age_score
        + df["性别"].astype(int)
        + df["吸烟史"].astype(int)
    )


def stratify(hard: pd.Series | np.ndarray, score: pd.Series | np.ndarray, low: int, high: int) -> np.ndarray:
    """Return 1=low, 2=medium, 3=high."""
    score_arr = np.asarray(score)
    hard_arr = np.asarray(hard, dtype=bool)
    out = np.ones_like(score_arr, dtype=int)
    out[score_arr > low] = 2
    out[score_arr >= high] = 3
    out[hard_arr] = 3
    return out


def level_name(level: int) -> str:
    return {1: "低", 2: "中", 3: "高"}[int(level)]


def make_apriori_rules(items: pd.DataFrame, outcome: np.ndarray) -> pd.DataFrame:
    """Small, dependency-free Apriori-style enumeration for high-risk antecedents."""
    item_names = list(items.columns)
    matrix = items.to_numpy(dtype=bool)
    high = np.asarray(outcome, dtype=bool)
    support_high = high.mean()
    min_support = 0.10
    min_confidence = 0.90
    min_lift = 1.05
    max_items = 4

    def is_redundant(names: list[str]) -> bool:
        names_set = set(names)
        mutually_exclusive = [
            {"活动<40", "活动40-59"},
            {"BMI超重", "BMI肥胖"},
            {"年龄60-69", "年龄≥70"},
        ]
        if any(len(names_set & group) >= 2 for group in mutually_exclusive):
            return True
        if "血脂异常≥1" in names_set and "血脂异常≥2" in names_set:
            return True
        if "血脂异常≥1" in names_set and names_set & {"TG升高", "TC升高", "LDL升高", "HDL偏低"}:
            return True
        return False

    rules: list[dict[str, object]] = []
    for size in range(1, max_items + 1):
        for combo in combinations(range(len(item_names)), size):
            names = [item_names[index] for index in combo]
            if is_redundant(names):
                continue

            antecedent = matrix[:, list(combo)].all(axis=1)
            support = antecedent.mean()
            if support < min_support:
                continue

            support_joint = (antecedent & high).mean()
            confidence = support_joint / support
            lift = confidence / support_high if support_high > 0 else 0.0
            if confidence >= min_confidence and lift >= min_lift:
                rules.append(
                    {
                        "组合": " ∧ ".join(names),
                        "项数": size,
                        "人数": int(antecedent.sum()),
                        "支持度": round(float(support), 3),
                        "置信度": round(float(confidence), 3),
                        "提升度": round(float(lift), 3),
                    }
                )

    if not rules:
        return pd.DataFrame(columns=["组合", "项数", "人数", "支持度", "置信度", "提升度"])

    return (
        pd.DataFrame(rules)
        .sort_values(["项数", "人数"], ascending=[True, False])
        .reset_index(drop=True)
    )


def save_main_outputs(df: pd.DataFrame) -> None:
    risk_summary = []
    for level in [1, 2, 3]:
        mask = df["risk_level"] == level
        risk_summary.append(
            {
                "等级": level_name(level),
                "人数": int(mask.sum()),
                "占比%": round(mask.mean() * 100, 1),
                "发病率%": round(df.loc[mask, "高血脂症二分类标签"].mean() * 100, 1),
            }
        )
    pd.DataFrame(risk_summary).to_csv(table_path("Q2_risk_stat.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(risk_summary).to_csv(
        table_path("Q2_stratification_summary.csv"), index=False, encoding="utf-8-sig"
    )

    calibration = (
        df.groupby("S_total")["高血脂症二分类标签"]
        .agg(n="count", **{"prevalence%": lambda item: item.mean() * 100})
        .reset_index()
    )
    calibration["prevalence%"] = calibration["prevalence%"].round(1)
    calibration.to_csv(table_path("Q2_calibration.csv"), index=False, encoding="utf-8-sig")

    full_cols = [
        "样本ID",
        "S_clin",
        "S_prog",
        "S_total",
        "R1",
        "hard_trigger",
        "risk_level",
        "高血脂症二分类标签",
    ]
    full = df[full_cols].rename(columns={"高血脂症二分类标签": "y"})
    full.to_csv(table_path("Q2_stratification_full.csv"), index=False, encoding="utf-8-sig")


def bootstrap_summary(risk_level: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    records = {"low_pct": [], "med_pct": [], "high_pct": [], "low_prev": [], "med_prev": [], "high_prev": []}
    n = len(y)
    for _ in range(1000):
        idx = RNG.integers(0, n, n)
        sampled_risk = risk_level[idx]
        sampled_y = y[idx]
        for level, key in [(1, "low"), (2, "med"), (3, "high")]:
            mask = sampled_risk == level
            records[f"{key}_pct"].append(mask.mean() * 100)
            records[f"{key}_prev"].append(sampled_y[mask].mean() * 100 if mask.any() else np.nan)

    rows = []
    labels = {
        "low_pct": "低风险占比(%)",
        "med_pct": "中风险占比(%)",
        "high_pct": "高风险占比(%)",
        "low_prev": "低层发病率(%)",
        "med_prev": "中层发病率(%)",
        "high_prev": "高层发病率(%)",
    }
    for key in ["low_pct", "med_pct", "high_pct", "low_prev", "med_prev", "high_prev"]:
        values = np.asarray(records[key], dtype=float)
        values = values[~np.isnan(values)]
        rows.append(
            {
                "指标": labels[key],
                "均值": round(float(values.mean()), 2),
                "2.5%": round(float(np.percentile(values, 2.5)), 2),
                "97.5%": round(float(np.percentile(values, 97.5)), 2),
            }
        )
    return pd.DataFrame(rows)


def sensitivity_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, np.ndarray]]]:
    base = df["risk_level"].to_numpy()
    hard = df["hard_trigger"].to_numpy(dtype=bool)
    s_clin = df["S_clin"].to_numpy()
    s_prog = df["S_prog"].to_numpy()

    def with_scaled_progression(scale: float) -> np.ndarray:
        scaled_prog = np.round(s_prog * scale).astype(int)
        return stratify(hard, s_clin + scaled_prog, LOW_CUT, HIGH_CUT)

    def with_thresholds(low: int, high: int) -> np.ndarray:
        return stratify(hard, df["S_total"].to_numpy(), low, high)

    scenarios = [
        ("baseline", base),
        ("S_prog x0.75", with_scaled_progression(0.75)),
        ("S_prog x1.25", with_scaled_progression(1.25)),
        ("cut(1,6)", with_thresholds(1, 6)),
        ("cut(3,6)", with_thresholds(3, 6)),
        ("cut(2,5)", with_thresholds(2, 5)),
        ("cut(2,7)", with_thresholds(2, 7)),
    ]

    rows = []
    for name, risk in scenarios:
        rows.append(
            {
                "scenario": name,
                "agree%": round(float((risk == base).mean() * 100), 1),
                "low%": round(float((risk == 1).mean() * 100), 1),
                "med%": round(float((risk == 2).mean() * 100), 1),
                "high%": round(float((risk == 3).mean() * 100), 1),
            }
        )
    return pd.DataFrame(rows), scenarios


def method_comparison(df: pd.DataFrame) -> pd.DataFrame:
    set_c_cols = [
        "TC（总胆固醇）",
        "TG（甘油三酯）",
        "LDL-C（低密度脂蛋白）",
        "HDL-C（高密度脂蛋白）",
        "空腹血糖",
        "血尿酸",
        "BMI",
        "ADL用厕",
        "ADL吃饭",
        "ADL步行",
        "ADL穿衣",
        "ADL洗澡",
        "IADL购物",
        "IADL做饭",
        "IADL理财",
        "IADL交通",
        "IADL服药",
    ]
    y = df["高血脂症二分类标签"].to_numpy()
    features = StandardScaler().fit_transform(df[set_c_cols].to_numpy())
    model = LogisticRegression(max_iter=1000, random_state=42).fit(features, y)
    probabilities = model.predict_proba(features)[:, 1]
    q33, q67 = np.quantile(probabilities, [1 / 3, 2 / 3])
    quantile_risk = np.where(probabilities >= q67, 3, np.where(probabilities >= q33, 2, 1))

    rows = []
    for name, risk in [
        ("直接概率分位切分", quantile_risk),
        ("硬规则+双轴评分卡(本方案)", df["risk_level"].to_numpy()),
    ]:
        row = {"方法": name}
        for level, column in [(1, "低风险"), (2, "中风险"), (3, "高风险")]:
            mask = risk == level
            row[column] = f"n={int(mask.sum())} ({mask.mean()*100:.1f}%), p={y[mask].mean()*100:.1f}%"
        rows.append(row)
    return pd.DataFrame(rows)


def save_visualization(
    df: pd.DataFrame,
    calibration: pd.DataFrame,
    boot: pd.DataFrame,
    sensitivity: pd.DataFrame,
    scenarios: list[tuple[str, np.ndarray]],
) -> None:
    y = df["高血脂症二分类标签"].to_numpy()
    auc_clin = roc_auc_score(y, df["S_clin"])
    auc_prog = roc_auc_score(y, df["S_prog"])
    auc_total = roc_auc_score(y, df["S_total"])

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Problem 2: Hard-rule + Two-Axis Score Card - Verification Summary (v3)", fontsize=13)

    ax = axes[0, 0]
    ax.plot(calibration["S_total"], calibration["prevalence%"], "-o", color="#1f77b4")
    ax.axvline(LOW_CUT + 0.5, ls="--", c="green", alpha=0.7, label="Low/Med cut")
    ax.axvline(HIGH_CUT - 0.5, ls="--", c="red", alpha=0.7, label="Med/High cut")
    for _, row in calibration.iterrows():
        ax.annotate(f"n={int(row['n'])}", (row["S_total"], row["prevalence%"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_title("(a) Calibration curve (monotone)")
    ax.set_xlabel("S_total")
    ax.set_ylabel("Observed prevalence (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    labels = ["Low", "Medium", "High"]
    counts = [(df["risk_level"] == level).sum() for level in [1, 2, 3]]
    prevalences = [
        df.loc[df["risk_level"] == level, "高血脂症二分类标签"].mean() * 100
        for level in [1, 2, 3]
    ]
    x = np.arange(3)
    width = 0.35
    ax2 = ax.twinx()
    ax.bar(x - width / 2, counts, width, label="# cases", color="#6ea8cf")
    ax2.bar(x + width / 2, prevalences, width, label="prevalence %", color="#e07b5c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("# cases")
    ax2.set_ylabel("prevalence (%)")
    ax2.set_ylim(0, 115)
    ax.set_title("(b) Three-level stratification")
    for idx, (count, prevalence) in enumerate(zip(counts, prevalences)):
        ax.text(idx - width / 2, count + 5, str(count), ha="center", fontsize=9)
        ax2.text(idx + width / 2, prevalence + 1, f"{prevalence:.1f}%", ha="center", fontsize=9)
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    ax = axes[0, 2]
    heat = df.pivot_table(index="S_prog", columns="S_clin", values="高血脂症二分类标签", aggfunc="mean")
    im = ax.imshow(heat.values, origin="lower", cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels(heat.columns)
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_xlabel("S_clin")
    ax.set_ylabel("S_prog")
    ax.set_title("(c) Prevalence heatmap: S_clin x S_prog")
    for row in range(heat.shape[0]):
        for col in range(heat.shape[1]):
            value = heat.values[row, col]
            if not np.isnan(value):
                ax.text(col, row, f"{value*100:.0f}", ha="center", va="center", color="white" if value > 0.5 else "black", fontsize=7)
    plt.colorbar(im, ax=ax, label="prevalence")

    ax = axes[1, 0]
    for name, score, auc, color in [
        ("S_clin", df["S_clin"], auc_clin, "#1f77b4"),
        ("S_prog", df["S_prog"], auc_prog, "#d62728"),
        ("S_total", df["S_total"], auc_total, "#2ca02c"),
    ]:
        fpr, tpr, _ = roc_curve(y, score)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color)
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.6)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("(d) ROC curves")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    source = {
        "R1: lipid abn >= 2 (2016 guideline)": int(df["R1"].sum()),
        "Score layer: S_total >= 6 (non-R1)": int(((df["S_total"] >= HIGH_CUT) & ~df["R1"].astype(bool)).sum()),
    }
    total_high = sum(source.values())
    bars = ax.barh(list(source.keys()), list(source.values()), color="#b36b6b")
    for bar, value in zip(bars, source.values()):
        ax.text(value + 3, bar.get_y() + bar.get_height() / 2, f"{value} ({value/total_high*100:.1f}%)", va="center", fontsize=9)
    ax.set_xlabel("# high-risk cases")
    ax.set_title(f"(e) High-risk trigger source (n={total_high})")
    ax.invert_yaxis()

    ax = axes[1, 2]
    x = np.arange(len(sensitivity))
    ax.bar(x, sensitivity["low%"], color="#6fbf73", label="Low")
    ax.bar(x, sensitivity["med%"], bottom=sensitivity["low%"], color="#f0ad4e", label="Medium")
    ax.bar(
        x,
        sensitivity["high%"],
        bottom=sensitivity["low%"] + sensitivity["med%"],
        color="#d9534f",
        label="High",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in scenarios], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("%")
    ax.set_title("(f) Sensitivity: composition stability")
    ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(figure_path("Q2_summary_v2.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_data()
    df["blood_lipid_abnormal_count"] = count_abnormal(df)
    df["S_clin"] = clinical_score(df)
    df["S_prog"] = progression_score(df)
    df["S_total"] = df["S_clin"] + df["S_prog"]
    df["R1"] = df["blood_lipid_abnormal_count"] >= 2
    df["hard_trigger"] = df["R1"].astype(int)
    df["risk_level"] = stratify(df["R1"], df["S_total"], LOW_CUT, HIGH_CUT)

    y = df["高血脂症二分类标签"].to_numpy()
    print("Problem 2 v3: R1-only + two-axis score card")
    print(f"R1 triggers: {int(df['R1'].sum())}, prevalence={y[df['R1'].to_numpy()].mean()*100:.1f}%")

    r1_uplift = df["R1"] & (df["S_total"] < HIGH_CUT)
    print(f"R1 independent uplift: {int(r1_uplift.sum())}, prevalence={df.loc[r1_uplift, '高血脂症二分类标签'].mean()*100:.1f}%")

    save_main_outputs(df)
    calibration = pd.read_csv(table_path("Q2_calibration.csv"), encoding="utf-8-sig")

    boot = bootstrap_summary(df["risk_level"].to_numpy(), y)
    boot.to_csv(table_path("Q2_bootstrap_CI.csv"), index=False, encoding="utf-8-sig")

    sensitivity, scenarios = sensitivity_summary(df)
    sensitivity.to_csv(table_path("Q2_sensitivity.csv"), index=False, encoding="utf-8-sig")

    method_compare = method_comparison(df)
    method_compare.to_csv(table_path("Q2_method_compare.csv"), index=False, encoding="utf-8-sig")

    phlegm_mask = df["体质标签"] == 5
    phlegm_df = df.loc[phlegm_mask].copy()
    apriori_items = pd.DataFrame(
        {
            "痰湿≥60": phlegm_df["痰湿质"] >= 60,
            "活动<40": phlegm_df["活动量表总分（ADL总分+IADL总分）"] < 40,
            "活动40-59": (phlegm_df["活动量表总分（ADL总分+IADL总分）"] >= 40)
            & (phlegm_df["活动量表总分（ADL总分+IADL总分）"] < 60),
            "BMI超重": (phlegm_df["BMI"] >= 24) & (phlegm_df["BMI"] < 28),
            "BMI肥胖": phlegm_df["BMI"] >= 28,
            "年龄60-69": phlegm_df["年龄组"] == 3,
            "年龄≥70": phlegm_df["年龄组"] >= 4,
            "男性": phlegm_df["性别"] == 1,
            "吸烟": phlegm_df["吸烟史"] == 1,
            "TG升高": phlegm_df["TG（甘油三酯）"] > 1.7,
            "TC升高": phlegm_df["TC（总胆固醇）"] > 6.2,
            "LDL升高": phlegm_df["LDL-C（低密度脂蛋白）"] > 3.1,
            "HDL偏低": phlegm_df["HDL-C（高密度脂蛋白）"] < 1.04,
            "血脂异常≥1": phlegm_df["blood_lipid_abnormal_count"] >= 1,
            "血脂异常≥2": phlegm_df["blood_lipid_abnormal_count"] >= 2,
        }
    )
    apriori_full = make_apriori_rules(apriori_items, phlegm_df["risk_level"].to_numpy() == 3)
    apriori_full.to_csv(table_path("Q2_apriori_rules_full.csv"), index=False, encoding="utf-8-sig")

    apriori_top = pd.concat(
        [apriori_full[apriori_full["项数"] == size].head(4) for size in range(1, 5)],
        ignore_index=True,
    ).head(10)
    apriori_top.to_csv(table_path("Q2_apriori_rules.csv"), index=False, encoding="utf-8-sig")

    save_visualization(df, calibration, boot, sensitivity, scenarios)

    risk_counts = df["risk_level"].value_counts().sort_index()
    risk_prev = df.groupby("risk_level")["高血脂症二分类标签"].mean() * 100
    print("Risk strata:")
    for level in [1, 2, 3]:
        print(f"  {level_name(level)}: n={int(risk_counts[level])}, prevalence={risk_prev[level]:.1f}%")
    print(f"AUC(S_total)={roc_auc_score(y, df['S_total']):.3f}")
    print(f"Minimum sensitivity agreement={sensitivity['agree%'].min():.1f}%")
    print("Outputs written to src/outputs/tables and src/outputs/figures.")


if __name__ == "__main__":
    main()
