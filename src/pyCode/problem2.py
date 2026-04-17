"""
问题2：高血脂症低/中/高三级风险预警模型
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from common import configure_plotting, load_data, save_csv, save_figure, save_text, set_random_seed


warnings.filterwarnings("ignore")


def count_abnormal(row: pd.Series) -> int:
    count = 0
    if row["TC（总胆固醇）"] > 6.2:
        count += 1
    if row["TG（甘油三酯）"] > 1.7:
        count += 1
    if row["LDL-C（低密度脂蛋白）"] > 3.1:
        count += 1
    if row["HDL-C（高密度脂蛋白）"] < 1.04:
        count += 1
    return count


def rule_risk(row: pd.Series) -> int:
    abnormal_count = row["血脂异常项数"]
    phlegm_score = row["痰湿质"]
    activity_score = row["活动量表总分（ADL总分+IADL总分）"]
    bmi = row["BMI"]
    if abnormal_count >= 2:
        return 3
    if abnormal_count >= 1 and (phlegm_score >= 60 or bmi >= 24):
        return 3
    if phlegm_score >= 80 and activity_score < 40:
        return 3
    if abnormal_count == 1:
        return 2
    if phlegm_score >= 60:
        return 2
    if 40 <= phlegm_score < 60 and activity_score < 50:
        return 2
    return 1


def model_risk(probability: float, q33: float, q67: float) -> int:
    if probability >= q67:
        return 3
    if probability >= q33:
        return 2
    return 1


def main() -> None:
    configure_plotting()
    set_random_seed()
    df = load_data()

    df["血脂异常项数"] = df.apply(count_abnormal, axis=1)

    features = [
        "TG（甘油三酯）",
        "TC（总胆固醇）",
        "LDL-C（低密度脂蛋白）",
        "HDL-C（高密度脂蛋白）",
        "血尿酸",
        "BMI",
        "空腹血糖",
        "痰湿质",
        "湿热质",
        "血瘀质",
        "气虚质",
        "阳虚质",
        "阴虚质",
        "平和质",
        "气郁质",
        "特禀质",
        "活动量表总分（ADL总分+IADL总分）",
        "ADL总分",
        "IADL总分",
        "年龄组",
        "性别",
        "吸烟史",
        "饮酒史",
    ]
    x = df[features].to_numpy()
    y = df["高血脂症二分类标签"].to_numpy()
    x_scaled = StandardScaler().fit_transform(x)

    models = {
        "Logistic": lambda: LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=1.0,
            random_state=42,
        ),
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "GBDT": lambda: GradientBoostingClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
        ),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results: dict[str, dict[str, float]] = {}
    for name, builder in models.items():
        aucs: list[float] = []
        accs: list[float] = []
        f1s: list[float] = []
        for train_idx, test_idx in skf.split(x_scaled, y):
            model = builder()
            x_train, x_test = (x_scaled[train_idx], x_scaled[test_idx]) if name == "Logistic" else (x[train_idx], x[test_idx])
            model.fit(x_train, y[train_idx])
            probabilities = model.predict_proba(x_test)[:, 1]
            predictions = (probabilities >= 0.5).astype(int)
            aucs.append(float(roc_auc_score(y[test_idx], probabilities)))
            accs.append(float(accuracy_score(y[test_idx], predictions)))
            f1s.append(float(f1_score(y[test_idx], predictions)))
        cv_results[name] = {
            "AUC_mean": float(np.mean(aucs)),
            "AUC_std": float(np.std(aucs)),
            "Acc": float(np.mean(accs)),
            "F1": float(np.mean(f1s)),
        }

    df_cv = pd.DataFrame(cv_results).T.round(4)
    print("==== 5-Fold CV 结果 ====")
    print(df_cv)
    save_csv(df_cv, "Q2_cv_results.csv")

    best_lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, random_state=42)
    best_lr.fit(x_scaled, y)
    probabilities = best_lr.predict_proba(x_scaled)[:, 1]
    df["预测发病概率"] = probabilities
    print(
        f"\nLogistic 预测概率分布: min={probabilities.min():.4f}, "
        f"中位数={np.median(probabilities):.4f}, max={probabilities.max():.4f}"
    )

    coef_abs = pd.Series(np.abs(best_lr.coef_[0]), index=features).sort_values(ascending=False)
    print("\n==== Logistic |beta| Top10 (标准化) ====")
    print(coef_abs.head(10).round(4))

    gbdt = GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
    gbdt.fit(x, y)
    fi_gbdt = pd.Series(gbdt.feature_importances_, index=features).sort_values(ascending=False)
    print("\n==== GBDT 特征重要度 Top10 (对照) ====")
    print(fi_gbdt.head(10).round(4))

    feature_importance = pd.concat(
        [coef_abs.rename("Logistic_|beta|"), fi_gbdt.rename("GBDT_importance")],
        axis=1,
    )
    save_csv(feature_importance, "Q2_feature_importance.csv")

    df["规则风险等级"] = df.apply(rule_risk, axis=1)
    q33, q67 = np.quantile(probabilities, [0.33, 0.67])
    print(f"\nLogistic 预测概率分位: q33={q33:.4f}, q67={q67:.4f}")
    df["模型风险等级"] = df["预测发病概率"].apply(lambda value: model_risk(value, q33, q67))
    df["最终风险等级"] = np.maximum(df["规则风险等级"], df["模型风险等级"])
    df["最终风险等级_名"] = df["最终风险等级"].map({1: "低风险", 2: "中风险", 3: "高风险"})

    print("\n==== 最终三级风险分布 ====")
    print(df["最终风险等级_名"].value_counts())
    print("\n各风险层实际高血脂发病率:")
    print(df.groupby("最终风险等级_名")["高血脂症二分类标签"].agg(["count", "mean"]).round(3))
    print("\n诊断标签 x 风险等级 交叉表:")
    print(pd.crosstab(df["高血脂症二分类标签"], df["最终风险等级_名"]))

    print("\n各风险层特征均值:")
    print(
        df.groupby("最终风险等级_名").agg(
            血脂异常项数=("血脂异常项数", "mean"),
            痰湿积分=("痰湿质", "mean"),
            活动量表总分=("活动量表总分（ADL总分+IADL总分）", "mean"),
            TG均值=("TG（甘油三酯）", "mean"),
            TC均值=("TC（总胆固醇）", "mean"),
            BMI均值=("BMI", "mean"),
            年龄组=("年龄组", "mean"),
            预测概率=("预测发病概率", "mean"),
        ).round(3)
    )

    dft = df[df["体质标签"] == 5].copy()
    print(
        f"\n[痰湿体质{len(dft)}人] 高风险{(dft['最终风险等级'] == 3).sum()}, "
        f"中风险{(dft['最终风险等级'] == 2).sum()}, 低风险{(dft['最终风险等级'] == 1).sum()}"
    )

    tree_features = [
        "痰湿质",
        "活动量表总分（ADL总分+IADL总分）",
        "TG（甘油三酯）",
        "TC（总胆固醇）",
        "LDL-C（低密度脂蛋白）",
        "HDL-C（高密度脂蛋白）",
        "BMI",
        "血尿酸",
        "年龄组",
    ]
    xt = dft[tree_features].to_numpy()
    yt = (dft["最终风险等级"] == 3).astype(int).to_numpy()
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=15, random_state=42)
    tree.fit(xt, yt)
    rules_text = export_text(tree, feature_names=tree_features)
    print("\n==== 痰湿体质高风险的决策规则 ====")
    print(rules_text)
    save_text(rules_text, "Q2_tree_rules_phlegm.txt")

    combos = [
        ("痰湿积分≥60 & 活动量表<40", (dft["痰湿质"] >= 60) & (dft["活动量表总分（ADL总分+IADL总分）"] < 40)),
        ("痰湿积分≥60 & 血脂异常≥1", (dft["痰湿质"] >= 60) & (dft["血脂异常项数"] >= 1)),
        ("TG>1.7 & BMI≥24", (dft["TG（甘油三酯）"] > 1.7) & (dft["BMI"] >= 24)),
        ("血脂异常≥2 & 痰湿积分≥40", (dft["血脂异常项数"] >= 2) & (dft["痰湿质"] >= 40)),
        ("痰湿积分≥60 & 年龄组≥3", (dft["痰湿质"] >= 60) & (dft["年龄组"] >= 3)),
    ]
    rows: list[dict[str, str | int]] = []
    for name, mask in combos:
        subset = dft[mask]
        if len(subset) == 0:
            continue
        rows.append(
            {
                "组合": name,
                "人数": int(len(subset)),
                "占痰湿比例": f"{len(subset) / len(dft) * 100:.1f}%",
                "实际发病率": f"{subset['高血脂症二分类标签'].mean() * 100:.1f}%",
                "高风险占比": f"{(subset['最终风险等级'] == 3).mean() * 100:.1f}%",
            }
        )
    core = pd.DataFrame(rows)
    print("\n==== 痰湿体质高风险核心特征组合 ====")
    print(core)
    save_csv(core, "Q2_core_combo.csv", index=False)

    fig = plt.figure(figsize=(16, 12))

    ax = plt.subplot(2, 3, 1)
    ax.bar(
        df_cv.index,
        df_cv["AUC_mean"],
        yerr=df_cv["AUC_std"],
        color=["#4C72B0", "#55A868", "#C44E52"],
        capsize=5,
    )
    for idx, value in enumerate(df_cv["AUC_mean"]):
        ax.text(idx, value + 0.01, f"{value:.3f}", ha="center", fontweight="bold")
    ax.set_ylim(0.7, 1.02)
    ax.set_ylabel("AUC")
    ax.set_title("5-Fold CV AUC")
    ax.grid(axis="y", alpha=0.3)

    ax = plt.subplot(2, 3, 2)
    top10 = coef_abs.head(10)
    ax.barh(range(10)[::-1], top10.to_numpy(), color="teal")
    ax.set_yticks(range(10)[::-1])
    ax.set_yticklabels([name[:14] for name in top10.index])
    ax.set_title("Logistic |beta| Top10")

    ax = plt.subplot(2, 3, 3)
    fpr, tpr, _ = roc_curve(y, probabilities)
    ax.plot(fpr, tpr, linewidth=2, label=f"Logistic AUC={roc_auc_score(y, probabilities):.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC Curve")
    ax.legend()

    ax = plt.subplot(2, 3, 4)
    risk_dist = (
        df.groupby("最终风险等级_名")["高血脂症二分类标签"]
        .agg(["count", "mean"])
        .reindex(["低风险", "中风险", "高风险"])
    )
    x_axis = np.arange(3)
    ax2 = ax.twinx()
    ax.bar(x_axis - 0.2, risk_dist["count"], 0.4, color="skyblue", label="# cases")
    ax2.bar(x_axis + 0.2, risk_dist["mean"], 0.4, color="salmon", label="actual rate")
    ax.set_xticks(x_axis)
    ax.set_xticklabels(["Low", "Medium", "High"])
    ax.set_ylabel("Count")
    ax2.set_ylabel("Actual prevalence")
    ax.set_title("Risk Stratification")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    ax = plt.subplot(2, 3, 5)
    cmap = {1: "green", 2: "orange", 3: "red"}
    for level in [1, 2, 3]:
        subset = dft[dft["最终风险等级"] == level]
        ax.scatter(
            subset["痰湿质"],
            subset["活动量表总分（ADL总分+IADL总分）"],
            c=cmap[level],
            alpha=0.6,
            label=f"Risk L{level}",
            s=25,
        )
    ax.axvline(60, color="k", linestyle=":", alpha=0.4)
    ax.axhline(40, color="k", linestyle=":", alpha=0.4)
    ax.set_xlabel("Phlegm Score")
    ax.set_ylabel("Activity Score")
    ax.set_title("Phlegm Constitution Map")
    ax.legend(fontsize=8)

    ax = plt.subplot(2, 3, 6)
    plot_tree(
        tree,
        feature_names=["Phlegm", "Act", "TG", "TC", "LDL", "HDL", "BMI", "UA", "Age"],
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=7,
        ax=ax,
    )
    ax.set_title("Decision Tree (Phlegm Constitution)")

    save_figure(fig, "Q2_summary.png")

    save_csv(
        df[
            [
                "样本ID",
                "体质标签",
                "痰湿质",
                "活动量表总分（ADL总分+IADL总分）",
                "TG（甘油三酯）",
                "TC（总胆固醇）",
                "LDL-C（低密度脂蛋白）",
                "HDL-C（高密度脂蛋白）",
                "BMI",
                "血脂异常项数",
                "预测发病概率",
                "规则风险等级",
                "模型风险等级",
                "最终风险等级",
                "最终风险等级_名",
                "高血脂症二分类标签",
            ]
        ],
        "Q2_risk_full.csv",
        index=False,
    )
    print("\n[OK] Q2 所有输出保存完成")


if __name__ == "__main__":
    main()
