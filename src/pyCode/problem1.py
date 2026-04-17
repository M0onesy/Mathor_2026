from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.metrics import f1_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from common import (
    ACTIVITY_FEATURES,
    ACTIVITY_ITEM_FEATURES,
    ACTIVITY_SUMMARY_FEATURES,
    AUXILIARY_WARNING_FEATURES,
    COLS,
    CONSTITUTION_FEATURES,
    CONSTITUTION_NAME_MAP,
    DEMOGRAPHIC_FEATURES,
    DIRECT_DIAGNOSTIC_FEATURES,
    LIPID_FEATURES,
    METABOLIC_FEATURES,
    RANDOM_SEED,
    bh_fdr,
    cohen_d,
    configure_plotting,
    figure_path,
    load_data,
    minmax_scale,
    rank_to_borda,
    save_json,
    save_table,
    save_workbook,
    set_random_seed,
)


warnings.filterwarnings("ignore")


def normality_pvalue(series: pd.Series) -> float:
    if series.nunique() < 8:
        return np.nan
    try:
        return float(stats.normaltest(series).pvalue)
    except Exception:
        return np.nan


def safe_auc(y_true: pd.Series, values: pd.Series) -> tuple[float, float]:
    raw_auc = roc_auc_score(y_true, values)
    adjusted_auc = raw_auc if raw_auc >= 0.5 else 1 - raw_auc
    return raw_auc, adjusted_auc - 0.5


def compute_feature_statistics(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    y = df[COLS.diagnosis]
    tanshi = df[COLS.tanshi]
    mi_scores = mutual_info_classif(df[features], y, random_state=RANDOM_SEED)
    rows = []

    for feature, mi_score in zip(features, mi_scores):
        series = df[feature]
        case = series[y == 1]
        control = series[y == 0]
        pearson_r, pearson_p = stats.pearsonr(series, tanshi)
        spearman_r, spearman_p = stats.spearmanr(series, tanshi)
        normal_case = normality_pvalue(case)
        normal_control = normality_pvalue(control)

        if (normal_case > 0.05) and (normal_control > 0.05):
            test_name = "Welch t"
            group_stat, group_p = stats.ttest_ind(case, control, equal_var=False)
        else:
            test_name = "Mann-Whitney U"
            group_stat, group_p = stats.mannwhitneyu(case, control, alternative="two-sided")

        raw_auc, auc_gap = safe_auc(y, series)
        case_mean = float(case.mean())
        control_mean = float(control.mean())
        direction = "病例组更高" if case_mean >= control_mean else "病例组更低"

        rows.append(
            {
                "指标": feature,
                "均值_确诊组": round(case_mean, 4),
                "均值_未确诊组": round(control_mean, 4),
                "正态性p_确诊组": round(normal_case, 6) if pd.notna(normal_case) else np.nan,
                "正态性p_未确诊组": round(normal_control, 6) if pd.notna(normal_control) else np.nan,
                "组间检验": test_name,
                "组间统计量": round(float(group_stat), 6),
                "组间p值": float(group_p),
                "效应量_Cohen_d": float(cohen_d(case, control)),
                "与痰湿Pearson_r": float(pearson_r),
                "Pearson_p值": float(pearson_p),
                "与痰湿Spearman_r": float(spearman_r),
                "Spearman_p值": float(spearman_p),
                "原始AUC": float(raw_auc),
                "预警AUC增益": float(auc_gap),
                "互信息": float(mi_score),
                "风险方向": direction,
            }
        )

    result = pd.DataFrame(rows)
    result["组间FDR"] = bh_fdr(result["组间p值"])
    result["Pearson_FDR"] = bh_fdr(result["Pearson_p值"])
    result["Spearman_FDR"] = bh_fdr(result["Spearman_p值"])
    result["|Pearson_r|"] = result["与痰湿Pearson_r"].abs()
    result["|Spearman_r|"] = result["与痰湿Spearman_r"].abs()
    result["|Cohen_d|"] = result["效应量_Cohen_d"].abs()
    return result.sort_values("指标").reset_index(drop=True)


def build_direct_ranking(feature_stats: pd.DataFrame) -> pd.DataFrame:
    direct = feature_stats[feature_stats["指标"].isin(DIRECT_DIAGNOSTIC_FEATURES)].copy()
    direct["Borda_AUC"] = rank_to_borda(direct["预警AUC增益"], ascending=False)
    direct["Borda_d"] = rank_to_borda(direct["|Cohen_d|"], ascending=False)
    direct["Borda_MI"] = rank_to_borda(direct["互信息"], ascending=False)
    direct["综合得分"] = (
        0.45 * minmax_scale(direct["Borda_AUC"])
        + 0.45 * minmax_scale(direct["Borda_d"])
        + 0.10 * minmax_scale(direct["Borda_MI"])
    )
    direct["分层结论"] = "直接诊断核心指标"
    return direct.sort_values("综合得分", ascending=False).reset_index(drop=True)


def auxiliary_core_score(df: pd.DataFrame, features: list[str]) -> pd.Series:
    stats_frame = compute_feature_statistics(df, features)
    stats_frame["相关性显著度"] = -np.log10(np.minimum(stats_frame["Pearson_FDR"], stats_frame["Spearman_FDR"]) + 1e-12)
    stats_frame["组间显著度"] = -np.log10(stats_frame["组间FDR"] + 1e-12)
    tanshi_borda = (
        rank_to_borda(stats_frame.set_index("指标")["|Pearson_r|"], ascending=False)
        + rank_to_borda(stats_frame.set_index("指标")["|Spearman_r|"], ascending=False)
        + rank_to_borda(stats_frame.set_index("指标")["相关性显著度"], ascending=False)
    ) / 3
    warning_borda = (
        rank_to_borda(stats_frame.set_index("指标")["预警AUC增益"], ascending=False)
        + rank_to_borda(stats_frame.set_index("指标")["|Cohen_d|"], ascending=False)
        + rank_to_borda(stats_frame.set_index("指标")["组间显著度"], ascending=False)
    ) / 3
    core = 0.5 * minmax_scale(tanshi_borda) + 0.5 * minmax_scale(warning_borda)
    core.name = "bootstrap_core_score"
    return core


def bootstrap_stability(df: pd.DataFrame, features: list[str], n_boot: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    sample_size = int(len(df) * 0.8)
    rank_sum = pd.Series(0.0, index=features)
    top5_count = pd.Series(0.0, index=features)

    for _ in range(n_boot):
        indices = rng.integers(0, len(df), size=sample_size)
        boot = df.iloc[indices].reset_index(drop=True)
        score = auxiliary_core_score(boot, features)
        ranks = score.rank(ascending=False, method="average")
        rank_sum += ranks
        top5_count += (ranks <= 5).astype(float)

    stability = pd.DataFrame(
        {
            "指标": features,
            "Bootstrap平均名次": rank_sum.values / n_boot,
            "Top5入选频率": top5_count.values / n_boot,
        }
    )
    stability["稳定性得分"] = minmax_scale(stability["Top5入选频率"])
    return stability


def build_auxiliary_ranking(feature_stats: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    auxiliary = feature_stats[feature_stats["指标"].isin(AUXILIARY_WARNING_FEATURES)].copy()
    auxiliary["相关性显著度"] = -np.log10(np.minimum(auxiliary["Pearson_FDR"], auxiliary["Spearman_FDR"]) + 1e-12)
    auxiliary["组间显著度"] = -np.log10(auxiliary["组间FDR"] + 1e-12)
    auxiliary["Borda_痰湿解释"] = (
        rank_to_borda(auxiliary["|Pearson_r|"], ascending=False)
        + rank_to_borda(auxiliary["|Spearman_r|"], ascending=False)
        + rank_to_borda(auxiliary["相关性显著度"], ascending=False)
    ) / 3
    auxiliary["Borda_预警能力"] = (
        rank_to_borda(auxiliary["预警AUC增益"], ascending=False)
        + rank_to_borda(auxiliary["|Cohen_d|"], ascending=False)
        + rank_to_borda(auxiliary["组间显著度"], ascending=False)
    ) / 3
    auxiliary = auxiliary.merge(stability, on="指标", how="left")
    auxiliary["综合得分"] = (
        0.4 * minmax_scale(auxiliary["Borda_痰湿解释"])
        + 0.4 * minmax_scale(auxiliary["Borda_预警能力"])
        + 0.2 * auxiliary["稳定性得分"]
    )
    auxiliary["分层结论"] = np.where(auxiliary["组间FDR"] <= 0.05, "辅助预警候选", "统计边缘候选")
    return auxiliary.sort_values("综合得分", ascending=False).reset_index(drop=True)


def compute_vif_table(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    x = sm.add_constant(df[features])
    rows = []
    for index, feature in enumerate(features, start=1):
        try:
            vif = variance_inflation_factor(x.values, index)
        except Exception:
            vif = np.inf
        rows.append({"指标": feature, "VIF": vif})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)


def constitution_corr_pairs(df: pd.DataFrame) -> pd.DataFrame:
    corr = df[CONSTITUTION_FEATURES].corr()
    rows = []
    for idx, left in enumerate(CONSTITUTION_FEATURES):
        for right in CONSTITUTION_FEATURES[idx + 1 :]:
            rows.append(
                {
                    "体质A": left,
                    "体质B": right,
                    "相关系数": float(corr.loc[left, right]),
                    "|相关系数|": abs(float(corr.loc[left, right])),
                }
            )
    return pd.DataFrame(rows).sort_values("|相关系数|", ascending=False).reset_index(drop=True)


def constitution_prevalence_table(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(COLS.constitution_label)[COLS.diagnosis]
        .agg(["count", "mean"])
        .rename(columns={"count": "人数", "mean": "高血脂患病率"})
        .reset_index()
    )
    grouped["体质"] = grouped[COLS.constitution_label].map(CONSTITUTION_NAME_MAP)
    grouped["高血脂患病率"] = grouped["高血脂患病率"].round(4)
    return grouped[["体质", "人数", "高血脂患病率"]]


def constitution_logit_table(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    selected_features = CONSTITUTION_FEATURES + ACTIVITY_ITEM_FEATURES + METABOLIC_FEATURES + DEMOGRAPHIC_FEATURES
    x = df[selected_features].copy()
    x[selected_features] = StandardScaler().fit_transform(x[selected_features])
    x = sm.add_constant(x)
    model = sm.GLM(df[COLS.diagnosis], x, family=sm.families.Binomial()).fit()
    table = model.summary2().tables[1].reset_index().rename(columns={"index": "变量"})
    table["OR"] = np.exp(table["Coef."])
    table["OR下界"] = np.exp(table["[0.025"])
    table["OR上界"] = np.exp(table["0.975]"])
    result = table[table["变量"].isin(CONSTITUTION_FEATURES)].copy()
    result["显著性"] = np.where(result["P>|z|"] <= 0.05, "显著", "不显著")
    result = result.rename(columns={"Coef.": "回归系数", "Std.Err.": "标准误", "z": "z值", "P>|z|": "p值"})
    result["|回归系数|"] = result["回归系数"].abs()
    return result.sort_values("|回归系数|", ascending=False).reset_index(drop=True), float(model.aic)


def run_l1_logit_experiment(
    df: pd.DataFrame,
    name: str,
    features: list[str],
    standardize: bool,
    include_total_scores: bool,
    include_direct_lipids: bool,
) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    steps = []
    if standardize:
        steps.append(("scaler", StandardScaler()))
    steps.append(
        (
            "model",
            LogisticRegressionCV(
                Cs=20,
                cv=cv,
                penalty="l1",
                solver="liblinear",
                scoring="roc_auc",
                max_iter=5000,
                random_state=RANDOM_SEED,
            ),
        )
    )
    pipeline = Pipeline(steps)
    x = df[features]
    y = df[COLS.diagnosis]
    oof_proba = cross_val_predict(pipeline, x, y, cv=cv, method="predict_proba")[:, 1]
    oof_pred = (oof_proba >= 0.5).astype(int)
    pipeline.fit(x, y)
    coef = pipeline.named_steps["model"].coef_[0]
    selected = [(feature, float(weight)) for feature, weight in zip(features, coef) if abs(weight) > 1e-8]
    selected_sorted = sorted(selected, key=lambda item: abs(item[1]), reverse=True)
    return {
        "配置": name,
        "是否标准化": "是" if standardize else "否",
        "是否含活动总分": "是" if include_total_scores else "否",
        "是否含直接血脂": "是" if include_direct_lipids else "否",
        "特征数": len(features),
        "AUC": float(roc_auc_score(y, oof_proba)),
        "F1": float(f1_score(y, oof_pred)),
        "非零特征数": len(selected_sorted),
        "主要选中变量": "；".join(f"{feature}({weight:.4f})" for feature, weight in selected_sorted[:8]) if selected_sorted else "无",
    }


def run_lasso_experiment(
    df: pd.DataFrame,
    name: str,
    features: list[str],
    standardize: bool,
    include_lipids: bool,
) -> dict:
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    steps = []
    if standardize:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", LassoCV(cv=cv, random_state=RANDOM_SEED, max_iter=20000)))
    pipeline = Pipeline(steps)
    x = df[features]
    y = df[COLS.tanshi]
    oof_pred = cross_val_predict(pipeline, x, y, cv=cv)
    pipeline.fit(x, y)
    coef = pipeline.named_steps["model"].coef_
    selected = [(feature, float(weight)) for feature, weight in zip(features, coef) if abs(weight) > 1e-8]
    selected_sorted = sorted(selected, key=lambda item: abs(item[1]), reverse=True)
    return {
        "配置": name,
        "是否标准化": "是" if standardize else "否",
        "是否含直接血脂": "是" if include_lipids else "否",
        "特征数": len(features),
        "OOF_R2": float(r2_score(y, oof_pred)),
        "OOF_RMSE": float(np.sqrt(mean_squared_error(y, oof_pred))),
        "alpha": float(pipeline.named_steps["model"].alpha_),
        "非零特征数": len(selected_sorted),
        "主要选中变量": "；".join(f"{feature}({weight:.4f})" for feature, weight in selected_sorted[:8]) if selected_sorted else "无",
    }


def plot_indicator_screening(direct_ranking: pd.DataFrame, auxiliary_ranking: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    direct_plot = direct_ranking.sort_values("综合得分")
    axes[0].barh(direct_plot["指标"], direct_plot["综合得分"], color="#C44E52")
    axes[0].set_title("直接诊断层关键指标")
    axes[0].set_xlabel("综合得分")

    aux_plot = auxiliary_ranking.head(10).sort_values("综合得分")
    colors = ["#4C72B0" if value <= 0.05 else "#B0B0B0" for value in aux_plot["组间FDR"]]
    axes[1].barh(aux_plot["指标"], aux_plot["综合得分"], color=colors)
    axes[1].set_title("辅助预警层 Top10 指标")
    axes[1].set_xlabel("综合得分")

    fig.tight_layout()
    fig.savefig(figure_path("problem1_indicator_screening.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_constitution_effects(prevalence: pd.DataFrame, constitution_logit: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    axes[0].bar(prevalence["体质"], prevalence["高血脂患病率"], color="#55A868")
    axes[0].set_title("九种体质原始患病率")
    axes[0].set_ylabel("患病率")
    axes[0].tick_params(axis="x", rotation=40)

    forest = constitution_logit.sort_values("OR")
    lower_error = forest["OR"] - forest["OR下界"]
    upper_error = forest["OR上界"] - forest["OR"]
    axes[1].errorbar(
        forest["OR"],
        forest["变量"],
        xerr=[lower_error, upper_error],
        fmt="o",
        color="#8172B3",
        ecolor="#8172B3",
        capsize=3,
    )
    axes[1].axvline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_title("体质积分调整后 OR (95%CI)")
    axes[1].set_xlabel("OR")

    fig.tight_layout()
    fig.savefig(figure_path("problem1_constitution_effects.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_collinearity_audit(activity_vif_raw: pd.DataFrame, activity_vif_clean: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    raw_plot = activity_vif_raw.copy()
    raw_plot["绘图VIF"] = raw_plot["VIF"].replace(np.inf, 1000)
    axes[0].barh(raw_plot["指标"], raw_plot["绘图VIF"], color="#DD8452")
    axes[0].set_title("活动变量原始 VIF 审计")
    axes[0].set_xlabel("VIF（inf 以 1000 截断显示）")

    clean_plot = activity_vif_clean.sort_values("VIF")
    axes[1].barh(clean_plot["指标"], clean_plot["VIF"], color="#55A868")
    axes[1].set_title("活动变量清洗后 VIF")
    axes[1].set_xlabel("VIF")

    fig.tight_layout()
    fig.savefig(figure_path("problem1_collinearity_audit.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_plotting()
    set_random_seed()
    df = load_data()

    print("=" * 72)
    print("问题1：共线性纠偏后的双层指标筛选与稳健性检验")
    print("=" * 72)

    all_stats = compute_feature_statistics(df, DIRECT_DIAGNOSTIC_FEATURES + AUXILIARY_WARNING_FEATURES)
    direct_ranking = build_direct_ranking(all_stats)
    stability = bootstrap_stability(df, AUXILIARY_WARNING_FEATURES)
    auxiliary_ranking = build_auxiliary_ranking(all_stats, stability)

    activity_vif_raw = compute_vif_table(df, ACTIVITY_FEATURES)
    activity_vif_clean = compute_vif_table(df, ACTIVITY_ITEM_FEATURES)
    constitution_vif = compute_vif_table(df, CONSTITUTION_FEATURES)
    constitution_pairs = constitution_corr_pairs(df)
    prevalence = constitution_prevalence_table(df)
    constitution_logit, model_aic = constitution_logit_table(df)

    l1logit_rows = [
        run_l1_logit_experiment(
            df,
            "正式口径：标准化+去总分+去直接血脂",
            CONSTITUTION_FEATURES + ACTIVITY_ITEM_FEATURES + METABOLIC_FEATURES + DEMOGRAPHIC_FEATURES,
            standardize=True,
            include_total_scores=False,
            include_direct_lipids=False,
        ),
        run_l1_logit_experiment(
            df,
            "泄漏对照：标准化+去总分+保留直接血脂",
            CONSTITUTION_FEATURES + ACTIVITY_ITEM_FEATURES + METABOLIC_FEATURES + DEMOGRAPHIC_FEATURES + LIPID_FEATURES,
            standardize=True,
            include_total_scores=False,
            include_direct_lipids=True,
        ),
        run_l1_logit_experiment(
            df,
            "共线性对照：标准化+保留总分+去直接血脂",
            CONSTITUTION_FEATURES + ACTIVITY_FEATURES + METABOLIC_FEATURES + DEMOGRAPHIC_FEATURES,
            standardize=True,
            include_total_scores=True,
            include_direct_lipids=False,
        ),
        run_l1_logit_experiment(
            df,
            "标准化对照：不标准化+去总分+去直接血脂",
            CONSTITUTION_FEATURES + ACTIVITY_ITEM_FEATURES + METABOLIC_FEATURES + DEMOGRAPHIC_FEATURES,
            standardize=False,
            include_total_scores=False,
            include_direct_lipids=False,
        ),
    ]
    l1logit_compare = pd.DataFrame(l1logit_rows).sort_values("AUC", ascending=False).reset_index(drop=True)

    lasso_rows = [
        run_lasso_experiment(
            df,
            "标准化+去总分+含血脂",
            ACTIVITY_ITEM_FEATURES + METABOLIC_FEATURES + LIPID_FEATURES,
            standardize=True,
            include_lipids=True,
        ),
        run_lasso_experiment(
            df,
            "标准化+去总分+不含血脂",
            ACTIVITY_ITEM_FEATURES + METABOLIC_FEATURES,
            standardize=True,
            include_lipids=False,
        ),
    ]
    lasso_compare = pd.DataFrame(lasso_rows).sort_values("OOF_R2", ascending=False).reset_index(drop=True)

    save_workbook(
        "problem1_tables",
        {
            "全量特征统计": all_stats.sort_values("预警AUC增益", ascending=False).reset_index(drop=True),
            "直接诊断层": direct_ranking,
            "辅助预警层": auxiliary_ranking,
            "活动VIF原始": activity_vif_raw,
            "活动VIF清洗后": activity_vif_clean,
            "体质VIF": constitution_vif,
            "体质相关对": constitution_pairs,
            "L1Logit诊断对照": l1logit_compare,
            "Lasso痰湿对照": lasso_compare,
            "体质患病率": prevalence,
            "体质Logit": constitution_logit,
        },
    )
    save_table(direct_ranking, "problem1_direct_ranking")
    save_table(auxiliary_ranking, "problem1_auxiliary_ranking")
    save_table(activity_vif_raw, "problem1_activity_vif_raw")
    save_table(activity_vif_clean, "problem1_activity_vif_clean")
    save_table(activity_vif_clean, "problem1_vif")
    save_table(constitution_vif, "problem1_constitution_vif")
    save_table(constitution_pairs, "problem1_constitution_corr_pairs")
    save_table(l1logit_compare, "problem1_l1logit_diagnosis_compare")
    save_table(lasso_compare, "problem1_lasso_tanshi_compare")
    save_table(constitution_logit, "problem1_constitution_logit")

    plot_indicator_screening(direct_ranking, auxiliary_ranking)
    plot_constitution_effects(prevalence, constitution_logit)
    plot_collinearity_audit(activity_vif_raw, activity_vif_clean)

    summary = {
        "top_direct_indicators": direct_ranking["指标"].head(4).tolist(),
        "top_auxiliary_indicators": auxiliary_ranking["指标"].head(6).tolist(),
        "raw_activity_vif_has_inf": bool(np.isinf(activity_vif_raw["VIF"]).any()),
        "clean_activity_vif_max": float(activity_vif_clean["VIF"].replace(np.inf, np.nan).max()),
        "constitution_vif_max": float(constitution_vif["VIF"].replace(np.inf, np.nan).max()),
        "max_constitution_abs_corr": float(constitution_pairs["|相关系数|"].max()),
        "formal_l1logit_auc": float(
            l1logit_compare.loc[l1logit_compare["配置"] == "正式口径：标准化+去总分+去直接血脂", "AUC"].iloc[0]
        ),
        "leakage_l1logit_auc": float(
            l1logit_compare.loc[l1logit_compare["配置"] == "泄漏对照：标准化+去总分+保留直接血脂", "AUC"].iloc[0]
        ),
        "tanshi_lasso_best_oof_r2": float(lasso_compare["OOF_R2"].max()),
        "constitution_model_aic": round(model_aic, 4),
    }
    save_json(summary, "problem1_summary")

    print("\n活动变量原始 VIF：")
    print(activity_vif_raw.to_string(index=False))
    print("\n活动变量清洗后 VIF：")
    print(activity_vif_clean.to_string(index=False))
    print("\n九种体质 VIF：")
    print(constitution_vif.to_string(index=False))
    print("\nL1-Logit 诊断对照：")
    print(l1logit_compare[["配置", "AUC", "F1", "非零特征数", "主要选中变量"]].to_string(index=False))
    print("\nLasso 痰湿对照：")
    print(lasso_compare[["配置", "OOF_R2", "OOF_RMSE", "非零特征数", "主要选中变量"]].to_string(index=False))
    print("\n图表与表格已输出到 src/outputs 目录。")


if __name__ == "__main__":
    main()
