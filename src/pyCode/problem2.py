from __future__ import annotations

import warnings
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, _tree

from common import (
    ACTIVITY_FEATURES,
    ACTIVITY_ITEM_FEATURES,
    COLS,
    CONSTITUTION_FEATURES,
    DEMOGRAPHIC_FEATURES,
    LINEAR_EARLY_SCREENING_FEATURES,
    LIPID_NORMAL_RANGES,
    METABOLIC_FEATURES,
    RANDOM_SEED,
    TREE_ACTIVITY_FEATURES,
    TREE_EARLY_SCREENING_FEATURES,
    configure_plotting,
    critic_weights,
    describe_age_group,
    describe_sex,
    find_sample_data_path,
    figure_path,
    get_lipid_flags,
    get_metabolic_flags,
    load_data,
    minmax_scale,
    save_json,
    save_table,
    save_workbook,
    set_random_seed,
)


warnings.filterwarnings("ignore")


RISK_COMPONENTS = [
    "lipid_burden",
    "lipid_excess",
    "early_risk",
    "metabolic_modifier",
    "function_modifier",
]
REPEATED_SPLITS = 5
REPEATED_REPEATS = 5
FIXED_SPLITS = 5
CI_BOOTSTRAPS = 500
THRESHOLD_BOOTSTRAPS = 200
RULE_BOOTSTRAPS = 100
AUC_DROP_TOLERANCE = 0.005


@dataclass
class ModelSpec:
    name: str
    pipeline: object
    features: list[str]
    feature_profile: str
    family: str


@dataclass
class ModelResult:
    name: str
    pipeline: object
    features: list[str]
    feature_profile: str
    oof_proba: np.ndarray
    oof_pred: np.ndarray
    metrics: dict[str, float]


def build_model_specs() -> dict[str, ModelSpec]:
    return {
        "Logistic回归": ModelSpec(
            name="Logistic回归",
            pipeline=Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=5000,
                            class_weight="balanced",
                            random_state=RANDOM_SEED,
                        ),
                    ),
                ]
            ),
            features=LINEAR_EARLY_SCREENING_FEATURES,
            feature_profile="线性清洗口径",
            family="logistic",
        ),
        "随机森林": ModelSpec(
            name="随机森林",
            pipeline=Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=500,
                            max_depth=6,
                            min_samples_leaf=8,
                            class_weight="balanced",
                            random_state=RANDOM_SEED,
                        ),
                    ),
                ]
            ),
            features=TREE_EARLY_SCREENING_FEATURES,
            feature_profile="树模型清洗口径",
            family="rf",
        ),
        "梯度提升树": ModelSpec(
            name="梯度提升树",
            pipeline=Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        GradientBoostingClassifier(
                            learning_rate=0.05,
                            n_estimators=180,
                            max_depth=3,
                            random_state=RANDOM_SEED,
                        ),
                    ),
                ]
            ),
            features=TREE_EARLY_SCREENING_FEATURES,
            feature_profile="树模型清洗口径",
            family="gbdt",
        ),
        "极端随机树": ModelSpec(
            name="极端随机树",
            pipeline=Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        ExtraTreesClassifier(
                            n_estimators=500,
                            max_depth=6,
                            min_samples_leaf=6,
                            class_weight="balanced",
                            random_state=RANDOM_SEED,
                        ),
                    ),
                ]
            ),
            features=TREE_EARLY_SCREENING_FEATURES,
            feature_profile="树模型清洗口径",
            family="extra_trees",
        ),
        "直方梯度提升树": ModelSpec(
            name="直方梯度提升树",
            pipeline=Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            learning_rate=0.05,
                            max_depth=4,
                            max_leaf_nodes=31,
                            min_samples_leaf=20,
                            random_state=RANDOM_SEED,
                        ),
                    ),
                ]
            ),
            features=TREE_EARLY_SCREENING_FEATURES,
            feature_profile="树模型清洗口径",
            family="hist_gbdt",
        ),
    }


def build_calibrated_spec(spec: ModelSpec) -> ModelSpec:
    return ModelSpec(
        name=f"{spec.name}+Sigmoid校准",
        pipeline=CalibratedClassifierCV(
            estimator=clone(spec.pipeline),
            method="sigmoid",
            cv=3,
        ),
        features=spec.features,
        feature_profile=spec.feature_profile,
        family=spec.family,
    )


def evaluate_predictions(y_true: pd.Series, proba: np.ndarray) -> dict[str, float]:
    pred = (proba >= 0.5).astype(int)
    return {
        "AUC": roc_auc_score(y_true, proba),
        "PR_AUC": average_precision_score(y_true, proba),
        "Brier分数": brier_score_loss(y_true, proba),
        "准确率": accuracy_score(y_true, pred),
        "F1": f1_score(y_true, pred),
        "精确率": precision_score(y_true, pred),
        "召回率": recall_score(y_true, pred),
    }


def evaluate_repeated_cv(df: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    x = df[spec.features]
    y = df[COLS.diagnosis]
    cv = RepeatedStratifiedKFold(
        n_splits=REPEATED_SPLITS,
        n_repeats=REPEATED_REPEATS,
        random_state=RANDOM_SEED,
    )
    rows = []
    for split_id, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
        estimator = clone(spec.pipeline)
        estimator.fit(x.iloc[train_idx], y.iloc[train_idx])
        proba = estimator.predict_proba(x.iloc[test_idx])[:, 1]
        metrics = evaluate_predictions(y.iloc[test_idx], proba)
        rows.append(
            {
                "模型": spec.name,
                "模型家族": spec.family,
                "特征数": len(spec.features),
                "特征口径": spec.feature_profile,
                "重复轮次": (split_id - 1) // REPEATED_SPLITS + 1,
                "折次": (split_id - 1) % REPEATED_SPLITS + 1,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def summarize_cv_detail(detail: pd.DataFrame) -> pd.DataFrame:
    summary = (
        detail.groupby("模型")
        .agg(
            模型家族=("模型家族", "first"),
            特征数=("特征数", "first"),
            特征口径=("特征口径", "first"),
            AUC均值=("AUC", "mean"),
            AUC标准差=("AUC", "std"),
            PR_AUC均值=("PR_AUC", "mean"),
            PR_AUC标准差=("PR_AUC", "std"),
            Brier均值=("Brier分数", "mean"),
            Brier标准差=("Brier分数", "std"),
            F1均值=("F1", "mean"),
            F1标准差=("F1", "std"),
            准确率均值=("准确率", "mean"),
            精确率均值=("精确率", "mean"),
            召回率均值=("召回率", "mean"),
        )
        .reset_index()
    )
    summary["AUC排序"] = summary["AUC均值"].rank(ascending=False, method="min")
    summary["F1排序"] = summary["F1均值"].rank(ascending=False, method="min")
    summary["Brier排序"] = summary["Brier均值"].rank(ascending=True, method="min")
    summary["综合排序分"] = summary["AUC排序"] + summary["F1排序"] + summary["Brier排序"]
    return summary.sort_values(
        ["综合排序分", "AUC均值", "AUC标准差"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def choose_best_family(summary: pd.DataFrame, specs: dict[str, ModelSpec]) -> ModelSpec:
    best_name = summary.iloc[0]["模型"]
    return specs[best_name]


def evaluate_calibration_compare(df: pd.DataFrame, base_spec: ModelSpec) -> tuple[pd.DataFrame, ModelSpec]:
    raw_detail = evaluate_repeated_cv(df, base_spec)
    calibrated_spec = build_calibrated_spec(base_spec)
    calibrated_detail = evaluate_repeated_cv(df, calibrated_spec)
    compare = summarize_cv_detail(pd.concat([raw_detail, calibrated_detail], ignore_index=True))
    raw_row = compare[compare["模型"] == base_spec.name].iloc[0]
    cal_row = compare[compare["模型"] == calibrated_spec.name].iloc[0]
    use_calibrated = bool(
        (cal_row["Brier均值"] < raw_row["Brier均值"])
        and (cal_row["AUC均值"] >= raw_row["AUC均值"] - AUC_DROP_TOLERANCE)
    )

    compare["是否采纳"] = "否"
    compare["选择理由"] = "原始版作为基线"
    if use_calibrated:
        compare.loc[compare["模型"] == calibrated_spec.name, "是否采纳"] = "是"
        compare.loc[compare["模型"] == calibrated_spec.name, "选择理由"] = "Brier更优且AUC未明显下降"
        compare.loc[compare["模型"] == base_spec.name, "选择理由"] = "校准后Brier更优"
        return compare, calibrated_spec

    compare.loc[compare["模型"] == base_spec.name, "是否采纳"] = "是"
    compare.loc[compare["模型"] == base_spec.name, "选择理由"] = "校准收益有限，保留原始版"
    compare.loc[compare["模型"] == calibrated_spec.name, "选择理由"] = "Brier改善不足或AUC下降"
    return compare, base_spec


def compute_fixed_oof_result(df: pd.DataFrame, spec: ModelSpec) -> ModelResult:
    x = df[spec.features]
    y = df[COLS.diagnosis]
    cv = StratifiedKFold(n_splits=FIXED_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    oof_proba = np.zeros(len(df), dtype=float)

    for train_idx, test_idx in cv.split(x, y):
        estimator = clone(spec.pipeline)
        estimator.fit(x.iloc[train_idx], y.iloc[train_idx])
        oof_proba[test_idx] = estimator.predict_proba(x.iloc[test_idx])[:, 1]

    oof_pred = (oof_proba >= 0.5).astype(int)
    metrics = evaluate_predictions(y, oof_proba)
    return ModelResult(
        name=spec.name,
        pipeline=spec.pipeline,
        features=spec.features,
        feature_profile=spec.feature_profile,
        oof_proba=oof_proba,
        oof_pred=oof_pred,
        metrics=metrics,
    )


def bootstrap_metric_intervals(
    y_true: pd.Series,
    proba: np.ndarray,
    pred: np.ndarray,
    n_boot: int = CI_BOOTSTRAPS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    y = y_true.to_numpy()
    records = []
    valid = 0
    while valid < n_boot:
        sample_idx = rng.integers(0, len(y), len(y))
        sample_y = y[sample_idx]
        if np.unique(sample_y).size < 2:
            continue
        sample_proba = proba[sample_idx]
        sample_pred = pred[sample_idx]
        records.append(
            {
                "bootstrap_id": valid + 1,
                "AUC": roc_auc_score(sample_y, sample_proba),
                "PR_AUC": average_precision_score(sample_y, sample_proba),
                "Brier分数": brier_score_loss(sample_y, sample_proba),
                "F1": f1_score(sample_y, sample_pred),
            }
        )
        valid += 1

    detail = pd.DataFrame(records)
    point = {
        "AUC": roc_auc_score(y, proba),
        "PR_AUC": average_precision_score(y, proba),
        "Brier分数": brier_score_loss(y, proba),
        "F1": f1_score(y, pred),
    }
    summary_rows = []
    for metric_name, point_value in point.items():
        values = detail[metric_name]
        summary_rows.append(
            {
                "指标": metric_name,
                "点估计": point_value,
                "均值": values.mean(),
                "标准差": values.std(ddof=1),
                "CI下界": values.quantile(0.025),
                "CI上界": values.quantile(0.975),
            }
        )
    return pd.DataFrame(summary_rows), detail


def build_model_summary_with_ci(
    selected_spec: ModelSpec,
    selected_result: ModelResult,
    ci_summary: pd.DataFrame,
    calibration_compare: pd.DataFrame,
) -> pd.DataFrame:
    calibrated = "是" if "Sigmoid校准" in selected_spec.name else "否"
    ci_map = ci_summary.set_index("指标").to_dict("index")
    raw_name = selected_spec.name.replace("+Sigmoid校准", "")
    repeated_row = calibration_compare[calibration_compare["模型"] == selected_spec.name].iloc[0]
    return pd.DataFrame(
        [
            {
                "最终主模型": selected_spec.name,
                "主模型家族": raw_name,
                "是否采用校准": calibrated,
                "特征口径": selected_spec.feature_profile,
                "特征数": len(selected_spec.features),
                "RepeatedCV_AUC均值": repeated_row["AUC均值"],
                "RepeatedCV_AUC标准差": repeated_row["AUC标准差"],
                "RepeatedCV_Brier均值": repeated_row["Brier均值"],
                "RepeatedCV_F1均值": repeated_row["F1均值"],
                "固定5折OOF_AUC": selected_result.metrics["AUC"],
                "AUC_95CI": f"[{ci_map['AUC']['CI下界']:.4f}, {ci_map['AUC']['CI上界']:.4f}]",
                "固定5折OOF_PR_AUC": selected_result.metrics["PR_AUC"],
                "PR_AUC_95CI": f"[{ci_map['PR_AUC']['CI下界']:.4f}, {ci_map['PR_AUC']['CI上界']:.4f}]",
                "固定5折OOF_Brier": selected_result.metrics["Brier分数"],
                "Brier_95CI": f"[{ci_map['Brier分数']['CI下界']:.4f}, {ci_map['Brier分数']['CI上界']:.4f}]",
                "固定5折OOF_F1": selected_result.metrics["F1"],
                "F1_95CI": f"[{ci_map['F1']['CI下界']:.4f}, {ci_map['F1']['CI上界']:.4f}]",
            }
        ]
    )


def get_feature_blocks(feature_profile: str) -> dict[str, list[str]]:
    activity_block = TREE_ACTIVITY_FEATURES if feature_profile == "树模型清洗口径" else ACTIVITY_ITEM_FEATURES
    return {
        "体质": CONSTITUTION_FEATURES,
        "活动/功能": activity_block,
        "代谢": METABOLIC_FEATURES,
        "人口学": DEMOGRAPHIC_FEATURES,
    }


def run_ablation_study(df: pd.DataFrame, final_spec: ModelSpec) -> pd.DataFrame:
    blocks = get_feature_blocks(final_spec.feature_profile)
    scenarios = {
        "全特征": final_spec.features,
        "仅体质": blocks["体质"],
        "仅活动/功能": blocks["活动/功能"],
        "仅代谢": blocks["代谢"],
        "仅人口学": blocks["人口学"],
        "去体质": blocks["活动/功能"] + blocks["代谢"] + blocks["人口学"],
        "去活动": blocks["体质"] + blocks["代谢"] + blocks["人口学"],
        "去代谢": blocks["体质"] + blocks["活动/功能"] + blocks["人口学"],
        "去人口学": blocks["体质"] + blocks["活动/功能"] + blocks["代谢"],
    }

    rows = []
    for scenario_name, features in scenarios.items():
        scenario_spec = ModelSpec(
            name=f"{final_spec.name}-{scenario_name}",
            pipeline=clone(final_spec.pipeline),
            features=features,
            feature_profile=final_spec.feature_profile,
            family=final_spec.family,
        )
        result = compute_fixed_oof_result(df, scenario_spec)
        rows.append(
            {
                "方案": scenario_name,
                "特征数": len(features),
                "AUC": result.metrics["AUC"],
                "PR_AUC": result.metrics["PR_AUC"],
                "Brier分数": result.metrics["Brier分数"],
                "F1": result.metrics["F1"],
            }
        )
    frame = pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)
    full_auc = frame.loc[frame["方案"] == "全特征", "AUC"].iloc[0]
    frame["相对全特征AUC变化"] = frame["AUC"] - full_auc
    return frame


def feature_direction(df: pd.DataFrame, feature: str) -> str:
    case_mean = df.loc[df[COLS.diagnosis] == 1, feature].mean()
    control_mean = df.loc[df[COLS.diagnosis] == 0, feature].mean()
    return "值越高风险越高" if case_mean >= control_mean else "值越低风险越高"


def interpret_feature(feature: str) -> str:
    if feature == COLS.uric_acid:
        return "尿酸异常提示代谢背景恶化，是轨道A最稳定的早筛信号"
    if feature == COLS.bmi:
        return "BMI偏离正常范围会放大代谢性风险"
    if feature == COLS.glucose:
        return "空腹血糖偏离正常范围提示糖脂代谢共病风险"
    if feature in CONSTITUTION_FEATURES:
        return "体质偏颇提供慢性易感背景，不直接等同于诊断标准"
    if feature in ACTIVITY_FEATURES:
        return "活动能力下降反映功能受限与生活方式风险累积"
    return "基础人口学变量提供分层修正信息"


def compute_permutation_importance(df: pd.DataFrame, model_result: ModelResult) -> pd.DataFrame:
    x = df[model_result.features]
    y = df[COLS.diagnosis]
    fitted = clone(model_result.pipeline)
    fitted.fit(x, y)
    importance = permutation_importance(
        fitted,
        x,
        y,
        scoring="roc_auc",
        n_repeats=15,
        random_state=RANDOM_SEED,
    )
    frame = (
        pd.DataFrame(
            {
                "特征": model_result.features,
                "置换重要性": importance.importances_mean,
            }
        )
        .sort_values("置换重要性", ascending=False)
        .reset_index(drop=True)
    )
    frame["风险方向"] = frame["特征"].map(lambda item: feature_direction(df, item))
    frame["证据来源"] = "最终主模型置换重要性 + 分位数风险表"
    frame["临床含义"] = frame["特征"].map(interpret_feature)
    return frame


def build_quantile_effect_table(df: pd.DataFrame, model_result: ModelResult, importance: pd.DataFrame) -> pd.DataFrame:
    chosen = [COLS.uric_acid, COLS.bmi, COLS.glucose]
    extra_feature = None
    for feature in importance["特征"]:
        if feature in chosen or feature in DEMOGRAPHIC_FEATURES:
            continue
        if df[feature].nunique() <= 6:
            continue
        extra_feature = feature
        break
    if extra_feature is not None:
        chosen.append(extra_feature)

    rows = []
    p_early = model_result.oof_proba
    for feature in chosen:
        try:
            bins = pd.qcut(df[feature], q=5, duplicates="drop")
        except ValueError:
            continue
        summary = (
            pd.DataFrame(
                {
                    "feature": df[feature],
                    "quantile": bins.astype(str),
                    "p_early": p_early,
                    "label": df[COLS.diagnosis],
                }
            )
            .groupby("quantile")
            .agg(
                样本数=("feature", "size"),
                特征均值=("feature", "mean"),
                p_early均值=("p_early", "mean"),
                确诊率=("label", "mean"),
            )
            .reset_index()
        )
        summary.insert(0, "特征", feature)
        summary = summary.rename(columns={"quantile": "分位组"})
        rows.append(summary)
    if not rows:
        return pd.DataFrame(columns=["特征", "分位组", "样本数", "特征均值", "p_early均值", "确诊率"])
    return pd.concat(rows, ignore_index=True)


def compute_subgroup_stability(df: pd.DataFrame, model_result: ModelResult) -> pd.DataFrame:
    rows = []
    subgroup_specs = [
        ("性别", [(0, describe_sex(0)), (1, describe_sex(1))]),
        ("年龄组", [(code, describe_age_group(code)) for code in sorted(df[COLS.age_group].unique())]),
        ("痰湿体质", [(0, "非痰湿体质"), (1, "痰湿体质")]),
    ]

    indicator_map = {
        "性别": COLS.sex,
        "年龄组": COLS.age_group,
    }

    for group_name, group_values in subgroup_specs:
        for value, label in group_values:
            if group_name == "痰湿体质":
                mask = (df[COLS.constitution_label] == 5) if value == 1 else (df[COLS.constitution_label] != 5)
            else:
                mask = df[indicator_map[group_name]] == value

            subgroup = df.loc[mask]
            subgroup_y = subgroup[COLS.diagnosis]
            subgroup_proba = model_result.oof_proba[mask.to_numpy()]
            subgroup_n = int(mask.sum())
            pos_n = int(subgroup_y.sum())
            neg_n = subgroup_n - pos_n
            note = ""
            auc = np.nan
            if subgroup_n < 40 or min(pos_n, neg_n) < 10:
                note = "仅供参考"
            if subgroup_y.nunique() == 2:
                auc = roc_auc_score(subgroup_y, subgroup_proba)
            else:
                note = "类别单一，仅供参考"
            rows.append(
                {
                    "分组维度": group_name,
                    "亚组": label,
                    "样本数": subgroup_n,
                    "阳性数": pos_n,
                    "阴性数": neg_n,
                    "AUC": auc,
                    "Brier分数": brier_score_loss(subgroup_y, subgroup_proba),
                    "备注": note or "稳定",
                }
            )
    return pd.DataFrame(rows)


def build_clinical_risk_score(
    df: pd.DataFrame,
    p_early: np.ndarray,
    weights: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lipid_flags = get_lipid_flags(df)
    metabolic_flags = get_metabolic_flags(df)

    severity_raw = pd.DataFrame(index=df.index)
    for feature, (lower, upper) in LIPID_NORMAL_RANGES.items():
        if feature == COLS.hdl:
            severity_raw[f"{feature}_超阈比例"] = ((lower - df[feature]).clip(lower=0) / lower)
        else:
            severity_raw[f"{feature}_超阈比例"] = ((df[feature] - upper).clip(lower=0) / upper)

    high_direction_count = lipid_flags[[f"{feature}_高" for feature in [COLS.tc, COLS.tg, COLS.ldl]]].sum(axis=1)
    low_direction_count = lipid_flags[[f"{COLS.hdl}_低"]].sum(axis=1)

    raw = pd.DataFrame(index=df.index)
    raw["lipid_burden_raw"] = (
        lipid_flags["血脂异常项数"] / 4
        + 0.25 * (high_direction_count > 0).astype(int)
        + 0.25 * (low_direction_count > 0).astype(int)
    )
    raw["lipid_excess_raw"] = severity_raw.sum(axis=1)
    raw["early_risk_raw"] = pd.Series(p_early, index=df.index)
    raw["metabolic_modifier_raw"] = metabolic_flags["代谢异常项数"] / 3
    raw["function_modifier_raw"] = (
        0.5 * minmax_scale(df[COLS.tanshi])
        + 0.5 * minmax_scale(100 - df[COLS.activity_total])
    )

    # ---- CRITIC 客观赋权 ----
    if weights is None:
        raw_for_critic = pd.DataFrame({
            "lipid_burden": raw["lipid_burden_raw"],
            "lipid_excess": raw["lipid_excess_raw"],
            "early_risk": raw["early_risk_raw"],
            "metabolic_modifier": raw["metabolic_modifier_raw"],
            "function_modifier": raw["function_modifier_raw"],
        })
        weights = critic_weights(raw_for_critic)
        print(f"  CRITIC 客观赋权结果: {', '.join(f'{k}={v:.4f}' for k, v in weights.items())}")

    component_df = pd.DataFrame(index=df.index)
    for component in RISK_COMPONENTS:
        normalized = minmax_scale(raw[f"{component}_raw"])
        component_df[f"{component}_norm"] = normalized
        component_df[component] = weights[component] * normalized
    component_df["R"] = 100 * component_df[list(RISK_COMPONENTS)].sum(axis=1)

    merged = pd.concat([component_df, raw, lipid_flags, metabolic_flags, severity_raw], axis=1)
    weight_frame = pd.DataFrame(
        [{"分量": key, "CRITIC权重": round(value, 4), "说明": "CRITIC客观赋权法自动计算"} for key, value in weights.items()]
    )
    return merged, weight_frame


def search_thresholds(score: pd.Series, label: pd.Series, min_group_size: int = 120) -> tuple[float, float, pd.DataFrame]:
    values = score.to_numpy(dtype=float)
    labels = label.to_numpy(dtype=float)
    candidates = np.unique(np.round(np.quantile(values, np.linspace(0.12, 0.88, 32)), 4))
    records = []
    best_record = None

    for low in candidates:
        for high in candidates:
            if high <= low:
                continue

            low_mask = values <= low
            mid_mask = (values > low) & (values <= high)
            high_mask = values > high
            counts = np.array([low_mask.sum(), mid_mask.sum(), high_mask.sum()], dtype=int)
            if counts.min() < min_group_size:
                continue

            prevalence = np.array(
                [
                    labels[low_mask].mean(),
                    labels[mid_mask].mean(),
                    labels[high_mask].mean(),
                ]
            )
            if not (prevalence[0] < prevalence[1] < prevalence[2]):
                continue

            score_means = np.array(
                [
                    values[low_mask].mean(),
                    values[mid_mask].mean(),
                    values[high_mask].mean(),
                ]
            )
            score_vars = np.array(
                [
                    values[low_mask].var(ddof=1),
                    values[mid_mask].var(ddof=1),
                    values[high_mask].var(ddof=1),
                ]
            )
            separation = (prevalence[1] - prevalence[0]) + (prevalence[2] - prevalence[1])
            balance_penalty = counts.std(ddof=0) / len(values)
            objective = separation - 0.2 * balance_penalty
            record = {
                "T_low": float(low),
                "T_high": float(high),
                "低风险样本数": int(counts[0]),
                "中风险样本数": int(counts[1]),
                "高风险样本数": int(counts[2]),
                "低风险患病率": float(prevalence[0]),
                "中风险患病率": float(prevalence[1]),
                "高风险患病率": float(prevalence[2]),
                "低风险评分均值": float(score_means[0]),
                "中风险评分均值": float(score_means[1]),
                "高风险评分均值": float(score_means[2]),
                "组间评分均值差_中低": float(score_means[1] - score_means[0]),
                "组间评分均值差_高中": float(score_means[2] - score_means[1]),
                "组间患病率差_中低": float(prevalence[1] - prevalence[0]),
                "组间患病率差_高中": float(prevalence[2] - prevalence[1]),
                "低风险组内方差": float(score_vars[0]),
                "中风险组内方差": float(score_vars[1]),
                "高风险组内方差": float(score_vars[2]),
                "目标函数": float(objective),
            }
            records.append(record)
            if best_record is None or record["目标函数"] > best_record["目标函数"]:
                best_record = record

    if best_record is None:
        raise RuntimeError("未找到满足样本量和单调患病率约束的风险阈值，请调整搜索空间。")

    records_df = pd.DataFrame(records).sort_values("目标函数", ascending=False).reset_index(drop=True)
    return best_record["T_low"], best_record["T_high"], records_df


def assign_risk_level(score: pd.Series, low: float, high: float) -> pd.Series:
    return pd.cut(score, bins=[-np.inf, low, high, np.inf], labels=["低风险", "中风险", "高风险"]).astype(str)


def risk_group_summary(df: pd.DataFrame, score_col: str, group_col: str) -> pd.DataFrame:
    return (
        df.groupby(group_col)
        .agg(
            样本数=(COLS.sample_id, "count"),
            评分均值=(score_col, "mean"),
            评分中位数=(score_col, "median"),
            确诊率=(COLS.diagnosis, "mean"),
        )
        .reindex(["低风险", "中风险", "高风险"])
        .reset_index()
        .rename(columns={group_col: "风险等级"})
    )


def compute_risk_group_stats(df: pd.DataFrame) -> pd.DataFrame:
    low = df.loc[df["风险等级"] == "低风险", "R"]
    mid = df.loc[df["风险等级"] == "中风险", "R"]
    high = df.loc[df["风险等级"] == "高风险", "R"]
    rows = []
    overall = stats.kruskal(low, mid, high)
    rows.append(
        {
            "检验": "Kruskal-Wallis",
            "比较": "低风险 vs 中风险 vs 高风险",
            "统计量": float(overall.statistic),
            "p值": float(overall.pvalue),
        }
    )
    for name, left, right in [
        ("低风险 vs 中风险", low, mid),
        ("中风险 vs 高风险", mid, high),
        ("低风险 vs 高风险", low, high),
    ]:
        test = stats.mannwhitneyu(left, right, alternative="two-sided")
        rows.append(
            {
                "检验": "Mann-Whitney U",
                "比较": name,
                "统计量": float(test.statistic),
                "p值": float(test.pvalue),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_threshold_stability(score: pd.Series, label: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    records = []
    for bootstrap_id in range(1, THRESHOLD_BOOTSTRAPS + 1):
        sample_idx = rng.integers(0, len(score), len(score))
        sampled_score = score.iloc[sample_idx].reset_index(drop=True)
        sampled_label = label.iloc[sample_idx].reset_index(drop=True)
        try:
            low, high, _ = search_thresholds(sampled_score, sampled_label)
            records.append(
                {
                    "bootstrap_id": bootstrap_id,
                    "T_low": low,
                    "T_high": high,
                    "是否成功": "是",
                }
            )
        except RuntimeError:
            records.append(
                {
                    "bootstrap_id": bootstrap_id,
                    "T_low": np.nan,
                    "T_high": np.nan,
                    "是否成功": "否",
                }
            )
    detail = pd.DataFrame(records)
    valid = detail[detail["是否成功"] == "是"].copy()
    summary = pd.DataFrame(
        [
            {
                "指标": "T_low",
                "均值": valid["T_low"].mean(),
                "标准差": valid["T_low"].std(ddof=1),
                "CI下界": valid["T_low"].quantile(0.025),
                "CI上界": valid["T_low"].quantile(0.975),
                "成功率": len(valid) / len(detail),
            },
            {
                "指标": "T_high",
                "均值": valid["T_high"].mean(),
                "标准差": valid["T_high"].std(ddof=1),
                "CI下界": valid["T_high"].quantile(0.025),
                "CI上界": valid["T_high"].quantile(0.975),
                "成功率": len(valid) / len(detail),
            },
        ]
    )
    return summary, detail


def compute_weight_sensitivity(
    base_components: pd.DataFrame,
    label: pd.Series,
    base_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    if base_weights is None:
        raw_for_critic = pd.DataFrame({
            c: base_components[f"{c}_norm"] for c in RISK_COMPONENTS
        })
        base_weights = critic_weights(raw_for_critic)
    rows = []
    scenarios = [("CRITIC基准权重", None, 1.0)]
    for component in RISK_COMPONENTS:
        scenarios.append((f"{component}+20%", component, 1.2))
        scenarios.append((f"{component}-20%", component, 0.8))

    for scenario_name, component, factor in scenarios:
        weights = base_weights.copy()
        if component is not None:
            weights[component] *= factor
            total = sum(weights.values())
            weights = {key: value / total for key, value in weights.items()}

        score = 100 * sum(
            weights[key] * base_components[f"{key}_norm"]
            for key in RISK_COMPONENTS
        )
        try:
            low, high, _ = search_thresholds(score, label)
            groups = assign_risk_level(score, low, high)
            summary = risk_group_summary(
                pd.DataFrame({"R": score, "风险等级": groups, COLS.diagnosis: label, COLS.sample_id: np.arange(len(score))}),
                "R",
                "风险等级",
            )
            prevalence = summary["确诊率"].tolist()
            rows.append(
                {
                    "场景": scenario_name,
                    "扰动分量": component or "无",
                    "权重因子": factor,
                    "T_low": low,
                    "T_high": high,
                    "低风险患病率": prevalence[0],
                    "中风险患病率": prevalence[1],
                    "高风险患病率": prevalence[2],
                    "是否单调": "是" if prevalence[0] < prevalence[1] < prevalence[2] else "否",
                    "权重配置": "；".join(f"{key}={value:.3f}" for key, value in weights.items()),
                }
            )
        except RuntimeError:
            rows.append(
                {
                    "场景": scenario_name,
                    "扰动分量": component or "无",
                    "权重因子": factor,
                    "T_low": np.nan,
                    "T_high": np.nan,
                    "低风险患病率": np.nan,
                    "中风险患病率": np.nan,
                    "高风险患病率": np.nan,
                    "是否单调": "否",
                    "权重配置": "搜索失败",
                }
            )
    return pd.DataFrame(rows)


def verify_label_consistency(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    flags = get_lipid_flags(df)
    agreement = (flags["任一血脂异常"] == df[COLS.diagnosis]).mean()
    table = pd.crosstab(flags["任一血脂异常"], df[COLS.diagnosis], rownames=["任一血脂异常"], colnames=["二分类标签"])
    return table.reset_index(), float(agreement)


def canonicalize_rule_conditions(conditions: list[str]) -> tuple[str, dict[str, dict[str, float]]]:
    bounds: dict[str, dict[str, float]] = {}
    for condition in conditions:
        feature, operator, threshold_text = condition.rsplit(" ", 2)
        threshold = float(threshold_text)
        entry = bounds.setdefault(feature, {})
        if operator == "<=":
            entry["upper"] = min(entry.get("upper", np.inf), threshold)
        else:
            entry["lower"] = max(entry.get("lower", -np.inf), threshold)

    signature_parts = []
    for feature in sorted(bounds):
        has_lower = "lower" in bounds[feature]
        has_upper = "upper" in bounds[feature]
        if has_lower and has_upper:
            signature_parts.append(f"{feature}:between")
        elif has_lower:
            signature_parts.append(f"{feature}:gt")
        else:
            signature_parts.append(f"{feature}:le")
    return " | ".join(signature_parts), bounds


def render_rule(bounds: dict[str, dict[str, float]]) -> str:
    pieces = []
    for feature in sorted(bounds):
        has_lower = "lower" in bounds[feature]
        has_upper = "upper" in bounds[feature]
        if has_lower and has_upper:
            pieces.append(f"{bounds[feature]['lower']:.4f} < {feature} <= {bounds[feature]['upper']:.4f}")
        elif has_lower:
            pieces.append(f"{feature} > {bounds[feature]['lower']:.4f}")
        else:
            pieces.append(f"{feature} <= {bounds[feature]['upper']:.4f}")
    return " 且 ".join(pieces)


def extract_bootstrap_rule_records(
    tree_model: DecisionTreeClassifier,
    feature_names: list[str],
    full_subset: pd.DataFrame,
    full_target: pd.Series,
) -> list[dict[str, object]]:
    tree_ = tree_model.tree_
    records: list[dict[str, object]] = []

    def recurse(node: int, conditions: list[str]) -> None:
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            feature = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            recurse(tree_.children_left[node], conditions + [f"{feature} <= {threshold:.4f}"])
            recurse(tree_.children_right[node], conditions + [f"{feature} > {threshold:.4f}"])
            return

        values = tree_.value[node][0]
        total = int(values.sum())
        if total == 0:
            return
        positive_rate = float(values[1] / total)
        if positive_rate < 0.6:
            return

        signature, bounds = canonicalize_rule_conditions(conditions)
        mask = np.ones(len(full_subset), dtype=bool)
        for feature, limit in bounds.items():
            if "lower" in limit:
                mask &= full_subset[feature].to_numpy() > limit["lower"]
            if "upper" in limit:
                mask &= full_subset[feature].to_numpy() <= limit["upper"]
        support = int(mask.sum())
        if support == 0:
            return

        records.append(
            {
                "signature": signature,
                "bounds": bounds,
                "rule_text": render_rule(bounds),
                "support": support,
                "purity": float(full_target.loc[mask].mean()),
                "diagnosis_rate": float(full_subset.loc[mask, COLS.diagnosis].mean()),
            }
        )

    recurse(0, [])
    return records


def bootstrap_rule_stability(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tree_features = [
        COLS.tc,
        COLS.tg,
        COLS.ldl,
        COLS.hdl,
        COLS.tanshi,
        COLS.activity_total,
        COLS.bmi,
        COLS.glucose,
        COLS.uric_acid,
        "p_early",
        "血脂异常项数",
    ]
    tanshi_subset = df[df[COLS.constitution_label] == 5].copy().reset_index(drop=True)
    target = (tanshi_subset["风险等级"] == "高风险").astype(int)
    rng = np.random.default_rng(RANDOM_SEED)
    all_records = []

    for bootstrap_id in range(1, RULE_BOOTSTRAPS + 1):
        sample_idx = rng.integers(0, len(tanshi_subset), len(tanshi_subset))
        sampled = tanshi_subset.iloc[sample_idx].reset_index(drop=True)
        sampled_target = target.iloc[sample_idx].reset_index(drop=True)
        tree_model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=12, random_state=RANDOM_SEED + bootstrap_id)
        tree_model.fit(sampled[tree_features], sampled_target)
        for record in extract_bootstrap_rule_records(tree_model, tree_features, tanshi_subset, target):
            record["bootstrap_id"] = bootstrap_id
            all_records.append(record)

    detail = pd.DataFrame(all_records)
    if detail.empty:
        empty = pd.DataFrame(columns=["规则", "出现频率", "平均支持样本数", "平均高风险纯度", "平均确诊率"])
        return empty, empty

    aggregated_rows = []
    for signature, group in detail.groupby("signature"):
        feature_bounds = {}
        for bounds in group["bounds"]:
            for feature, limit in bounds.items():
                feature_bounds.setdefault(feature, {"lower": [], "upper": []})
                if "lower" in limit:
                    feature_bounds[feature]["lower"].append(limit["lower"])
                if "upper" in limit:
                    feature_bounds[feature]["upper"].append(limit["upper"])

        median_bounds = {}
        for feature, limit_dict in feature_bounds.items():
            median_bounds[feature] = {}
            if limit_dict["lower"]:
                median_bounds[feature]["lower"] = float(np.median(limit_dict["lower"]))
            if limit_dict["upper"]:
                median_bounds[feature]["upper"] = float(np.median(limit_dict["upper"]))

        aggregated_rows.append(
            {
                "signature": signature,
                "规则": render_rule(median_bounds),
                "出现次数": int(group["bootstrap_id"].nunique()),
                "出现频率": group["bootstrap_id"].nunique() / RULE_BOOTSTRAPS,
                "平均支持样本数": group["support"].mean(),
                "平均高风险纯度": group["purity"].mean(),
                "平均确诊率": group["diagnosis_rate"].mean(),
            }
        )

    stability = pd.DataFrame(aggregated_rows).sort_values(
        ["出现频率", "平均高风险纯度", "平均支持样本数"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    stable_rules = stability[
        (stability["出现频率"] >= 0.08)
        & (stability["平均支持样本数"] >= 10)
        & (stability["平均高风险纯度"] >= 0.65)
    ].head(4)
    if stable_rules.empty:
        stable_rules = stability[stability["平均支持样本数"] >= 10].head(4)
    return stability, stable_rules.reset_index(drop=True)


def plot_early_model_performance(
    y_true: pd.Series,
    base_result: ModelResult,
    final_result: ModelResult,
    importance: pd.DataFrame,
) -> None:
    final_proba = final_result.oof_proba
    base_proba = base_result.oof_proba
    fpr, tpr, _ = roc_curve(y_true, final_proba)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, final_proba)
    frac_positive_base, mean_pred_base = calibration_curve(y_true, base_proba, n_bins=8, strategy="quantile")
    frac_positive_final, mean_pred_final = calibration_curve(y_true, final_proba, n_bins=8, strategy="quantile")

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes[0, 0].plot(fpr, tpr, color="#4C72B0", linewidth=2, label=f"AUC={roc_auc_score(y_true, final_proba):.4f}")
    axes[0, 0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0, 0].set_title("轨道A ROC 曲线")
    axes[0, 0].set_xlabel("False Positive Rate")
    axes[0, 0].set_ylabel("True Positive Rate")
    axes[0, 0].legend()

    axes[0, 1].plot(recall_curve, precision_curve, color="#55A868", linewidth=2)
    axes[0, 1].set_title("轨道A PR 曲线")
    axes[0, 1].set_xlabel("Recall")
    axes[0, 1].set_ylabel("Precision")

    axes[1, 0].plot(mean_pred_base, frac_positive_base, marker="o", color="#999999", label=base_result.name)
    axes[1, 0].plot(mean_pred_final, frac_positive_final, marker="o", color="#C44E52", label=final_result.name)
    axes[1, 0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1, 0].set_title("轨道A 校准对比")
    axes[1, 0].set_xlabel("Mean Predicted Probability")
    axes[1, 0].set_ylabel("Fraction of Positives")
    axes[1, 0].legend(fontsize=9)

    top_imp = importance.head(10).sort_values("置换重要性")
    axes[1, 1].barh(top_imp["特征"], top_imp["置换重要性"], color="#8172B3")
    axes[1, 1].set_title("轨道A 置换重要性 Top10")
    axes[1, 1].set_xlabel("Permutation Importance")

    fig.tight_layout()
    fig.savefig(figure_path("problem2_early_model_performance.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_risk_stratification(df: pd.DataFrame, low: float, high: float) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    diagnosed = df[df[COLS.diagnosis] == 1]["R"]
    undiagnosed = df[df[COLS.diagnosis] == 0]["R"]
    axes[0, 0].hist(undiagnosed, bins=25, alpha=0.7, color="#64B5CD", label="未确诊")
    axes[0, 0].hist(diagnosed, bins=25, alpha=0.7, color="#DD8452", label="确诊")
    axes[0, 0].axvline(low, color="gray", linestyle="--", linewidth=1.5)
    axes[0, 0].axvline(high, color="black", linestyle="--", linewidth=1.5)
    axes[0, 0].set_title("轨道B 风险评分与诊断分布")
    axes[0, 0].set_xlabel("风险评分 R")
    axes[0, 0].legend()

    summary = risk_group_summary(df, "R", "风险等级")
    axes[0, 1].bar(summary["风险等级"], summary["确诊率"], color=["#64B5CD", "#F2B134", "#C44E52"])
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].set_title("三级风险组的实际确诊率")

    order = ["低风险", "中风险", "高风险"]
    data = [df.loc[df["风险等级"] == group, "R"] for group in order]
    axes[1, 0].boxplot(data, labels=order, patch_artist=True)
    axes[1, 0].set_title("三级风险组的评分箱线图")
    axes[1, 0].set_ylabel("风险评分 R")

    for group, color in zip(order, ["#64B5CD", "#F2B134", "#C44E52"]):
        values = df.loc[df["风险等级"] == group, "R"].to_numpy()
        if len(np.unique(values)) > 1:
            density = stats.gaussian_kde(values)
            x_grid = np.linspace(values.min(), values.max(), 200)
            axes[1, 1].plot(x_grid, density(x_grid), color=color, linewidth=2, label=group)
    axes[1, 1].set_title("三级风险组的密度曲线")
    axes[1, 1].set_xlabel("风险评分 R")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(figure_path("problem2_risk_stratification.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_plotting()
    set_random_seed()
    df = load_data()

    print("=" * 72)
    print("问题2：双轨风险预警模型的稳健冲奖版优化分析")
    print("=" * 72)
    print(f"样例数据文件：{find_sample_data_path().name}")

    consistency_table, agreement = verify_label_consistency(df)
    print("\n关键事实校验：题目二分类标签与“任一血脂异常”完全一致。")
    print(consistency_table.to_string(index=False))
    print(f"一致率 = {agreement:.4f}")

    model_specs = build_model_specs()
    repeated_detail = pd.concat(
        [evaluate_repeated_cv(df, spec) for spec in model_specs.values()],
        ignore_index=True,
    )
    repeated_summary = summarize_cv_detail(repeated_detail)
    best_family_spec = choose_best_family(repeated_summary, model_specs)

    calibration_compare, selected_spec = evaluate_calibration_compare(df, best_family_spec)
    base_fixed_result = compute_fixed_oof_result(df, best_family_spec)
    final_fixed_result = compute_fixed_oof_result(df, selected_spec)
    df["p_early"] = final_fixed_result.oof_proba

    ci_summary, ci_detail = bootstrap_metric_intervals(
        df[COLS.diagnosis],
        final_fixed_result.oof_proba,
        final_fixed_result.oof_pred,
    )
    model_summary_with_ci = build_model_summary_with_ci(
        selected_spec,
        final_fixed_result,
        ci_summary,
        calibration_compare,
    )
    best_model_predictions = pd.DataFrame(
        {
            COLS.sample_id: df[COLS.sample_id],
            COLS.diagnosis: df[COLS.diagnosis],
            "p_early": final_fixed_result.oof_proba,
            "pred_label": final_fixed_result.oof_pred,
            "最终主模型": selected_spec.name,
        }
    )

    ablation = run_ablation_study(df, selected_spec)
    subgroup = compute_subgroup_stability(df, final_fixed_result)
    importance = compute_permutation_importance(df, final_fixed_result)
    quantile_effects = build_quantile_effect_table(df, final_fixed_result, importance)

    component_scores, weight_table = build_clinical_risk_score(df, final_fixed_result.oof_proba)
    df = pd.concat([df, component_scores], axis=1)
    low_threshold, high_threshold, threshold_search = search_thresholds(df["R"], df[COLS.diagnosis])
    df["风险等级"] = assign_risk_level(df["R"], low_threshold, high_threshold)
    group_summary = risk_group_summary(df, "R", "风险等级")
    group_stats = compute_risk_group_stats(df)

    threshold_bootstrap_summary, threshold_bootstrap_detail = bootstrap_threshold_stability(df["R"], df[COLS.diagnosis])
    weight_sensitivity = compute_weight_sensitivity(component_scores, df[COLS.diagnosis])
    rule_stability, stable_rules = bootstrap_rule_stability(df)
    stable_rules_export = stable_rules.drop(columns=["signature"], errors="ignore")

    predictions = df[
        [
            COLS.sample_id,
            "p_early",
            "lipid_burden",
            "lipid_excess",
            "early_risk",
            "metabolic_modifier",
            "function_modifier",
            "R",
            "风险等级",
            COLS.diagnosis,
        ]
    ].copy()

    save_workbook(
        "problem2_tables",
        {
            "轨道A重复CV明细": repeated_detail,
            "轨道A模型汇总": repeated_summary,
            "校准对比": calibration_compare,
            "主模型含CI": model_summary_with_ci,
            "主模型OOF结果": best_model_predictions,
            "轨道A消融实验": ablation,
            "轨道A亚组稳健性": subgroup,
            "轨道A特征解释": importance,
            "轨道A分位风险表": quantile_effects,
            "轨道B权重配置": weight_table,
            "阈值搜索Top20": threshold_search.head(20),
            "阈值Bootstrap摘要": threshold_bootstrap_summary,
            "阈值Bootstrap明细": threshold_bootstrap_detail,
            "权重敏感性": weight_sensitivity,
            "三级分层汇总": group_summary,
            "三级分层检验": group_stats,
            "规则稳定性": rule_stability,
            "稳定规则": stable_rules_export if not stable_rules_export.empty else pd.DataFrame({"说明": ["未提取到稳定规则"]}),
            "预测结果": predictions,
            "标签一致性": consistency_table,
        },
    )

    save_table(repeated_summary, "problem2_early_model_metrics")
    save_table(repeated_detail, "problem2_model_cv_detail")
    save_table(model_summary_with_ci, "problem2_model_summary_with_ci")
    save_table(best_model_predictions, "problem2_best_model_oof_predictions")
    save_table(calibration_compare, "problem2_calibration_compare")
    save_table(importance, "problem2_permutation_importance")
    save_table(quantile_effects, "problem2_quantile_effects")
    save_table(ablation, "problem2_ablation_study")
    save_table(subgroup, "problem2_subgroup_stability")
    save_table(threshold_bootstrap_summary, "problem2_threshold_bootstrap")
    save_table(weight_sensitivity, "problem2_weight_sensitivity")
    save_table(group_summary, "problem2_risk_group_summary")
    save_table(group_stats, "problem2_risk_group_stats")
    save_table(predictions, "problem2_predictions")
    save_table(rule_stability, "problem2_rule_stability")
    if not stable_rules_export.empty:
        save_table(stable_rules_export, "problem2_stable_rules")
        save_table(stable_rules_export, "problem2_high_risk_rules")

    plot_early_model_performance(df[COLS.diagnosis], base_fixed_result, final_fixed_result, importance)
    plot_risk_stratification(df, low_threshold, high_threshold)

    selected_ci = ci_summary.set_index("指标").to_dict("index")
    threshold_ci = threshold_bootstrap_summary.set_index("指标").to_dict("index")
    summary = {
        "best_early_model_family": best_family_spec.name,
        "final_early_model": selected_spec.name,
        "calibration_applied": "Sigmoid校准" in selected_spec.name,
        "best_early_model_auc": round(final_fixed_result.metrics["AUC"], 4),
        "best_early_model_auc_ci": [
            round(selected_ci["AUC"]["CI下界"], 4),
            round(selected_ci["AUC"]["CI上界"], 4),
        ],
        "best_early_model_f1": round(final_fixed_result.metrics["F1"], 4),
        "best_early_model_brier": round(final_fixed_result.metrics["Brier分数"], 4),
        "threshold_low": round(low_threshold, 4),
        "threshold_high": round(high_threshold, 4),
        "threshold_low_ci": [
            round(threshold_ci["T_low"]["CI下界"], 4),
            round(threshold_ci["T_low"]["CI上界"], 4),
        ],
        "threshold_high_ci": [
            round(threshold_ci["T_high"]["CI下界"], 4),
            round(threshold_ci["T_high"]["CI上界"], 4),
        ],
        "risk_group_prevalence": {
            row["风险等级"]: round(row["确诊率"], 4)
            for _, row in group_summary.iterrows()
        },
        "top_early_features": importance["特征"].head(6).tolist(),
        "stable_rule_count": int(len(stable_rules)),
        "label_rule_agreement": round(agreement, 4),
    }
    save_json(summary, "problem2_summary")

    print("\n轨道A repeated-CV 模型汇总：")
    print(
        repeated_summary[
            ["模型", "AUC均值", "AUC标准差", "PR_AUC均值", "Brier均值", "F1均值", "综合排序分"]
        ].to_string(index=False)
    )
    print("\n校准对比：")
    print(
        calibration_compare[
            ["模型", "AUC均值", "Brier均值", "F1均值", "是否采纳", "选择理由"]
        ].to_string(index=False)
    )
    print(f"\n最终无泄漏早筛主模型：{selected_spec.name}")
    print("\n轨道B 阈值与稳定区间：")
    print(f"  T_low = {low_threshold:.4f}")
    print(f"  T_high = {high_threshold:.4f}")
    print(threshold_bootstrap_summary.to_string(index=False))
    print("\n三级风险组结果：")
    print(group_summary.to_string(index=False))
    if not stable_rules_export.empty:
        print("\n痰湿体质稳定高风险规则：")
        print(stable_rules_export.to_string(index=False))
    print("\n图表与表格已输出到 src/outputs 目录。")


if __name__ == "__main__":
    main()
