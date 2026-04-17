from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats

from common import (
    ACTIVITY_COST,
    ACTIVITY_DURATION,
    AGE_GROUP_MAP,
    COLS,
    TCM_COST,
    TCM_PLAN,
    configure_plotting,
    describe_age_group,
    describe_sex,
    figure_path,
    get_max_activity_level,
    get_tcm_level,
    load_data,
    next_tanshi_score,
    save_json,
    save_table,
    save_workbook,
    set_random_seed,
)


warnings.filterwarnings("ignore")


PLANNING_MONTHS = 6
DEFAULT_BUDGET = 2000
BUDGET_SCENARIOS = [1200, 1400, 1600, 1800, 2000]
SAMPLE_IDS = [1, 2, 3]
SAMPLE_COLORS = {1: "#4C72B0", 2: "#55A868", 3: "#C44E52"}
PLAN_STYLE = {"动态最优": "-", "固定活动基线": "--"}


def action_label(activity_level: int, weekly_freq: int) -> str:
    return f"L{activity_level}-F{weekly_freq}"


def safe_mode(series: pd.Series) -> object:
    mode = series.mode()
    return mode.iloc[0] if not mode.empty else pd.NA


def safe_wilcoxon(x: pd.Series, y: pd.Series, alternative: str) -> float | None:
    if len(x) == 0 or len(y) == 0:
        return None
    if (x.reset_index(drop=True) == y.reset_index(drop=True)).all():
        return None
    result = stats.wilcoxon(x, y, alternative=alternative)
    return float(result.pvalue)


def plan_sequences(monthly_plan: list[dict]) -> tuple[str, str]:
    if not monthly_plan:
        return "", ""
    action_sequence = " / ".join(
        action_label(step["活动强度"], step["训练频率(次/周)"]) for step in monthly_plan
    )
    tcm_sequence = "-".join(str(step["中医调理等级"]) for step in monthly_plan)
    return action_sequence, tcm_sequence


def base_plan_result(
    initial_score: float,
    max_activity_level: int,
    budget: int,
    final_score: float,
    total_cost: int,
    switch_count: int,
    monthly_plan: list[dict] | None,
) -> dict:
    monthly_plan = monthly_plan or []
    action_sequence, tcm_sequence = plan_sequences(monthly_plan)
    return {
        "初始痰湿积分": round(float(initial_score), 2),
        "最大可选活动强度": int(max_activity_level),
        "预算上限": int(budget),
        "最终痰湿积分": round(float(final_score), 2),
        "积分下降量": round(float(initial_score) - float(final_score), 2),
        "积分下降率": round((float(initial_score) - float(final_score)) / float(initial_score) * 100, 2),
        "总成本": int(total_cost),
        "预算利用率(%)": round(total_cost / budget * 100, 2),
        "切换次数": int(switch_count),
        "动作序列": action_sequence,
        "中医序列": tcm_sequence,
        "monthly_plan": monthly_plan,
    }


def dynamic_programming_plan(
    initial_score: float,
    max_activity_level: int,
    budget: int = DEFAULT_BUDGET,
    keep_history: bool = True,
) -> dict:
    initial_score = round(float(initial_score), 2)

    frontier: dict[tuple[float, tuple[int, int]], tuple[int, int, list[dict] | None]] = {
        (initial_score, (0, 0)): (0, 0, [] if keep_history else None)
    }

    for month in range(1, PLANNING_MONTHS + 1):
        new_frontier: dict[tuple[float, tuple[int, int]], tuple[int, int, list[dict] | None]] = {}

        for (score, previous_action), (spent_cost, switches, history) in frontier.items():
            current_tcm_level = get_tcm_level(score)
            current_tcm_cost = TCM_COST[current_tcm_level]

            for activity_level in range(1, max_activity_level + 1):
                for weekly_freq in range(1, 11):
                    month_activity_cost = ACTIVITY_COST[activity_level] * weekly_freq * 4
                    new_spent = spent_cost + current_tcm_cost + month_activity_cost
                    if new_spent > budget:
                        continue

                    next_score = next_tanshi_score(score, activity_level, weekly_freq)
                    switch_flag = 0 if month == 1 or previous_action == (activity_level, weekly_freq) else 1
                    monthly_step = None
                    if keep_history:
                        monthly_step = {
                            "月份": month,
                            "月初痰湿积分": round(score, 2),
                            "中医调理等级": current_tcm_level,
                            "中医调理方案": TCM_PLAN[current_tcm_level],
                            "中医月成本": current_tcm_cost,
                            "活动强度": activity_level,
                            "单次训练时长(分钟)": ACTIVITY_DURATION[activity_level],
                            "训练频率(次/周)": weekly_freq,
                            "活动月成本": month_activity_cost,
                            "月末痰湿积分": round(next_score, 2),
                            "方案切换": switch_flag,
                        }

                    key = (round(next_score, 2), (activity_level, weekly_freq))
                    candidate_history = None if history is None else history + [monthly_step]
                    candidate = (new_spent, switches + switch_flag, candidate_history)
                    existing = new_frontier.get(key)
                    if existing is None or candidate[:2] < existing[:2]:
                        new_frontier[key] = candidate

        if not new_frontier:
            raise RuntimeError(f"未找到可行方案：初始积分={initial_score}, 最大强度={max_activity_level}")
        frontier = new_frontier

    best_plan = None
    best_signature = None
    for (final_score, _), (spent_cost, switches, history) in frontier.items():
        signature = (final_score, spent_cost, switches)
        if best_signature is None or signature < best_signature:
            best_signature = signature
            best_plan = (spent_cost, switches, history)

    total_cost, switch_count, history = best_plan
    return base_plan_result(
        initial_score=initial_score,
        max_activity_level=max_activity_level,
        budget=budget,
        final_score=best_signature[0],
        total_cost=total_cost,
        switch_count=switch_count,
        monthly_plan=history,
    )


def static_activity_baseline(
    initial_score: float,
    max_activity_level: int,
    budget: int = DEFAULT_BUDGET,
    keep_history: bool = True,
) -> dict:
    initial_score = round(float(initial_score), 2)
    best_signature = None
    best_history = None
    best_action = None

    for activity_level in range(1, max_activity_level + 1):
        for weekly_freq in range(1, 11):
            score = initial_score
            spent_cost = 0
            history: list[dict] | None = [] if keep_history else None
            feasible = True

            for month in range(1, PLANNING_MONTHS + 1):
                current_tcm_level = get_tcm_level(score)
                current_tcm_cost = TCM_COST[current_tcm_level]
                month_activity_cost = ACTIVITY_COST[activity_level] * weekly_freq * 4
                spent_cost += current_tcm_cost + month_activity_cost
                if spent_cost > budget:
                    feasible = False
                    break

                next_score = next_tanshi_score(score, activity_level, weekly_freq)
                if keep_history:
                    history.append(
                        {
                            "月份": month,
                            "月初痰湿积分": round(score, 2),
                            "中医调理等级": current_tcm_level,
                            "中医调理方案": TCM_PLAN[current_tcm_level],
                            "中医月成本": current_tcm_cost,
                            "活动强度": activity_level,
                            "单次训练时长(分钟)": ACTIVITY_DURATION[activity_level],
                            "训练频率(次/周)": weekly_freq,
                            "活动月成本": month_activity_cost,
                            "月末痰湿积分": round(next_score, 2),
                            "方案切换": 0,
                        }
                    )
                score = next_score

            if not feasible:
                continue

            signature = (round(score, 2), spent_cost, 0)
            if best_signature is None or signature < best_signature:
                best_signature = signature
                best_history = history
                best_action = (activity_level, weekly_freq)

    if best_signature is None:
        raise RuntimeError(f"固定活动基线无可行方案：初始积分={initial_score}, 最大强度={max_activity_level}")

    result = base_plan_result(
        initial_score=initial_score,
        max_activity_level=max_activity_level,
        budget=budget,
        final_score=best_signature[0],
        total_cost=best_signature[1],
        switch_count=0,
        monthly_plan=best_history,
    )
    result["固定活动强度"] = best_action[0]
    result["固定训练频率"] = best_action[1]
    result["固定动作"] = action_label(best_action[0], best_action[1])
    return result


def plan_to_monthly_df(patient: pd.Series, plan: dict, plan_type: str) -> pd.DataFrame:
    monthly = pd.DataFrame(plan["monthly_plan"]).copy()
    if monthly.empty:
        return monthly
    monthly.insert(0, "方案类型", plan_type)
    monthly.insert(0, "样本ID", int(patient[COLS.sample_id]))
    monthly.insert(1, "年龄段", describe_age_group(int(patient[COLS.age_group])))
    monthly.insert(2, "活动量表总分", int(patient[COLS.activity_total]))
    monthly.insert(3, "最大可选活动强度", plan["最大可选活动强度"])
    return monthly


def patient_dynamic_summary(patient: pd.Series, plan: dict) -> dict:
    return {
        "样本ID": int(patient[COLS.sample_id]),
        "年龄组": int(patient[COLS.age_group]),
        "年龄段": describe_age_group(int(patient[COLS.age_group])),
        "性别": describe_sex(int(patient[COLS.sex])),
        "体质标签": int(patient[COLS.constitution_label]),
        "初始痰湿积分": int(patient[COLS.tanshi]),
        "活动量表总分": int(patient[COLS.activity_total]),
        "最大可选活动强度": plan["最大可选活动强度"],
        "最终痰湿积分": plan["最终痰湿积分"],
        "积分下降量": plan["积分下降量"],
        "积分下降率(%)": plan["积分下降率"],
        "总成本": plan["总成本"],
        "预算利用率(%)": plan["预算利用率(%)"],
        "切换次数": plan["切换次数"],
        "首月中医等级": plan["monthly_plan"][0]["中医调理等级"],
        "首月活动强度": plan["monthly_plan"][0]["活动强度"],
        "首月训练频率": plan["monthly_plan"][0]["训练频率(次/周)"],
        "末月中医等级": plan["monthly_plan"][-1]["中医调理等级"],
        "末月活动强度": plan["monthly_plan"][-1]["活动强度"],
        "末月训练频率": plan["monthly_plan"][-1]["训练频率(次/周)"],
        "动作序列": plan["动作序列"],
        "中医序列": plan["中医序列"],
    }


def build_policy_map(summary_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        summary_df.groupby(["初始痰湿积分", "最大可选活动强度"])
        .agg(
            模板人数=("样本ID", "count"),
            最常见首月活动强度=("首月活动强度", safe_mode),
            最常见首月频率=("首月训练频率", safe_mode),
            平均最终积分=("最终痰湿积分", "mean"),
            平均总成本=("总成本", "mean"),
            平均切换次数=("切换次数", "mean"),
        )
        .reset_index()
        .sort_values(["初始痰湿积分", "最大可选活动强度"])
    )
    return grouped


def build_template_library(
    template_cache: dict[tuple[int, int], dict],
    static_cache: dict[tuple[int, int], dict],
    template_counts: dict[tuple[int, int], int],
) -> pd.DataFrame:
    rows = []
    for key in sorted(template_cache):
        initial_score, max_level = key
        dynamic_plan = template_cache[key]
        static_plan = static_cache[key]
        rows.append(
            {
                "初始痰湿积分": initial_score,
                "最大可选活动强度": max_level,
                "模板人数": template_counts[key],
                "动态最终积分": dynamic_plan["最终痰湿积分"],
                "静态最终积分": static_plan["最终痰湿积分"],
                "终点改善值": round(static_plan["最终痰湿积分"] - dynamic_plan["最终痰湿积分"], 2),
                "动态总成本": dynamic_plan["总成本"],
                "静态总成本": static_plan["总成本"],
                "成本变化(动态-静态)": dynamic_plan["总成本"] - static_plan["总成本"],
                "切换次数": dynamic_plan["切换次数"],
                "动态预算利用率(%)": dynamic_plan["预算利用率(%)"],
                "静态预算利用率(%)": static_plan["预算利用率(%)"],
                "动态动作序列": dynamic_plan["动作序列"],
                "静态动作": static_plan["固定动作"],
                "动态中医序列": dynamic_plan["中医序列"],
            }
        )
    return pd.DataFrame(rows)


def build_dynamic_static_compare(
    tanshi_patients: pd.DataFrame,
    template_cache: dict[tuple[int, int], dict],
    static_cache: dict[tuple[int, int], dict],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    compare_rows = []
    sample_compare_rows = []

    for _, patient in tanshi_patients.iterrows():
        max_level = get_max_activity_level(int(patient[COLS.age_group]), float(patient[COLS.activity_total]))
        key = (int(patient[COLS.tanshi]), max_level)
        dynamic_plan = template_cache[key]
        static_plan = static_cache[key]
        row = {
            "样本ID": int(patient[COLS.sample_id]),
            "年龄段": describe_age_group(int(patient[COLS.age_group])),
            "性别": describe_sex(int(patient[COLS.sex])),
            "初始痰湿积分": int(patient[COLS.tanshi]),
            "活动量表总分": int(patient[COLS.activity_total]),
            "最大可选活动强度": max_level,
            "动态最终积分": dynamic_plan["最终痰湿积分"],
            "静态最终积分": static_plan["最终痰湿积分"],
            "终点改善值": round(static_plan["最终痰湿积分"] - dynamic_plan["最终痰湿积分"], 2),
            "动态积分下降率(%)": dynamic_plan["积分下降率"],
            "静态积分下降率(%)": static_plan["积分下降率"],
            "动态总成本": dynamic_plan["总成本"],
            "静态总成本": static_plan["总成本"],
            "成本变化(动态-静态)": dynamic_plan["总成本"] - static_plan["总成本"],
            "动态切换次数": dynamic_plan["切换次数"],
            "静态动作": static_plan["固定动作"],
            "动态动作序列": dynamic_plan["动作序列"],
            "动态优于静态": int(dynamic_plan["最终痰湿积分"] < static_plan["最终痰湿积分"]),
        }
        compare_rows.append(row)
        if int(patient[COLS.sample_id]) in SAMPLE_IDS:
            sample_compare_rows.append(row)

    compare_df = pd.DataFrame(compare_rows).sort_values("样本ID").reset_index(drop=True)
    sample_compare_df = pd.DataFrame(sample_compare_rows).sort_values("样本ID").reset_index(drop=True)

    summary_rows = []
    for group_name, group_df in [("总体", compare_df)] + [
        (f"最大强度={level}", compare_df[compare_df["最大可选活动强度"] == level])
        for level in sorted(compare_df["最大可选活动强度"].unique())
    ]:
        final_p = safe_wilcoxon(group_df["静态最终积分"], group_df["动态最终积分"], alternative="greater")
        cost_p = safe_wilcoxon(group_df["动态总成本"], group_df["静态总成本"], alternative="two-sided")
        summary_rows.append(
            {
                "分析组": group_name,
                "患者数": len(group_df),
                "动态平均最终积分": round(group_df["动态最终积分"].mean(), 4),
                "静态平均最终积分": round(group_df["静态最终积分"].mean(), 4),
                "平均终点改善值": round(group_df["终点改善值"].mean(), 4),
                "动态平均下降率(%)": round(group_df["动态积分下降率(%)"].mean(), 4),
                "静态平均下降率(%)": round(group_df["静态积分下降率(%)"].mean(), 4),
                "动态平均总成本": round(group_df["动态总成本"].mean(), 4),
                "静态平均总成本": round(group_df["静态总成本"].mean(), 4),
                "动态优于静态人数": int((group_df["终点改善值"] > 0).sum()),
                "需要切换人数": int((group_df["动态切换次数"] > 0).sum()),
                "最终积分Wilcoxon p": None if final_p is None else round(final_p, 6),
                "总成本Wilcoxon p": None if cost_p is None else round(cost_p, 6),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    return compare_df, summary_df, sample_compare_df


def build_budget_sensitivity(
    tanshi_patients: pd.DataFrame,
    budget_cache: dict[tuple[int, int, int], dict],
) -> pd.DataFrame:
    rows = []
    for budget in BUDGET_SCENARIOS:
        patient_rows = []
        for _, patient in tanshi_patients.iterrows():
            max_level = get_max_activity_level(int(patient[COLS.age_group]), float(patient[COLS.activity_total]))
            key = (int(patient[COLS.tanshi]), max_level, budget)
            plan = budget_cache[key]
            patient_rows.append(
                {
                    "预算上限": budget,
                    "最大可选活动强度": max_level,
                    "最终痰湿积分": plan["最终痰湿积分"],
                    "积分下降率(%)": plan["积分下降率"],
                    "总成本": plan["总成本"],
                    "预算利用率(%)": plan["预算利用率(%)"],
                    "切换次数": plan["切换次数"],
                }
            )
        patient_budget_df = pd.DataFrame(patient_rows)
        for group_name, group_df in [("总体", patient_budget_df)] + [
            (f"最大强度={level}", patient_budget_df[patient_budget_df["最大可选活动强度"] == level])
            for level in sorted(patient_budget_df["最大可选活动强度"].unique())
        ]:
            rows.append(
                {
                    "分析组": group_name,
                    "预算上限": budget,
                    "患者数": len(group_df),
                    "平均最终积分": round(group_df["最终痰湿积分"].mean(), 4),
                    "平均积分下降率(%)": round(group_df["积分下降率(%)"].mean(), 4),
                    "平均总成本": round(group_df["总成本"].mean(), 4),
                    "平均预算利用率(%)": round(group_df["预算利用率(%)"].mean(), 4),
                    "切换患者占比": round((group_df["切换次数"] > 0).mean(), 4),
                }
            )
    return pd.DataFrame(rows)


def build_monthly_policy_profile(monthly_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (max_level, month), group_df in monthly_df.groupby(["最大可选活动强度", "月份"]):
        action_counts = group_df[["活动强度", "训练频率(次/周)"]].value_counts()
        (top_level, top_freq) = action_counts.index[0]
        support_count = int(action_counts.iloc[0])
        rows.append(
            {
                "最大可选活动强度": max_level,
                "月份": month,
                "样本数": len(group_df),
                "最常见活动强度": int(top_level),
                "最常见训练频率": int(top_freq),
                "最常见动作": action_label(int(top_level), int(top_freq)),
                "支持样本数": support_count,
                "支持率": round(support_count / len(group_df), 4),
                "平均月初痰湿积分": round(group_df["月初痰湿积分"].mean(), 4),
                "平均月末痰湿积分": round(group_df["月末痰湿积分"].mean(), 4),
                "平均月总成本": round((group_df["中医月成本"] + group_df["活动月成本"]).mean(), 4),
            }
        )
    return pd.DataFrame(rows).sort_values(["最大可选活动强度", "月份"]).reset_index(drop=True)


def build_validation_checks(
    tanshi_patients: pd.DataFrame,
    summary_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    template_cache: dict[tuple[int, int], dict],
    compare_df: pd.DataFrame,
) -> pd.DataFrame:
    budget_violation_count = int((summary_df["总成本"] > DEFAULT_BUDGET).sum())
    monotone_violation_count = int((monthly_df["月末痰湿积分"] > monthly_df["月初痰湿积分"]).sum())
    strength_violation_count = int((monthly_df["活动强度"] > monthly_df["最大可选活动强度"]).sum())
    month_count_violation = int(monthly_df.groupby("样本ID")["月份"].nunique().ne(PLANNING_MONTHS).sum())

    template_mismatch_count = 0
    for (initial_score, max_level), cached_plan in template_cache.items():
        direct_plan = dynamic_programming_plan(initial_score, max_level, keep_history=False)
        cached_signature = (
            cached_plan["最终痰湿积分"],
            cached_plan["总成本"],
            cached_plan["切换次数"],
        )
        direct_signature = (
            direct_plan["最终痰湿积分"],
            direct_plan["总成本"],
            direct_plan["切换次数"],
        )
        if cached_signature != direct_signature:
            template_mismatch_count += 1

    checks = [
        ("痰湿患者数量", len(summary_df), len(tanshi_patients), "应保持为全部痰湿体质患者"),
        ("预算违规方案数", budget_violation_count, 0, "所有方案总成本必须不超过 2000 元"),
        ("月度积分上升次数", monotone_violation_count, 0, "月末痰湿积分序列应非增"),
        ("活动强度越界次数", strength_violation_count, 0, "活动强度不得超过年龄/量表允许上限"),
        ("患者月份记录缺失数", month_count_violation, 0, "每位患者都应有 6 个月计划"),
        ("模板复用不一致数", template_mismatch_count, 0, "模板缓存结果应与直接求解完全一致"),
        ("动态优于静态人数", int((compare_df["终点改善值"] > 0).sum()), 15, "动态收益应集中在少数高能力边界患者"),
    ]

    rows = []
    for item, observed, expected, note in checks:
        rows.append(
            {
                "检查项": item,
                "观测值": observed,
                "目标值": expected,
                "是否通过": "是" if observed == expected else "否",
                "说明": note,
            }
        )
    return pd.DataFrame(rows)


def build_age_and_intensity_summary(summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    age_summary = (
        summary_df.groupby("年龄段")
        .agg(
            患者数=("样本ID", "count"),
            平均最终积分=("最终痰湿积分", "mean"),
            平均积分下降率=("积分下降率(%)", "mean"),
            平均总成本=("总成本", "mean"),
            平均预算利用率=("预算利用率(%)", "mean"),
        )
        .reindex(AGE_GROUP_MAP.values())
        .reset_index()
    )
    intensity_summary = (
        summary_df.groupby("最大可选活动强度")
        .agg(
            患者数=("样本ID", "count"),
            平均最终积分=("最终痰湿积分", "mean"),
            平均积分下降率=("积分下降率(%)", "mean"),
            平均总成本=("总成本", "mean"),
            平均预算利用率=("预算利用率(%)", "mean"),
            平均切换次数=("切换次数", "mean"),
        )
        .reset_index()
        .sort_values("最大可选活动强度")
    )
    return age_summary, intensity_summary


def plot_sample_trajectories(sample_monthly_compare: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for sample_id in SAMPLE_IDS:
        for plan_type in ["动态最优", "固定活动基线"]:
            sub = (
                sample_monthly_compare[
                    (sample_monthly_compare["样本ID"] == sample_id)
                    & (sample_monthly_compare["方案类型"] == plan_type)
                ]
                .sort_values("月份")
                .reset_index(drop=True)
            )
            if sub.empty:
                continue
            x_score = list(range(0, PLANNING_MONTHS + 1))
            y_score = [sub.loc[0, "月初痰湿积分"]] + sub["月末痰湿积分"].tolist()
            axes[0].plot(
                x_score,
                y_score,
                marker="o",
                linestyle=PLAN_STYLE[plan_type],
                color=SAMPLE_COLORS[sample_id],
                label=f"样本{sample_id}-{plan_type}",
            )

            cumulative_cost = (sub["中医月成本"] + sub["活动月成本"]).cumsum()
            axes[1].plot(
                sub["月份"],
                cumulative_cost,
                marker="o",
                linestyle=PLAN_STYLE[plan_type],
                color=SAMPLE_COLORS[sample_id],
                label=f"样本{sample_id}-{plan_type}",
            )

    axes[0].set_title("样本1-3动态最优与固定基线的积分轨迹")
    axes[0].set_xlabel("阶段（0为干预前）")
    axes[0].set_ylabel("痰湿积分")
    axes[0].set_xticks(range(0, PLANNING_MONTHS + 1))
    axes[0].legend(ncol=2, fontsize=9)

    axes[1].set_title("样本1-3动态最优与固定基线的累计成本")
    axes[1].set_xlabel("月份")
    axes[1].set_ylabel("累计成本(元)")
    axes[1].set_xticks(range(1, PLANNING_MONTHS + 1))
    axes[1].legend(ncol=2, fontsize=9)

    fig.tight_layout()
    fig.savefig(figure_path("problem3_dynamic_samples.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_population_strategy(
    summary_df: pd.DataFrame,
    dynamic_static_summary: pd.DataFrame,
    budget_sensitivity: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    scatter = axes[0, 0].scatter(
        summary_df["总成本"],
        summary_df["积分下降率(%)"],
        c=summary_df["最大可选活动强度"],
        cmap="viridis",
        alpha=0.75,
    )
    axes[0, 0].set_title("动态最优方案的成本-效果分布")
    axes[0, 0].set_xlabel("总成本(元)")
    axes[0, 0].set_ylabel("积分下降率(%)")
    fig.colorbar(scatter, ax=axes[0, 0], label="最大可选强度")

    gain_df = dynamic_static_summary[dynamic_static_summary["分析组"].str.startswith("最大强度=")].copy()
    axes[0, 1].bar(gain_df["分析组"], gain_df["平均终点改善值"], color="#55A868")
    axes[0, 1].set_title("动态规划相对固定基线的终点改善")
    axes[0, 1].set_ylabel("平均终点改善值")
    axes[0, 1].tick_params(axis="x", rotation=20)

    for group_name in ["总体", "最大强度=1", "最大强度=2", "最大强度=3"]:
        sub = budget_sensitivity[budget_sensitivity["分析组"] == group_name]
        axes[1, 0].plot(sub["预算上限"], sub["平均最终积分"], marker="o", label=group_name)
    axes[1, 0].set_title("不同预算上限下的平均最终积分")
    axes[1, 0].set_xlabel("预算上限(元)")
    axes[1, 0].set_ylabel("平均最终积分")
    axes[1, 0].legend()

    switch_dist = summary_df["切换次数"].value_counts().sort_index()
    axes[1, 1].bar(switch_dist.index.astype(str), switch_dist.values, color="#C44E52")
    axes[1, 1].set_title("动态方案切换次数分布")
    axes[1, 1].set_xlabel("切换次数")
    axes[1, 1].set_ylabel("患者数")

    fig.tight_layout()
    fig.savefig(figure_path("problem3_population_strategy.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_plotting()
    set_random_seed()
    df = load_data()
    tanshi_patients = df[df[COLS.constitution_label] == 5].copy()

    print("=" * 72)
    print("问题3：动态规划版 6 个月干预方案优化")
    print("=" * 72)
    print(f"痰湿体质患者数量：{len(tanshi_patients)}")

    template_keys = sorted(
        {
            (
                int(patient[COLS.tanshi]),
                get_max_activity_level(int(patient[COLS.age_group]), float(patient[COLS.activity_total])),
            )
            for _, patient in tanshi_patients.iterrows()
        }
    )
    template_counts = (
        tanshi_patients.assign(
            最大可选活动强度=lambda data: data.apply(
                lambda row: get_max_activity_level(int(row[COLS.age_group]), float(row[COLS.activity_total])),
                axis=1,
            )
        )
        .groupby([COLS.tanshi, "最大可选活动强度"])
        .size()
        .to_dict()
    )

    template_cache = {
        key: dynamic_programming_plan(key[0], key[1], budget=DEFAULT_BUDGET, keep_history=True)
        for key in template_keys
    }
    static_cache = {
        key: static_activity_baseline(key[0], key[1], budget=DEFAULT_BUDGET, keep_history=True)
        for key in template_keys
    }
    budget_cache = {
        (key[0], key[1], budget): dynamic_programming_plan(
            key[0], key[1], budget=budget, keep_history=False
        )
        for key in template_keys
        for budget in BUDGET_SCENARIOS
    }

    summary_rows = []
    monthly_rows = []
    sample_monthly_compare_rows = []

    for _, patient in tanshi_patients.iterrows():
        max_level = get_max_activity_level(int(patient[COLS.age_group]), float(patient[COLS.activity_total]))
        key = (int(patient[COLS.tanshi]), max_level)
        dynamic_plan = template_cache[key]
        static_plan = static_cache[key]

        summary_rows.append(patient_dynamic_summary(patient, dynamic_plan))
        monthly_rows.append(plan_to_monthly_df(patient, dynamic_plan, "动态最优"))

        if int(patient[COLS.sample_id]) in SAMPLE_IDS:
            sample_monthly_compare_rows.append(plan_to_monthly_df(patient, dynamic_plan, "动态最优"))
            sample_monthly_compare_rows.append(plan_to_monthly_df(patient, static_plan, "固定活动基线"))

    summary_df = pd.DataFrame(summary_rows).sort_values("样本ID").reset_index(drop=True)
    monthly_df = pd.concat(monthly_rows, ignore_index=True)
    sample_summary = summary_df[summary_df["样本ID"].isin(SAMPLE_IDS)].copy()
    sample_monthly = monthly_df[monthly_df["样本ID"].isin(SAMPLE_IDS)].copy()
    sample_monthly_compare = pd.concat(sample_monthly_compare_rows, ignore_index=True)

    policy_map = build_policy_map(summary_df)
    template_library = build_template_library(template_cache, static_cache, template_counts)
    dynamic_static_compare, dynamic_static_summary, sample_compare = build_dynamic_static_compare(
        tanshi_patients, template_cache, static_cache
    )
    budget_sensitivity = build_budget_sensitivity(tanshi_patients, budget_cache)
    monthly_policy_profile = build_monthly_policy_profile(monthly_df)
    validation_checks = build_validation_checks(
        tanshi_patients, summary_df, monthly_df, template_cache, dynamic_static_compare
    )
    age_summary, intensity_summary = build_age_and_intensity_summary(summary_df)

    save_workbook(
        "problem3_tables",
        {
            "全部患者汇总": summary_df,
            "全部患者月计划": monthly_df,
            "样本1-3汇总": sample_summary,
            "样本1-3动静对比": sample_compare,
            "样本1-3月计划": sample_monthly,
            "样本1-3动静月计划": sample_monthly_compare,
            "动静态汇总": dynamic_static_summary,
            "动静态逐人对比": dynamic_static_compare,
            "预算敏感性": budget_sensitivity,
            "模板库": template_library,
            "月度策略画像": monthly_policy_profile,
            "年龄段汇总": age_summary,
            "强度分层汇总": intensity_summary,
            "策略规律": policy_map,
            "可行性审计": validation_checks,
        },
    )

    save_table(summary_df, "problem3_all_patient_summary")
    save_table(sample_monthly, "problem3_sample_monthly_plan")
    save_table(sample_compare, "problem3_sample_plan_compare")
    save_table(sample_monthly_compare, "problem3_sample_monthly_compare")
    save_table(policy_map, "problem3_policy_map")
    save_table(template_library, "problem3_template_library")
    save_table(dynamic_static_compare, "problem3_dynamic_vs_static_compare")
    save_table(dynamic_static_summary, "problem3_dynamic_vs_static_summary")
    save_table(budget_sensitivity, "problem3_budget_sensitivity")
    save_table(monthly_policy_profile, "problem3_monthly_policy_profile")
    save_table(validation_checks, "problem3_validation_checks")
    save_table(age_summary, "problem3_age_group_summary")
    save_table(intensity_summary, "problem3_intensity_summary")

    plot_sample_trajectories(sample_monthly_compare)
    plot_population_strategy(summary_df, dynamic_static_summary, budget_sensitivity)

    overall_compare = dynamic_static_summary[dynamic_static_summary["分析组"] == "总体"].iloc[0]
    budget_2000 = budget_sensitivity[budget_sensitivity["预算上限"] == DEFAULT_BUDGET].copy()
    budget_1600 = budget_sensitivity[budget_sensitivity["预算上限"] == 1600].copy()

    summary = {
        "patient_count": int(len(summary_df)),
        "template_count": int(len(template_keys)),
        "average_final_score": round(summary_df["最终痰湿积分"].mean(), 4),
        "average_drop_rate": round(summary_df["积分下降率(%)"].mean(), 4),
        "average_total_cost": round(summary_df["总成本"].mean(), 4),
        "switch_patient_count": int((summary_df["切换次数"] > 0).sum()),
        "dynamic_better_than_static_count": int(overall_compare["动态优于静态人数"]),
        "dynamic_vs_static_mean_gain": float(overall_compare["平均终点改善值"]),
        "dynamic_vs_static_final_pvalue": None
        if pd.isna(overall_compare["最终积分Wilcoxon p"])
        else float(overall_compare["最终积分Wilcoxon p"]),
        "budget_1600_overall_final_score": float(
            budget_1600.loc[budget_1600["分析组"] == "总体", "平均最终积分"].iloc[0]
        ),
        "budget_2000_overall_final_score": float(
            budget_2000.loc[budget_2000["分析组"] == "总体", "平均最终积分"].iloc[0]
        ),
        "sample_1_3_final_scores": {
            str(int(row["样本ID"])): float(row["最终痰湿积分"]) for _, row in sample_summary.iterrows()
        },
    }
    save_json(summary, "problem3_summary")

    print("\n样本1-3最优方案汇总：")
    print(
        sample_summary[
            [
                "样本ID",
                "年龄段",
                "初始痰湿积分",
                "最大可选活动强度",
                "最终痰湿积分",
                "积分下降率(%)",
                "总成本",
                "切换次数",
            ]
        ].to_string(index=False)
    )

    print("\n动态 vs 固定活动基线：")
    print(dynamic_static_summary.to_string(index=False))

    print("\n预算敏感性（总体与各强度组）：")
    print(budget_sensitivity.to_string(index=False))

    print("\n可行性审计：")
    print(validation_checks.to_string(index=False))
    print("\n图表与表格已输出到 src/outputs 目录。")


if __name__ == "__main__":
    main()
