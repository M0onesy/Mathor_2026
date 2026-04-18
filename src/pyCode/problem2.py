"""
问题2（修订版 V2）：高血脂症三级风险预警
======================================
设计理念：
  ① 规避旧版"概率分位切分"的两个陷阱：标签泄漏污染、样本依赖阈值；
  ② 采用"硬规则层 + 双轴可解释评分卡"的双层预警体系；
  ③ 以 Apriori 关联规则代替单一决策树，挖掘痰湿体质高风险核心组合；
  ④ 增加"校准曲线 / Bootstrap 稳定性 / 敏感性分析"三重验证。

模型结构：
  Step A: 硬规则层（3 条 if-then，直接判高风险，完全匹配题干示例）
    R1: 血脂异常项数 ≥ 2   → 高风险
    R2: 血脂异常 ≥1 项 且 痰湿积分 ≥ 60   → 高风险（题干例 1）
    R3: 痰湿积分 ≥ 80 且 活动量表 < 40   → 高风险（题干例 2）
  Step B: 双轴评分层（仅对硬规则未触发者）
    S_clin 临床严重度 (0–7)  — 基于 2016 中国血脂指南阈值
    S_prog 进展风险 (0–9)    — 基于中医体质 + ASCVD 流行病学因素
    S_total = S_clin + S_prog ∈ [0, 15]
    阈值: S ≤ 2 → 低; 3 ≤ S ≤ 5 → 中; S ≥ 6 → 高
  Step C: Apriori 关联规则挖掘痰湿体质高风险核心组合
  Step D: 三项验证（校准 / Bootstrap / 敏感性）
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.metrics import roc_curve, roc_auc_score

from common import configure_plotting, figure_path, load_data, set_random_seed, table_path

warnings.filterwarnings("ignore")
configure_plotting()
set_random_seed()

# =====================================================================
# 基础工具
# =====================================================================

def count_abnormal(r: pd.Series) -> int:
    """血脂异常项数（与诊断标签定义 100% 一致）。"""
    c = 0
    if r["TC（总胆固醇）"] > 6.2: c += 1
    if r["TG（甘油三酯）"] > 1.7: c += 1
    if r["LDL-C（低密度脂蛋白）"] > 3.1: c += 1
    if r["HDL-C（高密度脂蛋白）"] < 1.04: c += 1
    return c


def s_clin_func(r: pd.Series) -> int:
    """临床严重度 S_clin ∈ [0, 7]。依据：《中国成人血脂异常防治指南(2016)》。"""
    s = 0
    tc = r["TC（总胆固醇）"]
    tg = r["TG（甘油三酯）"]
    ldl = r["LDL-C（低密度脂蛋白）"]
    hdl = r["HDL-C（高密度脂蛋白）"]
    if tc > 7.2: s += 2
    elif tc > 6.2: s += 1
    if tg > 2.3: s += 2
    elif tg > 1.7: s += 1
    if ldl > 4.1: s += 2
    elif ldl > 3.1: s += 1
    if hdl < 1.04: s += 1
    return s


def s_prog_func(r: pd.Series) -> int:
    """进展风险 S_prog ∈ [0, 9]。
    依据：中医体质学痰湿分级 + WHO/中国 BMI 分级 + ASCVD 流行病学危险因素。
    """
    s = 0
    phl = r["痰湿质"]
    bmi = r["BMI"]
    act = r["活动量表总分（ADL总分+IADL总分）"]
    age = int(r["年龄组"])
    sex = int(r["性别"])  # 1 = 男
    smk = int(r["吸烟史"])

    # 痰湿积分（三档）
    if phl >= 80:        s += 3
    elif phl >= 60:      s += 2
    elif phl >= 40:      s += 1
    # BMI（两档）
    if bmi >= 28:        s += 2
    elif bmi >= 24:      s += 1
    # 活动量表低
    if act < 40:         s += 1
    # 年龄 ≥60
    if age >= 3:         s += 1
    # 男性（心血管独立危险因素）
    if sex == 1:         s += 1
    # 吸烟
    if smk == 1:         s += 1
    return s


def hard_rule_trigger(r: pd.Series) -> tuple[int, str]:
    """硬规则层：返回 (触发规则号 0=未触发, 1/2/3=R1/R2/R3)."""
    n = r["血脂异常项数"]; phl = r["痰湿质"]
    act = r["活动量表总分（ADL总分+IADL总分）"]
    if n >= 2:
        return 1, "R1 血脂异常≥2项"
    if n >= 1 and phl >= 60:
        return 2, "R2 血脂异常≥1 且 痰湿积分≥60"
    if phl >= 80 and act < 40:
        return 3, "R3 痰湿积分≥80 且 活动量表<40"
    return 0, "未触发"


def classify_score(s: int, t_low: int = 2, t_high: int = 5) -> int:
    """评分层分层：S ≤ t_low 低；t_low < S ≤ t_high 中；S > t_high 高。"""
    if s <= t_low:   return 1  # 低
    if s <= t_high:  return 2  # 中
    return 3                    # 高


def final_risk(r: pd.Series, t_low: int = 2, t_high: int = 5) -> int:
    """最终风险等级：硬规则触发→3；否则用评分层。"""
    rule_id, _ = hard_rule_trigger(r)
    if rule_id > 0:
        return 3
    return classify_score(r["S_total"], t_low, t_high)


# =====================================================================
# Step 1: 数据与特征准备
# =====================================================================

df = load_data()
df["血脂异常项数"] = df.apply(count_abnormal, axis=1)
df["S_clin"] = df.apply(s_clin_func, axis=1)
df["S_prog"] = df.apply(s_prog_func, axis=1)
df["S_total"] = df["S_clin"] + df["S_prog"]

# 硬规则触发
rule_info = df.apply(hard_rule_trigger, axis=1)
df["硬规则编号"] = rule_info.map(lambda x: x[0])
df["硬规则描述"] = rule_info.map(lambda x: x[1])

# 最终分层
df["最终风险"] = df.apply(final_risk, axis=1)
df["风险等级"] = df["最终风险"].map({1: "低风险", 2: "中风险", 3: "高风险"})
n = len(df)

# =====================================================================
# Step 2: 分层结果统计
# =====================================================================
print("=" * 66); print("Step 2: 三级分层结果"); print("=" * 66)

risk_counts = df["风险等级"].value_counts().reindex(["低风险", "中风险", "高风险"])
print("\n三级风险分布：")
print(risk_counts)

risk_stat = df.groupby("风险等级", observed=True).agg(
    人数=("样本ID", "count"),
    实际发病率=("高血脂症二分类标签", "mean"),
    平均S_clin=("S_clin", "mean"),
    平均S_prog=("S_prog", "mean"),
    平均S_total=("S_total", "mean"),
    平均痰湿=("痰湿质", "mean"),
    平均活动=("活动量表总分（ADL总分+IADL总分）", "mean"),
    平均BMI=("BMI", "mean"),
).reindex(["低风险", "中风险", "高风险"]).round(3)
print("\n各层特征均值：")
print(risk_stat)
risk_stat.to_csv(table_path("Q2_risk_stat.csv"), encoding="utf-8-sig")

# 硬规则触发统计
print("\n硬规则触发分布：")
print(df["硬规则描述"].value_counts())
rule_trigger = df.groupby("硬规则描述")["高血脂症二分类标签"].agg(["count", "mean"]).round(3)
rule_trigger.to_csv(table_path("Q2_hard_rule_trigger.csv"), encoding="utf-8-sig")

# 诊断交叉表
print("\n诊断标签 × 风险等级 交叉表：")
print(pd.crosstab(df["高血脂症二分类标签"], df["风险等级"]))

# =====================================================================
# Step 3: 校准曲线（S_total vs 实际发病率）
# =====================================================================
print("\n" + "=" * 66); print("Step 3: 校准曲线"); print("=" * 66)

cal = df.groupby("S_total")["高血脂症二分类标签"].agg(["count", "mean"])
cal_percent = cal.copy()
cal_percent["mean"] = (cal_percent["mean"] * 100).round(1)
print("\nS_total 与发病率对应表（%）：")
print(cal_percent.rename(columns={"count": "人数", "mean": "发病率(%)"}))
cal_percent.to_csv(table_path("Q2_calibration.csv"), encoding="utf-8-sig")

# AUC
auc_total = roc_auc_score(df["高血脂症二分类标签"], df["S_total"])
auc_clin = roc_auc_score(df["高血脂症二分类标签"], df["S_clin"])
auc_prog = roc_auc_score(df["高血脂症二分类标签"], df["S_prog"])
print(f"\nS_clin AUC  = {auc_clin:.4f}")
print(f"S_prog AUC  = {auc_prog:.4f}")
print(f"S_total AUC = {auc_total:.4f}")

# =====================================================================
# Step 4: Bootstrap 稳定性（1000 次）
# =====================================================================
print("\n" + "=" * 66); print("Step 4: Bootstrap 稳定性 (1000 次)"); print("=" * 66)

rng = np.random.default_rng(42)
B = 1000
boot_dist = {"low_pct": [], "mid_pct": [], "high_pct": [],
             "low_rate": [], "mid_rate": [], "high_rate": []}
for b in range(B):
    idx = rng.integers(0, n, size=n)
    sb = df.iloc[idx]
    for lv, key_p, key_r in [(1, "low_pct", "low_rate"),
                              (2, "mid_pct", "mid_rate"),
                              (3, "high_pct", "high_rate")]:
        mask = sb["最终风险"] == lv
        boot_dist[key_p].append(mask.mean() * 100)
        boot_dist[key_r].append(
            sb.loc[mask, "高血脂症二分类标签"].mean() * 100 if mask.any() else np.nan
        )

boot_summary = pd.DataFrame({
    "指标": ["低风险占比(%)", "中风险占比(%)", "高风险占比(%)",
             "低层发病率(%)", "中层发病率(%)", "高层发病率(%)"],
    "均值": [np.mean(boot_dist[k]) for k in ["low_pct", "mid_pct", "high_pct",
                                              "low_rate", "mid_rate", "high_rate"]],
    "2.5%": [np.percentile(boot_dist[k], 2.5) for k in ["low_pct", "mid_pct", "high_pct",
                                                         "low_rate", "mid_rate", "high_rate"]],
    "97.5%": [np.percentile(boot_dist[k], 97.5) for k in ["low_pct", "mid_pct", "high_pct",
                                                           "low_rate", "mid_rate", "high_rate"]],
}).round(2)
print(boot_summary.to_string(index=False))
boot_summary.to_csv(table_path("Q2_bootstrap_CI.csv"), encoding="utf-8-sig", index=False)

# =====================================================================
# Step 5: 敏感性分析（权重 ±25% & 阈值扰动）
# =====================================================================
print("\n" + "=" * 66); print("Step 5: 敏感性分析"); print("=" * 66)

base_risk = df["最终风险"].values

def compute_risk_with_prog_scale(scale: float) -> np.ndarray:
    """把 S_prog 乘以 scale 后重新分层，硬规则不变。"""
    new_prog = (df["S_prog"] * scale).round().astype(int)
    new_total = df["S_clin"] + new_prog
    rule_ids = df["硬规则编号"].values
    out = np.zeros(n, dtype=int)
    s_arr = new_total.values
    for i in range(n):
        if rule_ids[i] > 0:
            out[i] = 3
        else:
            s = s_arr[i]
            if s <= 2: out[i] = 1
            elif s <= 5: out[i] = 2
            else: out[i] = 3
    return out

def compute_risk_with_threshold(t_low: int, t_high: int) -> np.ndarray:
    rule_ids = df["硬规则编号"].values
    s_tot = df["S_total"].values
    out = np.zeros(n, dtype=int)
    for i in range(n):
        if rule_ids[i] > 0:
            out[i] = 3
        else:
            s = s_tot[i]
            if s <= t_low: out[i] = 1
            elif s <= t_high: out[i] = 2
            else: out[i] = 3
    return out

sens_rows = []
for label, new_r in [
    ("S_prog 权重 ×0.75", compute_risk_with_prog_scale(0.75)),
    ("S_prog 权重 ×1.25", compute_risk_with_prog_scale(1.25)),
    ("阈值 (1, 5)",       compute_risk_with_threshold(1, 5)),
    ("阈值 (3, 5)",       compute_risk_with_threshold(3, 5)),
    ("阈值 (2, 4)",       compute_risk_with_threshold(2, 4)),
    ("阈值 (2, 6)",       compute_risk_with_threshold(2, 6)),
]:
    agree = (new_r == base_risk).mean() * 100
    sens_rows.append({"扰动": label, "一致率(%)": round(agree, 2),
                      "低(%)": round((new_r == 1).mean() * 100, 2),
                      "中(%)": round((new_r == 2).mean() * 100, 2),
                      "高(%)": round((new_r == 3).mean() * 100, 2)})
sens_df = pd.DataFrame(sens_rows)
print(sens_df.to_string(index=False))
sens_df.to_csv(table_path("Q2_sensitivity.csv"), encoding="utf-8-sig", index=False)

# =====================================================================
# Step 6: 痰湿体质 Apriori 关联规则
# =====================================================================
print("\n" + "=" * 66); print("Step 6: 痰湿体质 Apriori 关联规则"); print("=" * 66)

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    MLXTEND_OK = True
except Exception as e:
    print(f"[WARN] mlxtend 不可用，跳过 Apriori: {e}")
    MLXTEND_OK = False

dft = df[df["体质标签"] == 5].copy()
print(f"痰湿体质样本数: {len(dft)}  高风险: {(dft['最终风险']==3).sum()} 人")

top_rules = pd.DataFrame()
if MLXTEND_OK:
    items = pd.DataFrame({
        "痰湿≥60":          (dft["痰湿质"] >= 60).values,
        "BMI超重":          (dft["BMI"] >= 24).values,
        "BMI肥胖":          (dft["BMI"] >= 28).values,
        "活动<50":          (dft["活动量表总分（ADL总分+IADL总分）"] < 50).values,
        "活动<40":          (dft["活动量表总分（ADL总分+IADL总分）"] < 40).values,
        "年龄≥60":          (dft["年龄组"] >= 3).values,
        "男性":              (dft["性别"] == 1).values,
        "吸烟":              (dft["吸烟史"] == 1).values,
        "血脂异常≥1":        (dft["血脂异常项数"] >= 1).values,
        "血脂异常≥2":        (dft["血脂异常项数"] >= 2).values,
        "TG升高":           (dft["TG（甘油三酯）"] > 1.7).values,
        "TC升高":           (dft["TC（总胆固醇）"] > 6.2).values,
        "HDL偏低":           (dft["HDL-C（高密度脂蛋白）"] < 1.04).values,
        "高风险":            (dft["最终风险"] == 3).values,
    })

    freq = apriori(items, min_support=0.10, use_colnames=True, max_len=4)
    rules = association_rules(freq, metric="confidence", min_threshold=0.9)
    # 仅保留结论为 "高风险"、前件不含高风险的规则
    rules = rules[rules["consequents"].apply(lambda x: "高风险" in x and len(x) == 1)]
    rules = rules[rules["antecedents"].apply(lambda x: "高风险" not in x)]
    rules = rules[rules["lift"] >= 1.1]
    rules = rules.sort_values(by=["lift", "support"], ascending=[False, False])

    def fmt_set(s):
        return " ∧ ".join(sorted(s))
    rules_out = pd.DataFrame({
        "核心组合": rules["antecedents"].apply(fmt_set),
        "支持度": rules["support"].round(3),
        "置信度": rules["confidence"].round(3),
        "提升度": rules["lift"].round(3),
        "组合项数": rules["antecedents"].apply(len).values,
    })
    top_rules = rules_out.groupby("组合项数", group_keys=False).head(6)
    print("\nApriori 关联规则 (前件组合 → 高风险; 最多4项)：")
    print(top_rules.head(15).to_string(index=False))
    top_rules.to_csv(table_path("Q2_apriori_rules.csv"), encoding="utf-8-sig", index=False)

# 决策树对比（保留原版，用于论文图）
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
tree_feats = ["痰湿质", "活动量表总分（ADL总分+IADL总分）", "TG（甘油三酯）", "TC（总胆固醇）",
              "LDL-C（低密度脂蛋白）", "HDL-C（高密度脂蛋白）", "BMI", "血尿酸", "年龄组"]
Xt = dft[tree_feats].values
yt = (dft["最终风险"] == 3).astype(int).values
tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=15, random_state=42)
tree.fit(Xt, yt)
rules_text = export_text(tree, feature_names=tree_feats)
table_path("Q2_tree_rules.txt").write_text(rules_text, encoding="utf-8")

# =====================================================================
# Step 7: 六面板可视化（Q2_summary_v2.png）
# =====================================================================
print("\n" + "=" * 66); print("Step 7: 六面板可视化"); print("=" * 66)

fig = plt.figure(figsize=(16, 11))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

# (a) 校准曲线
ax = fig.add_subplot(gs[0, 0])
ax.plot(cal_percent.index, cal_percent["mean"], "o-", color="steelblue", lw=2, markersize=6)
for s_val, row in cal_percent.iterrows():
    ax.annotate(f"n={int(row['count'])}", (s_val, row["mean"]),
                xytext=(0, 7), textcoords="offset points",
                ha="center", fontsize=7, color="gray")
ax.axvline(2.5, color="green", ls="--", alpha=0.7, label="Low/Med cut")
ax.axvline(5.5, color="red", ls="--", alpha=0.7, label="Med/High cut")
ax.set_xlabel("S_total"); ax.set_ylabel("Observed prevalence (%)")
ax.set_title("(a) Calibration curve (monotone)")
ax.set_ylim(-5, 108)
ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")

# (b) 三级分层与发病率
ax = fig.add_subplot(gs[0, 1])
rd = df.groupby("风险等级")["高血脂症二分类标签"].agg(["count", "mean"]).reindex(
    ["低风险", "中风险", "高风险"])
x = np.arange(3); ax2 = ax.twinx()
ax.bar(x - 0.2, rd["count"], 0.4, color="skyblue", label="# cases")
ax2.bar(x + 0.2, rd["mean"] * 100, 0.4, color="salmon", label="prevalence %")
for i, (c, m) in enumerate(zip(rd["count"], rd["mean"])):
    ax.text(i - 0.2, c + 15, f"{int(c)}", ha="center", fontsize=9)
    ax2.text(i + 0.2, m * 100 + 3, f"{m*100:.1f}%", ha="center", fontsize=9, color="darkred")
ax.set_xticks(x); ax.set_xticklabels(["Low", "Medium", "High"])
ax.set_ylabel("# cases")
ax2.set_ylabel("prevalence (%)"); ax2.set_ylim(0, 115)
ax.set_title("(b) Three-level risk stratification")
ax.legend(loc="upper left", fontsize=8)
ax2.legend(loc="upper right", fontsize=8)

# (c) S_clin × S_prog 发病率热图
ax = fig.add_subplot(gs[0, 2])
pivot = df.pivot_table(index="S_prog", columns="S_clin",
                        values="高血脂症二分类标签", aggfunc="mean")
# 用 cnt_pivot 标注人数
cnt_pivot = df.pivot_table(index="S_prog", columns="S_clin",
                            values="高血脂症二分类标签", aggfunc="count")
im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto",
               origin="lower", vmin=0, vmax=1)
ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        c = cnt_pivot.values[i, j]
        if not np.isnan(c) and c > 0:
            ax.text(j, i, f"{int(c)}", ha="center", va="center",
                    fontsize=6, color="black")
ax.set_xlabel("S_clin"); ax.set_ylabel("S_prog")
ax.set_title("(c) Prevalence heatmap: S_clin × S_prog")
plt.colorbar(im, ax=ax, fraction=0.046, label="prevalence")

# (d) ROC：S_clin / S_prog / S_total
ax = fig.add_subplot(gs[1, 0])
for name, score, auc in [("S_clin", df["S_clin"], auc_clin),
                         ("S_prog", df["S_prog"], auc_prog),
                         ("S_total", df["S_total"], auc_total)]:
    fpr, tpr, _ = roc_curve(df["高血脂症二分类标签"], score)
    ax.plot(fpr, tpr, lw=1.8, label=f"{name} (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("(d) ROC curves")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# (e) 高风险来源构成
ax = fig.add_subplot(gs[1, 1])
high_df = df[df["最终风险"] == 3].copy()
def high_source(r):
    rid = r["硬规则编号"]
    if rid == 1: return "R1 blood-lipid ≥ 2"
    if rid == 2: return "R2 blood-lipid + phlegm ≥ 60"
    if rid == 3: return "R3 phlegm ≥ 80 & act < 40"
    return "Score S ≥ 6"
high_df["来源"] = high_df.apply(high_source, axis=1)
src_cnt = high_df["来源"].value_counts()
colors_src = ["#E74C3C", "#F39C12", "#27AE60", "#2980B9"]
bars = ax.barh(range(len(src_cnt)), src_cnt.values,
               color=colors_src[:len(src_cnt)])
ax.set_yticks(range(len(src_cnt))); ax.set_yticklabels(src_cnt.index, fontsize=9)
for i, v in enumerate(src_cnt.values):
    pct = v / src_cnt.sum() * 100
    ax.text(v + 8, i, f"{v} ({pct:.1f}%)", va="center", fontsize=8)
ax.set_xlabel("# high-risk cases")
ax.set_title(f"(e) High-risk trigger source (n={len(high_df)})")
ax.set_xlim(0, src_cnt.max() * 1.28)

# (f) 敏感性：权重扰动下三级占比
ax = fig.add_subplot(gs[1, 2])
labels_en = ["baseline", "S_prog×0.75", "S_prog×1.25",
             "(1,5)", "(3,5)", "(2,4)", "(2,6)"]
variants_r = [base_risk,
              compute_risk_with_prog_scale(0.75),
              compute_risk_with_prog_scale(1.25),
              compute_risk_with_threshold(1, 5),
              compute_risk_with_threshold(3, 5),
              compute_risk_with_threshold(2, 4),
              compute_risk_with_threshold(2, 6)]
low_pct = [(r == 1).mean() * 100 for r in variants_r]
mid_pct = [(r == 2).mean() * 100 for r in variants_r]
high_pct = [(r == 3).mean() * 100 for r in variants_r]
x_pos = np.arange(len(labels_en))
ax.bar(x_pos, low_pct, label="Low", color="#2ECC71")
ax.bar(x_pos, mid_pct, bottom=low_pct, label="Medium", color="#F39C12")
ax.bar(x_pos, high_pct, bottom=np.array(low_pct) + np.array(mid_pct),
       label="High", color="#E74C3C")
ax.set_xticks(x_pos); ax.set_xticklabels(labels_en, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Percentage (%)"); ax.set_ylim(0, 100)
ax.set_title("(f) Sensitivity: composition stability")
ax.legend(fontsize=8, loc="upper right")

plt.suptitle("Problem 2: Hard-rule + Two-Axis Score Card — Verification Summary",
             fontsize=13, fontweight="bold", y=0.995)
plt.savefig(figure_path("Q2_summary_v2.png"), dpi=150, bbox_inches="tight")
plt.close()

# =====================================================================
# Step 8: 输出全量分层表与摘要
# =====================================================================
print("\n" + "=" * 66); print("Step 8: 输出全量表格"); print("=" * 66)

out_cols = ["样本ID", "体质标签", "痰湿质", "活动量表总分（ADL总分+IADL总分）",
            "TG（甘油三酯）", "TC（总胆固醇）", "LDL-C（低密度脂蛋白）",
            "HDL-C（高密度脂蛋白）", "BMI", "年龄组", "性别", "吸烟史",
            "血脂异常项数", "S_clin", "S_prog", "S_total",
            "硬规则编号", "硬规则描述", "最终风险", "风险等级",
            "高血脂症二分类标签"]
df[out_cols].to_csv(table_path("Q2_stratification_full.csv"),
                    encoding="utf-8-sig", index=False)

summary = pd.DataFrame({
    "评价指标": ["S_clin AUC", "S_prog AUC", "S_total AUC",
                 "低风险发病率(%)", "中风险发病率(%)", "高风险发病率(%)",
                 "低层占比(%)", "中层占比(%)", "高层占比(%)",
                 "硬规则触发总数", "R1 触发数", "R2 触发数", "R3 触发数"],
    "数值": [f"{auc_clin:.4f}", f"{auc_prog:.4f}", f"{auc_total:.4f}",
             f"{risk_stat.loc['低风险','实际发病率']*100:.1f}",
             f"{risk_stat.loc['中风险','实际发病率']*100:.1f}",
             f"{risk_stat.loc['高风险','实际发病率']*100:.1f}",
             f"{risk_counts['低风险']/n*100:.1f}",
             f"{risk_counts['中风险']/n*100:.1f}",
             f"{risk_counts['高风险']/n*100:.1f}",
             (df['硬规则编号']>0).sum(),
             (df['硬规则编号']==1).sum(),
             (df['硬规则编号']==2).sum(),
             (df['硬规则编号']==3).sum()]
})
print("\n[最终摘要]")
print(summary.to_string(index=False))
summary.to_csv(table_path("Q2_summary_metrics.csv"),
               encoding="utf-8-sig", index=False)

print("\n[OK] Problem 2 V2 全部输出完成")
