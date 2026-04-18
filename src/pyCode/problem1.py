"""
问题1：关键指标筛选与九种体质贡献度分析（修订版）

针对题目严格要求：
  - 候选特征仅为血常规指标 + 中老年人活动量表评分（不含性别/吸烟/饮酒/年龄组）
  - 活动量表需考虑 ADL/IADL 各子项
  - 多重共线性诊断与处理
  - 指标重要度评估采用 5 种互补方法 + Borda 秩次聚合（而非 Z-score 加和）
"""
import numpy as np, pandas as pd, warnings
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import ElasticNetCV, LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from matplotlib import rcParams
from common import configure_plotting, figure_path, load_data, set_random_seed, table_path

warnings.filterwarnings('ignore')
configure_plotting()

set_random_seed()
df = load_data()

# ===================================================================
# Step 0: 数据清洗与质量检查
# ===================================================================
print("="*60); print("STEP 0: 数据质量检查"); print("="*60)

# 检查缺失、重复
print(f"总样本数: {len(df)}  列数: {len(df.columns)}")
print(f"缺失值总数: {df.isnull().sum().sum()}")
print(f"样本ID重复: {df['样本ID'].duplicated().sum()}")

# ADL/IADL 子项和与总分一致性
adl_items = ['ADL用厕','ADL吃饭','ADL步行','ADL穿衣','ADL洗澡']
iadl_items = ['IADL购物','IADL做饭','IADL理财','IADL交通','IADL服药']
adl_ok = (df[adl_items].sum(axis=1) == df['ADL总分']).all()
iadl_ok = (df[iadl_items].sum(axis=1) == df['IADL总分']).all()
tot_ok = (df['ADL总分'] + df['IADL总分'] == df['活动量表总分（ADL总分+IADL总分）']).all()
print(f"ADL子项之和=总分: {adl_ok};  IADL: {iadl_ok};  总分: {tot_ok}")

# 体质标签 vs 最高体质积分一致性
tizhi_cols = ['平和质','气虚质','阳虚质','阴虚质','痰湿质','湿热质','血瘀质','气郁质','特禀质']
argmax_id = df[tizhi_cols].values.argmax(axis=1) + 1
lab_consistency = (argmax_id == df['体质标签'].values).sum()
print(f"体质标签与最高积分体质一致的样本: {lab_consistency}/{len(df)} ({lab_consistency/len(df)*100:.2f}%)")
print(f"不一致样本数: {len(df) - lab_consistency}  (主要体质判定歧义, 保留但在分析中说明)")

# 诊断标签与血脂指标一致性
def abn_count(r):
    c=0
    if r['TC（总胆固醇）']>6.2: c+=1
    if r['TG（甘油三酯）']>1.7: c+=1
    if r['LDL-C（低密度脂蛋白）']>3.1: c+=1
    if r['HDL-C（高密度脂蛋白）']<1.04: c+=1
    return c
df['血脂异常项数'] = df.apply(abn_count, axis=1)
df['临床异常'] = (df['血脂异常项数']>=1).astype(int)
match_rate = (df['临床异常']==df['高血脂症二分类标签']).mean()
print(f"诊断标签 vs 血脂异常≥1 一致率: {match_rate*100:.1f}%")

# 异常值检测 (Tukey IQR 法)
print("\n异常值检测 (1.5×IQR 准则):")
for col in ['TC（总胆固醇）','TG（甘油三酯）','LDL-C（低密度脂蛋白）','HDL-C（高密度脂蛋白）',
            '空腹血糖','血尿酸','BMI']:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1-1.5*iqr, q3+1.5*iqr
    out = ((df[col]<lo) | (df[col]>hi)).sum()
    print(f"  {col}: outliers={out} ({out/len(df)*100:.1f}%); 范围[{df[col].min():.2f}, {df[col].max():.2f}]")

# 保留原始数据做分析（离群不删除，毕竟临床数据离群也有意义）
print("\n→ 处理策略: 保留所有 1000 条记录; 标注 65 例体质判定歧义但不剔除;")
print("             异常值多数为临床真实疾病表现, 保留; 后续分析中对离群稳健方法(Spearman, RF)不敏感.")

# ===================================================================
# Step 1: 候选特征清单 —— 严格限定为血常规 + 活动量表
# ===================================================================
print("\n"+"="*60); print("STEP 1: 候选特征清单"); print("="*60)

blood = ['TC（总胆固醇）','TG（甘油三酯）','LDL-C（低密度脂蛋白）',
         'HDL-C（高密度脂蛋白）','空腹血糖','血尿酸','BMI']
adl_sub = adl_items          # 5 项 ADL 子项
iadl_sub = iadl_items        # 5 项 IADL 子项
adl_total = ['ADL总分','IADL总分','活动量表总分（ADL总分+IADL总分）']

# Set A: 最细粒度 —— 血常规 + ADL/IADL 子项
SET_A = blood + adl_sub + iadl_sub              # 17 个特征
# Set B: 总分 —— 血常规 + ADL/IADL 总分
SET_B = blood + adl_total                        # 10 个特征

print(f"Set A (细粒度, {len(SET_A)}个): {SET_A}")
print(f"Set B (总分级, {len(SET_B)}个): {SET_B}")

# ===================================================================
# Step 2: 多重共线性诊断 (VIF)
# ===================================================================
print("\n"+"="*60); print("STEP 2: 多重共线性诊断"); print("="*60)

def vif_table(df_in, cols):
    X = StandardScaler().fit_transform(df_in[cols].values)
    rows=[]
    for i, c in enumerate(cols):
        try:
            v = variance_inflation_factor(X, i)
        except Exception:
            v = np.nan
        rows.append({'feature': c, 'VIF': v})
    return pd.DataFrame(rows).sort_values('VIF', ascending=False)

vif_A = vif_table(df, SET_A)
print("\nVIF (Set A, 细粒度):"); print(vif_A.round(2).to_string(index=False))
vif_B = vif_table(df, SET_B)
print("\nVIF (Set B, 总分):"); print(vif_B.round(2).to_string(index=False))

vif_A.to_csv(table_path('Q1_vif_setA.csv'), encoding='utf-8-sig', index=False)
vif_B.to_csv(table_path('Q1_vif_setB.csv'), encoding='utf-8-sig', index=False)

# Set B 中 "活动量表总分" = ADL总分+IADL总分 必然完美共线 -> 只保留总分 或 两分项
# 筛选策略: 使用 Set C = 血常规 + ADL/IADL 子项 (剔除总分), 这是既保留细节又避免完美共线
# 由于 ADL子项之和=ADL总分, 若同时纳入会完全共线, 故采取"子项 XOR 总分"
SET_C = blood + adl_sub + iadl_sub              # 同 Set A
print(f"\n选定 Set C = 血常规({len(blood)}) + ADL子项({len(adl_sub)}) + IADL子项({len(iadl_sub)}) = {len(SET_C)} 个")
print("(不纳入总分以避免完全共线 ADL_sum = ΣADL_i)")

# VIF 再检查 Set C (同 Set A)
vif_C = vif_A.copy()
# 去除超高共线特征 (VIF > 10, 除血脂相关)
to_keep = vif_C.copy()
print(f"\nSet C 中 VIF 最大的指标: {vif_C.iloc[0]['feature']} = {vif_C.iloc[0]['VIF']:.2f}")
# 允许血脂指标留存 (有明确临床意义), 只做标注

# ===================================================================
# Step 3: 相关性矩阵与热力图
# ===================================================================
print("\n"+"="*60); print("STEP 3: 相关性矩阵"); print("="*60)

corr_mat = df[SET_C].corr(method='spearman')
print("相关矩阵(Spearman) 绝对值 > 0.5 的对:")
for i in range(len(SET_C)):
    for j in range(i+1, len(SET_C)):
        v = corr_mat.iloc[i,j]
        if abs(v) > 0.5:
            print(f"  {SET_C[i]} - {SET_C[j]}: ρ = {v:.3f}")

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr_mat.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(SET_C))); ax.set_yticks(range(len(SET_C)))
labels_en = {
    'TC（总胆固醇）':'TC','TG（甘油三酯）':'TG','LDL-C（低密度脂蛋白）':'LDL-C',
    'HDL-C（高密度脂蛋白）':'HDL-C','空腹血糖':'FBS','血尿酸':'UA','BMI':'BMI',
    'ADL用厕':'ADL-toilet','ADL吃饭':'ADL-eat','ADL步行':'ADL-walk',
    'ADL穿衣':'ADL-dress','ADL洗澡':'ADL-bath',
    'IADL购物':'IADL-shop','IADL做饭':'IADL-cook','IADL理财':'IADL-finance',
    'IADL交通':'IADL-transp','IADL服药':'IADL-med'
}
en_labels = [labels_en.get(c, c[:10]) for c in SET_C]
ax.set_xticklabels(en_labels, rotation=45, ha='right')
ax.set_yticklabels(en_labels)
for i in range(len(SET_C)):
    for j in range(len(SET_C)):
        v = corr_mat.values[i,j]
        if abs(v) > 0.3:
            ax.text(j, i, f"{v:.2f}", ha='center', va='center',
                    color='white' if abs(v) > 0.6 else 'black', fontsize=7)
plt.colorbar(im, ax=ax)
ax.set_title('Spearman Correlation Matrix (Blood + ADL/IADL sub-items)')
plt.tight_layout()
plt.savefig(figure_path('Q1_corr_heatmap.png'), dpi=150)
plt.close()

# ===================================================================
# Step 4: 五种互补方法评估
# ===================================================================
print("\n"+"="*60); print("STEP 4: 五种互补方法特征评估"); print("="*60)
print("""
方法论 (5 种互补方法):
  M1. Spearman 秩相关        (非参数 · 单调 · 稳健)
  M2. 互信息 MI              (非参数 · 非线性 · 捕捉任意依赖)
  M3. 弹性网络回归           (L1+L2 联合惩罚 · 稀疏且稳健处理共线特征组)
  M4. 随机森林置换重要度     (嵌入式 · 非参数化 · 捕捉交互)
  M5. 偏相关                 (参数 · 控制其它变量后的净效应)

聚合: 对每种方法得到的指标排名, 使用 Borda 计数法 (排序聚合) 而非 Z-score 加和,
      以回避数值量纲不同、分布偏斜等问题, 得到稳健一致的最终排名.
""")

def borda_rank(score_dict_list, features, higher_better=True):
    """score_dict_list: list of dicts (feature -> score). higher_better=True 则 score 越高名次越前"""
    k = len(features)
    points = {f: 0 for f in features}
    for sd in score_dict_list:
        # 排名: higher_better ? 降序排 : 升序排
        sorted_feats = sorted(features,
                              key=lambda f: sd.get(f, 0),
                              reverse=higher_better)
        for rank, f in enumerate(sorted_feats):
            points[f] += (k - rank)  # 第1名得k分, 末位得1分
    return points

# ===== 4.1 表征痰湿体质严重程度 —— 以痰湿积分为连续目标 =====
print("\n--- 4.1 表征痰湿体质严重程度(痰湿积分为目标) ---")
y_phl = df['痰湿质'].values
X_phl = df[SET_C].values
Xs = StandardScaler().fit_transform(X_phl)

m1_phl, m2_phl, m3_phl, m4_phl, m5_phl = {}, {}, {}, {}, {}

# M1 Spearman
for f in SET_C:
    r, _ = stats.spearmanr(df[f].values, y_phl)
    m1_phl[f] = abs(r)

# M2 MI
mi = mutual_info_regression(X_phl, y_phl, random_state=42)
for i, f in enumerate(SET_C):
    m2_phl[f] = mi[i]

# M3 弹性网络
enet_l1_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
enet = ElasticNetCV(
    l1_ratio=enet_l1_ratios,
    alphas=200,
    cv=5,
    random_state=42,
    max_iter=20000,
)
enet.fit(Xs, y_phl)
print(f"M3 回归最优参数: best_alpha={enet.alpha_:.6f}, best_l1_ratio={enet.l1_ratio_:.2f}")
for i, f in enumerate(SET_C):
    m3_phl[f] = abs(enet.coef_[i])

# M4 RF 置换重要度
rf = RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_phl, y_phl)
from sklearn.inspection import permutation_importance
perm = permutation_importance(rf, X_phl, y_phl, n_repeats=10, random_state=42, n_jobs=-1)
for i, f in enumerate(SET_C):
    m4_phl[f] = perm.importances_mean[i]

# M5 偏相关 (各特征与 y 控制其他特征后的偏相关系数)
# 用 OLS 回归计算 β 再转偏相关
import statsmodels.api as sm
X_const = sm.add_constant(Xs)
ols = sm.OLS(y_phl, X_const).fit()
# partial r = t / sqrt(t^2 + df)
tvals = ols.tvalues[1:]  # 去掉 const
dfo = ols.df_resid
for i, f in enumerate(SET_C):
    m5_phl[f] = abs(tvals[i] / np.sqrt(tvals[i]**2 + dfo))

# Borda 聚合
borda_phl = borda_rank([m1_phl, m2_phl, m3_phl, m4_phl, m5_phl], SET_C, higher_better=True)
df_phl = pd.DataFrame({
    'feature': SET_C,
    'Spearman|r|': [m1_phl[f] for f in SET_C],
    'MI': [m2_phl[f] for f in SET_C],
    'ENet|β|': [m3_phl[f] for f in SET_C],
    'RF_perm': [m4_phl[f] for f in SET_C],
    'Partial_r': [m5_phl[f] for f in SET_C],
    'Borda_pts': [borda_phl[f] for f in SET_C],
}).sort_values('Borda_pts', ascending=False)
df_phl['Borda_排名'] = range(1, len(SET_C)+1)

print("表征痰湿严重程度的特征排序 (Borda 聚合):")
print(df_phl.round(4).to_string(index=False))
df_phl.to_csv(table_path('Q1_phlegm_ranking.csv'), encoding='utf-8-sig', index=False)

# ===== 4.2 预警高血脂发病风险 —— 以二分类标签为目标 =====
print("\n--- 4.2 预警高血脂发病风险(二分类标签为目标) ---")
y_risk = df['高血脂症二分类标签'].values
m1_r, m2_r, m3_r, m4_r, m5_r = {}, {}, {}, {}, {}

# M1 Spearman (二分类也可用)
for f in SET_C:
    r, _ = stats.spearmanr(df[f].values, y_risk)
    m1_r[f] = abs(r)

# M2 MI (classif)
mi = mutual_info_classif(X_phl, y_risk, random_state=42)
for i, f in enumerate(SET_C):
    m2_r[f] = mi[i]

# M3 弹性网络 Logistic
logit_Cs = np.logspace(-2, 2, 20)
logit_l1_ratios = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
lr_enet = LogisticRegressionCV(
    cv=5,
    penalty='elasticnet',
    solver='saga',
    Cs=logit_Cs,
    l1_ratios=logit_l1_ratios,
    scoring='roc_auc',
    max_iter=20000,
    random_state=42,
    n_jobs=-1,
)
lr_enet.fit(Xs, y_risk)
print(f"M3 分类最优参数: best_C={lr_enet.C_[0]:.6f}, best_l1_ratio={lr_enet.l1_ratio_[0]:.2f}")
for i, f in enumerate(SET_C):
    m3_r[f] = abs(lr_enet.coef_[0][i])

# M4 RF 置换
rfc = RandomForestClassifier(n_estimators=500, max_depth=10,
                              random_state=42, class_weight='balanced', n_jobs=-1)
rfc.fit(X_phl, y_risk)
perm_r = permutation_importance(rfc, X_phl, y_risk, n_repeats=10, random_state=42, n_jobs=-1)
for i, f in enumerate(SET_C):
    m4_r[f] = perm_r.importances_mean[i]

# M5 偏相关 (Logistic 回归 Wald 检验)
lr_full = sm.Logit(y_risk, X_const).fit(disp=0, maxiter=2000)
zvals = lr_full.tvalues[1:]
for i, f in enumerate(SET_C):
    m5_r[f] = abs(zvals[i]) / (abs(zvals[i]) + 1)  # 标准化到 (0,1)

borda_r = borda_rank([m1_r, m2_r, m3_r, m4_r, m5_r], SET_C, higher_better=True)
df_risk = pd.DataFrame({
    'feature': SET_C,
    'Spearman|r|': [m1_r[f] for f in SET_C],
    'MI': [m2_r[f] for f in SET_C],
    'ENet|β|': [m3_r[f] for f in SET_C],
    'RF_perm': [m4_r[f] for f in SET_C],
    'Wald': [m5_r[f] for f in SET_C],
    'Borda_pts': [borda_r[f] for f in SET_C],
}).sort_values('Borda_pts', ascending=False)
df_risk['Borda_排名'] = range(1, len(SET_C)+1)

print("预警高血脂发病风险的特征排序 (Borda 聚合):")
print(df_risk.round(4).to_string(index=False))
df_risk.to_csv(table_path('Q1_risk_ranking.csv'), encoding='utf-8-sig', index=False)

# ===================================================================
# Step 5: 九种体质对发病风险的贡献度
# ===================================================================
print("\n"+"="*60); print("STEP 5: 九种体质对发病风险的贡献度"); print("="*60)

# 方法1: 标签为定类变量的贡献度 -> 独热编码后各体质的Logistic beta
# 方法2: 体质积分的多元Logistic beta
# 方法3: 各体质发病率比 vs 整体
# 方法4: 卡方检验（体质标签 vs 诊断）
# 方法5: Cohen's h (效应量)

y = df['高血脂症二分类标签'].values
p_all = y.mean()  # 整体发病率

# Method A: 独热编码体质标签 + Logistic回归
X_oh = pd.get_dummies(df['体质标签'], prefix='体质').values
lr_oh = LogisticRegression(max_iter=2000, random_state=42)
lr_oh.fit(X_oh, y)
beta_oh = lr_oh.coef_[0]

# Method B: 体质积分多元 Logistic
Xtz = StandardScaler().fit_transform(df[tizhi_cols].values)
lr_tz = LogisticRegression(max_iter=2000, random_state=42)
lr_tz.fit(Xtz, y)
beta_score = lr_tz.coef_[0]

# Method C: 各体质人群发病率、相对风险RR、绝对风险差ARR
rows = []
for i, t in enumerate(tizhi_cols, 1):
    sub = df[df['体质标签'] == i]
    n = len(sub); p = sub['高血脂症二分类标签'].mean()
    rr = p / p_all
    arr = p - p_all
    # Cohen's h
    h = 2*np.arcsin(np.sqrt(p)) - 2*np.arcsin(np.sqrt(p_all))
    # 卡方 (该体质 vs 其它)
    others = df[df['体质标签'] != i]['高血脂症二分类标签']
    tab = pd.crosstab(
        pd.Series([1]*len(sub) + [0]*len(others)),
        pd.concat([sub['高血脂症二分类标签'], others]))
    chi2, pval, _, _ = stats.chi2_contingency(tab)
    rows.append({'编号': i, '体质': t, '样本量': n, '发病率': p,
                 'RR': rr, 'ARR': arr, 'Cohen_h': h, 'chi2_p': pval,
                 '独热β': beta_oh[i-1] if i-1 < len(beta_oh) else np.nan,
                 '积分多元β': beta_score[i-1]})
df_tz = pd.DataFrame(rows)

# 贡献度综合 (Borda 基于 |Cohen_h|, |RR-1|, |ARR|, |独热β|, |积分多元β|)
contrib_scores = [
    {r['体质']: abs(r['Cohen_h']) for _, r in df_tz.iterrows()},
    {r['体质']: abs(r['RR']-1)    for _, r in df_tz.iterrows()},
    {r['体质']: abs(r['ARR'])     for _, r in df_tz.iterrows()},
    {r['体质']: abs(r['独热β'])   for _, r in df_tz.iterrows()},
    {r['体质']: abs(r['积分多元β']) for _, r in df_tz.iterrows()},
]
borda_tz = borda_rank(contrib_scores, tizhi_cols, higher_better=True)
df_tz['Borda_pts'] = [borda_tz[t] for t in df_tz['体质']]
df_tz = df_tz.sort_values('Borda_pts', ascending=False).reset_index(drop=True)
df_tz['贡献排名'] = range(1, 10)

print("九种体质对发病风险的贡献度 (发病率降序):")
print(df_tz.round(4).to_string(index=False))
df_tz.to_csv(table_path('Q1_tizhi_contribution.csv'), encoding='utf-8-sig', index=False)

# ===================================================================
# Step 6: 可视化汇总
# ===================================================================
print("\n"+"="*60); print("STEP 6: 生成汇总图"); print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# (1) 痰湿严重程度 top-8
ax = axes[0, 0]
top8 = df_phl.head(8)
ax.barh(range(8)[::-1], top8['Borda_pts'].values, color='steelblue')
ax.set_yticks(range(8)[::-1])
ax.set_yticklabels([labels_en.get(f, f[:10]) for f in top8['feature']])
ax.set_xlabel('Borda points')
ax.set_title('Top-8 Indicators for Phlegm-damp Severity')

# (2) 发病风险 top-8
ax = axes[0, 1]
top8r = df_risk.head(8)
ax.barh(range(8)[::-1], top8r['Borda_pts'].values, color='firebrick')
ax.set_yticks(range(8)[::-1])
ax.set_yticklabels([labels_en.get(f, f[:10]) for f in top8r['feature']])
ax.set_xlabel('Borda points')
ax.set_title('Top-8 Indicators for Hyperlipidemia Risk')

# (3) 九种体质 Borda 贡献度
ax = axes[1, 0]
color_list = ['darkorange' if t=='痰湿质' else 'steelblue' for t in df_tz['体质']]
bars = ax.bar(range(9), df_tz['Borda_pts'], color=color_list)
name_en = {'平和质':'Peace','气虚质':'Qi-def','阳虚质':'Yang-def','阴虚质':'Yin-def',
           '痰湿质':'Phlegm','湿热质':'Damp-heat','血瘀质':'Blood-stasis',
           '气郁质':'Qi-stag','特禀质':'Const'}
ax.set_xticks(range(9))
ax.set_xticklabels([name_en[t] for t in df_tz['体质']], rotation=30)
ax.set_ylabel('Borda points')
ax.set_title('9 Constitutions Contribution (Borda rank)')
ax.grid(axis='y', alpha=0.3)

# (4) 九种体质发病率 vs 整体
ax = axes[1, 1]
sorted_df_tz = df_tz.sort_values('编号')  # 按原序1-9
color2 = ['darkorange' if t==5 else 'steelblue' for t in sorted_df_tz['编号']]
ax.bar(range(9), sorted_df_tz['发病率'], color=color2)
ax.axhline(p_all, color='k', ls='--', alpha=0.5, label=f'overall={p_all:.3f}')
ax.set_xticks(range(9))
ax.set_xticklabels([name_en[t] for t in sorted_df_tz['体质']], rotation=30)
ax.set_ylabel('Hyperlipidemia prevalence')
ax.set_title('Prevalence by Constitution Type')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(figure_path('Q1_summary.png'), dpi=150)
plt.close()
print("图已保存: Q1_summary.png, Q1_corr_heatmap.png")
print("数据已保存: Q1_*.csv")
