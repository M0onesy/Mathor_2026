import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# ==================== 1. 读取数据 ====================
# 请根据实际路径修改，推荐使用原始字符串或正斜杠
file_path = r"../Data/data.xlsx"   # 改成你的实际路径
df = pd.read_excel(file_path, sheet_name="Sheet1")

# ==================== 2. 定义特征和目标 ====================
# 血常规指标（7个）
blood_cols = ['HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 
              'TG（甘油三酯）', 'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI']

# 活动能力评分：使用 ADL总分 和 IADL总分（细分）
activity_cols = ['ADL总分', 'IADL总分']

# 合并所有特征
features = blood_cols + activity_cols
X = df[features].copy()

# 目标变量
y_phlegm = df['痰湿质']            # 痰湿积分（连续）
y_risk = df['高血脂症二分类标签']   # 高血脂标签（0/1）

# ==================== 3. 数据清洗 ====================
print("缺失值统计：")
print(X.isnull().sum())
# 如果有缺失，用中位数填充（本例中无缺失）
X.fillna(X.median(), inplace=True)

# ==================== 4. 标准化 ====================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
feature_names = X.columns.tolist()

# ==================== 5. LASSO回归（痰湿积分） - 手动设定alpha ====================
print("\n--- LASSO回归（痰湿积分） ---")
# 尝试一系列alpha，从大到小，直到出现非零系数
alphas_to_try = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
best_lasso_reg = None
best_alpha = None
for alpha in alphas_to_try:
    lasso_reg = Lasso(alpha=alpha, random_state=42, max_iter=10000)
    lasso_reg.fit(X_scaled, y_phlegm)
    if np.sum(lasso_reg.coef_ != 0) > 0:
        best_lasso_reg = lasso_reg
        best_alpha = alpha
        break
if best_lasso_reg is None:
    print("警告：即使在alpha=0.0001下，所有系数仍为0！建议改用随机森林回归。")
    # 此时可以改为随机森林，但为了代码完整性，仍使用最后一个模型
    best_lasso_reg = Lasso(alpha=0.0001, random_state=42, max_iter=10000)
    best_lasso_reg.fit(X_scaled, y_phlegm)

print(f"使用的alpha: {best_alpha if best_alpha else '0.0001 (但系数仍可能为零)'}")
coef_reg = pd.Series(best_lasso_reg.coef_, index=feature_names)
selected_reg = coef_reg[coef_reg != 0].index.tolist()
print(f"非零系数特征 ({len(selected_reg)}个): {selected_reg}")
if len(selected_reg) > 0:
    print("各特征系数：")
    print(coef_reg[coef_reg != 0].sort_values(key=abs, ascending=False))
else:
    print("所有系数均为0，无法筛选特征。")

# ==================== 6. LASSO逻辑回归（高血脂） ====================
print("\n--- LASSO逻辑回归（高血脂） ---")
# 交叉验证选择最佳C
C_values = np.logspace(-3, 1, 20)
best_clf = None
best_score = -np.inf
best_C = None
for C in C_values:
    lr = LogisticRegression(penalty='l1', solver='saga', C=C, max_iter=5000, random_state=42)
    scores = cross_val_score(lr, X_scaled, y_risk, cv=5, scoring='accuracy')
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_C = C
        best_clf = lr
print(f"最佳C值: {best_C:.4f}, 平均准确率: {best_score:.4f}")

# 使用最佳C训练最终模型
final_clf = LogisticRegression(penalty='l1', solver='saga', C=best_C, max_iter=5000, random_state=42)
final_clf.fit(X_scaled, y_risk)
coef_clf = pd.Series(final_clf.coef_[0], index=feature_names)
selected_clf = coef_clf[coef_clf != 0].index.tolist()
print(f"非零系数特征 ({len(selected_clf)}个): {selected_clf}")
print("各特征系数：")
print(coef_clf[coef_clf != 0].sort_values(key=abs, ascending=False))

# ==================== 7. 取交集 ====================
dual_key = list(set(selected_reg) & set(selected_clf))
print("\n=== 双能力关键指标（交集） ===")
print(dual_key)

if len(dual_key) == 0:
    print("警告：交集为空！取并集并按综合得分（|β_reg| * |β_clf|）排序。")
    reg_abs = coef_reg.abs()
    clf_abs = coef_clf.abs()
    combined_score = reg_abs * clf_abs
    combined_score = combined_score.sort_values(ascending=False)
    print("\n综合得分前5的特征：")
    print(combined_score.head(5))
    # 推荐前3个作为关键指标
    dual_key = combined_score.head(3).index.tolist()
    print(f"\n推荐选取: {dual_key}")
else:
    # 若交集非空，也计算综合得分排序
    reg_abs = coef_reg.abs()
    clf_abs = coef_clf.abs()
    combined_score = (reg_abs * clf_abs).loc[dual_key].sort_values(ascending=False)
    print("\n交集中的综合得分排序：")
    print(combined_score)

# ==================== 8. 输出完整系数表 ====================
print("\n--- 完整系数表（标准化后） ---")
coef_table = pd.DataFrame({
    '特征': feature_names,
    '痰湿积分系数': coef_reg.values,
    '高血脂系数': coef_clf.values
})
coef_table['|痰湿系数|'] = coef_table['痰湿积分系数'].abs()
coef_table['|高血脂系数|'] = coef_table['高血脂系数'].abs()
coef_table['综合得分'] = coef_table['|痰湿系数|'] * coef_table['|高血脂系数|']
coef_table = coef_table.sort_values('综合得分', ascending=False)
print(coef_table.to_string(index=False))

# ==================== 9. 保存结果 ====================
with open("dual_key_features.txt", "w", encoding="utf-8") as f:
    f.write("双能力关键指标：\n")
    f.write(", ".join(dual_key))
print("\n结果已保存到 dual_key_features.txt")