"""
问题2：高血脂症三级风险预警模型（修订版）

构建 "规则层 + 模型层" 融合预警体系：
  - 机器学习模块：L2-正则化 Logistic 回归(主) + RF/GBDT(对照)
  - 规则层：临床异常项数 + 痰湿积分 + 活动评分 + BMI
  - 融合：R = max(R_rule, R_model)
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score, roc_curve)
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt
from matplotlib import rcParams
from common import configure_plotting, figure_path, load_data, set_random_seed, table_path

configure_plotting()
set_random_seed()

df = load_data()

def count_abnormal(r):
    c=0
    if r['TC（总胆固醇）']>6.2: c+=1
    if r['TG（甘油三酯）']>1.7: c+=1
    if r['LDL-C（低密度脂蛋白）']>3.1: c+=1
    if r['HDL-C（高密度脂蛋白）']<1.04: c+=1
    return c
df['血脂异常项数'] = df.apply(count_abnormal, axis=1)

# 特征：血常规(7) + ADL子项(5) + IADL子项(5) + 痰湿积分(1) = 18 个
FEATS = ['TC（总胆固醇）','TG（甘油三酯）','LDL-C（低密度脂蛋白）','HDL-C（高密度脂蛋白）',
         '空腹血糖','血尿酸','BMI',
         'ADL用厕','ADL吃饭','ADL步行','ADL穿衣','ADL洗澡',
         'IADL购物','IADL做饭','IADL理财','IADL交通','IADL服药',
         '痰湿质']
X = df[FEATS].values
y = df['高血脂症二分类标签'].values
Xs = StandardScaler().fit_transform(X)

models = {
    'Logistic': lambda: LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0, random_state=42),
    'RandomForest': lambda: RandomForestClassifier(n_estimators=500, max_depth=8,
                                                    min_samples_leaf=5,
                                                    class_weight='balanced', random_state=42, n_jobs=-1),
    'GBDT': lambda: GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                                learning_rate=0.05, random_state=42)
}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv = {}
for name, builder in models.items():
    aucs, accs, f1s = [], [], []
    for tr, te in skf.split(Xs, y):
        m = builder()
        Xtr, Xte = (Xs[tr], Xs[te]) if name=='Logistic' else (X[tr], X[te])
        m.fit(Xtr, y[tr])
        pr = m.predict_proba(Xte)[:,1]
        yh = (pr>=0.5).astype(int)
        aucs.append(roc_auc_score(y[te], pr))
        accs.append(accuracy_score(y[te], yh))
        f1s.append(f1_score(y[te], yh))
    cv[name] = {'AUC': np.mean(aucs), 'AUC_std': np.std(aucs),
                'Acc': np.mean(accs), 'F1': np.mean(f1s)}
df_cv = pd.DataFrame(cv).T.round(4)
print("="*60); print("5-Fold CV Results:"); print("="*60)
print(df_cv)
df_cv.to_csv(table_path('Q2_cv_results.csv'), encoding='utf-8-sig')

# 主模型: Logistic (概率光滑, 利于分层)
best = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0, random_state=42)
best.fit(Xs, y)
proba = best.predict_proba(Xs)[:, 1]
df['预测发病概率'] = proba
print(f"\nLogistic 预测概率分布: min={proba.min():.4f}, median={np.median(proba):.4f}, max={proba.max():.4f}")

coef = pd.Series(np.abs(best.coef_[0]), index=FEATS).sort_values(ascending=False)
print("\n==== Logistic |β| Top10 (标准化) ====")
print(coef.head(10).round(4))
coef.to_csv(table_path('Q2_feature_importance.csv'), encoding='utf-8-sig')

# 规则层
def rule_risk(r):
    n = r['血脂异常项数']; phl = r['痰湿质']
    act = r['活动量表总分（ADL总分+IADL总分）']; bmi = r['BMI']
    if n >= 2: return 3
    if n >= 1 and (phl >= 60 or bmi >= 24): return 3
    if phl >= 80 and act < 40: return 3
    if n == 1: return 2
    if phl >= 60: return 2
    if 40 <= phl < 60 and act < 50: return 2
    return 1
df['规则风险'] = df.apply(rule_risk, axis=1)

# 模型层: 分位映射
q33, q67 = np.quantile(proba, [0.33, 0.67])
print(f"\n模型概率分位: q33={q33:.4f}, q67={q67:.4f}")
def model_risk(p):
    if p>=q67: return 3
    if p>=q33: return 2
    return 1
df['模型风险'] = df['预测发病概率'].apply(model_risk)

# 融合
df['最终风险'] = np.maximum(df['规则风险'], df['模型风险'])
df['风险等级'] = df['最终风险'].map({1:'低风险', 2:'中风险', 3:'高风险'})

print("\n==== 三级风险分布 ====")
print(df['风险等级'].value_counts())
print("\n各风险层实际发病率:")
print(df.groupby('风险等级', observed=True)['高血脂症二分类标签'].agg(['count','mean']).round(3))
print("\n诊断x风险交叉表:")
print(pd.crosstab(df['高血脂症二分类标签'], df['风险等级']))
print("\n各风险层特征:")
print(df.groupby('风险等级', observed=True).agg(
    血脂异常项数=('血脂异常项数','mean'),
    痰湿积分=('痰湿质','mean'),
    活动量表总分=('活动量表总分（ADL总分+IADL总分）','mean'),
    TG=('TG（甘油三酯）','mean'),
    TC=('TC（总胆固醇）','mean'),
    BMI=('BMI','mean'),
    年龄组=('年龄组','mean'),
    预测概率=('预测发病概率','mean')).round(3))

# 痰湿体质分析
dft = df[df['体质标签']==5].copy()
print(f"\n[痰湿体质 {len(dft)} 人]  高={int((dft['最终风险']==3).sum())} 中={int((dft['最终风险']==2).sum())} 低={int((dft['最终风险']==1).sum())}")

tree_feats = ['痰湿质','活动量表总分（ADL总分+IADL总分）','TG（甘油三酯）','TC（总胆固醇）',
              'LDL-C（低密度脂蛋白）','HDL-C（高密度脂蛋白）','BMI','血尿酸','年龄组']
Xt = dft[tree_feats].values
yt = (dft['最终风险']==3).astype(int).values
tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=15, random_state=42)
tree.fit(Xt, yt)
rules = export_text(tree, feature_names=tree_feats)
print("\n痰湿体质高风险决策树规则:")
print(rules)
table_path('Q2_tree_rules.txt').write_text(rules, encoding='utf-8')

combos = [
    ('痰湿积分>=60 & 活动量表<40', (dft['痰湿质']>=60) & (dft['活动量表总分（ADL总分+IADL总分）']<40)),
    ('痰湿积分>=60 & 血脂异常>=1', (dft['痰湿质']>=60) & (dft['血脂异常项数']>=1)),
    ('TG>1.7 & BMI>=24', (dft['TG（甘油三酯）']>1.7) & (dft['BMI']>=24)),
    ('血脂异常>=2 & 痰湿积分>=40', (dft['血脂异常项数']>=2) & (dft['痰湿质']>=40)),
    ('痰湿积分>=60 & 年龄组>=3', (dft['痰湿质']>=60) & (dft['年龄组']>=3)),
]
rows=[]
for nm, mask in combos:
    sub = dft[mask]
    if len(sub):
        rows.append({'组合': nm, '人数': len(sub),
                     '占痰湿比例': f"{len(sub)/len(dft)*100:.1f}%",
                     '实际发病率': f"{sub['高血脂症二分类标签'].mean()*100:.1f}%",
                     '高风险占比': f"{(sub['最终风险']==3).mean()*100:.1f}%"})
core = pd.DataFrame(rows)
print("\n痰湿体质高风险核心特征组合:"); print(core)
core.to_csv(table_path('Q2_core_combo.csv'), encoding='utf-8-sig', index=False)

# ================ 绘图 ================
fig = plt.figure(figsize=(16, 12))

ax = plt.subplot(2, 3, 1)
ax.bar(df_cv.index, df_cv['AUC'], yerr=df_cv['AUC_std'],
       color=['#4C72B0','#55A868','#C44E52'], capsize=5)
for i, v in enumerate(df_cv['AUC']):
    ax.text(i, v+0.01, f"{v:.3f}", ha='center', fontweight='bold')
ax.set_ylim(0.7, 1.02); ax.set_ylabel('AUC')
ax.set_title('5-Fold CV AUC'); ax.grid(axis='y', alpha=0.3)

ax = plt.subplot(2, 3, 2)
top10 = coef.head(10)
labels_en = {'TG（甘油三酯）':'TG','TC（总胆固醇）':'TC','LDL-C（低密度脂蛋白）':'LDL-C',
             'HDL-C（高密度脂蛋白）':'HDL-C','空腹血糖':'FBS','血尿酸':'UA','BMI':'BMI',
             'ADL用厕':'ADL-toilet','ADL吃饭':'ADL-eat','ADL步行':'ADL-walk',
             'ADL穿衣':'ADL-dress','ADL洗澡':'ADL-bath',
             'IADL购物':'IADL-shop','IADL做饭':'IADL-cook','IADL理财':'IADL-finance',
             'IADL交通':'IADL-trans','IADL服药':'IADL-med','痰湿质':'Phlegm'}
ax.barh(range(10)[::-1], top10.values, color='teal')
ax.set_yticks(range(10)[::-1])
ax.set_yticklabels([labels_en.get(s, s[:12]) for s in top10.index])
ax.set_title('Logistic |β| Top10')

ax = plt.subplot(2, 3, 3)
fpr, tpr, _ = roc_curve(y, proba)
ax.plot(fpr, tpr, lw=2, label=f'Logistic AUC={roc_auc_score(y,proba):.3f}')
ax.plot([0,1],[0,1], 'k--', alpha=0.5)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('ROC Curve'); ax.legend()

ax = plt.subplot(2, 3, 4)
rd = df.groupby('风险等级')['高血脂症二分类标签'].agg(['count','mean']).reindex(['低风险','中风险','高风险'])
x = np.arange(3); ax2 = ax.twinx()
ax.bar(x-0.2, rd['count'], 0.4, color='skyblue', label='# cases')
ax2.bar(x+0.2, rd['mean'], 0.4, color='salmon', label='actual rate')
ax.set_xticks(x); ax.set_xticklabels(['Low','Medium','High'])
ax.set_ylabel('Count'); ax2.set_ylabel('Actual prevalence')
ax.set_title('Risk Stratification')
ax.legend(loc='upper left'); ax2.legend(loc='upper right')

ax = plt.subplot(2, 3, 5)
cmap = {1:'green', 2:'orange', 3:'red'}
for lv in [1,2,3]:
    sub = dft[dft['最终风险']==lv]
    ax.scatter(sub['痰湿质'], sub['活动量表总分（ADL总分+IADL总分）'],
               c=cmap[lv], alpha=0.6, label=f'Risk L{lv}', s=25)
ax.axvline(60, color='k', ls=':', alpha=0.4)
ax.axhline(40, color='k', ls=':', alpha=0.4)
ax.set_xlabel('Phlegm Score'); ax.set_ylabel('Activity Score')
ax.set_title('Phlegm Constitution Map'); ax.legend(fontsize=8)

ax = plt.subplot(2, 3, 6)
plot_tree(tree, feature_names=['Phlegm','Act','TG','TC','LDL','HDL','BMI','UA','Age'],
          filled=True, rounded=True, max_depth=3, fontsize=7, ax=ax)
ax.set_title('Decision Tree (Phlegm Constitution)')

plt.tight_layout()
plt.savefig(figure_path('Q2_summary.png'), dpi=150)
plt.close()

df[['样本ID','体质标签','痰湿质','活动量表总分（ADL总分+IADL总分）',
    'TG（甘油三酯）','TC（总胆固醇）','LDL-C（低密度脂蛋白）','HDL-C（高密度脂蛋白）',
    'BMI','血脂异常项数','预测发病概率','规则风险','模型风险',
    '最终风险','风险等级','高血脂症二分类标签']].to_csv(
    table_path('Q2_risk_full.csv'), encoding='utf-8-sig', index=False)
print("\n[OK] Q2 全部输出完成")
