"""
问题3：痰湿体质患者 6 个月个性化干预方案优化（DP）
决策: 每月 L_t (由 P_{t-1} 决定) + s_t (强度) + f_t (周频次)
动力学: P_t = P_{t-1}*(1 - 0.03s - 0.01max(0,f-5)) 当 f>=5; 否则 P_t = P_{t-1}
目标: J = α P_6 + β Cost + γ Σmax(0,f-5) + δ Σ(3-s)
参数: α=1.0, β=0.01, γ=0.5, δ=0.3
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from common import configure_plotting, figure_path, load_data, table_path

configure_plotting()

df = load_data()
dft = df[df['体质标签']==5].copy()
if dft.empty:
    raise ValueError("数据中未找到痰湿体质（体质标签=5）样本，无法执行问题三优化。")
print(f"痰湿体质共 {len(dft)} 人")

C_L = {1:30, 2:80, 3:130}
C_S = {1:3, 2:5, 3:8}
N_WK = 4

def next_phlegm(P_prev, s, f):
    if f < 5: return P_prev
    decay = s*0.03 + max(0, f-5)*0.01
    return max(P_prev*(1-decay), 0)

def level_from(P):
    if P >= 62: return 3
    elif P >= 59: return 2
    else: return 1

def feasible_s(age, act):
    S = set()
    if age in [1,2]: S |= {1,2,3}
    elif age in [3,4]: S |= {1,2}
    else: S |= {1}
    if act < 40: S &= {1}
    elif act < 60: S &= {1,2}
    return sorted(S)

def optimize(P0, age, act, budget=2000,
             alpha=1.0, beta=0.01, gamma=0.5, delta=0.3, max_states=2000):
    S_base = feasible_s(age, act)
    states = {(round(P0,2), 0.0): (0.0, [])}
    best = (float('inf'), None, None, None)
    for t in range(1, 7):
        new_states = {}
        for (P_prev, cost), (pen, path) in states.items():
            L = level_from(P_prev)
            cL = C_L[L]
            for s in S_base:
                for f in range(1, 11):
                    mc = cL + C_S[s]*f*N_WK
                    nc = cost + mc
                    if nc > budget: continue
                    P_new = next_phlegm(P_prev, s, f)
                    dpen = beta*mc + gamma*max(0, f-5) + delta*(3-s)
                    new_pen = pen + dpen
                    new_path = path + [(L, s, f, round(P_new,2), mc)]
                    key = (round(P_new,2), round(nc,1))
                    if key not in new_states or new_states[key][0] > new_pen:
                        new_states[key] = (new_pen, new_path)
        if t == 6:
            for (P6, cost), (pen, path) in new_states.items():
                obj = alpha*P6 + pen
                if obj < best[0]:
                    best = (obj, cost, P6, path)
        if len(new_states) > max_states:
            sorted_st = sorted(new_states.items(),
                               key=lambda x: x[0][0] + 0.02*x[0][1] + x[1][0])[:max_states]
            new_states = dict(sorted_st)
        states = new_states
    return best

# 样本 1 / 2 / 3
samples = dft[dft['样本ID'].isin([1,2,3])].copy()
if len(samples) < 3:
    samples = dft.head(3).copy()
print("\n前3位患者:")
print(samples[['样本ID','体质标签','痰湿质','活动量表总分（ADL总分+IADL总分）','年龄组','性别']])

results = []
for _, row in samples.iterrows():
    P0 = row['痰湿质']; age = int(row['年龄组'])
    act = row['活动量表总分（ADL总分+IADL总分）']
    print(f"\n=== 样本 {int(row['样本ID'])} ===")
    print(f"P0={P0}, age_group={age}, act={act}")
    r = optimize(P0, age, act)
    obj, cost, P6, path = r
    print(f"最优目标={obj:.3f}, 总成本={cost:.1f}, P6={P6:.2f} (降 {(P0-P6)/P0*100:.1f}%)")
    print("月 | L | s | f | P_end | 月成本")
    for t, (L, s, f, P, c) in enumerate(path, 1):
        print(f" {t} | {L} | {s} | {f} | {P:6.2f} | {c:.1f}")
    results.append({'样本ID': int(row['样本ID']), 'P0': P0, 'age': age, 'act': act,
                    'total_cost': cost, 'P6': P6, 'path': path})

demo_rows = []
for r in results:
    for t, (L, s, f, P, c) in enumerate(r['path'], 1):
        demo_rows.append({'样本ID': r['样本ID'], '月份': t, '调理分级': L,
                          '活动强度': s, '周训练次数': f, '月末痰湿积分': P,
                          '本月成本(元)': c})
pd.DataFrame(demo_rows).to_csv(table_path('Q3_demo_3patients.csv'), encoding='utf-8-sig', index=False)

# 全部求解
print("\n对全部痰湿体质患者求解...")
all_results = []
for _, row in dft.iterrows():
    P0 = row['痰湿质']; age = int(row['年龄组'])
    act = row['活动量表总分（ADL总分+IADL总分）']
    r = optimize(P0, age, act, max_states=1000)
    obj, cost, P6, path = r
    if path is None: continue
    avg_s = np.mean([p[1] for p in path])
    avg_f = np.mean([p[2] for p in path])
    max_L = max(p[0] for p in path)
    all_results.append({'样本ID': int(row['样本ID']), 'P0': P0, 'age': age, 'act': act,
                       'BMI': row['BMI'], '性别': row['性别'],
                       'total_cost': cost, 'P6': P6, 'reduction_pct': (P0-P6)/P0*100,
                       'avg_s': avg_s, 'avg_f': avg_f, 'max_L': max_L})
all_df = pd.DataFrame(all_results)
print(all_df[['P0','total_cost','P6','reduction_pct','avg_s','avg_f']].describe().round(2))
all_df.to_csv(table_path('Q3_all_phlegm.csv'), encoding='utf-8-sig', index=False)

print("\n==== 按初始痰湿积分分组 ====")
all_df['P0_bin'] = pd.cut(all_df['P0'], [-1, 60, 62, 100], labels=['<60','60-62','≥62'])
print(all_df.groupby('P0_bin', observed=True)[['total_cost','P6','reduction_pct','avg_s','avg_f']].mean().round(2))
print("\n==== 按年龄组分组 ====")
print(all_df.groupby('age')[['total_cost','P6','reduction_pct','avg_s','avg_f']].mean().round(2))
print("\n==== 按活动能力分组 ====")
all_df['act_bin'] = pd.cut(all_df['act'], [-1, 40, 60, 100], labels=['<40','40-60','≥60'])
print(all_df.groupby('act_bin', observed=True)[['total_cost','P6','reduction_pct','avg_s','avg_f']].mean().round(2))

# 图表
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

ax = axes[0,0]
sc = ax.scatter(all_df['P0'], all_df['P6'], alpha=0.5, c=all_df['total_cost'], cmap='viridis')
ax.plot([50,70],[50,70],'k--',alpha=0.4,label='y=x')
ax.set_xlabel('Initial P0'); ax.set_ylabel('P6'); ax.set_title('P0 vs P6')
ax.legend(); plt.colorbar(sc, ax=ax, label='Cost')

ax = axes[0,1]
ax.hist(all_df['total_cost'], bins=30, color='steelblue', edgecolor='k')
ax.axvline(2000, color='r', ls='--', label='Budget')
ax.set_xlabel('Total Cost'); ax.set_ylabel('# Patients'); ax.set_title('Cost Distribution'); ax.legend()

ax = axes[0,2]
ax.hist(all_df['reduction_pct'], bins=30, color='forestgreen', edgecolor='k')
ax.set_xlabel('Reduction %'); ax.set_ylabel('# Patients'); ax.set_title('Reduction Distribution')

ax = axes[1,0]
age_sum = all_df.groupby('age')[['avg_s','avg_f']].mean()
ax.plot(age_sum.index, age_sum['avg_s'], 'o-', label='Avg s', color='crimson')
ax.plot(age_sum.index, age_sum['avg_f'], 's-', label='Avg f', color='navy')
ax.set_xlabel('Age Group'); ax.set_title('Strategy by Age')
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1,1]
ax.scatter(all_df['P0'], all_df['reduction_pct'], alpha=0.5, color='purple')
ax.set_xlabel('P0'); ax.set_ylabel('Reduction %'); ax.set_title('P0 vs Reduction')
ax.grid(alpha=0.3)

ax = axes[1,2]
ax.scatter(all_df['act'], all_df['avg_s'], alpha=0.5, color='teal')
ax.axvline(40, color='k', ls='--', alpha=0.4); ax.axvline(60, color='k', ls='--', alpha=0.4)
ax.set_xlabel('Activity'); ax.set_ylabel('Avg s'); ax.set_title('Activity vs Strategy')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(figure_path('Q3_summary.png'), dpi=150)
plt.close()

# Demo 轨迹
demo = pd.read_csv(table_path('Q3_demo_3patients.csv'))
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
P0s = [64, 58, 59]
infos = ['ID=1: P0=64, Female, 50-59, Act=38',
         'ID=2: P0=58, Male, 40-49, Act=40',
         'ID=3: P0=59, Female, 40-49, Act=63']
for i, pid in enumerate(demo['样本ID'].unique()):
    sub = demo[demo['样本ID']==pid].sort_values('月份')
    ax = axes[i]
    months = [0] + list(sub['月份'])
    vals = [P0s[i]] + list(sub['月末痰湿积分'])
    ax.plot(months, vals, 'o-', color='steelblue', lw=2, markersize=7)
    for _, row in sub.iterrows():
        ax.annotate(f"L{row['调理分级']}s{row['活动强度']}f{row['周训练次数']}",
                   xy=(row['月份'], row['月末痰湿积分']),
                   xytext=(-5,8), textcoords='offset points', fontsize=8)
    ax.set_xlabel('Month'); ax.set_ylabel('Phlegm Score')
    ax.set_title(infos[i])
    ax.grid(alpha=0.3); ax.set_xticks(range(7))
    total_cost = sub['本月成本(元)'].sum()
    final_P = vals[-1]
    reduction = (P0s[i] - final_P)/P0s[i]*100
    ax.text(0.5, 0.05, f"Cost={total_cost}CNY  Reduction={reduction:.1f}%",
            transform=ax.transAxes, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout()
plt.savefig(figure_path('Q3_demo_trajectories.png'), dpi=150)
plt.close()
print("\n[OK] Q3 全部输出完成")
