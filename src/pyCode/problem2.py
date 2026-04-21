# -*- coding: utf-8 -*-
"""
================================================================================
  问题二：中老年高血脂三级风险预警模型
  范式转换：诊断重述 -> 筛查工具（剔除血脂四项，消除标签泄漏）
  方法体系：
    - WOE-Logistic 评分卡（Siddiqi 2006 经典范式）作为可解释主线
    - GradientBoostingClassifier + RandomForest 作为非线性对照
    - Youden 指数 + Bootstrap 置信区间 + 特异度约束双阈值
    - Cochran-Armitage 趋势检验 + NNT 临床锚定
    - Isotonic 校准 + HL 检验 + DeLong + 决策曲线 DCA
    - 痰湿子集三路径挖掘：Apriori + 决策树 + 置换重要度
================================================================================
依赖: pandas numpy scipy scikit-learn matplotlib openpyxl
运行: python src/pyCode/problem2.py
输出: src/outputs/figures/Q2_*.png, src/outputs/tables/Q2_results.json 与配套 CSV/TXT
================================================================================
"""

import os, json, warnings, itertools
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2, norm

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, roc_curve, f1_score,
    confusion_matrix, cohen_kappa_score,
)

from common import configure_plotting, figure_path, load_data, set_random_seed, table_path

RNG = 42
configure_plotting()
set_random_seed(RNG)

# =============================================================================
# 【配置】
# =============================================================================
DATA_PATH = os.environ.get("Q2_DATA_PATH")

COL_MAP = {
    "label":     "高血脂症二分类标签",
    "phlegm":    "痰湿质",
    "fpg":       "空腹血糖",
    "ua":        "血尿酸",
    "bmi":       "BMI",
    "act":       "活动量表总分（ADL总分+IADL总分）",
    "age_group": "年龄组",
    "gender":    "性别",
    "smoking":   "吸烟史",
    "drinking":  "饮酒史",
    "tizhi":     "体质标签",
    "tc":        "TC（总胆固醇）",
    "tg":        "TG（甘油三酯）",
    "ldl":       "LDL-C（低密度脂蛋白）",
    "hdl":       "HDL-C（高密度脂蛋白）",
}
FEATURES_CONT = ["phlegm", "fpg", "ua", "bmi", "act"]
FEATURES_ORD  = ["age_group"]
FEATURES_CAT  = ["gender", "smoking", "drinking"]
FEATURES_ALL  = FEATURES_CONT + FEATURES_ORD + FEATURES_CAT
LIPID_BLACKLIST = ["tc", "tg", "ldl", "hdl"]

PDO = 20
BASE_POINTS = 100
BASE_ODDS = 1.0


# =============================================================================
# 模块 A — 数据加载 + 泄漏自查 + 预处理
# =============================================================================
def module_A_preprocess(path=DATA_PATH):
    df = pd.read_excel(path) if path else load_data()
    inv = {v: k for k, v in COL_MAP.items() if v in df.columns}
    missing = [v for v in COL_MAP.values() if v not in df.columns]
    if missing:
        raise KeyError(f"Excel 中缺失以下列: {missing}")
    df = df.rename(columns=inv)

    abn = ((df["tc"] > 6.2) | (df["tg"] > 1.7) |
           (df["ldl"] > 3.1) | (df["hdl"] < 1.04)).astype(int)
    consist = float((abn == df["label"]).mean())
    pos_rate = float(df["label"].mean())
    n_phlegm = int((df["tizhi"] == 5).sum())
    print(f"[A] n={len(df)}, 阳性率={pos_rate:.4f}, "
          f"泄漏一致率={consist:.4f}, 痰湿体质 n={n_phlegm}")
    print(f"[A] Layer-1 血脂异常人数={int(abn.sum())} "
          f"(覆盖率 {abn.sum()/len(df):.2%})")

    X = df[FEATURES_ALL].copy()
    y = df["label"].astype(int).values
    tizhi = df["tizhi"].astype(int).values
    abn_full = abn.values.astype(int)

    X[FEATURES_CONT] = SimpleImputer(strategy="median") \
        .fit_transform(X[FEATURES_CONT])
    X[FEATURES_ORD + FEATURES_CAT] = SimpleImputer(strategy="most_frequent") \
        .fit_transform(X[FEATURES_ORD + FEATURES_CAT])
    X = X.astype(float)

    Xtr, Xte, ytr, yte, ttr, tte, abn_tr, abn_te = train_test_split(
        X, y, tizhi, abn_full,
        test_size=0.30, stratify=y, random_state=RNG)

    return dict(X_raw=X, y=y, tizhi=tizhi, abn_full=abn_full,
                Xtr_raw=Xtr, Xte_raw=Xte,
                ytr=ytr, yte=yte, ttr=ttr, tte=tte,
                abn_tr=abn_tr, abn_te=abn_te,
                leak_rate=consist, pos_rate=pos_rate, n_phlegm=n_phlegm)


# =============================================================================
# 模块 B — WOE / IV 编码
# =============================================================================
def _make_bins_cont(x, n_bins=5):
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return np.array([x.min() - 1, x.max() + 1])
    return edges


def woe_encode_train(Xtr, ytr, cont=FEATURES_CONT):
    y = np.asarray(ytr)
    n1_total = int(y.sum()); n0_total = len(y) - n1_total
    eps = 0.5
    woe_map, edges_map, iv_summary = {}, {}, {}

    for f in Xtr.columns:
        x = Xtr[f].values
        if f in cont:
            edges = _make_bins_cont(x, n_bins=5)
            bins = np.digitize(x, edges[1:-1])
            labels = [(float(edges[i]), float(edges[i + 1]))
                      for i in range(len(edges) - 1)]
        else:
            uniq = sorted(np.unique(x).tolist())
            labels = [(float(v), float(v)) for v in uniq]
            mapping = {v: i for i, v in enumerate(uniq)}
            bins = np.array([mapping[v] for v in x])
            edges = uniq

        rows = []
        for b_id, (lo, hi) in enumerate(labels):
            mask = (bins == b_id)
            n = int(mask.sum())
            n1 = int(y[mask].sum())
            n0 = n - n1
            p1 = (n1 + eps) / (n1_total + eps * len(labels))
            p0 = (n0 + eps) / (n0_total + eps * len(labels))
            woe = float(np.log(p1 / p0))
            iv_i = float((p1 - p0) * woe)
            rows.append([b_id, lo, hi, n, n1, n0, woe, iv_i])
        tbl = pd.DataFrame(rows, columns=["bin_id", "lo", "hi",
                                          "n", "n1", "n0", "woe", "iv_i"])
        woe_map[f] = tbl
        edges_map[f] = edges
        iv_summary[f] = float(tbl["iv_i"].sum())
    return woe_map, edges_map, iv_summary


def woe_transform(X, woe_map, edges_map, cont=FEATURES_CONT):
    out = pd.DataFrame(index=X.index)
    for f in X.columns:
        x = X[f].values
        if f in cont:
            edges = edges_map[f]
            bins = np.digitize(x, edges[1:-1])
        else:
            uniq = edges_map[f]
            mapping = {v: i for i, v in enumerate(uniq)}
            bins = np.array([mapping.get(v, 0) for v in x])
        woe_vals = woe_map[f].set_index("bin_id")["woe"].to_dict()
        default = list(woe_vals.values())[0]
        out[f] = np.array([woe_vals.get(int(b), default) for b in bins])
    return out


# =============================================================================
# 模块 C — 三模型 5 折 CV
# =============================================================================
def bootstrap_ci(y, p, metric_fn, B=1000, alpha=0.05, seed=RNG):
    rng = np.random.default_rng(seed)
    n = len(y); scores = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        scores.append(metric_fn(y[idx], p[idx]))
    lo, hi = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (round(float(lo), 4), round(float(hi), 4),
            round(float(np.mean(scores)), 4))


def module_C_cv_train(d, woe_map, edges_map):
    Xtr_woe = woe_transform(d["Xtr_raw"], woe_map, edges_map)
    Xte_woe = woe_transform(d["Xte_raw"], woe_map, edges_map)
    ytr = d["ytr"]

    models = {
        "LR_WOE": (LogisticRegression(
            C=1.0, solver="liblinear", class_weight="balanced",
            max_iter=1000, random_state=RNG), Xtr_woe, Xte_woe),
        "GBDT":   (GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=20, random_state=RNG),
            d["Xtr_raw"], d["Xte_raw"]),
        "RF":     (RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=10,
            class_weight="balanced", n_jobs=-1, random_state=RNG),
            d["Xtr_raw"], d["Xte_raw"]),
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    oof = {k: np.zeros(len(ytr)) for k in models}
    fold_aucs = {k: [] for k in models}

    for tr, va in skf.split(np.zeros(len(ytr)), ytr):
        for k, (m, Xt, _) in models.items():
            m.fit(Xt.iloc[tr], ytr[tr])
            oof[k][va] = m.predict_proba(Xt.iloc[va])[:, 1]
            fold_aucs[k].append(roc_auc_score(ytr[va], oof[k][va]))

    cv_summary = {}
    for k, (m, Xt, Xev) in models.items():
        mu, sd = np.mean(fold_aucs[k]), np.std(fold_aucs[k])
        lo, hi, _ = bootstrap_ci(ytr, oof[k], roc_auc_score)
        cv_summary[k] = dict(
            auc_mean=round(float(mu), 4),
            auc_std=round(float(sd), 4),
            auc_95ci=(lo, hi),
            brier=round(float(brier_score_loss(ytr, oof[k])), 4),
            fold_aucs=[round(float(a), 4) for a in fold_aucs[k]],
        )
        print(f"[C] {k}: AUC={mu:.4f}±{sd:.4f}  95%CI=({lo},{hi})  "
              f"Brier={cv_summary[k]['brier']}")
        m.fit(Xt, ytr)
    return models, oof, cv_summary, Xtr_woe, Xte_woe


def delong_test(y, p1, p2):
    y = np.asarray(y); p1 = np.asarray(p1); p2 = np.asarray(p2)
    auc1, auc2 = roc_auc_score(y, p1), roc_auc_score(y, p2)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    m, n = len(pos), len(neg)
    def v_struct(p):
        V10 = np.zeros(m); V01 = np.zeros(n)
        for i, pi in enumerate(pos):
            V10[i] = np.mean((p[pi] > p[neg]) + 0.5 * (p[pi] == p[neg]))
        for j, nj in enumerate(neg):
            V01[j] = np.mean((p[pos] > p[nj]) + 0.5 * (p[pos] == p[nj]))
        return V10, V01
    V10_1, V01_1 = v_struct(p1)
    V10_2, V01_2 = v_struct(p2)
    S10 = np.cov(np.vstack([V10_1, V10_2]))
    S01 = np.cov(np.vstack([V01_1, V01_2]))
    S = S10 / m + S01 / n
    var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var_diff <= 0:
        return auc1, auc2, 0.0, 1.0
    z = (auc1 - auc2) / np.sqrt(var_diff)
    pval = 2 * (1 - norm.cdf(abs(z)))
    return float(auc1), float(auc2), float(z), float(pval)


# =============================================================================
# 模块 D — WOE → 评分卡
# =============================================================================
def module_D_scorecard(Xtr_woe, ytr, woe_map, edges_map,
                        PDO=PDO, base_points=BASE_POINTS,
                        base_odds=BASE_ODDS):
    """
    约定: 医学惯例 "分数越高 = 风险越高"
      ⇒ Points_bin = round( + factor * β_j * WOE_bin )
    """
    lr = LogisticRegression(C=1.0, solver="liblinear",
                            class_weight=None, max_iter=1000,
                            random_state=RNG)
    lr.fit(Xtr_woe, ytr)
    intercept = float(lr.intercept_[0])
    coefs = {f: float(b) for f, b in zip(Xtr_woe.columns, lr.coef_[0])}

    factor = PDO / np.log(2)
    offset = base_points - factor * np.log(base_odds)
    base_adjust = -factor * intercept

    scorecard = {}
    for f in Xtr_woe.columns:
        tbl = woe_map[f].copy()
        tbl["points"] = (factor * coefs[f] * tbl["woe"]).round().astype(int)
        tbl["beta"] = coefs[f]
        scorecard[f] = tbl

    print(f"[D] PDO={PDO}, Base={base_points}, Offset={offset:.2f}, "
          f"Intercept adjust={base_adjust:.2f}")
    for f, b in coefs.items():
        print(f"    β[{f}] = {b:+.4f}")
    return dict(card=scorecard, intercept=intercept, coefs=coefs,
                factor=float(factor), offset=float(offset),
                base_adjust=float(base_adjust),
                base_points=base_points, PDO=PDO)


def apply_scorecard(X_raw, sc, woe_map, edges_map):
    total = np.full(len(X_raw), sc["offset"] + sc["base_adjust"], dtype=float)
    for f in X_raw.columns:
        tbl = sc["card"][f]
        b2p = tbl.set_index("bin_id")["points"].to_dict()
        x = X_raw[f].values
        if f in FEATURES_CONT:
            edges = edges_map[f]
            bins = np.digitize(x, edges[1:-1])
        else:
            uniq = edges_map[f]
            mapping = {v: i for i, v in enumerate(uniq)}
            bins = np.array([mapping.get(v, 0) for v in x])
        total += np.array([b2p.get(int(b), 0) for b in bins])
    return total


def score_to_prob(score, sc):
    logit = (score - sc["offset"]) / sc["factor"]
    return 1 / (1 + np.exp(-logit))


# =============================================================================
# 模块 E — 多准则阈值
# =============================================================================
def module_E_thresholds(y, p, B=1000, seed=RNG, label="prob", save=True,
                        sp_target=0.70):
    fpr, tpr, thr = roc_curve(y, p)
    # sklearn thr[0] 为 inf 虚拟阈值，剔除
    finite = np.isfinite(thr)
    fpr, tpr, thr = fpr[finite], tpr[finite], thr[finite]
    j_idx = np.argmax(tpr - fpr)
    t_youden = float(thr[j_idx])
    J_max = float(tpr[j_idx] - fpr[j_idx])

    d_ = np.sqrt(fpr ** 2 + (1 - tpr) ** 2)
    t_near01 = float(thr[np.argmin(d_)])
    t_eer = float(thr[np.argmin(np.abs(fpr - (1 - tpr)))])

    lo_p, hi_p = float(np.quantile(p, 0.01)), float(np.quantile(p, 0.99))
    grid = np.linspace(lo_p, hi_p, 500)
    f1s = [f1_score(y, (p >= g).astype(int), zero_division=0) for g in grid]
    t_f1 = float(grid[int(np.argmax(f1s))])

    # Sp≥sp_target 下取最小阈值（最宽松的仍满足 Sp 约束的阈值）
    # 即 "最小 c 使得 Sp(c) ≥ sp_target"，作为低危组上界 cL
    mask = (1 - fpr) >= sp_target
    if mask.any():
        t_sp_tgt = float(thr[mask].min())
    else:
        t_sp_tgt = float(np.quantile(p, 0.25))

    rng = np.random.default_rng(seed); bs = []
    for _ in range(B):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        f_, t_, th_ = roc_curve(y[idx], p[idx])
        bs.append(float(th_[np.argmax(t_ - f_)]))
    ci_lo, ci_hi = np.percentile(bs, [2.5, 97.5])

    out = dict(
        Youden=round(t_youden, 4), J_max=round(J_max, 4),
        Youden_CI=(round(float(ci_lo), 4), round(float(ci_hi), 4)),
        Near01=round(t_near01, 4), EER=round(t_eer, 4),
        F1=round(t_f1, 4), Sp70=round(t_sp_tgt, 4),
        sp_target=sp_target,
    )
    print(f"[E] 阈值 ({label}): {out}")

    if save:
        plt.figure(figsize=(7, 5))
        order = np.argsort(thr)
        thr_s, tpr_s, fpr_s = thr[order], tpr[order], fpr[order]
        plt.plot(thr_s, tpr_s, label="Sensitivity (TPR)", lw=1.8)
        plt.plot(thr_s, 1 - fpr_s, label="Specificity (TNR)", lw=1.8)
        plt.plot(thr_s, tpr_s - fpr_s, "--", label="Youden J", lw=1.8)
        for name, t in [("Youden", t_youden),
                        ("Sp>=0.70", t_sp_tgt), ("F1", t_f1)]:
            plt.axvline(t, ls=":", alpha=0.5)
        plt.xlabel(f"Threshold ({label})")
        plt.ylabel("Metric value"); plt.legend(loc="best")
        plt.title(f"Multi-criterion threshold search ({label})")
        plt.tight_layout()
        plt.savefig(figure_path("Q2_threshold_search.png"), dpi=150)
        plt.close()
    return out


def cost_sensitive_threshold(y, p, C_FN=5, C_FP=1):
    lo, hi = float(np.quantile(p, 0.01)), float(np.quantile(p, 0.99))
    grid = np.linspace(lo, hi, 300)
    costs = []
    for t in grid:
        pr = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pr, labels=[0, 1]).ravel()
        costs.append(C_FN * fn + C_FP * fp)
    i_opt = int(np.argmin(costs))
    return dict(t_opt=round(float(grid[i_opt]), 4),
                min_cost=int(min(costs)), C_FN=C_FN, C_FP=C_FP)


# =============================================================================
# 模块 F — 三级分层 + Cochran-Armitage
# =============================================================================
def cochran_armitage(y, level_int):
    """2×k 线性趋势检验，已修正方差公式 V = n1·n0/N² · Σ(s_j − s̄)²"""
    y = np.asarray(y, dtype=float)
    s = np.asarray(level_int, dtype=float)
    N = len(y); n1 = float(y.sum()); n0 = N - n1
    if n0 == 0 or n1 == 0:
        return 0.0, 1.0
    s_bar = s.mean()
    T = float(np.sum(y * (s - s_bar)))
    V = (n1 * n0 / N**2) * float(np.sum((s - s_bar) ** 2))
    if V <= 0:
        return 0.0, 1.0
    Z = T / np.sqrt(V)
    pval = 2 * (1 - norm.cdf(abs(Z)))
    return float(Z), float(pval)


def module_F_stratify(y, p, cH, scheme="A", sp_target=0.70):
    """
    scheme='A': cH=Youden, cL = max{c: Sp(c) >= sp_target}
       默认 sp_target=0.70 而非 0.95。原因：本数据阳性率 79.3%，
       阴性样本仅 20.7%，Sp=0.95 要求对应的阈值在分布右尾，
       与 Youden 阈值几乎重合，导致中层样本为 0。放宽至 0.70
       可保证三层均有足够样本，同时维持 "低危组假阳性 ≤ 30%"
       的可接受假阳率水平（与 FRS/ASCVD 的低危定义相容）。
    """
    p = np.asarray(p); y = np.asarray(y)
    if scheme == "A":
        fpr, tpr, thr = roc_curve(y, p)
        # sklearn 将 thr[0] 设为 inf (虚拟阈值)，需剔除
        finite = np.isfinite(thr)
        fpr, tpr, thr = fpr[finite], tpr[finite], thr[finite]
        mask = (1 - fpr) >= sp_target
        if mask.any():
            cL = float(thr[mask].min())
        else:
            cL = float(np.quantile(p, 1/4))
        cL = min(cL, cH - 1e-3)
    elif scheme == "B":
        fpr, tpr, thr = roc_curve(y, p)
        cH = float(thr[np.argmax(tpr - fpr)])
        m_lo = p < cH
        if m_lo.sum() > 10 and len(np.unique(y[m_lo])) > 1:
            fpr2, tpr2, thr2 = roc_curve(y[m_lo], p[m_lo])
            cL = float(thr2[np.argmax(tpr2 - fpr2)])
            cL = min(cL, cH - 1e-3)
        else:
            cL = cH - 0.05
    elif scheme == "C":
        cL, cH = np.quantile(p, [1/3, 2/3])
    elif scheme == "D":
        cL, cH = 1/16, 1/3
    else:
        raise ValueError(scheme)

    level = np.where(p < cL, 0, np.where(p < cH, 1, 2))
    strata = []
    for k, name in enumerate(["低", "中", "高"]):
        mk = (level == k)
        pr = float(y[mk].mean()) if mk.sum() else None
        strata.append(dict(
            level=name, n=int(mk.sum()),
            pos_rate=round(pr, 4) if pr is not None else None,
            NNT=round(1.0 / pr, 2) if (pr and pr > 0) else None,
        ))
    Z, pv = cochran_armitage(y, level)
    return dict(scheme=scheme,
                cL=round(float(cL), 4), cH=round(float(cH), 4),
                strata=strata, CA_Z=round(Z, 4), CA_p=pv, level=level)


def module_F_twolayer(y, score, abn, cL, cH):
    """
    两层融合分层:
      Layer 1 硬规则: abn=1 (血脂四项任一异常) → 直接判定为 高风险
      Layer 2 评分卡: abn=0 残余样本按 score 与阈值 {cL,cH} 分层 低/中/高
      融合: 低/中 来自 Layer 2; 高 = Layer 1 全部 ∪ Layer 2 高
    """
    y = np.asarray(y); score = np.asarray(score); abn = np.asarray(abn).astype(int)
    # Layer 2 评分分层 (仅 abn=0 时起效)
    level_l2 = np.where(score < cL, 0, np.where(score < cH, 1, 2))
    # 融合: abn=1 全部 → 高 (level=2)
    final_level = np.where(abn == 1, 2, level_l2)
    strata = []
    for k, name in enumerate(["低", "中", "高"]):
        mk = (final_level == k)
        n_ = int(mk.sum())
        pr = float(y[mk].mean()) if n_ > 0 else None
        strata.append(dict(
            level=name, n=n_,
            pos_rate=round(pr, 4) if pr is not None else None,
            NNT=round(1.0 / pr, 2) if (pr and pr > 0) else None,
        ))
    layer1_n = int((abn == 1).sum())
    layer2_counts = {name: int(((abn == 0) & (level_l2 == k)).sum())
                     for k, name in enumerate(["低", "中", "高"])}
    Z, pv = cochran_armitage(y, final_level)
    return dict(cL=round(float(cL), 4), cH=round(float(cH), 4),
                strata=strata,
                layer1_n=layer1_n,
                layer2_counts=layer2_counts,
                CA_Z=round(Z, 4), CA_p=pv, level=final_level)


# =============================================================================
# 模块 G — 校准 + HL
# =============================================================================
def hosmer_lemeshow(y, p, g=10):
    df = pd.DataFrame({"y": y, "p": p})
    df["decile"] = pd.qcut(df["p"], q=g, duplicates="drop", labels=False)
    o1 = df.groupby("decile")["y"].sum().values
    e1 = df.groupby("decile")["p"].sum().values
    n = df.groupby("decile")["y"].count().values
    o0 = n - o1; e0 = n - e1
    e1s = np.where(e1 == 0, 1e-9, e1); e0s = np.where(e0 == 0, 1e-9, e0)
    chi_sq = float(np.sum((o1 - e1) ** 2 / e1s) +
                   np.sum((o0 - e0) ** 2 / e0s))
    dfree = max(int(df["decile"].nunique()) - 2, 1)
    pval = float(1 - chi2.cdf(chi_sq, dfree))
    return chi_sq, pval, dfree


def calibration_intercept_slope(y, p):
    eps = 1e-6
    logit = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    X = np.column_stack([np.ones_like(logit), logit])
    lr = LogisticRegression(fit_intercept=False, C=1e6, solver="lbfgs",
                            max_iter=1000)
    lr.fit(X, y)
    return float(lr.coef_[0][0]), float(lr.coef_[0][1])


def module_G_calibrate(model_factory, Xtr, ytr, Xte, yte, name="GBDT"):
    model = model_factory()
    try:
        cal = CalibratedClassifierCV(model, method="isotonic", cv=5)
        cal.fit(Xtr, ytr)
        p_cal = cal.predict_proba(Xte)[:, 1]
    except Exception as e:
        print(f"[G] Isotonic 失败 ({e}), 回退原概率。")
        model.fit(Xtr, ytr); p_cal = model.predict_proba(Xte)[:, 1]

    pt, pp = calibration_curve(yte, p_cal, n_bins=10, strategy="quantile")
    plt.figure(figsize=(5.5, 5))
    plt.plot(pp, pt, "o-", lw=2, label=f"{name} (isotonic)")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed positive rate")
    plt.legend(); plt.title(f"Calibration curve ({name})")
    plt.tight_layout()
    plt.savefig(figure_path("Q2_calibration.png"), dpi=150); plt.close()

    chi_sq, hl_p, dfree = hosmer_lemeshow(yte, p_cal)
    a, b = calibration_intercept_slope(yte, p_cal)
    brier = float(brier_score_loss(yte, p_cal))
    pi = float(yte.mean()); bss = 1 - brier / max(pi * (1 - pi), 1e-9)
    stats_ = dict(Brier=round(brier, 4), BSS=round(bss, 4),
                  HL_chi2=round(chi_sq, 3), HL_df=dfree,
                  HL_p=round(hl_p, 4),
                  calib_intercept=round(a, 4),
                  calib_slope=round(b, 4))
    print(f"[G] 校准 ({name}): {stats_}")
    return p_cal, stats_


# =============================================================================
# 模块 H — ROC + DeLong
# =============================================================================
def module_H_roc_delong(y, probs_dict):
    plt.figure(figsize=(6, 5)); aucs = {}
    for k, p in probs_dict.items():
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p); aucs[k] = auc
        plt.plot(fpr, tpr, lw=1.8, label=f"{k}  AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("FPR (1 - Specificity)"); plt.ylabel("TPR (Sensitivity)")
    plt.title("Three-model ROC (test set)")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(figure_path("Q2_roc_curves.png"), dpi=150); plt.close()

    keys = list(probs_dict.keys()); mat = {}
    for k1, k2 in itertools.combinations(keys, 2):
        a1, a2, z, pv = delong_test(y, probs_dict[k1], probs_dict[k2])
        mat[f"{k1}_vs_{k2}"] = dict(auc1=round(a1, 4), auc2=round(a2, 4),
                                    z=round(z, 3), p=round(pv, 6))
    print("[H] DeLong 两两比较:")
    for k, v in mat.items():
        print(f"    {k}: {v}")
    return mat, aucs


# =============================================================================
# 模块 I — 决策曲线 DCA + 成本敏感
# =============================================================================
def module_I_decision_curve(y, p, C_FN=5, C_FP=1):
    thresholds = np.arange(0.01, 0.99, 0.01); N = len(y)
    nb_model, nb_all = [], []; pi = y.mean()
    for pt in thresholds:
        pr = (p >= pt).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pr, labels=[0, 1]).ravel()
        nb_model.append(tp / N - fp / N * (pt / (1 - pt)))
        nb_all.append(pi - (1 - pi) * (pt / (1 - pt)))
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(thresholds, nb_model, lw=2, label="Model")
    plt.plot(thresholds, nb_all, "--", label="Treat all")
    plt.axhline(0, color="k", ls=":", label="Treat none")
    plt.xlabel("Threshold probability $p_t$"); plt.ylabel("Net benefit")
    plt.title("Decision Curve Analysis"); plt.ylim(-0.05, 0.5)
    plt.legend(); plt.tight_layout()
    plt.savefig(figure_path("Q2_dca.png"), dpi=150); plt.close()
    cs = cost_sensitive_threshold(y, p, C_FN=C_FN, C_FP=C_FP)
    print(f"[I] Cost-sensitive: {cs}")
    return cs


# =============================================================================
# 模块 J — 痰湿子集规则挖掘
# =============================================================================
def manual_apriori(bin_df, target_col="high_risk",
                   min_support=0.10, min_conf=0.60, min_lift=1.20,
                   max_ant_len=3, top_k=10):
    items = [c for c in bin_df.columns if c != target_col]
    N = len(bin_df); target_sup = bin_df[target_col].mean()
    sup1 = {frozenset([it]): bin_df[it].mean() for it in items}
    sup1 = {k: v for k, v in sup1.items() if v >= min_support}
    all_itemsets = {**sup1}
    current = list(sup1.keys())
    for k in range(2, max_ant_len + 1):
        new_sets = set()
        for a, b in itertools.combinations(current, 2):
            u = a | b
            if len(u) == k:
                new_sets.add(u)
        next_level = []
        for s in new_sets:
            cols = list(s)
            sup = bin_df[cols].all(axis=1).mean()
            if sup >= min_support:
                all_itemsets[s] = sup
                next_level.append(s)
        current = next_level
        if not current:
            break
    rules = []
    for ant, sup_a in all_itemsets.items():
        if target_col in ant:
            continue
        cols = list(ant)
        mask = bin_df[cols].all(axis=1)
        if mask.sum() == 0:
            continue
        sup_ab = float((mask & bin_df[target_col].astype(bool)).mean())
        conf = sup_ab / sup_a if sup_a > 0 else 0
        lift = conf / target_sup if target_sup > 0 else 0
        if conf >= min_conf and lift >= min_lift:
            rules.append([sorted(list(ant)), float(sup_a), sup_ab,
                          float(conf), float(lift), int(mask.sum())])
    df = pd.DataFrame(rules, columns=["antecedents", "support",
                                       "sup_ab", "confidence",
                                       "lift", "n_match"])
    df = df.sort_values("lift", ascending=False).head(top_k)
    return df, float(target_sup)


def module_J_phlegm_rules(X_raw, y, tizhi):
    mask = (tizhi == 5)
    Xs, ys = X_raw[mask].copy(), y[mask]
    if len(Xs) < 30:
        print("[J] 痰湿子集样本不足"); return None
    print(f"[J] 痰湿子集 n={len(Xs)}, 阳性率={ys.mean():.4f}")

    bin_df = pd.DataFrame(index=Xs.index)
    for c in FEATURES_CONT:
        if c in Xs.columns:
            med = float(Xs[c].median())
            bin_df[f"{c}_hi"] = (Xs[c] > med).astype(int)
    if "age_group" in Xs.columns:
        bin_df["age_ge3"] = (Xs["age_group"] >= 3).astype(int)
    for c in FEATURES_CAT:
        if c in Xs.columns:
            bin_df[c] = Xs[c].astype(int)
    bin_df["high_risk"] = ys
    bin_df = bin_df.astype(bool)

    rules_df, base_sup = None, float(ys.mean())
    try:
        rules_df, base_sup = manual_apriori(
            bin_df, target_col="high_risk",
            min_support=0.10, min_conf=0.60, min_lift=1.10,
            max_ant_len=3, top_k=10)
        print(f"[J] Apriori base={base_sup:.4f}, Top-{len(rules_df)}:")
        for _, r in rules_df.iterrows():
            print(f"    {{{', '.join(r['antecedents'])}}} -> high_risk  "
                  f"sup={r['support']:.3f} conf={r['confidence']:.3f} "
                  f"lift={r['lift']:.3f}  (n={r['n_match']})")
    except Exception as e:
        print(f"[J] Apriori 失败: {e}")

    dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10,
                                class_weight="balanced",
                                random_state=RNG).fit(Xs, ys)
    tree_txt = export_text(dt, feature_names=list(Xs.columns), max_depth=4)
    print(f"[J] 决策树:\n{tree_txt}")

    imp = None
    try:
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                        learning_rate=0.05,
                                        random_state=RNG).fit(Xs, ys)
        pi = permutation_importance(gb, Xs, ys, n_repeats=30,
                                    random_state=RNG, scoring="roc_auc")
        imp = [(c, round(float(pi.importances_mean[i]), 4),
                round(float(pi.importances_std[i]), 4))
               for i, c in enumerate(Xs.columns)]
        imp.sort(key=lambda t: -t[1])
        print("[J] 痰湿子集置换重要度 Top-5:")
        for c, m, s in imp[:5]:
            print(f"    {c:14s} ΔAUC={m:+.4f} ± {s:.4f}")
    except Exception as e:
        print(f"[J] 置换重要度失败: {e}")

    return dict(apriori=rules_df, tree_text=tree_txt,
                permutation=imp, base_sup=float(ys.mean()))


# =============================================================================
# 模块 K — 稳健性
# =============================================================================
def module_K_robustness(models, d, probs_te, score_total, cH_score):
    yte = d["yte"]
    gb_model, _, Xte_gb = models["GBDT"]
    pi = permutation_importance(gb_model, Xte_gb, yte, n_repeats=30,
                                random_state=RNG, scoring="roc_auc")
    perm = [(c, round(float(pi.importances_mean[i]), 4),
             round(float(pi.importances_std[i]), 4))
            for i, c in enumerate(Xte_gb.columns)]
    perm.sort(key=lambda t: -t[1])
    print("[K] GBDT 置换重要度:")
    for c, m, s in perm:
        print(f"    {c:14s} ΔAUC={m:+.4f} ± {s:.4f}")

    perturb = []
    for dH in [-10, -5, 0, 5, 10]:
        cH_p = cH_score + dH
        r = module_F_stratify(yte, score_total, cH_p, scheme="A")
        perturb.append({
            "cH": round(float(cH_p), 3), "cL": r["cL"],
            "n": [s["n"] for s in r["strata"]],
            "pos": [s["pos_rate"] for s in r["strata"]],
            "CA_Z": r["CA_Z"], "CA_p": r["CA_p"],
        })
    print("[K] 阈值扰动:")
    for p in perturb:
        print(f"    cH={p['cH']} n={p['n']} pos={p['pos']} "
              f"CA_p={p['CA_p']:.2e}")

    level_dict = {}
    for k, p in probs_te.items():
        from sklearn.metrics import roc_curve as _rc
        fpr, tpr, thr = _rc(yte, p)
        cH_k = float(thr[np.argmax(tpr - fpr)])
        r = module_F_stratify(yte, p, cH_k, scheme="A")
        level_dict[k] = r["level"]
    kappa = {}
    for k1, k2 in itertools.combinations(level_dict.keys(), 2):
        kappa[f"{k1}_vs_{k2}"] = round(float(cohen_kappa_score(
            level_dict[k1], level_dict[k2], weights="linear")), 4)
    print(f"[K] 分层一致性 κ: {kappa}")
    return dict(perm_importance=perm, threshold_perturb=perturb, kappa=kappa)


# =============================================================================
# 模块 L — 六面板汇总图
# =============================================================================
def module_L_summary_plot(yte, probs_dict, p_cal, strata_main,
                          thr_result, dca_cost, rules, score_total):
    fig = plt.figure(figsize=(17, 10))
    ax1 = plt.subplot(2, 3, 1)
    for k, p in probs_dict.items():
        fpr, tpr, _ = roc_curve(yte, p)
        auc = roc_auc_score(yte, p)
        ax1.plot(fpr, tpr, lw=1.8, label=f"{k} AUC={auc:.3f}")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
    ax1.legend(fontsize=8); ax1.set_title("(a) Three-model ROC")

    ax2 = plt.subplot(2, 3, 2)
    pt, pp = calibration_curve(yte, p_cal, n_bins=10, strategy="quantile")
    ax2.plot(pp, pt, "o-", lw=2)
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("Observed")
    ax2.set_title("(b) Calibration (GBDT+Isotonic)")

    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(score_total[yte == 0], bins=30, alpha=0.6,
             label=f"Negative (n={int((yte==0).sum())})", color="#2ca02c")
    ax3.hist(score_total[yte == 1], bins=30, alpha=0.6,
             label=f"Positive (n={int((yte==1).sum())})", color="#d62728")
    ax3.axvline(strata_main["cL"], color="blue", ls="--",
                label=f"cL={strata_main['cL']:.1f}")
    ax3.axvline(strata_main["cH"], color="red", ls="--",
                label=f"cH={strata_main['cH']:.1f}")
    ax3.set_xlabel("Scorecard total"); ax3.set_ylabel("Frequency")
    ax3.set_title("(c) Score distribution by diagnosis"); ax3.legend(fontsize=8)

    ax4 = plt.subplot(2, 3, 4)
    names = [s["level"] for s in strata_main["strata"]]
    ns = [s["n"] for s in strata_main["strata"]]
    prs = [s["pos_rate"] if s["pos_rate"] is not None else 0
           for s in strata_main["strata"]]
    bars = ax4.bar(names, ns, color=["#2ca02c", "#ff7f0e", "#d62728"])
    ax4b = ax4.twinx()
    ax4b.plot(names, prs, "ko-", lw=2, ms=9); ax4b.set_ylim(0, 1.05)
    for b, n, pr in zip(bars, ns, prs):
        ax4.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                 f"n={n}", ha="center", va="bottom", fontsize=9)
        ax4b.text(b.get_x() + b.get_width()/2, pr + 0.03,
                  f"{pr*100:.1f}%", ha="center", fontsize=9)
    ax4.set_ylabel("N"); ax4b.set_ylabel("Positive rate")
    ax4.set_title(f"(d) Three-tier stratification  CA Z={strata_main['CA_Z']:.2f}, "
                  f"p={strata_main['CA_p']:.1e}")

    ax5 = plt.subplot(2, 3, 5)
    N = len(yte); tg = np.arange(0.01, 0.991, 0.01); nb = []; pi = yte.mean()
    for pt_ in tg:
        pr = (p_cal >= pt_).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, pr, labels=[0, 1]).ravel()
        nb.append(tp / N - fp / N * (pt_ / (1 - pt_)))
    ax5.plot(tg, nb, lw=2, label="Model")
    ax5.plot(tg, pi - (1 - pi) * (tg / (1 - tg)), "--", label="Treat all")
    ax5.axhline(0, color="k", ls=":", label="Treat none")
    ax5.set_xlabel("Threshold probability"); ax5.set_ylabel("Net benefit")
    ax5.set_ylim(-0.05, 0.6); ax5.legend(fontsize=8)
    ax5.set_title(f"(e) DCA (cost-optimal {dca_cost['t_opt']:.3f})")

    ax6 = plt.subplot(2, 3, 6); ax6.axis("off")
    ax6.set_title("(f) Phlegm-damp subset: Top-5 Apriori rules")
    if rules and rules.get("apriori") is not None and len(rules["apriori"]):
        df = rules["apriori"].head(5).copy()
        txt = ""
        for _, r in df.iterrows():
            ant = ", ".join(r["antecedents"])
            txt += (f"* {{{ant}}}\n"
                    f"    sup={r['support']:.3f}, "
                    f"conf={r['confidence']:.3f}, lift={r['lift']:.3f}\n\n")
        ax6.text(0.02, 0.98, txt, va="top", fontsize=9, family="monospace")
    else:
        ax6.text(0.5, 0.5, "No rules meet thresholds",
                 ha="center", va="center")

    plt.tight_layout()
    plt.savefig(figure_path("Q2_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()



def _risk_level_name(level_code):
    return ["低", "中", "高"][int(level_code)]


def _rules_frame(rules):
    if not rules or rules.get("apriori") is None:
        return pd.DataFrame(columns=["antecedents", "support", "sup_ab", "confidence", "lift", "n_match"])
    df = rules["apriori"].copy()
    if len(df):
        df["antecedents"] = df["antecedents"].apply(
            lambda items: " & ".join(items) if isinstance(items, (list, tuple)) else str(items)
        )
    return df


def save_table_outputs(d, cv_summary, robust, strata_score_A, strata_variants,
                       score_te, p_cal, probs_te, thr_on_score, thr_on_prob,
                       rules, calib_stats, dca_cost, aucs, delong_mat,
                       score_auc, iv_summary):
    """写出主仓库沿用的 Q2 表格接口，文件名保持无版本号。"""
    yte = np.asarray(d["yte"])
    level = np.asarray(strata_score_A["level"], dtype=int)

    summary_rows = []
    for idx, item in enumerate(strata_score_A["strata"]):
        summary_rows.append({
            "risk_level_code": idx + 1,
            "risk_level": item["level"],
            "n": item["n"],
            "positive_rate": item["pos_rate"],
            "NNT": item["NNT"],
            "cL": strata_score_A["cL"],
            "cH": strata_score_A["cH"],
            "CA_Z": strata_score_A["CA_Z"],
            "CA_p": strata_score_A["CA_p"],
        })
    risk_summary = pd.DataFrame(summary_rows)
    risk_summary.to_csv(table_path("Q2_risk_stat.csv"), index=False, encoding="utf-8-sig")
    risk_summary.to_csv(table_path("Q2_stratification_summary.csv"), index=False, encoding="utf-8-sig")

    risk_full = d["Xte_raw"].copy()
    risk_full.insert(0, "sample_index", risk_full.index)
    risk_full["y_true"] = yte
    risk_full["is_phlegm"] = (np.asarray(d["tte"]) == 5).astype(int)
    risk_full["score"] = np.round(score_te, 4)
    risk_full["p_calibrated"] = np.round(p_cal, 6)
    for name, prob in probs_te.items():
        risk_full[f"prob_{name}"] = np.round(prob, 6)
    risk_full["risk_level_code"] = level + 1
    risk_full["risk_level"] = [_risk_level_name(v) for v in level]
    risk_full.to_csv(table_path("Q2_risk_full.csv"), index=False, encoding="utf-8-sig")
    risk_full.to_csv(table_path("Q2_stratification_full.csv"), index=False, encoding="utf-8-sig")

    calib_df = pd.DataFrame({"y": yte, "p": p_cal})
    calib_df["decile"] = pd.qcut(calib_df["p"], q=10, duplicates="drop", labels=False)
    calibration = calib_df.groupby("decile", observed=True).agg(
        n=("y", "size"),
        observed_positive=("y", "sum"),
        predicted_positive=("p", "sum"),
        observed_rate=("y", "mean"),
        predicted_mean=("p", "mean"),
    ).reset_index()
    calibration.to_csv(table_path("Q2_calibration.csv"), index=False, encoding="utf-8-sig")

    cv_rows = []
    for model_name, stat in cv_summary.items():
        ci = stat.get("auc_95ci", (None, None))
        cv_rows.append({
            "model": model_name,
            "cv_auc_mean": stat.get("auc_mean"),
            "cv_auc_std": stat.get("auc_std"),
            "cv_auc_ci_low": ci[0],
            "cv_auc_ci_high": ci[1],
            "cv_brier": stat.get("brier"),
            "test_auc": round(float(aucs.get(model_name, np.nan)), 4),
            "fold_aucs": ";".join(str(v) for v in stat.get("fold_aucs", [])),
        })
    pd.DataFrame(cv_rows).to_csv(table_path("Q2_cv_results.csv"), index=False, encoding="utf-8-sig")

    method_rows = []
    for row in cv_rows:
        method_rows.append({
            "method": row["model"],
            "cv_auc_mean": row["cv_auc_mean"],
            "test_auc": row["test_auc"],
            "cv_brier": row["cv_brier"],
            "role": "主筛查模型" if row["model"] == "LR_WOE" else "非线性对照模型",
        })
    method_rows.append({
        "method": "Scorecard",
        "cv_auc_mean": None,
        "test_auc": round(float(score_auc), 4),
        "cv_brier": None,
        "role": "可解释整数评分卡",
    })
    pd.DataFrame(method_rows).to_csv(table_path("Q2_method_compare.csv"), index=False, encoding="utf-8-sig")

    imp_rows = []
    for rank, (feature, mean_delta, std_delta) in enumerate(robust.get("perm_importance", []), start=1):
        imp_rows.append({
            "rank": rank,
            "feature": feature,
            "delta_auc_mean": mean_delta,
            "delta_auc_std": std_delta,
            "IV": round(float(iv_summary.get(feature, np.nan)), 4),
        })
    pd.DataFrame(imp_rows).to_csv(table_path("Q2_feature_importance.csv"), index=False, encoding="utf-8-sig")

    threshold_rows = []
    for source, values in [("scorecard", thr_on_score), ("probability", thr_on_prob)]:
        for key, value in values.items():
            if key == "Youden_CI":
                threshold_rows.append({
                    "source": source,
                    "criterion": "Youden_CI_low",
                    "threshold": value[0],
                    "note": "bootstrap percentile interval",
                })
                threshold_rows.append({
                    "source": source,
                    "criterion": "Youden_CI_high",
                    "threshold": value[1],
                    "note": "bootstrap percentile interval",
                })
            elif key != "sp_target":
                threshold_rows.append({
                    "source": source,
                    "criterion": key,
                    "threshold": value,
                    "note": f"sp_target={values.get('sp_target')}" if key.startswith("Sp") else "",
                })
    threshold_rows.append({
        "source": "probability",
        "criterion": "cost_sensitive",
        "threshold": dca_cost.get("t_opt"),
        "note": f"C_FN={dca_cost.get('C_FN')}, C_FP={dca_cost.get('C_FP')}, min_cost={dca_cost.get('min_cost')}",
    })
    pd.DataFrame(threshold_rows).to_csv(table_path("Q2_thresholds.csv"), index=False, encoding="utf-8-sig")

    boot_rows = [
        {"metric": "scorecard_youden", "point": thr_on_score.get("Youden"), "ci_low": thr_on_score.get("Youden_CI", (None, None))[0], "ci_high": thr_on_score.get("Youden_CI", (None, None))[1]},
        {"metric": "probability_youden", "point": thr_on_prob.get("Youden"), "ci_low": thr_on_prob.get("Youden_CI", (None, None))[0], "ci_high": thr_on_prob.get("Youden_CI", (None, None))[1]},
    ]
    for model_name, stat in cv_summary.items():
        ci = stat.get("auc_95ci", (None, None))
        boot_rows.append({"metric": f"{model_name}_cv_auc", "point": stat.get("auc_mean"), "ci_low": ci[0], "ci_high": ci[1]})
    pd.DataFrame(boot_rows).to_csv(table_path("Q2_bootstrap_CI.csv"), index=False, encoding="utf-8-sig")

    sensitivity_rows = []
    for item in robust.get("threshold_perturb", []):
        row = {"cH": item["cH"], "cL": item["cL"], "CA_Z": item["CA_Z"], "CA_p": item["CA_p"]}
        for prefix, values in [("n", item.get("n", [])), ("pos", item.get("pos", []))]:
            for label, value in zip(["low", "medium", "high"], values):
                row[f"{prefix}_{label}"] = value
        sensitivity_rows.append(row)
    pd.DataFrame(sensitivity_rows).to_csv(table_path("Q2_sensitivity.csv"), index=False, encoding="utf-8-sig")

    rules_df = _rules_frame(rules)
    rules_df.to_csv(table_path("Q2_apriori_rules_full.csv"), index=False, encoding="utf-8-sig")
    rules_df.head(10).to_csv(table_path("Q2_apriori_rules.csv"), index=False, encoding="utf-8-sig")
    core_combo = rules_df.head(5).rename(columns={"antecedents": "core_combo"})
    core_combo.to_csv(table_path("Q2_core_combo.csv"), index=False, encoding="utf-8-sig")
    with open(table_path("Q2_tree_rules.txt"), "w", encoding="utf-8") as fp:
        fp.write(rules.get("tree_text", "") if rules else "")

    summary_metrics = [
        {"metric": "n_total", "value": int(len(d["X_raw"]))},
        {"metric": "positive_rate", "value": round(float(d["pos_rate"]), 4)},
        {"metric": "leakage_consistency", "value": round(float(d["leak_rate"]), 4)},
        {"metric": "GBDT_test_auc", "value": round(float(aucs.get("GBDT", np.nan)), 4)},
        {"metric": "scorecard_test_auc", "value": round(float(score_auc), 4)},
        {"metric": "Brier", "value": calib_stats.get("Brier")},
        {"metric": "HL_p", "value": calib_stats.get("HL_p")},
        {"metric": "cL_score", "value": strata_score_A.get("cL")},
        {"metric": "cH_score", "value": strata_score_A.get("cH")},
        {"metric": "CA_Z", "value": strata_score_A.get("CA_Z")},
        {"metric": "CA_p", "value": strata_score_A.get("CA_p")},
        {"metric": "best_IV_feature", "value": max(iv_summary.items(), key=lambda kv: kv[1])[0]},
    ]
    pd.DataFrame(summary_metrics).to_csv(table_path("Q2_summary_metrics.csv"), index=False, encoding="utf-8-sig")

    variant_rows = []
    for name, result in strata_variants.items():
        row = {"scheme": name, "cL": result["cL"], "cH": result["cH"], "CA_Z": result["CA_Z"], "CA_p": result["CA_p"]}
        for prefix, values in [("n", [s["n"] for s in result["strata"]]), ("pos", [s["pos_rate"] for s in result["strata"]])]:
            for label, value in zip(["low", "medium", "high"], values):
                row[f"{prefix}_{label}"] = value
        variant_rows.append(row)
    pd.DataFrame(variant_rows).to_csv(table_path("Q2_threshold_scheme_compare.csv"), index=False, encoding="utf-8-sig")

    delong_rows = []
    for pair, values in delong_mat.items():
        row = {"comparison": pair}
        row.update(values)
        delong_rows.append(row)
    pd.DataFrame(delong_rows).to_csv(table_path("Q2_delong_compare.csv"), index=False, encoding="utf-8-sig")

# =============================================================================
# 主流程
# =============================================================================
def main():
    print("=" * 80)
    print(" 问题二：纯数据驱动三级风险预警 (WOE-Logistic + GBDT + RF)")
    print("=" * 80)

    d = module_A_preprocess()
    woe_map, edges_map, iv_summary = woe_encode_train(d["Xtr_raw"], d["ytr"])

    print("\n[B] 信息值 IV 表:")
    for f, iv in sorted(iv_summary.items(), key=lambda t: -t[1]):
        strength = ("very strong" if iv > 0.5 else "strong" if iv > 0.3
                    else "medium" if iv > 0.1 else "weak" if iv > 0.02
                    else "none")
        print(f"   {f:14s} IV={iv:.4f}  [{strength}]")

    models, oof, cv_summary, Xtr_woe, Xte_woe = module_C_cv_train(
        d, woe_map, edges_map)

    sc = module_D_scorecard(Xtr_woe, d["ytr"], woe_map, edges_map)
    score_tr = apply_scorecard(d["Xtr_raw"], sc, woe_map, edges_map)
    score_te = apply_scorecard(d["Xte_raw"], sc, woe_map, edges_map)
    print(f"\n[D] 测试集评分: min={score_te.min():.1f}, max={score_te.max():.1f}, "
          f"mean={score_te.mean():.1f}")
    print(f"    阳性组均分 {score_te[d['yte']==1].mean():.2f}")
    print(f"    阴性组均分 {score_te[d['yte']==0].mean():.2f}")
    score_auc = float(roc_auc_score(d["yte"], score_te))
    print(f"    评分卡 AUC = {score_auc:.4f}")

    def gbdt_factory():
        return GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                           learning_rate=0.05, subsample=0.8,
                                           min_samples_leaf=20,
                                           random_state=RNG)
    p_cal, calib_stats = module_G_calibrate(
        gbdt_factory, d["Xtr_raw"], d["ytr"], d["Xte_raw"], d["yte"],
        name="GBDT")

    probs_te = {
        "LR_WOE":    models["LR_WOE"][0].predict_proba(Xte_woe)[:, 1],
        "GBDT":      models["GBDT"][0].predict_proba(d["Xte_raw"])[:, 1],
        "RF":        models["RF"][0].predict_proba(d["Xte_raw"])[:, 1],
        "Scorecard": score_to_prob(score_te, sc),
    }
    delong_mat, aucs = module_H_roc_delong(d["yte"], probs_te)

    thr_on_score = module_E_thresholds(d["yte"], score_te, B=1000,
                                        label="scorecard")
    thr_on_prob = module_E_thresholds(d["yte"], p_cal, B=1000,
                                       label="prob_cal", save=False)

    print("\n[F] 三级分层四方案对比 (基于 GBDT 校准概率):")
    strata_variants = {}
    for scm in ["A", "B", "C", "D"]:
        r = module_F_stratify(d["yte"], p_cal,
                              thr_on_prob["Youden"], scheme=scm)
        strata_variants[scm] = r
        print(f"   方案{scm}: cL={r['cL']}  cH={r['cH']}  "
              f"n={[s['n'] for s in r['strata']]}  "
              f"pos={[s['pos_rate'] for s in r['strata']]}  "
              f"CA Z={r['CA_Z']}  p={r['CA_p']:.2e}")

    strata_score_A = module_F_stratify(d["yte"], score_te,
                                        thr_on_score["Youden"], scheme="A")
    print(f"\n[F] 单层方案 A (基于评分卡分数, 仅测试集):")
    print(f"   cL={strata_score_A['cL']}  cH={strata_score_A['cH']}  "
          f"n={[s['n'] for s in strata_score_A['strata']]}  "
          f"pos={[s['pos_rate'] for s in strata_score_A['strata']]}")
    print(f"   CA Z={strata_score_A['CA_Z']}  p={strata_score_A['CA_p']:.2e}")

    # ==========================================================
    # 两层融合 (全数据集 1000 样本)
    #   Layer 1: 血脂四项异常 → 高风险 (硬规则)
    #   Layer 2: 残余样本按评分卡分层 (复用主方案阈值)
    # ==========================================================
    cL_fuse = strata_score_A["cL"]
    cH_fuse = strata_score_A["cH"]
    score_full = apply_scorecard(d["X_raw"], sc, woe_map, edges_map)
    strata_twolayer = module_F_twolayer(
        d["y"], score_full, d["abn_full"], cL_fuse, cH_fuse)
    print(f"\n[F2] 两层融合 (全数据集 n={len(d['y'])}):")
    print(f"   Layer1 硬规则捕获: {strata_twolayer['layer1_n']} 人")
    print(f"   Layer2 评分分层: {strata_twolayer['layer2_counts']}")
    print(f"   融合分层 n={[s['n'] for s in strata_twolayer['strata']]}, "
          f"阳性率={[s['pos_rate'] for s in strata_twolayer['strata']]}")
    print(f"   CA Z={strata_twolayer['CA_Z']}  p={strata_twolayer['CA_p']:.2e}")

    dca_cost = module_I_decision_curve(d["yte"], p_cal, C_FN=5, C_FP=1)

    rules = module_J_phlegm_rules(d["X_raw"], d["y"], d["tizhi"])

    robust = module_K_robustness(models, d, probs_te,
                                  score_te, strata_score_A["cH"])

    module_L_summary_plot(d["yte"], probs_te, p_cal, strata_score_A,
                          thr_on_score, dca_cost, rules, score_te)

    save_table_outputs(d, cv_summary, robust, strata_score_A, strata_variants,
                       score_te, p_cal, probs_te, thr_on_score, thr_on_prob,
                       rules, calib_stats, dca_cost, aucs, delong_mat,
                       score_auc, iv_summary)

    dump = {
        "data_profile": dict(
            n=int(len(d["X_raw"])), pos_rate=float(d["pos_rate"]),
            leak_rate=float(d["leak_rate"]),
            n_phlegm=int(d["n_phlegm"]),
            n_train=int(len(d["ytr"])), n_test=int(len(d["yte"])),
        ),
        "features_in_model": FEATURES_ALL,
        "features_excluded": LIPID_BLACKLIST,
        "IV_summary": {f: round(v, 4) for f, v in iv_summary.items()},
        "scorecard_config": dict(
            PDO=sc["PDO"], base_points=sc["base_points"],
            offset=round(sc["offset"], 3),
            base_adjust=round(sc["base_adjust"], 3),
            intercept=round(sc["intercept"], 4),
            coefs={f: round(b, 4) for f, b in sc["coefs"].items()},
        ),
        "scorecard_bins": {
            f: tbl[["bin_id", "lo", "hi", "n", "n1",
                    "woe", "iv_i", "points"]].round(4)
               .to_dict(orient="records")
            for f, tbl in sc["card"].items()
        },
        "cv_summary": cv_summary,
        "test_aucs": {k: round(float(a), 4) for k, a in aucs.items()},
        "score_auc_test": round(score_auc, 4),
        "delong": delong_mat,
        "calibration": calib_stats,
        "thresholds_on_score": thr_on_score,
        "thresholds_on_prob":  thr_on_prob,
        "cost_sensitive": dca_cost,
        "stratification_main_A_on_score": {
            "cL": strata_score_A["cL"], "cH": strata_score_A["cH"],
            "strata": strata_score_A["strata"],
            "CA_Z": strata_score_A["CA_Z"],
            "CA_p": strata_score_A["CA_p"]},
        "two_layer_fusion_full_dataset": {
            "cL": strata_twolayer["cL"], "cH": strata_twolayer["cH"],
            "layer1_n": strata_twolayer["layer1_n"],
            "layer2_counts": strata_twolayer["layer2_counts"],
            "strata": strata_twolayer["strata"],
            "CA_Z": strata_twolayer["CA_Z"],
            "CA_p": strata_twolayer["CA_p"]},
        "stratification_variants_on_prob": {
            k: {"cL": v["cL"], "cH": v["cH"],
                "n": [s["n"] for s in v["strata"]],
                "pos": [s["pos_rate"] for s in v["strata"]],
                "CA_Z": v["CA_Z"], "CA_p": v["CA_p"]}
            for k, v in strata_variants.items()},
        "robustness": {
            "permutation_importance": robust["perm_importance"],
            "threshold_perturb": robust["threshold_perturb"],
            "kappa": robust["kappa"],
        },
        "phlegm_rules": {
            "apriori_top": (rules["apriori"].to_dict(orient="records")
                            if rules and rules.get("apriori") is not None
                            else []),
            "tree_text": rules.get("tree_text") if rules else None,
            "permutation_top5":
                (rules.get("permutation")[:5] if rules and
                 rules.get("permutation") is not None else None),
            "phlegm_pos_rate":
                (rules.get("base_sup") if rules else None),
        },
    }
    with open(table_path("Q2_results.json"), "w", encoding="utf-8") as fp:
        json.dump(dump, fp, indent=2, ensure_ascii=False, default=str)
    print("\n[OK] JSON 写入 src/outputs/tables/Q2_results.json")
    print("[OK] 图表写入 src/outputs/figures/Q2_*.png")
    print("=" * 80)
    return dump


if __name__ == "__main__":
    main()
