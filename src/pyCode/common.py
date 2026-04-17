from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RANDOM_SEED = 42
CODE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = CODE_ROOT.parent
PROJECT_ROOT = SRC_ROOT.parent
DATA_ROOT = SRC_ROOT / "Data"
OUTPUT_ROOT = SRC_ROOT / "outputs"
FIGURE_DIR = OUTPUT_ROOT / "figures"
TABLE_DIR = OUTPUT_ROOT / "tables"


@dataclass(frozen=True)
class Columns:
    sample_id: str = "样本ID"
    constitution_label: str = "体质标签"
    pinghe: str = "平和质"
    qixu: str = "气虚质"
    yangxu: str = "阳虚质"
    yinxu: str = "阴虚质"
    tanshi: str = "痰湿质"
    shire: str = "湿热质"
    xueyu: str = "血瘀质"
    qiyu: str = "气郁质"
    tebing: str = "特禀质"
    adl_toilet: str = "ADL用厕"
    adl_eat: str = "ADL吃饭"
    adl_walk: str = "ADL步行"
    adl_dress: str = "ADL穿衣"
    adl_bath: str = "ADL洗澡"
    adl_total: str = "ADL总分"
    iadl_shop: str = "IADL购物"
    iadl_cook: str = "IADL做饭"
    iadl_finance: str = "IADL理财"
    iadl_transport: str = "IADL交通"
    iadl_medication: str = "IADL服药"
    iadl_total: str = "IADL总分"
    activity_total: str = "活动量表总分（ADL总分+IADL总分）"
    hdl: str = "HDL-C（高密度脂蛋白）"
    ldl: str = "LDL-C（低密度脂蛋白）"
    tg: str = "TG（甘油三酯）"
    tc: str = "TC（总胆固醇）"
    glucose: str = "空腹血糖"
    uric_acid: str = "血尿酸"
    bmi: str = "BMI"
    diagnosis: str = "高血脂症二分类标签"
    lipid_type: str = "血脂异常分型标签（确诊病例）"
    age_group: str = "年龄组"
    sex: str = "性别"
    smoking: str = "吸烟史"
    drinking: str = "饮酒史"


COLS = Columns()

CONSTITUTION_FEATURES = [
    COLS.pinghe,
    COLS.qixu,
    COLS.yangxu,
    COLS.yinxu,
    COLS.tanshi,
    COLS.shire,
    COLS.xueyu,
    COLS.qiyu,
    COLS.tebing,
]

ADL_FEATURES = [
    COLS.adl_toilet,
    COLS.adl_eat,
    COLS.adl_walk,
    COLS.adl_dress,
    COLS.adl_bath,
    COLS.adl_total,
]

IADL_FEATURES = [
    COLS.iadl_shop,
    COLS.iadl_cook,
    COLS.iadl_finance,
    COLS.iadl_transport,
    COLS.iadl_medication,
    COLS.iadl_total,
]

ACTIVITY_ITEM_FEATURES = ADL_FEATURES[:5] + IADL_FEATURES[:5]
ACTIVITY_SUMMARY_FEATURES = [COLS.adl_total, COLS.iadl_total, COLS.activity_total]
TREE_ACTIVITY_FEATURES = ACTIVITY_ITEM_FEATURES + [COLS.activity_total]
ACTIVITY_FEATURES = ACTIVITY_ITEM_FEATURES + ACTIVITY_SUMMARY_FEATURES
LIPID_FEATURES = [COLS.tc, COLS.tg, COLS.ldl, COLS.hdl]
METABOLIC_FEATURES = [COLS.glucose, COLS.uric_acid, COLS.bmi]
DEMOGRAPHIC_FEATURES = [COLS.age_group, COLS.sex, COLS.smoking, COLS.drinking]

DIRECT_DIAGNOSTIC_FEATURES = LIPID_FEATURES
AUXILIARY_WARNING_FEATURES = METABOLIC_FEATURES + ACTIVITY_ITEM_FEATURES
LINEAR_EARLY_SCREENING_FEATURES = CONSTITUTION_FEATURES + ACTIVITY_ITEM_FEATURES + METABOLIC_FEATURES + DEMOGRAPHIC_FEATURES
TREE_EARLY_SCREENING_FEATURES = CONSTITUTION_FEATURES + TREE_ACTIVITY_FEATURES + METABOLIC_FEATURES + DEMOGRAPHIC_FEATURES
EARLY_SCREENING_FEATURES = TREE_EARLY_SCREENING_FEATURES

CONSTITUTION_NAME_MAP = {
    1: "平和质",
    2: "气虚质",
    3: "阳虚质",
    4: "阴虚质",
    5: "痰湿质",
    6: "湿热质",
    7: "血瘀质",
    8: "气郁质",
    9: "特禀质",
}

AGE_GROUP_MAP = {
    1: "40-49岁",
    2: "50-59岁",
    3: "60-69岁",
    4: "70-79岁",
    5: "80-89岁",
}

SEX_MAP = {0: "女", 1: "男"}

LIPID_NORMAL_RANGES = {
    COLS.tc: (3.1, 6.2),
    COLS.tg: (0.56, 1.7),
    COLS.ldl: (2.07, 3.1),
    COLS.hdl: (1.04, 1.55),
}

METABOLIC_NORMAL_RANGES = {
    COLS.glucose: (3.9, 6.1),
    COLS.bmi: (18.5, 23.9),
}

TCM_COST = {1: 30, 2: 80, 3: 130}
TCM_PLAN = {
    1: "基础调理：饮食调理 + 穴位按摩",
    2: "中度调理：饮食调理 + 穴位按摩 + 八段锦",
    3: "强化调理：饮食调理 + 穴位按摩 + 八段锦 + 中药代茶饮",
}
ACTIVITY_COST = {1: 3, 2: 5, 3: 8}
ACTIVITY_DURATION = {1: 10, 2: 20, 3: 30}


def ensure_output_dirs() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def configure_plotting() -> None:
    matplotlib.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
    plt.style.use("seaborn-v0_8-whitegrid")


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)


def find_sample_data_path() -> Path:
    candidates = [
        path
        for path in DATA_ROOT.glob("*样例数据.xlsx")
        if not path.name.startswith("~$")
    ]
    if not candidates:
        raise FileNotFoundError("未找到样例数据文件，请确认 Excel 文件已放入 src/Data 目录。")
    candidates.sort(key=lambda item: ("附件1" not in item.name, item.name))
    return candidates[0]


def load_data() -> pd.DataFrame:
    ensure_output_dirs()
    return pd.read_excel(find_sample_data_path())


def figure_path(name: str) -> Path:
    ensure_output_dirs()
    return FIGURE_DIR / name


def table_path(name: str) -> Path:
    ensure_output_dirs()
    return TABLE_DIR / name


def save_figure(fig: plt.Figure, name: str, dpi: int = 160) -> Path:
    path = figure_path(name)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> tuple[Path, Path]:
    csv_path = table_path(f"{name}.csv")
    xlsx_path = table_path(f"{name}.xlsx")
    df.to_csv(csv_path, index=index, encoding="utf-8-sig")
    df.to_excel(xlsx_path, index=index)
    return csv_path, xlsx_path


def save_workbook(name: str, sheets: Dict[str, pd.DataFrame], index: bool = False) -> Path:
    path = table_path(f"{name}.xlsx")
    with pd.ExcelWriter(path) as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name[:31], index=index)
    return path


def save_json(data: dict, name: str) -> Path:
    path = table_path(f"{name}.json")
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def minmax_scale(series: pd.Series) -> pd.Series:
    denom = series.max() - series.min()
    if abs(denom) < 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.min()) / denom


def rank_to_borda(series: pd.Series, ascending: bool = False) -> pd.Series:
    ranks = series.rank(ascending=ascending, method="average")
    return len(series) - ranks + 1


def bh_fdr(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for idx in range(n - 1, -1, -1):
        rank = idx + 1
        value = ranked[idx] * n / rank
        running = min(running, value)
        adjusted[idx] = running
    result = np.empty(n, dtype=float)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def cohen_d(x: pd.Series, y: pd.Series) -> float:
    x_var = x.var(ddof=1)
    y_var = y.var(ddof=1)
    pooled = ((len(x) - 1) * x_var + (len(y) - 1) * y_var) / max(len(x) + len(y) - 2, 1)
    if pooled <= 0:
        return 0.0
    return (x.mean() - y.mean()) / np.sqrt(pooled)


def get_lipid_flags(df: pd.DataFrame) -> pd.DataFrame:
    flags = {}
    for feature, (lower, upper) in LIPID_NORMAL_RANGES.items():
        flags[f"{feature}_低"] = (df[feature] < lower).astype(int)
        flags[f"{feature}_高"] = (df[feature] > upper).astype(int)
        if feature == COLS.hdl:
            flags[f"{feature}_异常"] = flags[f"{feature}_低"]
        else:
            flags[f"{feature}_异常"] = flags[f"{feature}_高"]
    flags_df = pd.DataFrame(flags, index=df.index)
    flags_df["血脂异常项数"] = flags_df[[f"{feature}_异常" for feature in LIPID_FEATURES]].sum(axis=1)
    flags_df["任一血脂异常"] = (flags_df["血脂异常项数"] > 0).astype(int)
    return flags_df


def get_metabolic_flags(df: pd.DataFrame) -> pd.DataFrame:
    flags = {
        "血糖异常": (
            (df[COLS.glucose] < METABOLIC_NORMAL_RANGES[COLS.glucose][0])
            | (df[COLS.glucose] > METABOLIC_NORMAL_RANGES[COLS.glucose][1])
        ).astype(int),
        "BMI异常": (
            (df[COLS.bmi] < METABOLIC_NORMAL_RANGES[COLS.bmi][0])
            | (df[COLS.bmi] > METABOLIC_NORMAL_RANGES[COLS.bmi][1])
        ).astype(int),
    }
    male_high = (df[COLS.sex] == 1) & (df[COLS.uric_acid] > 428)
    female_high = (df[COLS.sex] == 0) & (df[COLS.uric_acid] > 357)
    male_low = (df[COLS.sex] == 1) & (df[COLS.uric_acid] < 208)
    female_low = (df[COLS.sex] == 0) & (df[COLS.uric_acid] < 155)
    flags["尿酸异常"] = (male_high | female_high | male_low | female_low).astype(int)
    flags_df = pd.DataFrame(flags, index=df.index)
    flags_df["代谢异常项数"] = flags_df.sum(axis=1)
    return flags_df


def lipid_excess_ratio(value: float, lower: float, upper: float) -> float:
    if value < lower:
        return (lower - value) / max(lower, 1e-6)
    if value > upper:
        return (value - upper) / max(upper, 1e-6)
    return 0.0


def describe_age_group(code: int) -> str:
    return AGE_GROUP_MAP.get(int(code), f"未知年龄组({code})")


def describe_sex(code: int) -> str:
    return SEX_MAP.get(int(code), f"未知性别({code})")


def get_tcm_level(score: float) -> int:
    if score <= 58:
        return 1
    if score <= 61:
        return 2
    return 3


def get_max_activity_level(age_group: int, activity_score: float) -> int:
    if age_group <= 2:
        age_limit = 3
    elif age_group <= 4:
        age_limit = 2
    else:
        age_limit = 1

    if activity_score < 40:
        score_limit = 1
    elif activity_score < 60:
        score_limit = 2
    else:
        score_limit = 3
    return min(age_limit, score_limit)


def monthly_drop_rate(activity_level: int, weekly_frequency: int) -> float:
    if weekly_frequency < 5:
        return 0.0
    return (activity_level - 1) * 0.03 + (weekly_frequency - 5) * 0.01


def next_tanshi_score(score: float, activity_level: int, weekly_frequency: int) -> float:
    rate = monthly_drop_rate(activity_level, weekly_frequency)
    return round(score * (1 - rate), 2)
