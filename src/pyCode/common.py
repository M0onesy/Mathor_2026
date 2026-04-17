from __future__ import annotations

import shutil
from pathlib import Path

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
        if path.is_file() and not path.name.startswith("~$")
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


def save_csv(df: pd.DataFrame, name: str, index: bool = True) -> Path:
    path = table_path(name)
    df.to_csv(path, index=index, encoding="utf-8-sig")
    return path


def save_text(text: str, name: str) -> Path:
    path = table_path(name)
    path.write_text(text, encoding="utf-8")
    return path


def save_figure(fig: plt.Figure, name: str, dpi: int = 150) -> Path:
    path = figure_path(name)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def copy_to_tables(source: Path, dest_name: str | None = None) -> Path:
    ensure_output_dirs()
    destination = table_path(dest_name or source.name)
    shutil.copy2(source, destination)
    return destination
