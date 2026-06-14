"""
数据清洗工具 - 基于《Python金融大数据风控建模实战：基于机器学习》第4章
适用于 Lending Club 数据集

书中对应章节：
- 第4章：数据清洗与预处理（数据集成、数据清洗、探索性数据分析）
- 第19章：Lending Club 数据集实战（数据清洗与预处理）

注意：好坏样本定义（§19.2.2）属于独立步骤，在 label_definition.py 中完成。
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. 数据集成（书 §4.1）
# ─────────────────────────────────────────────

def load_data(filepath: str,
              nrows: Optional[int] = None,
              encoding: str = "utf-8") -> pd.DataFrame:
    """
    §4.1 数据集成
    读取 CSV，跳过 LendingClub 文件中常见的多余说明行。
    """
    df = pd.read_csv(
        filepath,
        nrows=nrows,
        encoding=encoding,
        low_memory=False,
    )

    # 部分 LendingClub 文件在列名行之后有一行全空行，检测到则删除
    if df.iloc[0].isnull().all():
        df = df.iloc[1:].reset_index(drop=True)
        logger.info("检测到首行为全空行，已删除")

    logger.info("数据加载完成：%d 行 × %d 列", *df.shape)
    return df


# ─────────────────────────────────────────────
# 2. 数据清洗（书 §4.2）
# ─────────────────────────────────────────────

class LendingClubCleaner:
    """
    §4.2 数据清洗

    包含以下步骤：
      1. 去除重复行
      2. 类型转换（百分号、就业年限、贷款期限 → 数值）
      3. 去除无信息列（纯 ID、URL、自由文本）
      4. 去除高缺失率列
      5. 去除单值列（无区分度）
      6. 保留 NaN，供后续分箱阶段单独归为缺失 bin

    注意：
    - 目标泄露列（还款事后变量）需结合业务理解后在特征工程阶段手动指定剔除，
      此处不做自动删除，避免误删有用字段。
    - loan_status 保留，供后续好坏样本定义步骤使用。
    """

    # 无任何建模价值的固定列（纯标识符 / 链接 / 自由文本）
    DROP_ID_COLS = ["id", "member_id", "url", "desc"]

    def __init__(self, missing_threshold: float = 0.3):
        """
        Parameters
        ----------
        missing_threshold : float
            列缺失率超过该阈值则直接删除（默认 30%）。
            低于阈值的缺失值保留为 NaN，由后续分箱阶段单独归为一个 bin。
        """
        self.missing_threshold = missing_threshold
        self._dropped_cols: list = []

    # ── 类型转换辅助方法 ────────────────────────

    @staticmethod
    def _parse_percent_cols(df: pd.DataFrame) -> pd.DataFrame:
        """将形如 '13.99%' 的列转为 float。"""
        for col in df.columns:
            if df[col].dtype == object:
                sample = df[col].dropna().head(200)
                if len(sample) and sample.str.endswith("%").mean() > 0.8:
                    df[col] = df[col].str.rstrip("%").astype(float)
                    logger.info("百分号列转换：%s", col)
        return df

    @staticmethod
    def _parse_emp_length(df: pd.DataFrame,
                          col: str = "emp_length") -> pd.DataFrame:
        """'10+ years' → 10，'< 1 year' → 0，其余 → NaN。"""
        if col not in df.columns:
            return df
        mapping = {
            "< 1 year": 0,
            "1 year": 1, "2 years": 2, "3 years": 3,
            "4 years": 4, "5 years": 5, "6 years": 6,
            "7 years": 7, "8 years": 8, "9 years": 9,
            "10+ years": 10,
        }
        df[col] = df[col].map(mapping)
        return df

    @staticmethod
    def _parse_term(df: pd.DataFrame,
                    col: str = "term") -> pd.DataFrame:
        """' 36 months' → 36。"""
        if col not in df.columns:
            return df
        df[col] = df[col].str.strip().str.extract(r"(\d+)").astype(float)
        return df

    # ── 主流程 ──────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """执行完整清洗流程，返回清洗后的 DataFrame。"""
        df = df.copy()
        n_rows_raw, n_cols_raw = df.shape

        # Step 1：去除重复行
        n_dup = df.duplicated().sum()
        if n_dup:
            df.drop_duplicates(inplace=True)
            logger.info("去除重复行：%d 条", n_dup)
        else:
            logger.info("无重复行。")

        # Step 2：类型转换
        df = self._parse_percent_cols(df)
        df = self._parse_emp_length(df)
        df = self._parse_term(df)

        # Step 2b：处理混合类型列
        for col in df.select_dtypes(include="object").columns:
            converted = pd.to_numeric(df[col], errors="coerce")
            success_rate = converted.notna().sum() / max(df[col].notna().sum(), 1)
            if success_rate >= 0.8:
                df[col] = converted
                logger.info("混合类型列强制转为数值：%s（成功率 %.1f%%）",
                            col, success_rate * 100)

        # Step 3：去除无信息列（ID / URL / 自由文本）
        id_cols = [c for c in self.DROP_ID_COLS if c in df.columns]
        df.drop(columns=id_cols, inplace=True)
        self._dropped_cols.extend(id_cols)
        logger.info("去除无信息列：%s", id_cols)

        # Step 4：去除高缺失率列
        missing_rate = df.isnull().mean()
        high_missing = missing_rate[missing_rate > self.missing_threshold].index.tolist()
        df.drop(columns=high_missing, inplace=True)
        self._dropped_cols.extend(high_missing)
        logger.info(
            "去除高缺失率列（>%.0f%%）%d 个：%s",
            self.missing_threshold * 100,
            len(high_missing),
            high_missing,
        )

        # Step 5：去除单值列（无区分度）
        single_val_cols = [
            c for c in df.columns
            if df[c].nunique(dropna=False) <= 1
        ]
        df.drop(columns=single_val_cols, inplace=True)
        self._dropped_cols.extend(single_val_cols)
        logger.info("去除单值列 %d 个：%s", len(single_val_cols), single_val_cols)

        # Step 6：统计剩余缺失值，保留 NaN
        missing_summary = df.isnull().sum()
        missing_summary = missing_summary[missing_summary > 0]
        if len(missing_summary):
            logger.info(
                "以下 %d 列存在缺失值，保留 NaN 供分箱阶段单独处理：\n%s",
                len(missing_summary),
                missing_summary.to_string(),
            )
        else:
            logger.info("无剩余缺失值列。")

        logger.info(
            "数据清洗完成：%d→%d 行，%d→%d 列",
            n_rows_raw, len(df), n_cols_raw, len(df.columns),
        )
        return df

    @property
    def dropped_columns(self) -> list:
        return self._dropped_cols


# ─────────────────────────────────────────────
# 3. 探索性数据分析（书 §4.3）
# ─────────────────────────────────────────────

def _get_col_types(df: pd.DataFrame,
                   cat_threshold: int = 10) -> tuple[list, list]:
    """
    将列分为连续型和类别型。

    判断逻辑（按优先级）：
    1. dtype 为 object → 类别型（混合类型列已在清洗阶段由 Step 2b 强制转换，
       能转成数值的不会再以 object 出现在这里）
    2. dtype 为数值型，唯一值数量 <= cat_threshold → 视为类别型（编码变量）
    3. dtype 为数值型，唯一值数量 > cat_threshold  → 连续型
    """
    cont_cols, cat_cols = [], []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            cat_cols.append(col)
        elif df[col].nunique() <= cat_threshold:
            cat_cols.append(col)
        else:
            cont_cols.append(col)
    return cont_cols, cat_cols


def eda_report(df: pd.DataFrame,
               cat_threshold: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    §4.3 探索性数据分析 — 统计摘要表

    连续型变量（图4-2）：count、mean、std、min、25%、50%、75%、max、null_rate
    类别型变量：count、n_unique、top_value、top_freq、null_rate

    Parameters
    ----------
    cat_threshold : 数值列唯一值数量 <= 该值时视为类别型（默认10）

    Returns
    -------
    cont_report : 连续型变量统计表
    cat_report  : 类别型变量统计表
    """
    cont_cols, cat_cols = _get_col_types(df, cat_threshold)
    logger.info("列类型识别：连续型 %d 列，类别型 %d 列", len(cont_cols), len(cat_cols))

    # 连续型统计（对齐书中图4-2的输出）
    cont_rows = []
    for col in cont_cols:
        s = df[col].dropna()
        cont_rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "count": s.count(),
            "null_rate": round(df[col].isnull().mean(), 4),
            "mean": round(s.mean(), 6),
            "std": round(s.std(), 6),
            "min": s.min(),
            "25%": s.quantile(0.25),
            "50%": s.quantile(0.50),
            "75%": s.quantile(0.75),
            "max": s.max(),
        })
    cont_report = (
        pd.DataFrame(cont_rows).set_index("column")
        if cont_rows else pd.DataFrame()
    )

    # 类别型统计
    cat_rows = []
    for col in cat_cols:
        non_null = df[col].notna().sum()
        vc = df[col].value_counts()
        top_val = vc.index[0] if len(vc) else np.nan
        top_freq = round(vc.iloc[0] / non_null, 4) if len(vc) and non_null else np.nan
        cat_rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "count": non_null,
            "null_rate": round(df[col].isnull().mean(), 4),
            "n_unique": df[col].nunique(),
            "top_value": top_val,
            "top_freq": top_freq,
        })
    cat_report = (
        pd.DataFrame(cat_rows).set_index("column")
        if cat_rows else pd.DataFrame()
    )

    logger.info("EDA 统计摘要生成完毕")
    return cont_report, cat_report


# ─────────────────────────────────────────────
# 4. EDA 可视化（书 §4.3 图4-3、图4-4）
# ─────────────────────────────────────────────

def plot_missing(df: pd.DataFrame,
                 figsize: tuple = (14, 5),
                 save_path: Optional[str] = None) -> None:
    """
    图4-3：各变量缺失值比例柱状图（对应书中 missingno.bar）

    用 matplotlib 原生实现，无需安装 missingno，效果等价。
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]

    null_rate = df.isnull().mean().sort_values()
    non_null_count = df.notnull().sum().loc[null_rate.index]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(null_rate)), 1 - null_rate, color="#555555", width=0.6)

    # 顶部标注非空数量（对应 missingno.bar labels=True）
    for i, (col, cnt) in enumerate(non_null_count.items()):
        ax.text(i, (1 - null_rate[col]) + 0.01, str(cnt),
                ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(range(len(null_rate)))
    ax.set_xticklabels(null_rate.index, rotation=90, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Non-null Rate")
    ax.set_title("Fig 1  Missing Value Analysis for Variables")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info("缺失值图已保存：%s", save_path)
    else:
        plt.show()
    plt.close()


def plot_boxplot(df: pd.DataFrame,
                 cat_threshold: int = 10,
                 cols_per_row: int = 4,
                 figsize_per_col: tuple = (3.5, 4),
                 save_path: Optional[str] = None) -> None:
    """
    图4-4：连续型变量箱线图（用于异常值分析）

    每行最多 cols_per_row 个子图，自动换行。
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]

    cont_cols, _ = _get_col_types(df, cat_threshold)
    if not cont_cols:
        logger.warning("未找到连续型变量，跳过箱线图。")
        return

    n = len(cont_cols)
    n_rows = (n + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(
        n_rows, cols_per_row,
        figsize=(figsize_per_col[0] * cols_per_row, figsize_per_col[1] * n_rows),
    )
    axes = np.array(axes).flatten()

    for i, col in enumerate(cont_cols):
        data = df[col].dropna()
        axes[i].boxplot(data, vert=True, patch_artist=True,
                        boxprops=dict(facecolor="white", color="black"),
                        medianprops=dict(color="black"),
                        flierprops=dict(marker="o", markersize=3,
                                        markerfacecolor="gray", linestyle="none"))
        axes[i].set_title(col, fontsize=9)
        axes[i].tick_params(labelsize=8)

    # 隐藏多余子图
    for j in range(len(cont_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Fig 2 Outliers Analysis for Variables", fontsize=11, y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("箱线图已保存：%s", save_path)
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────
# 5. Pipeline 入口
# ─────────────────────────────────────────────

def run_cleaning_pipeline(
    filepath: str,
    output_path: str = "cleaned_lending_club.csv",
    nrows: Optional[int] = None,
    missing_threshold: float = 0.3,
    cat_threshold: int = 10,
    plot: bool = True,
) -> pd.DataFrame:
    """
    数据清洗 Pipeline（书 §4）

    Parameters
    ----------
    filepath          : 原始 CSV 路径
    output_path       : 清洗后保存路径
    nrows             : 仅读取前 N 行（调试用）
    missing_threshold : 高缺失率删除阈值
    cat_threshold     : 数值列唯一值 <= 该值时视为类别型
    plot              : 是否生成并保存 EDA 图（图1、图2）

    Returns
    -------
    清洗后的 DataFrame（含 loan_status，待后续好坏样本定义）
    """
    # 1. 数据集成
    df = load_data(filepath, nrows=nrows)

    # 2. 数据清洗
    cleaner = LendingClubCleaner(missing_threshold=missing_threshold)
    df_clean = cleaner.fit_transform(df)

    # 3. 探索性数据分析 — 统计摘要
    cont_report, cat_report = eda_report(df_clean, cat_threshold=cat_threshold)

    cont_path = output_path.replace(".csv", "_eda_continuous.csv")
    cat_path  = output_path.replace(".csv", "_eda_categorical.csv")
    cont_report.to_csv(cont_path)
    cat_report.to_csv(cat_path)
    logger.info("连续型 EDA 报告已保存：%s", cont_path)
    logger.info("类别型 EDA 报告已保存：%s", cat_path)

    # 4. 探索性数据分析 — 可视化（图1、图2）
    if plot:
        missing_fig = output_path.replace(".csv", "_fig43_missing.png")
        boxplot_fig = output_path.replace(".csv", "_fig44_boxplot.png")
        plot_missing(df_clean, save_path=missing_fig)
        plot_boxplot(df_clean, cat_threshold=cat_threshold, save_path=boxplot_fig)

    # 5. 保存清洗结果
    df_clean.to_csv(output_path, index=False)
    logger.info("清洗后数据已保存至：%s", output_path)

    return df_clean
