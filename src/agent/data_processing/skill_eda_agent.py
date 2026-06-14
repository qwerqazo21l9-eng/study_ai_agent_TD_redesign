"""
skill_eda_agent.py — 信贷EDA Agent Skill
书籍参考：《Python金融大数据风控建模实战》§3.1.2（好坏样本定义）+ §4.3（EDA可视化）

工作流（LLM驱动，工具执行）：
  Step 1: Tool 1  → 获取所有列名+类型，传给LLM
  Step 2: LLM     → 识别借贷最终状态列
  Step 3: Tool 2  → 获取该列所有唯一值，传给LLM
  Step 4: LLM     → 按M0/M2+定义输出 good_values / bad_values JSON
  Step 5: Tool 3  → 按定义打标签过滤数据
  Step 6: Tool 4/5→ 批量绘图（连续型+类别型）
  Step 7: Tool 6  → 汇总生成 HTML 报告
"""

import json
import logging
import os
from typing import Dict, Any, Tuple, List

import pandas as pd
from src.utils.llm import get_llm
from src.agent.data_processing.tools_eda import (
    get_column_names,
    get_column_unique_values,
    filter_by_target_def,
    plot_numeric_distribution,
    plot_categorical_distribution,
    generate_html_report,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ════════════════════════════════════════════════════════════
# Skill System Prompt
# ════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
# Role
你是一个**自动化信贷数据EDA专家**。你的任务是从清洗后的贷款数据中：
1. 识别表示"贷款最终还款结果"的目标列
2. 根据教材滚动率分析（M0/M1+）定义好坏样本
3. 指导生成每个特征的好坏样本分布可视化报告

# Core Knowledge: 好坏样本定义（Python金融大数据风控建模实战书§3.1.2）

教材定义（滚动率分析 Roll Rate Analysis）：
- **好样本 M0**：观察窗口结束时逾期天数为0（正常还款或已结清）
  - Lending Club 中对应：Fully Paid
- **坏样本 M1+**：观察窗口结束时逾期天数 ≥ 30天，包括严重违约、已核销、长期逾期
  - Lending Club 中对应：Charged Off, Default, Late (31-120 days)
  - 注意：Late (31-120 days) 逾期已超过30天，属于坏样本，**不可放入 ignore_values**
- **忽略样本**：M1（逾期1-30天，仍在观察窗口内，结果未确定）
  - Lending Club 中对应：Current, In Grace Period, Late (16-30 days)
  - 注意："Late" 值只有 Late (16-30 days) 才是忽略样本，其余一律归入坏样本

# Task 1: 识别目标列

给定列名列表和数据类型，选出最能代表"贷款最终结果"的列。
候选关键词：loan_status, status, loan_state, final_status, account_status, collection_status

**输出格式（JSON）：**
```json
{"target_column": "列名"}
```
只输出JSON，不要有其他文字。

# Task 2: 定义好坏样本

给定目标列的所有唯一值，按照M0/M1+定义分类。

**输出格式（JSON）：**
```json
{
  "target_column": "列名",
  "good_values": ["值1", "值2"],
  "bad_values":  ["值A", "值B"],
  "ignore_values": ["值X"],
  "reasoning": "简短说明每个值归类的理由"
}
```
只输出JSON，不要有其他文字。reasoning 字段用中文简要解释。

# Constraints
- 严格按照教材M0/M1+定义，不要发明新规则
- 如果某个值含义不明确，优先放入 ignore_values，宁可保守也不要错误打标
- 输出必须是合法JSON，不要在JSON外面包裹 markdown 代码块
"""


# ════════════════════════════════════════════════════════════
# LLM调用封装（异步）
# ════════════════════════════════════════════════════════════

async def _call_llm(messages: list) -> str:
    """使用当前项目的 get_llm() 调用 LLM，返回文本内容。"""
    llm = get_llm()
    response = await llm.ainvoke(messages)
    return response.content.strip()


async def _parse_json_response(text: str) -> Dict:
    """从LLM输出中提取JSON，兼容带 ```json ... ``` 包裹的情况。"""
    for fence in ("```json", "```"):
        text = text.replace(fence, "")
    text = text.strip().strip("`").strip()
    return json.loads(text)


# ════════════════════════════════════════════════════════════
# Step 2: LLM 识别目标列
# ════════════════════════════════════════════════════════════

async def llm_identify_target_column(
    col_info: Dict[str, str],
) -> str:
    """
    Step 2: 将列名+类型送给LLM，让它选出目标列。

    Parameters
    ----------
    col_info : {列名: dtype} 字典

    Returns
    -------
    目标列名（字符串）
    """
    col_list = "\n".join(f"  {name}: {dtype}" for name, dtype in col_info.items())
    user_msg = f"""以下是数据集的所有列名及数据类型：

{col_list}

请从中选出最能代表"贷款最终还款结果"的一列，按 Task 1 的格式输出 JSON。"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]
    raw = await _call_llm(messages)
    logger.info("[LLM Step 2 Raw] %s", raw)

    result = await _parse_json_response(raw)
    target = result["target_column"]
    print(f"\n[LLM → Step 2] 识别目标列：'{target}'")
    return target


# ════════════════════════════════════════════════════════════
# Step 4: LLM 分析好坏样本
# ════════════════════════════════════════════════════════════

async def llm_define_good_bad(
    target_column: str,
    unique_values: List[Any],
) -> Dict:
    """
    Step 4: 将唯一值送给LLM，让它按M0/M1+定义输出分类。

    Returns
    -------
    dict with keys: target_column, good_values, bad_values, ignore_values, reasoning
    """
    vals_str = json.dumps(unique_values, ensure_ascii=False)
    user_msg = f"""目标列名：{target_column}
该列的所有唯一值：{vals_str}

请按照 Task 2 的格式，将上述值分类为好样本(M0)、坏样本(M1+)、忽略样本，输出 JSON。"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]
    raw = await _call_llm(messages)
    logger.info("[LLM Step 4 Raw] %s", raw)

    result = await _parse_json_response(raw)

    print(f"\n[LLM → Step 4] 好坏样本定义：")
    print(f"   好样本 M0  : {result['good_values']}")
    print(f"   坏样本 M1+ : {result['bad_values']}")
    print(f"   忽略       : {result.get('ignore_values', [])}")
    print(f"   LLM说明    : {result.get('reasoning','')}")
    return result


# ════════════════════════════════════════════════════════════
# 主 Skill 入口：run_eda_skill
# ════════════════════════════════════════════════════════════

async def run_eda_skill(
    df: pd.DataFrame,
    output_dir: str = "eda_output",
    html_path: str = "eda_dist_report.html",
    cat_threshold: int = 10,
    max_categories: int = 20,
) -> tuple[str, Dict]:
    """
    信贷EDA Skill 主入口。

    Parameters
    ----------
    df             : 已清洗的 DataFrame（来自 data_cleaning.py 输出）
    output_dir     : 图片输出目录
    html_path      : 最终 HTML 报告路径
    cat_threshold  : 数值列唯一值 ≤ 该值时视为类别型
    max_categories : 类别型变量最多显示的类别数（超过则跳过）

    Returns
    -------
    (HTML 文件绝对路径, 结果字典)
    """
    print("=" * 60)
    print("信贷EDA Agent Skill 启动")
    print("=" * 60)

    # ── Step 1: Tool 1 — 获取列名 ─────────────────────────
    print("\n[Step 1] Tool 1: 获取所有列名+类型")
    col_info = get_column_names(df)
    print(f"   共 {len(col_info)} 列")

    # ── Step 2: LLM — 识别目标列 ───────────────────────────
    print("\n[Step 2] LLM: 识别借贷最终状态列...")
    target_col = await llm_identify_target_column(col_info)

    # 验证列存在
    if target_col not in df.columns:
        raise ValueError(f"LLM 返回的列 '{target_col}' 不在数据中，请检查。")

    # ── Step 3: Tool 2 — 获取唯一值 ────────────────────────
    print(f"\n[Step 3] Tool 2: 获取 '{target_col}' 的所有唯一值")
    unique_vals = get_column_unique_values(df, target_col)
    print(f"   唯一值：{unique_vals}")

    # ── Step 4: LLM — 定义好坏样本 ─────────────────────────
    print("\n[Step 4] LLM: 分析好坏样本定义（M0/M1+）...")
    definition = await llm_define_good_bad(target_col, unique_vals)
    good_values = definition["good_values"]
    bad_values  = definition["bad_values"]

    if not good_values or not bad_values:
        raise ValueError("LLM 未能识别出有效的好/坏样本值，请检查数据和System Prompt。")

    # ── Step 5: Tool 3 — 过滤打标签 ────────────────────────
    print("\n[Step 5] Tool 3: 按好坏定义过滤数据、打标签")
    filtered_df, good_mask, bad_mask = filter_by_target_def(
        df, target_col, good_values, bad_values
    )
    filtered_df = filtered_df.reset_index(drop=True)

    total_n   = len(filtered_df)
    good_n    = good_mask.sum()
    bad_n     = bad_mask.sum()
    good_r    = round(good_n / total_n * 100, 2) if total_n else 0
    bad_r     = round(bad_n  / total_n * 100, 2) if total_n else 0

    # ── Step 6: Tool 4/5 — 批量绘图 ────────────────────────
    print("\n[Step 6] 批量绘制变量分布图...")

    # 排除目标列和 ID 类列
    exclude = {target_col, "id", "member_id", "url", "desc", "target"}
    feature_cols = [c for c in filtered_df.columns if c not in exclude]

    numeric_cols     = []
    categorical_cols = []

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(filtered_df[col]):
            if filtered_df[col].nunique() > cat_threshold:
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
        else:
            categorical_cols.append(col)

    print(f"   连续型特征：{len(numeric_cols)} 个")
    print(f"   类别型特征：{len(categorical_cols)} 个")

    continuous_images   = []
    categorical_images  = []
    skipped_features    = []

    for i, col in enumerate(numeric_cols, 1):
        print(f"   [连续 {i:3d}/{len(numeric_cols)}] {col}", end="", flush=True)
        path = plot_numeric_distribution(
            filtered_df, col, good_mask, bad_mask, output_dir
        )
        if path:
            continuous_images.append(path)
            print(" ✓")
        else:
            skipped_features.append(col)
            print(" (跳过)")

    for i, col in enumerate(categorical_cols, 1):
        print(f"   [类别 {i:3d}/{len(categorical_cols)}] {col}", end="", flush=True)
        path = plot_categorical_distribution(
            filtered_df, col, good_mask, bad_mask, output_dir,
            max_categories=max_categories,
        )
        if path:
            categorical_images.append(path)
            print(" ✓")
        else:
            skipped_features.append(col)
            print(" (跳过)")

    print(f"\n   生成连续型图：{len(continuous_images)} 张")
    print(f"   生成类别型图：{len(categorical_images)} 张")
    if skipped_features:
        print(f"   跳过（无有效数据）：{len(skipped_features)} 个 → {skipped_features}")

    # ── Step 7: Tool 6 — 生成 HTML 报告 ────────────────────
    print("\n[Step 7] Tool 6: 生成 HTML 报告（图片内嵌，自包含）...")

    stats = {
        "total_samples":  total_n,
        "good_count":     int(good_n),
        "bad_count":      int(bad_n),
        "good_ratio":     good_r,
        "bad_ratio":      bad_r,
        "feature_count":  len(feature_cols),
        "good_values":    good_values,
        "bad_values":     bad_values,
        "target_column":  target_col,
    }

    html_abs = generate_html_report(
        continuous_images=continuous_images,
        categorical_images=categorical_images,
        stats=stats,
        output_path=html_path,
    )

    # 兼容新拆分报告：如果工具写了两份 HTML（continuous/categorical），
    # 在这里把 categorical 那份也一起推给主控 Agent/frontend。
    cat_html = ""
    try:
        _p = Path(html_abs)
        _cat = _p.with_name(f"{_p.stem}_cat{_p.suffix or '.html'}")
        if _cat.exists():
            cat_html = str(_cat.resolve())
    except Exception:
        cat_html = ""

    print("\n" + "=" * 60)
    print(f"✅ 完成！HTML 报告：{html_abs}")
    if cat_html:
        print(f"   类别型报告：{cat_html}")
    print("=" * 60)

    result_dict = {
        "html_path":      html_abs,
        "html_path_cat":  cat_html,
        "target_column":  target_col,
        "good_values":    good_values,
        "bad_values":     bad_values,
        "n_total":        total_n,
        "n_good":         int(good_n),
        "n_bad":          int(bad_n),
        "good_ratio":     good_r,
        "bad_ratio":      bad_r,
        "n_features":     len(feature_cols),
        "n_numeric_cols":  len(numeric_cols),
        "n_categorical_cols": len(categorical_cols),
        "n_continuous_plots":   len(continuous_images),
        "n_categorical_plots":  len(categorical_images),
        "n_skipped":     len(skipped_features),
        "skipped_features": skipped_features,
    }

    return html_abs, result_dict
