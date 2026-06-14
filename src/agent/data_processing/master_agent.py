"""
master_agent.py — 主控 Agent 入口脚本

编排两阶段流水线：
  Phase 1: 数据清洗 → skill_clean_agent.run_cleaning_skill
  Phase 2: 好坏样本EDA → skill_eda_agent.run_eda_skill
  Phase 3: LLM 汇总两个阶段的返回结果，生成最终摘要

外部调用入口：run_data_processing_pipeline()
"""

import json
import logging
import os
from typing import Dict, Any

import pandas as pd
from src.utils.llm import get_llm
from src.agent.data_processing.skill_clean_agent import run_cleaning_skill
from src.agent.data_processing.skill_eda_agent import run_eda_skill

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ═════════════════════════════════════════════════════════════
# 主控 Agent System Prompt
# ═════════════════════════════════════════════════════════════

SYSTEM_PROMPT_MASTER = """
# Role
你是 **信贷数据分析主管 Agent**，负责编排两阶段数据流水线并输出最终自然语言总结。

# Pipeline 结构

```
Phase 1 — 数据清洗：
  工具: skill_clean_agent.run_cleaning_skill
  输入: 原始 CSV 路径 + 清洗参数
  输出: cleaned_df（含 loan_status 列）+ EDA 统计摘要 + 缺失值报告

Phase 2 — 好坏样本EDA：
  工具: skill_eda_agent.run_eda_skill
  输入: Phase 1 清洗后的 DataFrame
  输出: M0/M2+ 好坏样本定义 + 所有特征的分布可视化 HTML 报告
```

# To-Do List

- [x] Phase 1 数据清洗 → 已执行，结构化结果已就绪
- [x] Phase 2 好坏样本EDA → 已执行，分布报告已生成
- [ ] **最终总结** → 结合两个 Phase 的关键指标，输出一段自然语言数据分析总结

# 当前任务：输出最终总结

你现在处于 To-Do List 的最后一步。请根据下面提供的两阶段结构化数据，
输出一段**自然语言的数据分析总结**。

## 总结应包含以下要点：

1. **数据规模概述**：原始数据与清洗后的行数/列数变化
2. **好坏样本分布**：好样本(M0)和坏样本(M1+)的数量和占比
3. **特征概况**：连续型/类别型特征数量，绘图覆盖情况
4. **交付物清单**：生成了哪些文件（CSV、HTML 报告等）

## 输出要求

- **直接输出自然语言段落**，不要输出 JSON，不要用代码块包裹
- 简洁专业，控制在 **150-250 字**以内
- 使用中文
- 用分段或要点形式呈现均可，但要保证是一段连贯的总结

## 约束
- 只输出总结文本，不要有其他内容
- 如果任一 Phase 失败，summary 应描述原因和当前状态
"""


# ═════════════════════════════════════════════════════════════
# LLM 调用（异步）
# ═════════════════════════════════════════════════════════════

async def _generate_natural_summary(phase1: dict, phase2: dict) -> str:
    """调用 LLM 生成自然语言数据分析总结段落。"""
    llm = get_llm()

    # 构建结构化的用户消息，让 LLM 一目了然
    user_msg = f"""以下是两阶段数据处理的完整结果，请按 To-Do List 最后一步输出总结。

## Phase 1 — 数据清洗

| 指标 | 值 |
|------|-----|
| 原始行数 | {phase1.get('n_raw_rows', '?'):,} |
| 清洗后行数 | {phase1.get('n_clean_rows', '?'):,} |
| 原始列数 | {phase1['n_raw_cols']} |
| 清洗后列数 | {phase1['n_clean_cols']}（删除 {phase1.get('n_dropped_cols', 0)} 列） |
| 连续型特征 | {phase1['n_continuous']} |
| 类别型特征 | {phase1['n_categorical']} |
| 含缺失值列 | {phase1.get('n_missing_cols', 0)} |
| 清洗后 CSV | {phase1.get('cleaned_csv', '-')} |
| 清洗报告 HTML | {phase1.get('eda_report_html', '-')} |

## Phase 2 — 好坏样本 EDA

| 指标 | 值 |
|------|-----|
| 目标列 | {phase2.get('target_column', '?')} |
| 好样本(M0) | {phase2.get('good_values', [])} |
| 坏样本(M1+) | {phase2.get('bad_values', [])} |
| 有效样本总数 | {phase2.get('n_total', 0):,} |
| 好样本数 | {phase2.get('n_good', 0):,}（{phase2.get('good_ratio', 0)}%） |
| 坏样本数 | {phase2.get('n_bad', 0):,}（{phase2.get('bad_ratio', 0)}%） |
| 特征总数 | {phase2.get('n_features', 0)}（连续 {phase2.get('n_numeric_cols', 0)} + 类别 {phase2.get('n_categorical_cols', 0)}） |
| 生成分布图 | {phase2.get('n_continuous_plots', 0)} 连续 + {phase2.get('n_categorical_plots', 0)} 类别（跳过 {phase2.get('n_skipped', 0)}） |
| 分布报告 HTML | {phase2.get('dist_report_html', '-')} |

请输出自然语言总结。"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_MASTER},
        {"role": "user",   "content": user_msg},
    ]

    response = await llm.ainvoke(messages)
    summary_text = response.content.strip()

    # 去除可能的 markdown 代码块包裹（兼容 LLM 偶尔输出代码块）
    for fence in ("```json", "```markdown", "```text", "```"):
        summary_text = summary_text.replace(fence, "")
    summary_text = summary_text.strip().strip("`").strip()

    logger.info("[MASTER] LLM 总结生成完成（%d 字）", len(summary_text))
    return summary_text


# ═════════════════════════════════════════════════════════════
# 主入口：run_data_processing_pipeline
# ═════════════════════════════════════════════════════════════

async def run_data_processing_pipeline(
    filepath: str,
    output_dir: str = None,
    nrows: int = 200000,
    missing_threshold: float = 0.3,
    cat_threshold: int = 10,
    max_categories: int = 20,
    plot: bool = True,
) -> Dict:
    """
    两阶段数据自动化处理流水线主入口。

    Parameters
    ----------
    filepath          : 原始 CSV 文件路径
    output_dir       : 输出目录（清洗后 CSV、报告、图片均保存在此）
                        为 None 时使用 config.data_processing.output_dir
    nrows            : 仅读取前 N 行（调试用，默认 200000）
    missing_threshold : 高缺失率列删除阈值（默认 0.3）
    cat_threshold    : 数值列唯一值 <= 该值时视为类别型（默认 10）
    max_categories   : 类别型变量最多显示的类别数（默认 20）
    plot             : 是否生成 EDA 图（默认 True）

    Returns
    -------
    dict: 包含 status, summary, deliverables 等键的完整结果
    """
    # 解析输出目录
    if output_dir is None:
        from src.utils.config_loader import config
        output_dir = config.data_processing.get("output_dir", "./output/data_processing")

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    clean_csv_path  = os.path.join(output_dir, "lc_clean.csv")
    clean_html_path = os.path.join(output_dir, "eda_report.html")
    eda_html_path   = os.path.join(output_dir, "eda_dist_report.html")
    eda_img_dir     = os.path.join(output_dir, "eda_dist_images")

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + "   🔷 信贷数据分析主控 Agent — 两阶段流水线".ljust(36) + "║")
    print("╚" + "═" * 58 + "╝")

    # ══════════════════════════════════════
    # Phase 1: 数据清洗
    # ══════════════════════════════════════

    print("\n\n" + "━" * 60)
    print(">>> Phase 1: 数据清洗")
    print("━" * 60)

    phase1_result = await run_cleaning_skill(
        filepath=filepath,
        output_path=clean_csv_path,
        nrows=nrows,
        missing_threshold=missing_threshold,
        cat_threshold=cat_threshold,
        plot=plot,
        html_output=os.path.basename(clean_html_path),
    )

    if phase1_result["status"] != "success":
        print(f"\n[MASTER] ❌ Phase 1 失败：{phase1_result['message']}")
        return {
            "status": "error",
            "stage": "phase1",
            "message": phase1_result["message"],
            "summary": f"Phase 1 数据清洗失败：{phase1_result['message']}",
        }

    # 提取 Phase 1 摘要
    p1_summary = {
        "status":         "success",
        "n_clean_rows":   phase1_result["n_clean_rows"],
        "n_clean_cols":   phase1_result["n_clean_cols"],
        "n_raw_rows":     phase1_result["n_raw_rows"],
        "n_raw_cols":     phase1_result["n_raw_cols"],
        "n_dropped_cols": phase1_result["n_dropped_cols"],
        "n_continuous":   phase1_result["n_continuous"],
        "n_categorical":  phase1_result["n_categorical"],
        "n_missing_cols": phase1_result["n_missing_cols"],
        "cleaned_csv":    phase1_result["cleaned_csv"],
        "eda_report_html": phase1_result.get("eda_report_html", ""),
    }

    # ══════════════════════════════════════
    # Phase 2: 好坏样本 EDA
    # ══════════════════════════════════════

    print("\n\n" + "━" * 60)
    print(">>> Phase 2: 好坏样本 EDA")
    print("━" * 60)

    print(f"[MASTER] 加载 Phase 1 输出的清洗数据：{phase1_result['cleaned_csv']}")
    df_clean = pd.read_csv(phase1_result["cleaned_csv"], low_memory=False)
    print(f"[MASTER] 数据形状：{df_clean.shape}")

    html_path, eda_result = await run_eda_skill(
        df=df_clean,
        output_dir=eda_img_dir,
        html_path=eda_html_path,
        cat_threshold=cat_threshold,
        max_categories=max_categories,
    )

    # 基于 eda_result 构建 p2_summary
    p2_summary = {"status": "success", **eda_result}
    p2_summary["dist_report_html"] = eda_result.get("html_path", "")

    # ══════════════════════════════════════
    # Phase 3: LLM 自然语言总结
    # ══════════════════════════════════════

    print("\n\n" + "━" * 60)
    print(">>> Phase 3: LLM 生成自然语言数据分析总结")
    print("━" * 60)

    try:
        summary_text = await _generate_natural_summary(p1_summary, p2_summary)
    except Exception as e:
        logger.warning("[MASTER] LLM 总结生成失败，使用默认摘要：%s", e)
        summary_text = (
            f"数据清洗完成：原始 {p1_summary['n_raw_rows']:,} 行 {p1_summary['n_raw_cols']} 列 "
            f"→ 清洗后 {p1_summary['n_clean_rows']:,} 行 {p1_summary['n_clean_cols']} 列"
            f"（删除 {p1_summary['n_dropped_cols']} 列）。"
            f"好坏样本分析：{p2_summary['n_total']:,} 个有效样本中，"
            f"好样本(M0) {p2_summary['n_good']:,}（{p2_summary['good_ratio']}%），"
            f"坏样本(M1+) {p2_summary['n_bad']:,}（{p2_summary['bad_ratio']}%）。"
            f"已生成 {p2_summary.get('n_continuous_plots', 0)} 张连续特征分布图 + "
            f"{p2_summary.get('n_categorical_plots', 0)} 张类别特征分布图。"
        )

    # 保存总结到 summary.md
    summary_md_path = os.path.join(output_dir, "summary.md")
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(f"# 数据分析总结\n\n{summary_text}\n")

    # 构建交付物列表
    deliverables = {
        "cleaned_csv":      p1_summary["cleaned_csv"],
        "eda_report_html":  p1_summary.get("eda_report_html", ""),
        "dist_report_html": p2_summary.get("dist_report_html", ""),
        "summary_md":       summary_md_path,
    }

    # ══════════════════════════════════════
    # 最终输出
    # ══════════════════════════════════════

    print("\n\n" + "╔" + "═" * 58 + "╗")
    print("║" + "   ✅ 主控 Agent 流水线执行完毕".ljust(37) + "║")
    print("╚" + "═" * 58 + "╝")

    print(f"""
  ┌─ Phase 1 数据清洗 ──────────────────────────────
  │ 行数: {p1_summary['n_raw_rows']:,} → {p1_summary['n_clean_rows']:,}
  │ 列数: {p1_summary['n_raw_cols']} → {p1_summary['n_clean_cols']}（删 {p1_summary['n_dropped_cols']}）
  │ 连续型: {p1_summary['n_continuous']}  类别型: {p1_summary['n_categorical']}  含缺失: {p1_summary['n_missing_cols']}
  │ 输出: {p1_summary['cleaned_csv']}
  │ EDA报告: {p1_summary.get('eda_report_html', '-')}
  │
  ├─ Phase 2 好坏样本EDA ──────────────────────────
  │ 目标列: {p2_summary['target_column']}
  │ 好样本 M0:  {p2_summary['good_values']} → {p2_summary['n_good']:,} ({p2_summary['good_ratio']}%)
  │ 坏样本 M1+: {p2_summary['bad_values']} → {p2_summary['n_bad']:,} ({p2_summary['bad_ratio']}%)
  │ 特征: {p2_summary.get('n_numeric_cols',0)}连续 + {p2_summary.get('n_categorical_cols',0)}类别 = 共{p2_summary['n_features']}个（已绘图 {p2_summary['n_continuous_plots']}+{p2_summary['n_categorical_plots']}={p2_summary['n_continuous_plots']+p2_summary['n_categorical_plots']}张，跳过{p2_summary.get('n_skipped',0)}个）
  │ 输出: {p2_summary.get('dist_report_html', '-')}
  │
  └─ LLM 总结 ────────────────────────────────────
{summary_text}
""")

    print(f"\n  📋 交付物：")
    for k, v in deliverables.items():
        print(f"     - {k}: {v}")

    return {
        "status": "success",
        "summary": summary_text,
        "deliverables": deliverables,
        "phase1": p1_summary,
        "phase2": p2_summary,
    }
