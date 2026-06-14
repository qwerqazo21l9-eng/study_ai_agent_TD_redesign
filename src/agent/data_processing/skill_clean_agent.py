"""
skill_clean_agent.py — 数据清洗 Skill（Phase 1）

负责：
  1. 接收主控 Agent 传来的数据源配置
  2. 验证参数合法性（由 LLM 完成）
  3. 调用 tools_clean 执行完整清洗 pipeline
  4. 返回结构化结果给主控 Agent

LLM 调用方式：使用当前项目的 get_llm()（LangChain ChatOpenAI 兼容接口）
"""

import json
import logging
from typing import Dict, Any

from src.utils.llm import get_llm
from src.agent.data_processing.tools_clean import run_cleaning_with_report

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ══════════════════════════════════════════════════════════════
# Phase 1 System Prompt
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT_CLEAN = """
# Role
你是**数据清洗 Phase 管理者**。你的职责是接收主控 Agent 传递的数据源配置参数，
验证其合法性，然后调用清洗工具执行完整的 Lending Club 数据清洗 pipeline。

# 你的任务

主控 Agent 会给你一个 JSON 配置，你只需做以下事：

1. **验证参数**：检查 filepath 是否存在、参数范围是否合理
2. **返回确认**：如果一切正常，原样返回配置并添加 `"action": "proceed"`
3. **报错**：如果参数不合法，返回 `"action": "error"` 并说明原因

# 输出格式

```json
{
  "action": "proceed",
  "message": "参数验证通过",
  "config": {
    "filepath": "...",
    "output_path": "...",
    "nrows": 200000,
    "missing_threshold": 0.3,
    "cat_threshold": 10,
    "plot": true,
    "html_output": "eda_report.html"
  }
}
```

如果出错：
```json
{
  "action": "error",
  "message": "错误原因",
  "config": null
}
```

# 约束
- 只输出 JSON，不要有额外文字
- 不要修改主控 Agent 传来的配置参数
- filepath 不存在时直接报错
- missing_threshold 应在 0.1~0.9 之间
- nrows 应为正整数
"""


# ══════════════════════════════════════════════════════════════
# LLM 调用封装（异步）
# ══════════════════════════════════════════════════════════════

async def _call_llm(messages: list) -> str:
    """
    使用当前项目的 get_llm() 调用 LLM。
    messages: list of LangChain Message objects 或 dicts
    """
    llm = get_llm()
    response = await llm.ainvoke(messages)
    return response.content.strip()


def _parse_json(text: str) -> Dict:
    for fence in ("```json", "```"):
        text = text.replace(fence, "")
    text = text.strip().strip("`").strip()
    return json.loads(text)


# ══════════════════════════════════════════════════════════════
# Phase 1 Skill 入口
# ══════════════════════════════════════════════════════════════

async def run_cleaning_skill(
    filepath: str,
    output_path: str = "lc_clean.csv",
    nrows: int = 200000,
    missing_threshold: float = 0.3,
    cat_threshold: int = 10,
    plot: bool = True,
    html_output: str = "eda_report.html",
) -> Dict:
    """
    Phase 1 Skill 主入口：数据清洗 + EDA 报告

    Parameters
    ----------
    filepath         : 原始 CSV 绝对路径
    output_path      : 清洗后保存路径
    nrows            : 仅读取前 N 行
    missing_threshold: 高缺失率删除阈值
    cat_threshold    : 数值列唯一值 <= 该值视为类别型
    plot             : 是否生成 EDA 图
    html_output      : EDA HTML 报告文件名

    Returns
    -------
    dict with keys:
        status, cleaned_csv, n_raw_rows, n_clean_rows, n_clean_cols,
        n_continuous, n_categorical, n_missing_cols,
        eda_report_html, message
    """
    config = {
        "filepath": filepath,
        "output_path": output_path,
        "nrows": nrows,
        "missing_threshold": missing_threshold,
        "cat_threshold": cat_threshold,
        "plot": plot,
        "html_output": html_output,
    }

    print("\n" + "=" * 60)
    print("[Phase 1 Skill] 数据清洗 Skill 启动")
    print("=" * 60)

    # ── Step 1: LLM 验证参数 ──────────────────────────
    print("\n[Phase 1 Skill Step 1] LLM 验证参数...")
    user_msg = f"""请验证以下清洗配置参数：

{json.dumps(config, ensure_ascii=False, indent=2)}
"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_CLEAN},
        {"role": "user", "content": user_msg},
    ]
    raw = await _call_llm(messages)
    logger.info("[Phase 1 LLM Raw] %s", raw)

    validation = _parse_json(raw)
    print(f"   验证结果：{validation['action']} — {validation.get('message', '')}")

    if validation["action"] == "error":
        return {
            "status": "error",
            "message": validation["message"],
            "cleaned_csv": None,
        }

    # ── Step 2: 执行清洗 Pipeline ─────────────────────
    print("\n[Phase 1 Skill Step 2] 执行数据清洗 Pipeline...")
    result = run_cleaning_with_report(
        filepath=config["filepath"],
        output_path=config["output_path"],
        nrows=config["nrows"],
        missing_threshold=config["missing_threshold"],
        cat_threshold=config["cat_threshold"],
        plot=config["plot"],
        html_output=config["html_output"],
    )

    result["status"] = "success"
    result["message"] = "Phase 1 数据清洗完成"

    # ── 打印摘要 ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("[Phase 1 Skill] 清洗完成，数据摘要：")
    print(f"   行数: {result['n_raw_rows']:,} → {result['n_clean_rows']:,}")
    print(f"   列数: {result['n_raw_cols']} → {result['n_clean_cols']}（删 {result['n_dropped_cols']}）")
    print(f"   连续型: {result['n_continuous']}, 类别型: {result['n_categorical']}")
    print(f"   含缺失列: {result['n_missing_cols']}")
    if result.get("eda_report_html"):
        print(f"   EDA报告: {result['eda_report_html']}")
    print("=" * 60)

    return result
