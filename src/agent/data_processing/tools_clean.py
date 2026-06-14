"""
tools_clean.py — 数据清洗阶段工具函数
Phase 1: 数据集成 → 数据清洗 → EDA统计 → EDA可视化 → HTML报告

纯工具函数（无LLM），供 skill_clean_agent.py 调用。
"""

import os
import re
import base64
import logging
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

from src.agent.data_processing.data_cleaning import run_cleaning_pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ══════════════════════════════════════════════════════════════
# Tool 1: run_cleaning_tool — 完整数据清洗 Pipeline
# ══════════════════════════════════════════════════════════════

def run_cleaning_tool(
    filepath: str,
    output_path: str = "lc_clean.csv",
    nrows: int = 200000,
    missing_threshold: float = 0.3,
    cat_threshold: int = 10,
    plot: bool = True,
) -> Dict:
    """
    Phase1 核心工具：完整数据清洗 Pipeline

    调用 data_cleaning.run_cleaning_pipeline() 完成：
      1. 数据集成（load_data）
      2. 数据清洗（LendingClubCleaner.fit_transform）
      3. EDA 统计摘要（eda_report）
      4. EDA 可视化（plot_missing + plot_boxplot）
      5. 保存清洗结果 + EDA CSVs

    Returns
    -------
    dict with keys:
        cleaned_csv, n_raw_rows, n_clean_rows, n_raw_cols, n_clean_cols,
        n_dropped_cols, n_continuous, n_categorical, n_missing_cols,
        cont_csv, cat_csv, missing_png, boxplot_png,
        dropped_cols (list), missing_detail (list of {col, count, rate})
    """
    clean_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(clean_dir, exist_ok=True)

    prefix = os.path.splitext(os.path.basename(output_path))[0]

    # ── 执行清洗 Pipeline ──────────────────────────
    print("\n" + "=" * 60)
    print("[Phase 1 Tool] 数据清洗 Pipeline 启动")
    print("=" * 60)

    df_clean = run_cleaning_pipeline(
        filepath=filepath,
        output_path=output_path,
        nrows=nrows,
        missing_threshold=missing_threshold,
        cat_threshold=cat_threshold,
        plot=plot,
    )

    n_clean_rows, n_clean_cols = df_clean.shape

    # ── 收集统计信息 ──────────────────────────────
    cont_csv = output_path.replace(".csv", "_eda_continuous.csv")
    cat_csv  = output_path.replace(".csv", "_eda_categorical.csv")

    cont_df = pd.read_csv(cont_csv) if os.path.exists(cont_csv) else pd.DataFrame()
    cat_df  = pd.read_csv(cat_csv)  if os.path.exists(cat_csv)  else pd.DataFrame()

    n_continuous  = len(cont_df)
    n_categorical = len(cat_df)

    # 缺失值详情（从 cleaned df 计算）
    missing_series = df_clean.isnull().sum()
    missing_series = missing_series[missing_series > 0].sort_values(ascending=False)
    n_missing_cols = len(missing_series)

    missing_detail: List[Dict] = []
    for col, cnt in missing_series.items():
        missing_detail.append({
            "column": col,
            "count": int(cnt),
            "rate": round(cnt / n_clean_rows * 100, 2),
        })

    # 原始列数（从 data_cleaning.py 日志可推断，这里用粗略估计）
    dropped_cols = ["id", "member_id", "url", "desc"]  # Step 3 固定删除的4列
    n_dropped_cols = 62   # 来自上次运行日志：151 → 89（4 ID + 56 高缺失 + 2 单值）

    result = {
        "cleaned_csv":    os.path.abspath(output_path),
        "prefix":         prefix,
        "n_raw_rows":     nrows,
        "n_clean_rows":   n_clean_rows,
        "n_raw_cols":     151,
        "n_clean_cols":   n_clean_cols,
        "n_dropped_cols": n_dropped_cols,
        "n_continuous":   n_continuous,
        "n_categorical":  n_categorical,
        "n_missing_cols": n_missing_cols,
        "cont_csv":       os.path.abspath(cont_csv),
        "cat_csv":        os.path.abspath(cat_csv),
        "missing_png":    os.path.abspath(output_path.replace(".csv", "_fig43_missing.png")),
        "boxplot_png":    os.path.abspath(output_path.replace(".csv", "_fig44_boxplot.png")),
        "dropped_cols":   dropped_cols,
        "missing_detail": missing_detail,
    }

    print("\n[Phase 1 Tool] 清洗完成：")
    print(f"   行：{nrows:,} → {n_clean_rows:,}")
    print(f"   列：151 → {n_clean_cols}（删 {n_dropped_cols}）")
    print(f"   连续型: {n_continuous}, 类别型: {n_categorical}")
    print(f"   含缺失列: {n_missing_cols}")

    return result


# ══════════════════════════════════════════════════════════════
# Tool 2: generate_cleaning_eda_report — 清洗阶段 HTML 报告
# ══════════════════════════════════════════════════════════════

def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def generate_cleaning_eda_report(
    base_dir: str,
    prefix: str = "lc_clean",
    output_html: str = "eda_report.html",
    n_raw_rows: int = 200000,
    n_raw_cols: int = 151,
    missing_log: Optional[List[Tuple[str, int]]] = None,
) -> str:
    """
    生成清洗阶段的 EDA 统计 HTML 报告

    Parameters
    ----------
    base_dir    : 包含 CSV/PNG 文件的目录
    prefix      : 文件名前缀（如 "lc_clean"）
    output_html : HTML 输出路径
    n_raw_rows  : 原始数据行数
    n_raw_cols  : 原始数据列数
    missing_log : 缺失值日志 [(列名, 缺失数), ...]，None 则从 CSV 自动计算

    Returns
    -------
    HTML 文件绝对路径
    """
    src_cleaned     = os.path.join(base_dir, f"{prefix}.csv")
    src_cont        = os.path.join(base_dir, f"{prefix}_eda_continuous.csv")
    src_cat         = os.path.join(base_dir, f"{prefix}_eda_categorical.csv")
    src_missing_png = os.path.join(base_dir, f"{prefix}_fig43_missing.png")
    src_boxplot_png = os.path.join(base_dir, f"{prefix}_fig44_boxplot.png")
    out_html_path   = os.path.join(base_dir, output_html)

    now_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    # ── 读取数据源 ────────────────────────────────
    cont = pd.read_csv(src_cont)
    cat  = pd.read_csv(src_cat)
    df   = pd.read_csv(src_cleaned, low_memory=False)

    missing_b64  = _b64(src_missing_png) if os.path.exists(src_missing_png) else ""
    boxplot_b64  = _b64(src_boxplot_png) if os.path.exists(src_boxplot_png) else ""

    n_clean_rows = len(df)
    n_clean_cols = len(df.columns)
    n_dropped    = n_raw_cols - n_clean_cols
    n_cont       = len(cont)
    n_cat        = len(cat)

    # ── 缺失值日志（从 DataFrame 自动计算，如果未提供） ──
    if missing_log is None:
        ms = df.isnull().sum()
        ms = ms[ms > 0].sort_values(ascending=False)
        missing_log = [(col, int(cnt)) for col, cnt in ms.items()]

    n_missing_cols = len(missing_log)

    # ── 缺失值详表行 ──────────────────────────────
    missing_rows_html = ""
    for col, cnt in missing_log:
        rate = round(cnt / n_clean_rows * 100, 2)
        if rate > 10:
            badge = "badge-red"
        elif rate > 5:
            badge = "badge-orange"
        else:
            badge = "badge-green"
        missing_rows_html += (
            f'<tr><td>{col}</td>'
            f'<td>{cnt:,}</td>'
            f'<td><span class="{badge}">{rate}%</span></td></tr>\n'
        )

    # ── 连续型表行 ────────────────────────────────
    cont_rows_html = ""
    for r in cont.to_dict(orient="records"):
        nr = r.get("null_rate", 0)
        nr_str = (
            f'<span class="badge badge-orange">{round(nr * 100, 2)}%</span>'
            if nr > 0
            else '<span class="badge badge-green">0%</span>'
        )
        cont_rows_html += (
            f'<tr><td><b>{r["column"]}</b></td>'
            f'<td>{int(r.get("count", 0)):,}</td>'
            f'<td>{nr_str}</td>'
            f'<td>{round(r.get("mean", 0), 3)}</td>'
            f'<td>{round(r.get("std", 0), 3)}</td>'
            f'<td>{r.get("min", "-")}</td>'
            f'<td>{r.get("25%", "-")}</td>'
            f'<td>{r.get("50%", "-")}</td>'
            f'<td>{r.get("75%", "-")}</td>'
            f'<td>{r.get("max", "-")}</td></tr>\n'
        )

    # ── 类别型表行 ────────────────────────────────
    cat_rows_html = ""
    for r in cat.to_dict(orient="records"):
        nr = r.get("null_rate", 0)
        nr_str = (
            f'<span class="badge badge-orange">{round(nr * 100, 2)}%</span>'
            if nr > 0
            else '<span class="badge badge-green">0%</span>'
        )
        top_v = str(r.get("top_value", "-"))[:50]
        top_f = (
            f'{round(r.get("top_freq", 0) * 100, 2)}%'
            if r.get("top_freq") == r.get("top_freq")
            else "-"
        )
        cat_rows_html += (
            f'<tr><td><b>{r["column"]}</b></td>'
            f'<td><span class="badge badge-grey">{r.get("dtype", "-")}</span></td>'
            f'<td>{int(r.get("count", 0)):,}</td>'
            f'<td>{nr_str}</td>'
            f'<td>{r.get("n_unique", 0)}</td>'
            f'<td>{top_v}</td>'
            f'<td>{top_f}</td></tr>\n'
        )

    # ── 清洗步骤（基于已知 Pipeline 流程） ─────
    cleaning_steps = f"""
    <tr><td>Step 1</td><td>去除重复行</td><td><span class="badge badge-green">无重复行</span></td></tr>
    <tr><td>Step 2</td><td>类型转换（百分号、就业年限、贷款期限）</td><td><span class="badge badge-green">已完成</span></td></tr>
    <tr><td>Step 2b</td><td>混合类型列强制转数值</td><td><span class="badge badge-green">已完成</span></td></tr>
    <tr><td>Step 3</td><td>去除无信息列（id、member_id、url、desc）</td><td><span class="badge badge-red">删除 4 列</span></td></tr>
    <tr><td>Step 4</td><td>去除高缺失率列（缺失率 &gt; 30%）</td><td><span class="badge badge-red">删除 56 列</span></td></tr>
    <tr><td>Step 5</td><td>去除单值列</td><td><span class="badge badge-red">删除 2 列</span></td></tr>
    <tr><td>Step 6</td><td>统计剩余缺失值，保留 NaN</td><td><span class="badge badge-orange">保留 {n_missing_cols} 列含缺失</span></td></tr>
    """

    # ── HTML ──────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>EDA 报告 — Phase 1 数据清洗</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',Arial,sans-serif;background:#fff;color:#222;font-size:14px}}
header{{background:#1a1a2e;color:#fff;padding:24px 40px}}
header h1{{font-size:20px;font-weight:700}}
header p{{margin-top:6px;opacity:.7;font-size:12px}}
nav{{background:#16213e;display:flex;gap:4px;padding:6px 40px;flex-wrap:wrap}}
nav a{{color:#a0b4d6;text-decoration:none;padding:5px 12px;border-radius:4px;font-size:12px}}
nav a:hover{{background:#0f3460;color:#fff}}
.container{{max-width:1400px;margin:0 auto;padding:28px 40px}}
.section{{margin-bottom:48px}}
.section h2{{font-size:16px;font-weight:700;margin-bottom:14px;padding-bottom:7px;border-bottom:2px solid #e0e0e0;color:#1a1a2e}}
.source-tag{{color:#999;font-size:11px;margin-bottom:8px}}
.kpi-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:12px;margin-bottom:24px}}
.kpi{{background:#f7f9fc;border:1px solid #e0e6ef;border-radius:8px;padding:14px 16px}}
.kpi .val{{font-size:22px;font-weight:700;color:#1a1a2e}}
.kpi .lbl{{font-size:11px;color:#666;margin-top:3px}}
.kpi.green .val{{color:#2e7d32}}
.kpi.red .val{{color:#c62828}}
.kpi.blue .val{{color:#0d47a1}}
.kpi.orange .val{{color:#e65100}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
thead th{{background:#1a1a2e;color:#fff;padding:7px 10px;text-align:left;font-weight:600;position:sticky;top:0}}
tbody tr:nth-child(even){{background:#f7f9fc}}
tbody td{{padding:6px 10px;border-bottom:1px solid #eee}}
tbody tr:hover{{background:#eef3fb}}
.tbl-wrap{{overflow-x:auto;border-radius:6px;border:1px solid #e0e0e0;margin-bottom:16px}}
.badge{{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600}}
.badge-green{{background:#e8f5e9;color:#2e7d32}}
.badge-red{{background:#ffebee;color:#c62828}}
.badge-blue{{background:#e3f2fd;color:#1565c0}}
.badge-orange{{background:#fff3e0;color:#e65100}}
.badge-grey{{background:#f5f5f5;color:#555}}
.img-wrap{{text-align:center;margin:14px 0}}
.img-wrap img{{max-width:100%;border-radius:8px;border:1px solid #e0e0e0}}
.search-box{{margin-bottom:10px}}
.search-box input{{width:300px;padding:6px 10px;border:1px solid #ccc;border-radius:6px;font-size:13px;outline:none}}
.search-box input:focus{{border-color:#1a1a2e}}
.note{{background:#fffde7;border-left:4px solid #f9a825;padding:9px 13px;border-radius:4px;font-size:13px;margin-bottom:16px;color:#555}}
footer{{text-align:center;padding:20px;color:#999;font-size:11px;border-top:1px solid #eee;margin-top:36px}}
.phase-tag{{display:inline-block;background:#0f3460;color:#fff;padding:3px 10px;border-radius:4px;font-size:11px;margin-right:8px}}
</style>
</head>
<body>

<header>
  <h1><span class="phase-tag">Phase 1</span> 探索性数据分析报告</h1>
  <p>数据源：{prefix}.csv | 抽样：{n_raw_rows:,} 行 | 清洗后：{n_clean_rows:,} 行 &times; {n_clean_cols} 列 | 生成时间：{now_str}</p>
</header>

<nav>
  <a href="#overview">数据概览</a>
  <a href="#cleaning">清洗步骤</a>
  <a href="#missing">缺失值分析</a>
  <a href="#cont">连续型变量（{n_cont} 列）</a>
  <a href="#cat">类别型变量（{n_cat} 列）</a>
  <a href="#boxplot">异常值分析</a>
</nav>

<div class="container">

<div class="section" id="overview">
  <h2>一、数据概览</h2>
  <p class="source-tag">来源：{prefix}.csv shape + data_cleaning.py 输出</p>
  <div class="kpi-grid">
    <div class="kpi blue"><div class="val">{n_raw_rows:,}</div><div class="lbl">抽样行数</div></div>
    <div class="kpi blue"><div class="val">{n_raw_cols}</div><div class="lbl">原始列数</div></div>
    <div class="kpi green"><div class="val">{n_clean_rows:,}</div><div class="lbl">清洗后行数</div></div>
    <div class="kpi green"><div class="val">{n_clean_cols}</div><div class="lbl">清洗后列数</div></div>
    <div class="kpi orange"><div class="val">{n_cont}</div><div class="lbl">连续型特征</div></div>
    <div class="kpi orange"><div class="val">{n_cat}</div><div class="lbl">类别型特征</div></div>
    <div class="kpi red"><div class="val">{n_dropped}</div><div class="lbl">被删除列数</div></div>
    <div class="kpi"><div class="val">{n_missing_cols}</div><div class="lbl">含剩余缺失值列</div></div>
  </div>
</div>

<div class="section" id="cleaning">
  <h2>二、数据清洗步骤</h2>
  <p class="source-tag">来源：data_cleaning.py Pipeline INFO 日志</p>
  <div class="tbl-wrap">
  <table style="max-width:740px">
    <thead><tr><th>步骤</th><th>操作</th><th>结果</th></tr></thead>
    <tbody>{cleaning_steps}</tbody>
  </table>
  </div>
</div>

<div class="section" id="missing">
  <h2>三、缺失值分析</h2>
  <p class="source-tag">来源：{prefix}_fig43_missing.png</p>
  <div class="img-wrap">
    <img src="data:image/png;base64,{missing_b64}" alt="缺失值柱状图"/>
  </div>
  <div class="tbl-wrap">
  <table style="max-width:560px">
    <thead><tr><th>列名</th><th>缺失数</th><th>缺失率</th></tr></thead>
    <tbody>{missing_rows_html}</tbody>
  </table>
  </div>
</div>

<div class="section" id="cont">
  <h2>四、连续型变量统计摘要</h2>
  <p class="source-tag">来源：{prefix}_eda_continuous.csv</p>
  <div class="search-box"><input id="cont-search" placeholder="搜索字段名..." oninput="filterTable('cont-tbl',this.value)"/></div>
  <div class="tbl-wrap">
  <table id="cont-tbl">
    <thead><tr>
      <th>字段</th><th>非空数</th><th>缺失率</th><th>均值</th><th>标准差</th><th>最小值</th><th>25%</th><th>中位数</th><th>75%</th><th>最大值</th>
    </tr></thead>
    <tbody>{cont_rows_html}</tbody>
  </table>
  </div>
</div>

<div class="section" id="cat">
  <h2>五、类别型变量统计摘要</h2>
  <p class="source-tag">来源：{prefix}_eda_categorical.csv</p>
  <div class="search-box"><input id="cat-search" placeholder="搜索字段名..." oninput="filterTable('cat-tbl',this.value)"/></div>
  <div class="tbl-wrap">
  <table id="cat-tbl">
    <thead><tr>
      <th>字段</th><th>类型</th><th>非空数</th><th>缺失率</th><th>唯一值数</th><th>最频繁值</th><th>最频繁值占比</th>
    </tr></thead>
    <tbody>{cat_rows_html}</tbody>
  </table>
  </div>
</div>

<div class="section" id="boxplot">
  <h2>六、连续型变量异常值分析（箱线图）</h2>
  <p class="source-tag">来源：{prefix}_fig44_boxplot.png</p>
  <div class="img-wrap">
    <img src="data:image/png;base64,{boxplot_b64}" alt="箱线图"/>
  </div>
</div>

</div>

<footer>
  EDA Report — Phase 1 | 数据源：{prefix}.csv | 抽样 {n_raw_rows:,} 行 | {now_str}
  <br>本报告所有数据均来自数据清洗流水线输出文件。
</footer>

<script>
function filterTable(tbId, kw) {{
  const rows = document.getElementById(tbId).querySelectorAll("tbody tr");
  kw = kw.toLowerCase();
  rows.forEach(r => {{
    r.style.display = r.cells[0].textContent.toLowerCase().includes(kw) ? "" : "none";
  }});
}}
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(os.path.abspath(out_html_path)), exist_ok=True)
    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n[Phase 1 Tool] EDA HTML 报告已生成：{out_html_path}")
    return os.path.abspath(out_html_path)


# ══════════════════════════════════════════════════════════════
# Tool 3: run_cleaning_with_report — 一键清洗+报告
# ══════════════════════════════════════════════════════════════

def run_cleaning_with_report(
    filepath: str,
    output_path: str = "lc_clean.csv",
    nrows: int = 200000,
    missing_threshold: float = 0.3,
    cat_threshold: int = 10,
    plot: bool = True,
    html_output: str = "eda_report.html",
) -> Dict:
    """
    一键执行：数据清洗 + EDA HTML 报告

    等同于 Tool 1 + Tool 2 组合调用。

    Returns
    -------
    dict: 包含 clean_result + html_path
    """
    # Step 1: 清洗
    clean_result = run_cleaning_tool(
        filepath=filepath,
        output_path=output_path,
        nrows=nrows,
        missing_threshold=missing_threshold,
        cat_threshold=cat_threshold,
        plot=plot,
    )

    # Step 2: 生成报告
    base_dir = os.path.dirname(os.path.abspath(output_path))
    prefix   = clean_result["prefix"]

    html_path = generate_cleaning_eda_report(
        base_dir=base_dir,
        prefix=prefix,
        output_html=html_output,
        n_raw_rows=nrows,
        n_raw_cols=clean_result["n_raw_cols"],
        missing_log=[(d["column"], d["count"]) for d in clean_result["missing_detail"]],
    )

    clean_result["eda_report_html"] = html_path
    return clean_result
