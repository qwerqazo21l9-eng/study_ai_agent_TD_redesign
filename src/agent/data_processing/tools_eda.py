"""
tools_eda.py — 信贷EDA Agent 工具集

包含以下6个工具：
  Tool 1: get_column_names          - 返回所有列名+类型，供LLM识别目标列
  Tool 2: get_column_unique_values  - 获取指定列所有去重值
  Tool 3: filter_by_target_def      - 按好/坏定义打标签，返回过滤后DataFrame
  Tool 4: plot_numeric_distribution - 连续型变量好/坏分布直方图+箱线图（书图4-5）
  Tool 5: plot_categorical_distribution - 类别型变量占比柱状图+坏样本比率折线（书图4-6）
  Tool 6: generate_html_report      - 汇总所有图片，生成自包含HTML报告
"""

import os
import base64
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # 非交互后端，避免 Windows 弹窗
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

logger = logging.getLogger(__name__)

# ── 字体：优先使用系统中文字体，兜底英文 ────────────────────────
def _setup_font():
    """尝试设置中文字体，失败则保持默认。"""
    candidates = [
        "Microsoft YaHei", "SimHei", "SimSun",
        "PingFang SC", "Noto Sans CJK SC",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return "default"

_FONT = _setup_font()


# ══════════════════════════════════════════════════════════════
# Tool 1: get_column_names
# ══════════════════════════════════════════════════════════════

def get_column_names(df: pd.DataFrame) -> Dict[str, str]:
    """
    返回所有列名及其 dtype，供 LLM 识别目标列。

    Returns
    -------
    dict: {列名: dtype字符串}
    """
    return {col: str(df[col].dtype) for col in df.columns}


# ══════════════════════════════════════════════════════════════
# Tool 2: get_column_unique_values
# ══════════════════════════════════════════════════════════════

def get_column_unique_values(df: pd.DataFrame, column_name: str) -> List[Any]:
    """
    获取指定列所有去重非空值，供 LLM 分析好坏样本定义。

    Parameters
    ----------
    df          : 数据集
    column_name : 目标列名

    Returns
    -------
    list: 去重后的值列表（已排序）

    Raises
    ------
    ValueError: 列不存在时抛出
    """
    if column_name not in df.columns:
        raise ValueError(f"列 '{column_name}' 不存在于数据中，可用列：{list(df.columns)}")

    vals = df[column_name].dropna().unique().tolist()
    try:
        vals = sorted(vals, key=str)
    except Exception:
        pass
    logger.info("[Tool 2] %s 的唯一值（%d 个）：%s", column_name, len(vals), vals)
    return vals


# ══════════════════════════════════════════════════════════════
# Tool 3: filter_by_target_def
# ══════════════════════════════════════════════════════════════

def filter_by_target_def(
    df: pd.DataFrame,
    target_column: str,
    good_values: List[Any],
    bad_values: List[Any],
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    按好坏样本定义打标签、过滤数据。
    忽略样本（既不在 good_values 也不在 bad_values）直接丢弃。

    Parameters
    ----------
    df            : 原始 DataFrame
    target_column : 目标列名
    good_values   : 好样本对应的原始值列表
    bad_values    : 坏样本对应的原始值列表

    Returns
    -------
    filtered_df       : 只保留好/坏样本的 DataFrame（新增 'target' 列，0=好，1=坏）
    good_mask_filtered: 过滤后 DataFrame 中标记好样本的布尔 Series
    bad_mask_filtered : 过滤后 DataFrame 中标记坏样本的布尔 Series
    """
    good_mask = df[target_column].isin(good_values)
    bad_mask  = df[target_column].isin(bad_values)
    keep_mask = good_mask | bad_mask

    filtered = df[keep_mask].copy()
    filtered["target"] = 0
    filtered.loc[bad_mask[keep_mask].values, "target"] = 1

    good_f = good_mask[keep_mask].reset_index(drop=True)
    bad_f  = bad_mask[keep_mask].reset_index(drop=True)

    logger.info(
        "[Tool 3] 原始：%d 行 | 好样本：%d | 坏样本：%d | 过滤后：%d",
        len(df), good_mask.sum(), bad_mask.sum(), len(filtered),
    )
    print(f"[Tool 3] 原始数据: {len(df):,} 行")
    print(f"         好样本 ({good_values}): {good_mask.sum():,} 行")
    print(f"         坏样本 ({bad_values}):  {bad_mask.sum():,} 行")
    print(f"         过滤后保留: {len(filtered):,} 行")

    return filtered, good_f, bad_f


# ══════════════════════════════════════════════════════════════
# Tool 4: plot_numeric_distribution（书 §4.3 图4-5）
# ══════════════════════════════════════════════════════════════

def plot_numeric_distribution(
    df: pd.DataFrame,
    column_name: str,
    good_mask: pd.Series,
    bad_mask: pd.Series,
    output_dir: str,
) -> Optional[str]:
    """
    绘制连续型变量在好/坏样本下的叠加直方图（书图4-5风格）。
    x轴：变量值，y轴：频数，好/坏样本分色叠加，单图输出。

    Returns
    -------
    图片路径，数据不足时返回 None
    """
    good_data = df[good_mask][column_name].dropna()
    bad_data  = df[bad_mask][column_name].dropna()

    if len(good_data) == 0 and len(bad_data) == 0:
        logger.warning("跳过 %s: 无有效数据", column_name)
        return None

    # 统计信息
    valid_n    = len(good_data) + len(bad_data)
    valid_rate = round(valid_n / len(df) * 100, 2)
    good_mean  = round(good_data.mean(), 2) if len(good_data) else float("nan")
    bad_mean   = round(bad_data.mean(),  2) if len(bad_data)  else float("nan")
    good_std   = round(good_data.std(),  2) if len(good_data)  else float("nan")
    bad_std    = round(bad_data.std(),   2) if len(bad_data)  else float("nan")

    bins = min(50, max(10, int(valid_n ** 0.5)))

    fig, ax = plt.subplots(figsize=(9, 5))

    # 叠加直方图
    if len(good_data):
        ax.hist(good_data, bins=bins, alpha=0.55, label="好样本",
                color="#2196F3", edgecolor="white", linewidth=0.4)
    if len(bad_data):
        ax.hist(bad_data,  bins=bins, alpha=0.55, label="坏样本",
                color="#F44336", edgecolor="white", linewidth=0.4)

    ax.set_xlabel(column_name, fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title(
        f"Variable: {column_name}\n"
        f"valid={valid_rate}%  |  "
        f"Mean(好)={good_mean}, Mean(坏)={bad_mean}  |  "
        f"Std(好)={good_std}, Std(坏)={bad_std}",
        fontsize=9,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    safe = column_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    out_dir = os.path.join(output_dir, "continuous")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{safe}.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return path


# ══════════════════════════════════════════════════════════════
# Tool 5: plot_categorical_distribution（书 §4.3 图4-6）
# ══════════════════════════════════════════════════════════════

def plot_categorical_distribution(
    df: pd.DataFrame,
    column_name: str,
    good_mask: pd.Series,
    bad_mask: pd.Series,
    output_dir: str,
    max_categories: int = 20,
) -> Optional[str]:
    """
    绘制类别型变量分布：柱状图（各类别占比）+ 折线图（坏样本比率）。
    参考书中图4-6风格。

    跳过条件：
      - 无有效数据
      - 类别数 > max_categories（文字太密）

    Returns
    -------
    图片路径，跳过时返回 None
    """
    # 仅保留 good 或 bad 样本行，去除目标列中的 NaN
    keep = good_mask | bad_mask
    df_temp = df[keep][[column_name]].copy()
    df_temp["_target"] = bad_mask[keep].astype(int).values

    df_temp = df_temp.dropna(subset=[column_name])

    if len(df_temp) == 0:
        logger.warning("跳过 %s: 无有效数据", column_name)
        return None

    n_unique = df_temp[column_name].nunique()
    if n_unique > max_categories:
        logger.info("跳过 %s: 类别过多 (%d > %d)", column_name, n_unique, max_categories)
        return None

    valid_rate = round(len(df_temp) / len(df) * 100, 2)

    # 每类别统计（书中逻辑：bin_rate=占比，bad_rate=坏样本率）
    stats = (
        df_temp.groupby(column_name)["_target"]
        .agg(bin_count="count", bad_count="sum")
        .reset_index()
    )
    stats["bin_rate"] = stats["bin_count"] / len(df_temp)
    stats["bad_rate"] = stats["bad_count"] / stats["bin_count"]
    stats = stats.sort_values("bad_rate", ascending=False).reset_index(drop=True)

    x = np.arange(len(stats))
    labels = [str(v)[:20] for v in stats[column_name]]

    fig, ax1 = plt.subplots(figsize=(max(10, len(stats) * 0.8), 6))

    # 柱状图：占比（书中 color='black'，alpha=0.5）
    ax1.bar(x, stats["bin_rate"], color="steelblue", alpha=0.5,
            label="bin_rate (占比)", width=0.5)
    ax1.set_xlabel(column_name, fontsize=11)
    ax1.set_ylabel(column_name, fontsize=11)   # 书中 y轴标签用变量名
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)

    # 折线图：坏样本比率（书中 color='green'，alpha=0.5）
    ax2 = ax1.twinx()
    ax2.plot(x, stats["bad_rate"], color="green", alpha=0.7,
             linewidth=2, marker="o", markersize=5, label="bad_rate (坏样本率)")
    ax2.set_ylabel("Bad Rate", fontsize=11, color="green")
    ax2.tick_params(axis="y", labelcolor="green")
    ax2.set_ylim(0, max(stats["bad_rate"].max() * 1.3, 0.05))

    # 数值标签
    for xi, (br, bdr) in enumerate(zip(stats["bin_rate"], stats["bad_rate"])):
        ax1.text(xi, br + 0.003, f"{br:.1%}", ha="center", va="bottom",
                 fontsize=8, color="steelblue")
        ax2.text(xi, bdr + 0.005, f"{bdr:.1%}", ha="center", va="bottom",
                 fontsize=8, color="green")

    plt.title(f"valid rate={valid_rate}%", fontsize=11)

    lines1, lbls1 = ax1.get_legend_handles_labels()
    lines2, lbls2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbls1 + lbls2, loc="upper right", fontsize=9)

    plt.tight_layout()

    safe = column_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    out_dir = os.path.join(output_dir, "categorical")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{safe}.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return path


# ══════════════════════════════════════════════════════════════
# Tool 6: generate_html_report
# ══════════════════════════════════════════════════════════════

def _img_to_b64(path: str) -> str:
    """将本地图片文件编码为 base64 data URI，使 HTML 完全自包含。"""
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def generate_html_report(
    continuous_images: List[str],
    categorical_images: List[str],
    stats: Dict,
    output_path: str,
) -> str:
    """
    生成完全自包含的 HTML 可视化报告（所有图片内嵌 base64）。

    Parameters
    ----------
    continuous_images  : 连续型变量图片路径列表
    categorical_images : 类别型变量图片路径列表
    stats              : 统计摘要字典，键：
                           total_samples, good_count, bad_count,
                           good_ratio, bad_ratio, feature_count,
                           good_values, bad_values, target_column
    output_path        : 输出 HTML 文件路径

    Returns
    -------
    HTML 文件绝对路径
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 编码图片 ────────────────────────────────
    def _img_cards(paths: List[str], grid_id: str) -> str:
        if not paths:
            return '<p style="color:#999;padding:20px;">（无图表）</p>'
        cards = []
        for p in paths:
            name = os.path.basename(p).replace(".png", "")
            b64  = _img_to_b64(p)
            cards.append(
                f'<div class="img-card" data-name="{name.lower()}">'
                f'<img src="{b64}" alt="{name}" onclick="openModal(this.src)">'
                f'<div class="caption">{name}</div>'
                f"</div>"
            )
        return "\n".join(cards)

    cont_cards = _img_cards(continuous_images, "cont-grid")
    cat_cards  = _img_cards(categorical_images,  "cat-grid")

    good_vals_str = ", ".join(str(v) for v in stats.get("good_values", []))
    bad_vals_str  = ", ".join(str(v) for v in stats.get("bad_values",  []))

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>信贷EDA报告 — 好坏样本分布</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:"Segoe UI",Arial,sans-serif;background:#fff;color:#222;font-size:14px}}

/* ── Header ── */
header{{background:#1a1a2e;color:#fff;padding:26px 40px}}
header h1{{font-size:20px;font-weight:700;margin-bottom:6px}}
header p{{font-size:12px;opacity:.75}}

/* ── KPI ── */
.kpi-bar{{display:flex;flex-wrap:wrap;gap:12px;padding:20px 40px;background:#f7f9fc;border-bottom:1px solid #e0e0e0}}
.kpi{{background:#fff;border:1px solid #e0e6ef;border-radius:8px;padding:12px 20px;min-width:130px;text-align:center}}
.kpi .val{{font-size:22px;font-weight:700;color:#1a1a2e}}
.kpi .lbl{{font-size:11px;color:#666;margin-top:3px}}
.kpi.good .val{{color:#2e7d32}}
.kpi.bad  .val{{color:#c62828}}
.kpi.blue .val{{color:#1565c0}}

/* ── Definition box ── */
.def-box{{display:flex;gap:16px;padding:16px 40px;border-bottom:1px solid #e0e0e0;background:#fff}}
.def-card{{flex:1;border-radius:8px;padding:12px 16px;border-left:4px solid}}
.def-card.good{{border-color:#2e7d32;background:#f1f8e9}}
.def-card.bad {{border-color:#c62828;background:#fff8f8}}
.def-card h3{{font-size:13px;margin-bottom:4px}}
.def-card p {{font-size:12px;color:#555;word-break:break-all}}

/* ── Nav ── */
nav{{display:flex;gap:6px;padding:10px 40px;border-bottom:1px solid #e0e0e0;background:#fff}}
nav button{{padding:7px 18px;border:1px solid #ccc;border-radius:6px;background:#fff;
            cursor:pointer;font-size:13px;transition:all .2s}}
nav button:hover{{background:#f0f0f0}}
nav button.active{{background:#1a1a2e;color:#fff;border-color:#1a1a2e}}

/* ── Content ── */
.container{{max-width:1600px;margin:0 auto;padding:24px 40px}}
.tab-panel{{display:none}}
.tab-panel.active{{display:block}}
.section-title{{font-size:15px;font-weight:700;margin:0 0 14px;padding-bottom:6px;
                border-bottom:2px solid #e0e0e0;color:#1a1a2e}}

/* ── Search ── */
.search-row{{margin-bottom:14px}}
.search-row input{{padding:7px 12px;width:280px;border:1px solid #ccc;
                   border-radius:6px;font-size:13px;outline:none}}
.search-row input:focus{{border-color:#1a1a2e}}
.count-label{{font-size:12px;color:#888;margin-left:8px}}

/* ── Grid ── */
.img-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(460px,1fr));gap:20px}}
.img-card{{border:1px solid #e0e0e0;border-radius:8px;overflow:hidden;
           background:#fafafa;transition:box-shadow .2s;cursor:pointer}}
.img-card:hover{{box-shadow:0 4px 16px rgba(0,0,0,.12)}}
.img-card img{{width:100%;display:block;background:#fff}}
.img-card .caption{{padding:8px 12px;font-size:12px;color:#555;text-align:center;
                    background:#fff;border-top:1px solid #eee;font-weight:500}}

/* ── Modal ── */
.modal{{display:none;position:fixed;z-index:1000;inset:0;background:rgba(0,0,0,.88);cursor:pointer}}
.modal img{{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
            max-width:92vw;max-height:92vh}}
.modal .close-btn{{position:absolute;top:16px;right:28px;color:#fff;font-size:36px;cursor:pointer}}

/* ── Source tag ── */
.src{{font-size:11px;color:#999;margin-bottom:10px}}

footer{{text-align:center;padding:20px;color:#999;font-size:11px;
        border-top:1px solid #eee;margin-top:30px}}
</style>
</head>
<body>
<header>
  <h1>信贷数据 EDA — 好/坏样本分布报告</h1>
  <p>数据源：lc_clean.csv &nbsp;|&nbsp; 生成时间：{now}</p>
</header>

<!-- KPI -->
<div class="kpi-bar">
  <div class="kpi blue"><div class="val">{stats.get("total_samples",0):,}</div><div class="lbl">总样本（清洗后）</div></div>
  <div class="kpi good"><div class="val">{stats.get("good_count",0):,}</div><div class="lbl">好样本</div></div>
  <div class="kpi bad"> <div class="val">{stats.get("bad_count",0):,}</div> <div class="lbl">坏样本</div></div>
  <div class="kpi good"><div class="val">{stats.get("good_ratio",0):.1f}%</div><div class="lbl">好样本占比</div></div>
  <div class="kpi bad"> <div class="val">{stats.get("bad_ratio",0):.1f}%</div> <div class="lbl">坏样本占比</div></div>
  <div class="kpi">     <div class="val">{stats.get("feature_count",0)}</div>     <div class="lbl">特征数量</div></div>
  <div class="kpi">     <div class="val">{len(continuous_images)}</div>           <div class="lbl">连续型变量图</div></div>
  <div class="kpi">     <div class="val">{len(categorical_images)}</div>          <div class="lbl">类别型变量图</div></div>
</div>

<!-- 好坏定义 -->
<div class="def-box">
  <div class="def-card good">
    <h3>✅ 好样本</h3>
    <p>目标列：<b>{stats.get("target_column","")}</b> &nbsp;|&nbsp; 对应值：{good_vals_str}</p>
    <p style="margin-top:4px;font-size:11px;color:#888">数据来源：LLM 分析 loan_status 唯一值后返回的 good_values</p>
  </div>
  <div class="def-card bad">
    <h3>❌ 坏样本</h3>
    <p>目标列：<b>{stats.get("target_column","")}</b> &nbsp;|&nbsp; 对应值：{bad_vals_str}</p>
    <p style="margin-top:4px;font-size:11px;color:#888">数据来源：LLM 分析 loan_status 唯一值后返回的 bad_values</p>
  </div>
</div>

<!-- Nav -->
<nav>
  <button class="active" onclick="switchTab('cont',this)">📈 连续型变量（{len(continuous_images)} 张）</button>
  <button onclick="switchTab('cat',this)">📊 类别型变量（{len(categorical_images)} 张）</button>
</nav>

<div class="container">

  <!-- 连续型 -->
  <div id="tab-cont" class="tab-panel active">
    <div class="src">数据来源：tools_eda.plot_numeric_distribution()</div>
    <div class="section-title">连续型变量：好样本 vs 坏样本分布对比（叠加直方图）</div>
    <div class="search-row">
      <input id="cont-search" placeholder="搜索变量名..." oninput="filterGrid('cont-grid',this.value)"/>
      <span class="count-label" id="cont-count">{len(continuous_images)} 张</span>
    </div>
    <div class="img-grid" id="cont-grid">
      {cont_cards}
    </div>
  </div>

  <!-- 类别型 -->
  <div id="tab-cat" class="tab-panel">
    <div class="src">数据来源：tools_eda.plot_categorical_distribution()</div>
    <div class="section-title">类别型变量：各取值占比柱状图 + 坏样本率折线图</div>
    <div class="search-row">
      <input id="cat-search" placeholder="搜索变量名..." oninput="filterGrid('cat-grid',this.value)"/>
      <span class="count-label" id="cat-count">{len(categorical_images)} 张</span>
    </div>
    <div class="img-grid" id="cat-grid">
      {cat_cards}
    </div>
  </div>

</div>

<footer>信贷EDA报告 | 好/坏样本分布可视化</footer>

<!-- Modal -->
<div class="modal" id="modal" onclick="closeModal()">
  <span class="close-btn">×</span>
  <img id="modal-img" src="" alt=""/>
</div>

<script>
function switchTab(name, btn) {{
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
  document.querySelectorAll("nav button").forEach(b => b.classList.remove("active"));
  document.getElementById("tab-" + name).classList.add("active");
  btn.classList.add("active");
}}
function filterGrid(gridId, kw) {{
  kw = kw.toLowerCase();
  let vis = 0;
  document.querySelectorAll("#" + gridId + " .img-card").forEach(c => {{
    const show = c.dataset.name.includes(kw);
    c.style.display = show ? "" : "none";
    if (show) vis++;
  }});
  const prefix = gridId === "cont-grid" ? "cont" : "cat";
  const el = document.getElementById(prefix + "-count");
  if (el) el.textContent = vis + " 张";
}}
function openModal(src) {{
  document.getElementById("modal-img").src = src;
  document.getElementById("modal").style.display = "block";
}}
function closeModal() {{
  document.getElementById("modal").style.display = "none";
}}
document.addEventListener("keydown", e => {{ if (e.key === "Escape") closeModal(); }});
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info("[Tool 6] HTML 报告已生成：%s", output_path)
    return os.path.abspath(output_path)
