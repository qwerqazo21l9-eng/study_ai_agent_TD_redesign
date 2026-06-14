"""
data_processing_tool.py — DataProcessingTool 定义

继承 BaseTool，封装两阶段数据自动化处理流水线。
用户说"请对 xxx 表格进行数据处理"时，Agent 会调用此工具。
"""

import asyncio
import logging
from typing import Optional

from src.agent.tools.base import BaseTool, ToolDefinition, ToolParameter, ToolResult, ToolStatus
from src.utils.config_loader import config

logger = logging.getLogger(__name__)


class DataProcessingTool(BaseTool):
    """
    数据自动化处理工具

    封装 Phase 1（数据清洗）+ Phase 2（好坏样本 EDA）两阶段流水线。
    用户说"请对 xxx.csv 进行数据处理"时触发。
    """

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="data_processing",
            description=(
                "对 CSV 表格执行数据自动化处理：数据清洗 + EDA 分析报告。"
                "自动完成缺失值处理、类型转换、去重、好坏样本识别、分布可视化，"
                "并生成 HTML 报告。"
            ),
            parameters=[
                ToolParameter(
                    name="file_path",
                    description="CSV 文件路径（绝对路径或相对于项目的路径）",
                    type="string",
                    required=True,
                ),
                ToolParameter(
                    name="missing_threshold",
                    description="高缺失率列删除阈值（默认 0.3，即缺失率 > 30% 的列会被删除）",
                    type="number",
                    required=False,
                    default=0.3,
                ),
                ToolParameter(
                    name="cat_threshold",
                    description="数值列唯一值数量 ≤ 该值时视为类别型（默认 10）",
                    type="number",
                    required=False,
                    default=10,
                ),
                ToolParameter(
                    name="nrows",
                    description="仅读取前 N 行（调试用，默认 200000）",
                    type="number",
                    required=False,
                    default=200000,
                ),
            ],
            category="data",
            version="1.0.0",
            author="arthur",
            tags=["data-cleaning", "eda", "lending-club", "risk-control"],
        )

    async def _execute(self, **kwargs) -> ToolResult:
        """
        执行数据自动化处理流水线

        kwargs:
            file_path (str, required): CSV 文件路径
            missing_threshold (float, optional): 缺失率阈值，默认 0.3
            cat_threshold (int, optional): 类别型阈值，默认 10
            nrows (int, optional): 读取行数，默认 200000
        """
        file_path = kwargs.get("file_path")
        if not file_path:
            return ToolResult(
                status=ToolStatus.ERROR,
                error="缺少必需参数：file_path",
            )

        import os
        if not os.path.exists(file_path):
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"文件不存在：{file_path}",
            )

        missing_threshold = kwargs.get("missing_threshold", 0.3)
        cat_threshold    = int(kwargs.get("cat_threshold", 10))
        nrows            = int(kwargs.get("nrows", 200000))

        # 从 config 读取输出目录
        output_dir = config.data_processing.get(
            "output_dir", "./output/data_processing"
        )
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        try:
            from src.agent.data_processing.master_agent import run_data_processing_pipeline

            result = await run_data_processing_pipeline(
                filepath=file_path,
                output_dir=output_dir,
                nrows=nrows,
                missing_threshold=missing_threshold,
                cat_threshold=cat_threshold,
                plot=True,
            )

            if result.get("status") == "success":
                deliverables = result.get("deliverables", {})
                summary = result.get("summary", "数据处理完成")

                content_parts = [summary, "\n交付物："]
                for k, v in deliverables.items():
                    content_parts.append(f"  - {k}: {v}")

                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    content="\n".join(content_parts),
                    data=result,
                )
            else:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    error=result.get("summary", "数据处理失败"),
                    data=result,
                )

        except Exception as e:
            logger.error("[DataProcessingTool] 执行失败：%s", e, exc_info=True)
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"数据处理工具执行异常：{e}",
            )
