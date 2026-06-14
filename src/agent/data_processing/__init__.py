"""
数据自动化处理模块（Phase 1 清洗 + Phase 2 EDA）

导出：
  - run_data_processing_pipeline: 两阶段流水线主入口
  - DataProcessingTool: BaseTool 封装（在 tools/data_processing_tool.py）
"""

from src.agent.data_processing.master_agent import run_data_processing_pipeline

__all__ = ["run_data_processing_pipeline"]
