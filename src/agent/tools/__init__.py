"""
工具系统 - MCP 风格的工具调用框架

提供：
1. BaseTool：工具基类，定义标准接口
2. ToolRegistry：工具注册中心，支持动态注册
3. 内置工具：WebSearch 网页搜索
4. MCP 预留接口：支持外部工具服务扩展
"""

from src.agent.tools.base import BaseTool, ToolResult
from src.agent.tools.registry import ToolRegistry, get_registry
from src.agent.tools.web_search import WebSearchTool, DuckDuckGoSearch, TavilySearch

__all__ = [
    "BaseTool",
    "ToolResult",
    "ToolRegistry",
    "get_registry",
    "WebSearchTool",
    "DuckDuckGoSearch",
    "TavilySearch",
]
