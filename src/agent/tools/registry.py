"""
ToolRegistry - 工具注册中心

功能：
1. 工具注册与注销
2. 工具查找与获取
3. 工具列表导出
4. MCP 扩展接口预留

使用示例：
    # 单例模式
    registry = get_registry()
    registry.register(WebSearchTool(provider="duckduckgo"))

    # 获取工具
    tool = registry.get("web_search")
    result = await tool.execute(query="...")

    # 列出所有工具
    for name, tool in registry.list_tools():
        print(f"{name}: {tool.description}")
"""

from typing import Optional, Iterator
from dataclasses import dataclass, field
from collections import defaultdict

from src.agent.tools.base import BaseTool, ToolDefinition, ToolResult
from src.utils.logger import logger


@dataclass
class ToolInfo:
    """工具注册信息"""
    tool: BaseTool
    enabled: bool = True
    priority: int = 0  # 优先级，数值越大越优先被选用
    metadata: dict = field(default_factory=dict)


class ToolRegistry:
    """
    工具注册中心

    统一管理所有可用工具，提供：
    - 注册/注销
    - 获取
    - 列表
    - MCP 扩展接口
    """

    def __init__(self):
        self._tools: dict[str, ToolInfo] = {}
        self._categories: dict[str, set[str]] = defaultdict(set)

    def register(
        self,
        tool: BaseTool,
        enabled: bool = True,
        priority: int = 0,
        **metadata,
    ) -> None:
        """
        注册工具

        Args:
            tool: 工具实例
            enabled: 是否启用
            priority: 优先级
            **metadata: 额外元信息
        """
        name = tool.name

        if name in self._tools:
            logger.warning(f"Tool {name} already registered, overwriting")

        self._tools[name] = ToolInfo(
            tool=tool,
            enabled=enabled,
            priority=priority,
            metadata=metadata,
        )

        # 更新分类索引
        category = tool.definition.category
        self._categories[category].add(name)

        logger.info(f"Registered tool: {name} (category={category}, priority={priority})")

    def unregister(self, name: str) -> bool:
        """
        注销工具

        Args:
            name: 工具名称

        Returns:
            是否成功注销
        """
        if name not in self._tools:
            logger.warning(f"Tool {name} not found")
            return False

        tool_info = self._tools.pop(name)

        # 更新分类索引
        category = tool_info.tool.definition.category
        self._categories[category].discard(name)

        logger.info(f"Unregistered tool: {name}")
        return True

    def get(self, name: str) -> Optional[BaseTool]:
        """
        获取工具

        Args:
            name: 工具名称

        Returns:
            工具实例，若不存在返回 None
        """
        tool_info = self._tools.get(name)
        if tool_info is None:
            return None

        if not tool_info.enabled:
            logger.debug(f"Tool {name} is disabled")
            return None

        return tool_info.tool

    def get_tool_info(self, name: str) -> Optional[ToolInfo]:
        """获取工具及其元信息"""
        return self._tools.get(name)

    def list_tools(
        self,
        category: Optional[str] = None,
        enabled_only: bool = True,
    ) -> Iterator[tuple[str, BaseTool]]:
        """
        列出工具

        Args:
            category: 按分类过滤
            enabled_only: 只返回启用的工具

        Yields:
            (name, tool) 元组
        """
        if category:
            names = self._categories.get(category, set())
        else:
            names = self._tools.keys()

        for name in names:
            tool_info = self._tools.get(name)
            if tool_info is None:
                continue

            if enabled_only and not tool_info.enabled:
                continue

            yield name, tool_info.tool

    def list_definitions(self) -> list[ToolDefinition]:
        """获取所有工具定义（用于 LLM Function Calling）"""
        definitions = []

        for _, tool in self.list_tools(enabled_only=True):
            definitions.append(tool.definition)

        return definitions

    def enable(self, name: str) -> bool:
        """启用工具"""
        tool_info = self._tools.get(name)
        if tool_info is None:
            return False

        tool_info.enabled = True
        logger.info(f"Enabled tool: {name}")
        return True

    def disable(self, name: str) -> bool:
        """禁用工具"""
        tool_info = self._tools.get(name)
        if tool_info is None:
            return False

        tool_info.enabled = False
        logger.info(f"Disabled tool: {name}")
        return True

    def get_by_tag(self, tag: str) -> Iterator[tuple[str, BaseTool]]:
        """通过标签查找工具"""
        for name, tool in self._tools.items():
            if tag in tool.tool.definition.tags:
                yield name, tool.tool

    def clear(self) -> None:
        """清空所有工具"""
        self._tools.clear()
        self._categories.clear()
        logger.info("Cleared all tools")

    @property
    def count(self) -> int:
        """工具数量"""
        return len(self._tools)

    @property
    def enabled_count(self) -> int:
        """启用的工具数量"""
        return sum(1 for t in self._tools.values() if t.enabled)

    def __len__(self) -> int:
        return self.count

    def __contains__(self, name: str) -> bool:
        """检查工具是否存在"""
        return name in self._tools


# ============ 单例模式 ============

_global_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """
    获取全局工具注册中心（单例）

    Returns:
        全局 ToolRegistry 实例
    """
    global _global_registry

    if _global_registry is None:
        _global_registry = ToolRegistry()

    return _global_registry


def reset_registry() -> None:
    """重置全局注册中心（主要用于测试）"""
    global _global_registry
    _global_registry = None


# ============ MCP 扩展接口（预留） ============

class MCPToolAdapter:
    """
    MCP 协议工具适配器

    预留接口，用于接入外部 MCP 服务

    MCP (Model Context Protocol) 是标准化工具调用的协议，
    支持远程工具服务扩展
    """

    def __init__(self, server_url: str, auth_token: Optional[str] = None):
        """
        Args:
            server_url: MCP 服务器地址
            auth_token: 认证令牌
        """
        self.server_url = server_url
        self.auth_token = auth_token
        self._tools: dict[str, BaseTool] = {}
        self._connected = False

    async def connect(self) -> bool:
        """
        连接 MCP 服务器

        Returns:
            是否连接成功
        """
        # TODO: 实现 MCP 协议握手
        # 1. HTTP/WebSocket 连接
        # 2. 交换协议版本
        # 3. 获取可用工具列表
        raise NotImplementedError("MCP 适配器待实现")

    async def disconnect(self) -> None:
        """断开 MCP 服务器"""
        # TODO: 实现断开逻辑
        raise NotImplementedError("MCP 适配器待实现")

    async def call_tool(self, name: str, **kwargs) -> ToolResult:
        """
        调用 MCP 服务器上的工具

        Args:
            name: 工具名称
            **kwargs: 工具参数

        Returns:
            工具执行结果
        """
        # TODO: 实现工具调用
        raise NotImplementedError("MCP 适配器待实现")

    def get_available_tools(self) -> list[ToolDefinition]:
        """
        获取 MCP 服务器上可用的工具列表

        Returns:
            工具定义列表
        """
        # TODO: 实现工具列表获取
        raise NotImplementedError("MCP 适配器待实现")


def create_mcp_adapter(
    server_url: str,
    auth_token: Optional[str] = None,
) -> MCPToolAdapter:
    """
    工厂函数：创建 MCP 适配器

    Args:
        server_url: MCP 服务器地址
        auth_token: 认证令牌

    Returns:
        MCP 适配器实例
    """
    return MCPToolAdapter(server_url=server_url, auth_token=auth_token)


# ============ 便捷工具初始化函数 ============

def init_default_tools(provider: str = "duckduckgo") -> ToolRegistry:
    """
    初始化默认工具集

    Args:
        provider: WebSearch 工具的 provider

    Returns:
        配置好的工具注册中心
    """
    from src.agent.tools.web_search import WebSearchTool

    registry = get_registry()

    # 注册 WebSearch
    registry.register(
        WebSearchTool(provider=provider),
        priority=10,
    )

    return registry
