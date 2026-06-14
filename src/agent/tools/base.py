"""
BaseTool - 工具基类定义

MCP (Model Context Protocol) 风格的工具接口：
1. 标准化工具描述
2. 统一的执行接口
3. 类型安全的参数和结果
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum

from src.utils.logger import logger


class ToolStatus(Enum):
    """工具执行状态"""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"


@dataclass
class ToolResult:
    """
    工具执行结果

    统一的结果格式，便于 Agent 层处理
    """
    status: ToolStatus
    content: str = ""
    data: Optional[dict] = None
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == ToolStatus.SUCCESS

    @property
    def is_empty(self) -> bool:
        """判断结果是否为空"""
        return not self.content and not self.data

    def to_context(self) -> str:
        """转换为上下文字符串"""
        if not self.success:
            return f"[工具调用失败] {self.error or '未知错误'}"

        if self.content:
            return self.content

        if self.data:
            # 格式化 JSON 数据
            import json
            return json.dumps(self.data, ensure_ascii=False, indent=2)

        return "[工具返回空结果]"


@dataclass
class ToolParameter:
    """
    工具参数定义

    用于工具描述和参数校验
    """
    name: str
    description: str
    type: str = "string"  # string, number, boolean, array, object
    required: bool = True
    default: Any = None
    enum: Optional[list] = None


@dataclass
class ToolDefinition:
    """
    工具定义

    包含工具的完整元信息
    """
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    category: str = "general"  # general, search, memory, external
    version: str = "1.0.0"
    author: Optional[str] = None
    tags: list[str] = field(default_factory=list)

    def to_openai_format(self) -> dict:
        """
        转换为 OpenAI Function Calling 格式

        便于 LLM 调用
        """
        params = {}
        required = []

        for p in self.parameters:
            param_dict = {
                "type": p.type,
                "description": p.description,
            }
            if p.enum:
                param_dict["enum"] = p.enum
            if p.default is not None:
                param_dict["default"] = p.default

            params[p.name] = param_dict
            if p.required:
                required.append(p.name)

        result = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": params,
                    "required": required,
                },
            },
        }

        return result


class BaseTool(ABC):
    """
    工具基类

    所有工具必须继承此类并实现：
    1. _execute: 核心执行逻辑
    2. definition: 工具定义（属性）

    使用示例：
        class MyTool(BaseTool):
            @property
            def definition(self) -> ToolDefinition:
                return ToolDefinition(
                    name="my_tool",
                    description="我的工具",
                    parameters=[...]
                )

            async def _execute(self, **kwargs) -> ToolResult:
                # 实现逻辑
                ...
    """

    def __init__(self):
        self._definition = self.definition

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """工具定义，必须实现"""
        pass

    @property
    def name(self) -> str:
        return self._definition.name

    @property
    def description(self) -> str:
        return self._definition.description

    def get_parameter(self, name: str) -> Optional[ToolParameter]:
        """获取参数定义"""
        for p in self._definition.parameters:
            if p.name == name:
                return p
        return None

    def validate_params(self, params: dict) -> tuple[bool, Optional[str]]:
        """
        验证参数

        Returns:
            (is_valid, error_message)
        """
        for p in self._definition.parameters:
            if p.required and p.name not in params:
                return False, f"缺少必需参数: {p.name}"

            if p.name in params:
                value = params[p.name]
                expected_type = p.type

                # 类型检查
                if expected_type == "string" and not isinstance(value, str):
                    return False, f"参数 {p.name} 必须是字符串"
                elif expected_type == "number" and not isinstance(value, (int, float)):
                    return False, f"参数 {p.name} 必须是数字"
                elif expected_type == "boolean" and not isinstance(value, bool):
                    return False, f"参数 {p.name} 必须是布尔值"
                elif expected_type == "array" and not isinstance(value, list):
                    return False, f"参数 {p.name} 必须是数组"

                # 枚举检查
                if p.enum and value not in p.enum:
                    return False, f"参数 {p.name} 必须是 {p.enum} 之一"

        return True, None

    async def execute(self, **kwargs) -> ToolResult:
        """
        执行工具

        模板方法模式：
        1. 参数验证
        2. 调用 _execute
        3. 异常处理

        Args:
            **kwargs: 工具参数

        Returns:
            ToolResult: 执行结果
        """
        # 参数验证
        is_valid, error = self.validate_params(kwargs)
        if not is_valid:
            logger.warning(f"Tool {self.name}: invalid params - {error}")
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"参数验证失败: {error}",
            )

        # 执行
        try:
            result = await self._execute(**kwargs)
            logger.info(f"Tool {self.name}: executed successfully")
            return result
        except Exception as e:
            logger.error(f"Tool {self.name}: execution failed - {e}")
            return ToolResult(
                status=ToolStatus.ERROR,
                error=str(e),
            )

    @abstractmethod
    async def _execute(self, **kwargs) -> ToolResult:
        """
        核心执行逻辑，子类必须实现

        Args:
            **kwargs: 经过验证的工具参数

        Returns:
            ToolResult: 执行结果
        """
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name})>"
