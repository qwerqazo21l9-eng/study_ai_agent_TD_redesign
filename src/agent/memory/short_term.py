"""
短期记忆模块 - 会话消息历史管理

功能：
- 基于 LangGraph messages 的会话历史
- Token 计数与截断
- 摘要压缩
- 多会话隔离
"""

from typing import Optional
from dataclasses import dataclass
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from src.utils.logger import logger


@dataclass
class MemoryConfig:
    """记忆配置"""
    max_messages: int = 20           # 最大消息数
    max_tokens: int = 4000           # 最大 Token 数
    auto_summarize_threshold: float = 0.8  # 触发摘要的比例
    session_id: str = "default"       # 会话 ID


class ShortTermMemory:
    """
    短期记忆 - 管理当前会话的消息历史

    设计原则：
    1. 与 LangGraph StateGraph.messages 协同工作
    2. 支持 Token 计数和自动截断
    3. 预留摘要压缩接口
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self._messages: list[BaseMessage] = []

    @property
    def messages(self) -> list[BaseMessage]:
        """获取当前消息历史"""
        return self._messages.copy()

    def add_user_message(self, content: str) -> None:
        """添加用户消息"""
        self._messages.append(HumanMessage(content=content))

    def add_ai_message(self, content: str) -> None:
        """添加 AI 消息"""
        self._messages.append(AIMessage(content=content))

    def add_message(self, message: BaseMessage) -> None:
        """添加任意消息"""
        self._messages.append(message)

    def get_recent_messages(self, n: int = 10) -> list[BaseMessage]:
        """获取最近的 n 条消息"""
        return self._messages[-n:] if self._messages else []

    def get_conversation_context(self, n: int = 5) -> str:
        """
        获取对话上下文（用于 prompt）

        Args:
            n: 保留最近 n 轮对话（每轮 = user + ai）

        Returns:
            格式化后的对话历史
        """
        # 保留最近 n 轮完整对话
        recent = self._messages[-n * 2:] if self._messages else []

        if not recent:
            return ""

        # 格式化
        parts = []
        for msg in recent:
            role = "用户" if isinstance(msg, HumanMessage) else "助手"
            parts.append(f"{role}：{msg.content}")

        return "\n".join(parts)

    def clear(self) -> None:
        """清空记忆"""
        self._messages = []
        logger.info(f"ShortTermMemory cleared for session: {self.config.session_id}")

    def should_summarize(self) -> bool:
        """
        检查是否需要摘要压缩

        基于消息数量判断
        """
        return len(self._messages) >= self.config.max_messages

    def truncate(self, keep_recent: int = 10) -> list[BaseMessage]:
        """
        截断旧消息，保留最近的 keep_recent 条

        Args:
            keep_recent: 保留最近的消息数

        Returns:
            被截断的消息（可用于摘要）
        """
        if len(self._messages) <= keep_recent:
            return []

        truncated = self._messages[:-keep_recent]
        self._messages = self._messages[-keep_recent:]

        logger.info(
            f"Truncated {len(truncated)} messages, "
            f"kept {len(self._messages)} messages"
        )

        return truncated

    def get_session_id(self) -> str:
        """获取会话 ID"""
        return self.config.session_id

    def set_session_id(self, session_id: str) -> None:
        """设置会话 ID"""
        self.config.session_id = session_id
        self.clear()  # 切换会话时清空

    def __len__(self) -> int:
        """消息数量"""
        return len(self._messages)


def create_short_term_memory(
    max_messages: int = 20,
    session_id: str = "default",
) -> ShortTermMemory:
    """
    工厂函数：创建短期记忆实例
    """
    config = MemoryConfig(
        max_messages=max_messages,
        session_id=session_id,
    )
    return ShortTermMemory(config)
