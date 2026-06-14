"""
记忆模块基类定义
"""

from abc import ABC, abstractmethod
from typing import Optional, Any
from langchain_core.messages import BaseMessage


class BaseMemory(ABC):
    """
    记忆基类 - 定义记忆模块的统一接口
    """

    @abstractmethod
    def add_user_message(self, content: str) -> None:
        """添加用户消息"""
        pass

    @abstractmethod
    def add_ai_message(self, content: str) -> None:
        """添加 AI 消息"""
        pass

    @abstractmethod
    def add_message(self, message: BaseMessage) -> None:
        """添加任意消息"""
        pass

    @abstractmethod
    def get_messages(self) -> list[BaseMessage]:
        """获取所有消息"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """清空记忆"""
        pass

    @abstractmethod
    def get_session_id(self) -> str:
        """获取会话 ID"""
        pass

    @abstractmethod
    def set_session_id(self, session_id: str) -> None:
        """设置会话 ID"""
        pass
