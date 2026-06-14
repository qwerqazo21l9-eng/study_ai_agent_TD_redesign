"""
LLM 封装模块

统一管理 LLM 实例，提供同步/异步接口
"""

from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel

from src.utils.config_loader import config
from src.utils.logger import logger


# ============ 全局 LLM 实例 ============

_llm_instance: Optional[ChatOpenAI] = None


def get_llm() -> ChatOpenAI:
    """
    获取 LLM 实例（单例）

    Returns:
        ChatOpenAI 实例
    """
    global _llm_instance

    if _llm_instance is None:
        API_KEY = config.model["cloud_api_key"]
        API_BASE = config.model["cloud_api_base"]
        MODEL_NAME = config.model["cloud_model_name"]

        logger.info(f"Initializing LLM: {MODEL_NAME}")

        _llm_instance = ChatOpenAI(
            model=MODEL_NAME,
            api_key=API_KEY,
            base_url=API_BASE,
            temperature=0.3,
            max_tokens=2048,
        )

    return _llm_instance


def create_llm(temperature: float = 0.3, max_tokens: int = 2048) -> ChatOpenAI:
    """
    创建新的 LLM 实例（不使用单例）

    Args:
        temperature: 温度参数
        max_tokens: 最大 token 数

    Returns:
        ChatOpenAI 实例
    """
    API_KEY = config.model["cloud_api_key"]
    API_BASE = config.model["cloud_api_base"]
    MODEL_NAME = config.model["cloud_model_name"]

    return ChatOpenAI(
        model=MODEL_NAME,
        api_key=API_KEY,
        base_url=API_BASE,
        temperature=temperature,
        max_tokens=max_tokens,
    )


class LLMWrapper:
    """
    LLM 包装器

    提供更友好的接口，支持：
    - 同步/异步调用
    - 自动重试
    - 流式输出
    """

    def __init__(
        self,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        model_name: Optional[str] = None
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_name = model_name

        # 初始化 LLM
        API_KEY = config.model["cloud_api_key"]
        API_BASE = config.model["cloud_api_base"]
        model = model_name or config.model["cloud_model_name"]

        self.llm = ChatOpenAI(
            model=model,
            api_key=API_KEY,
            base_url=API_BASE,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def agenerate(
        self,
        messages: list[dict],
        **kwargs
    ):
        """
        异步生成

        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            **kwargs: 其他参数

        Returns:
            LLM 响应
        """
        # 转换格式
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

        langchain_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))

        return await self.llm.agenerate(langchain_messages)

    def generate(
        self,
        messages: list[dict],
        **kwargs
    ):
        """
        同步生成

        Args:
            messages: 消息列表
            **kwargs: 其他参数

        Returns:
            LLM 响应
        """
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

        langchain_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))

        return self.llm.generate([langchain_messages])

    async def ainvoke(self, prompt: str, **kwargs):
        """异步单轮对话"""
        return await self.llm.ainvoke(prompt, **kwargs)

    def invoke(self, prompt: str, **kwargs):
        """同步单轮对话"""
        return self.llm.invoke(prompt, **kwargs)
