"""
记忆管理器 - 统一管理短期和长期记忆

功能：
- 自动决策：何时读取记忆、何时写入记忆
- 短期记忆：会话消息历史
- 长期记忆：摘要向量库
- 跨会话上下文增强
"""

from typing import Optional
from dataclasses import dataclass, field

from src.utils.logger import logger
from src.agent.memory.short_term import ShortTermMemory, MemoryConfig
from src.agent.memory.long_term import LongTermMemory, LongTermConfig


@dataclass
class MemoryManagerConfig:
    """记忆管理器配置"""
    short_term: MemoryConfig = field(default_factory=MemoryConfig)
    long_term: LongTermConfig = field(default_factory=LongTermConfig)

    # 写入策略
    auto_summarize: bool = True       # 自动摘要存入长期记忆
    summarize_after_turns: int = 10    # 多少轮对话后触发摘要

    # 读取策略
    retrieve_long_term: bool = True   # 检索长期记忆作为上下文
    long_term_top_k: int = 3          # 长期记忆检索数量

    # 上下文组装
    include_recent_turns: int = 5     # 组装上下文时包含最近几轮


class MemoryManager:
    """
    记忆管理器 - 统一入口

    设计原则：
    1. 短期记忆：当前会话的消息历史，透明管理
    2. 长期记忆：跨会话摘要，按需检索
    3. 自动决策：何时摘要、何时检索

    工作流程：
    ┌─────────────────────────────────────────────────┐
    │                    MemoryManager                  │
    ├─────────────────────────────────────────────────┤
    │                                                   │
    │  add_message() ──→ 检查是否需要摘要 ──→ 摘要并存入长期 │
    │                                                   │
    │  get_context() ──→ 短期记忆 + 长期记忆 ──→ 组装上下文 │
    │                                                   │
    │  clear() ──→ 清空短期记忆（长期记忆保留）             │
    │                                                   │
    └─────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        config: Optional[MemoryManagerConfig] = None,
        embedder=None,
    ):
        self.config = config or MemoryManagerConfig()

        # 初始化子模块
        self._short_term = ShortTermMemory(config=self.config.short_term)
        self._long_term = LongTermMemory(
            config=self.config.long_term,
            embedder=embedder,
        )

        # 内部状态
        self._turn_count = 0  # 对话轮次计数

        logger.info("MemoryManager initialized")

    # ==================== 写入接口 ====================

    def add_user_message(self, content: str) -> None:
        """添加用户消息"""
        self._short_term.add_user_message(content)
        self._turn_count += 1

        # 检查是否需要摘要
        if self.config.auto_summarize:
            self._check_and_summarize()

    def add_ai_message(self, content: str) -> None:
        """添加 AI 消息"""
        self._short_term.add_ai_message(content)

    def add_message(self, message) -> None:
        """添加任意消息（LangChain 格式）"""
        self._short_term.add_message(message)
        if hasattr(message, "type") and message.type == "human":
            self._turn_count += 1

        if self.config.auto_summarize:
            self._check_and_summarize()

    def _check_and_summarize(self) -> None:
        """检查是否需要摘要并存入长期记忆"""
        if self._turn_count >= self.config.summarize_after_turns:
            self._turn_count = 0  # 重置计数
            # 触发摘要（异步）
            self._trigger_summarize()

    def _trigger_summarize(self) -> None:
        """触发摘要并存入长期记忆"""
        messages = self._short_term.messages
        if len(messages) < 4:
            return

        # 异步存入（不阻塞当前流程）
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 在已有事件循环中创建任务
                asyncio.create_task(
                    self._long_term.generate_summary(
                        messages=messages,
                        session_id=self._short_term.get_session_id(),
                    )
                )
            else:
                loop.run_until_complete(
                    self._long_term.generate_summary(
                        messages=messages,
                        session_id=self._short_term.get_session_id(),
                    )
                )
            logger.info("Triggered memory summarization")
        except Exception as e:
            logger.warning(f"Failed to trigger summarization: {e}")

    # ==================== 读取接口 ====================

    async def get_context(self, query: Optional[str] = None) -> dict:
        """
        获取增强后的上下文

        Args:
            query: 可选的查询内容，用于检索长期记忆

        Returns:
            {
                "short_term": 短期记忆上下文,
                "long_term": 长期记忆检索结果,
                "recent_turns": 最近几轮对话,
                "full_context": 组装后的完整上下文
            }
        """
        result = {
            "short_term": self._short_term.get_conversation_context(
                n=self.config.include_recent_turns
            ),
            "recent_turns": self._short_term.get_recent_messages(
                n=self.config.include_recent_turns * 2
            ),
            "long_term": [],
            "full_context": "",
        }

        # 检索长期记忆
        if query and self.config.retrieve_long_term:
            entries = await self._long_term.retrieve(
                query=query,
                top_k=self.config.long_term_top_k,
                session_filter=None,
            )
            result["long_term"] = entries

            # 组装完整上下文
            result["full_context"] = self._assemble_context(
                short_term=result["short_term"],
                long_term=entries,
            )
        else:
            result["full_context"] = result["short_term"]

        return result

    def _assemble_context(
        self,
        short_term: str,
        long_term: list,
    ) -> str:
        """组装完整上下文"""
        parts = []

        # 长期记忆优先（跨会话的重要信息）
        if long_term:
            parts.append("【相关历史记忆】")
            for entry in long_term:
                parts.append(f"- {entry.content}")
            parts.append("")

        # 短期记忆（当前会话）
        if short_term:
            parts.append("【当前对话】")
            parts.append(short_term)

        return "\n".join(parts)

    def get_short_term_messages(self) -> list:
        """获取短期记忆中的所有消息"""
        return self._short_term.messages

    # ==================== 管理接口 ====================

    def clear(self, clear_long_term: bool = False) -> None:
        """
        清空记忆

        Args:
            clear_long_term: 是否清空长期记忆（默认不清空）
        """
        self._short_term.clear()
        self._turn_count = 0

        if clear_long_term:
            self._long_term.clear()
            logger.info("Cleared all memories (short + long term)")
        else:
            logger.info("Cleared short term memory (long term preserved)")

    def new_session(self, session_id: str) -> None:
        """切换到新会话"""
        self._short_term.set_session_id(session_id)
        self._turn_count = 0
        logger.info(f"Switched to new session: {session_id}")

    def get_session_id(self) -> str:
        """获取当前会话 ID"""
        return self._short_term.get_session_id()

    # ==================== 状态查询 ====================

    def get_stats(self) -> dict:
        """获取记忆统计"""
        return {
            "short_term_count": len(self._short_term),
            "long_term_count": len(self._long_term.get_all()),
            "current_session": self._short_term.get_session_id(),
            "turn_count": self._turn_count,
        }

    def should_summarize(self) -> bool:
        """检查是否应该摘要"""
        return self._turn_count >= self.config.summarize_after_turns


# ==================== 单例管理 ====================

_global_manager: Optional[MemoryManager] = None


def get_memory_manager() -> Optional[MemoryManager]:
    """获取全局记忆管理器实例"""
    return _global_manager


def init_memory_manager(
    config: Optional[MemoryManagerConfig] = None,
    embedder=None,
) -> MemoryManager:
    """
    初始化全局记忆管理器

    Args:
        config: 配置
        embedder: 嵌入模型（用于长期记忆向量检索）

    Returns:
        记忆管理器实例
    """
    global _global_manager
    _global_manager = MemoryManager(config=config, embedder=embedder)
    return _global_manager


def create_memory_manager(
    embedder=None,
    auto_summarize: bool = True,
    summarize_after_turns: int = 10,
) -> MemoryManager:
    """
    工厂函数：创建记忆管理器

    Args:
        embedder: 嵌入模型
        auto_summarize: 自动摘要
        summarize_after_turns: 摘要触发轮数

    Returns:
        记忆管理器实例
    """
    config = MemoryManagerConfig(
        auto_summarize=auto_summarize,
        summarize_after_turns=summarize_after_turns,
    )
    return MemoryManager(config=config, embedder=embedder)
