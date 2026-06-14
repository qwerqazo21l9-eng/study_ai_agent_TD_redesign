"""
长期记忆模块 - 摘要向量库

功能：
- 将重要对话摘要存入向量库
- 跨会话检索相关历史
- 自动摘要生成
"""

import json
from typing import Optional, Any
from dataclasses import dataclass
from datetime import datetime
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from src.utils.logger import logger
from src.utils.llm import LLMWrapper


@dataclass
class LongTermConfig:
    """长期记忆配置"""
    enabled: bool = True
    max_memories: int = 100           # 最大记忆条数
    similarity_threshold: float = 0.7  # 相似度阈值
    auto_store_threshold: int = 10    # 自动存入的消息数阈值
    memory_db_path: str = "./vector_db/memory_index"  # 记忆向量库路径


class MemoryEntry:
    """
    记忆条目 - 存储在向量库中的单条记忆

    结构：
    - content: 摘要内容
    - session_id: 来源会话
    - timestamp: 创建时间
    - keywords: 关键词（用于快速过滤）
    """

    def __init__(
        self,
        content: str,
        session_id: str = "default",
        keywords: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ):
        self.content = content
        self.session_id = session_id
        self.timestamp = datetime.now().isoformat()
        self.keywords = keywords or []
        self.metadata = metadata or {}

    def to_document(self) -> Document:
        """转换为 LangChain Document"""
        return Document(
            page_content=self.content,
            metadata={
                "session_id": self.session_id,
                "timestamp": self.timestamp,
                "keywords": json.dumps(self.keywords, ensure_ascii=False),
                **self.metadata,
            }
        )

    @classmethod
    def from_document(cls, doc: Document) -> "MemoryEntry":
        """从 Document 恢复"""
        metadata = doc.metadata
        keywords_str = metadata.get("keywords", "[]")
        try:
            keywords = json.loads(keywords_str)
        except (json.JSONDecodeError, TypeError):
            keywords = []

        entry = cls(
            content=doc.page_content,
            session_id=metadata.get("session_id", "default"),
            keywords=keywords,
            metadata={k: v for k, v in metadata.items()
                     if k not in ["session_id", "timestamp", "keywords"]},
        )
        return entry

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "content": self.content,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "keywords": self.keywords,
            "metadata": self.metadata,
        }


class LongTermMemory:
    """
    长期记忆 - 基于向量库的跨会话记忆

    工作流程：
    1. 对话结束后，检测是否需要存入长期记忆
    2. 生成摘要（LLM）
    3. 提取关键词
    4. 存入向量库
    5. 检索时，查找相关记忆作为上下文
    """

    def __init__(
        self,
        config: Optional[LongTermConfig] = None,
        embedder=None,
    ):
        self.config = config or LongTermConfig()
        self._embedder = embedder
        self._vector_store = None  # 延迟初始化

    def _ensure_vector_store(self):
        """延迟初始化向量存储"""
        if self._vector_store is None:
            try:
                from langchain_community.vectorstores import FAISS
                import os

                path = self.config.memory_db_path

                if os.path.exists(path) and (
                    os.path.exists(os.path.join(path, "index.faiss"))
                    or os.path.exists(os.path.join(path, "index.pkl"))
                ):
                    self._vector_store = FAISS.load_local(
                        path,
                        self._embedder,
                        allow_dangerous_deserialization=True,
                    )
                    logger.info(f"Loaded memory vector store from {path}")
                else:
                    os.makedirs(path, exist_ok=True)
                    logger.info(f"Created new memory vector store at {path}")
            except ImportError as e:
                logger.warning(f"FAISS not available: {e}")
                self._vector_store = None

    async def store(
        self,
        summary: str,
        session_id: str = "default",
        keywords: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> bool:
        """
        存入长期记忆

        Args:
            summary: 摘要内容
            session_id: 会话 ID
            keywords: 关键词
            metadata: 额外元数据

        Returns:
            是否成功
        """
        if not self.config.enabled:
            return False

        entry = MemoryEntry(
            content=summary,
            session_id=session_id,
            keywords=keywords,
            metadata=metadata,
        )

        try:
            self._ensure_vector_store()

            if self._vector_store is None:
                # 简单内存存储
                self._memories = getattr(self, "_memories", [])
                self._memories.append(entry)
                logger.info(f"Stored memory in memory (total: {len(self._memories)})")
                return True

            # 存入向量库
            doc = entry.to_document()
            self._vector_store.add_documents([doc])

            # 持久化
            self._vector_store.save_local(self.config.memory_db_path)

            logger.info(f"Stored memory: {summary[:50]}...")
            return True

        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            return False

    async def generate_summary(
        self,
        messages: list,
        session_id: str = "default",
    ) -> Optional[MemoryEntry]:
        """
        生成对话摘要并存入长期记忆

        Args:
            messages: 对话消息列表
            session_id: 会话 ID

        Returns:
            生成的记忆条目
        """
        if len(messages) < 4:  # 至少 2 轮对话
            return None

        try:
            # 构建摘要 prompt
            conversation = self._format_messages(messages)

            summary_prompt = f"""请总结以下对话的核心内容，生成一段简洁的摘要：

{conversation}

要求：
1. 提取关键信息、决策、结论
2. 保留重要的上下文
3. 100-200 字以内
4. 只输出摘要，不要其他内容
"""

            # 调用 LLM
            llm = LLMWrapper(temperature=0.3)
            response = await llm.agenerate([[HumanMessage(content=summary_prompt)]])
            summary = response.generations[0][0].text.strip()

            # 提取关键词（简单实现）
            keywords = self._extract_keywords(summary)

            # 存入
            entry = MemoryEntry(
                content=summary,
                session_id=session_id,
                keywords=keywords,
                metadata={"message_count": len(messages)},
            )

            await self.store(
                summary=summary,
                session_id=session_id,
                keywords=keywords,
                metadata={"message_count": len(messages)},
            )

            return entry

        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return None

    def _format_messages(self, messages: list) -> str:
        """格式化消息列表"""
        parts = []
        for msg in messages:
            if hasattr(msg, "type"):
                role = "用户" if msg.type == "human" else "助手"
            else:
                role = "未知"

            content = msg.content if hasattr(msg, "content") else str(msg)
            parts.append(f"{role}：{content}")

        return "\n".join(parts)

    def _extract_keywords(self, text: str, top_n: int = 5) -> list[str]:
        """
        简单关键词提取（基于频率）

        TODO: 可升级为 LLM 提取或 KeyBERT
        """
        # 简单停用词
        stopwords = {"的", "了", "是", "在", "我", "有", "和", "就", "不", "人",
                    "都", "一", "一个", "上", "也", "很", "到", "说", "要",
                    "去", "你", "会", "着", "没有", "看", "好", "自己", "这"}

        # 分字/词统计
        words = []
        for char in text:
            if char not in stopwords and len(char.strip()) > 0:
                words.append(char)

        # 取最常见的
        from collections import Counter
        counter = Counter(words)
        return [word for word, _ in counter.most_common(top_n)]

    async def retrieve(
        self,
        query: str,
        top_k: int = 3,
        session_filter: Optional[str] = None,
    ) -> list[MemoryEntry]:
        """
        检索相关记忆

        Args:
            query: 查询内容
            top_k: 返回数量
            session_filter: 可选，按会话 ID 过滤

        Returns:
            相关记忆列表
        """
        if not self.config.enabled:
            return []

        try:
            self._ensure_vector_store()

            if self._vector_store is None:
                # 内存模式
                memories = getattr(self, "_memories", [])
                # 简单过滤
                if session_filter:
                    memories = [m for m in memories if m.session_id == session_filter]
                return memories[:top_k]

            # 向量检索
            docs = self._vector_store.similarity_search(query, k=top_k)

            entries = [MemoryEntry.from_document(doc) for doc in docs]

            # 按会话过滤
            if session_filter:
                entries = [e for e in entries if e.session_id == session_filter]

            return entries

        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []

    def get_all(self) -> list[MemoryEntry]:
        """获取所有记忆"""
        try:
            self._ensure_vector_store()

            if self._vector_store is None:
                return getattr(self, "_memories", [])

            docs = self._vector_store.similarity_search("*", k=1000)
            return [MemoryEntry.from_document(doc) for doc in docs]

        except Exception as e:
            logger.error(f"Failed to get all memories: {e}")
            return []

    def clear(self) -> None:
        """清空长期记忆"""
        self._memories = []
        self._vector_store = None
        logger.info("LongTermMemory cleared")


def create_long_term_memory(
    embedder=None,
    memory_db_path: str = "./vector_db/memory_index",
) -> LongTermMemory:
    """
    工厂函数：创建长期记忆实例
    """
    config = LongTermConfig(
        memory_db_path=memory_db_path,
    )
    return LongTermMemory(config=config, embedder=embedder)
