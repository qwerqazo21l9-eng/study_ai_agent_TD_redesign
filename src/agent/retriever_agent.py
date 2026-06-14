"""
Retriever Agent - Hybrid RAG + 工具调用

职责：
1. 封装 Hybrid RAG 检索
2. 协调联网搜索（可选）
3. 拼接检索上下文
"""

from typing import Optional

from src.agent.schema import AgentState, SearchResult, QueryType
from src.rag.vector_db_manager import VectorDBManager
from src.utils.logger import logger


class RetrieverAgent:
    """
    Retriever Agent

    负责：
    1. 执行 Hybrid RAG 检索
    2. 调用联网搜索（当需要时）
    3. 合并多源结果
    4. 生成带上下文的检索结果
    """

    def __init__(self):
        self.vector_db: Optional[VectorDBManager] = None

    def _get_vector_db(self) -> VectorDBManager:
        """延迟加载向量库"""
        if self.vector_db is None:
            self.vector_db = VectorDBManager()
        return self.vector_db

    async def retrieve(self, state: AgentState) -> AgentState:
        """
        执行检索

        Args:
            state: 当前状态（包含 query 和 intent）

        Returns:
            更新后的状态（包含 search_results 和 context）
        """
        query = state["query"]
        intent = state.get("intent")
        query_type = state.get("query_type", QueryType.KNOWLEDGE)

        logger.info(f"Retriever processing query: {query[:50]}...")

        try:
            # 1. 知识库检索
            logger.info("Retriever: starting knowledge base search")
            search_results = await self._retrieve_from_knowledge_base(query, intent)
            logger.info(f"Retriever: KB search returned {type(search_results)}")
            state["search_results"] = search_results

            # 2. 联网搜索（如果需要）
            if query_type in [QueryType.WEB_SEARCH, QueryType.HYBRID]:
                web_results = await self._retrieve_from_web(query, intent)
                state["web_results"] = web_results

            # 3. 拼接上下文
            logger.info("Retriever: building context")
            context = self._build_context(state)
            state["context"] = context
            logger.info(f"Retriever: context built, length={len(context) if context else 0}")

            logger.info(f"Retriever done: {len(search_results or [])} knowledge results, "
                       f"{len(state.get('web_results') or [])} web results")

        except Exception as e:
            import traceback
            logger.error(f"Retrieval failed: {e}\n{traceback.format_exc()}")
            state["error"] = f"检索失败: {str(e)}"
            state["context"] = ""
            state["search_results"] = []

        return state

    async def _retrieve_from_knowledge_base(
        self,
        query: str,
        intent
    ) -> list[SearchResult]:
        """
        从知识库检索

        Args:
            query: 查询文本
            intent: 意图分析结果

        Returns:
            检索结果列表
        """
        try:
            db = self._get_vector_db()

            # 使用 Hybrid 检索
            if db.use_hybrid:
                results = db.hybrid_search(query)
            else:
                results = db.similarity_search(query)
                # 转换为统一格式
                from langchain_core.documents import Document
                results = [(doc, score, {}) for doc, score in results]

            # 转换为 SearchResult
            search_results = []
            for doc, score, details in results:
                source = doc.metadata.get("source", "Unknown")
                search_results.append(SearchResult(
                    document=doc,
                    score=score,
                    source=source,
                    rank_info=details
                ))

            # 详细日志：记录粗排和精排结果
            logger.info(f"Knowledge base: {len(search_results) if search_results else 0} results")
            for i, sr in enumerate(search_results[:5]):
                content_preview = sr.document.page_content[:80] if sr.document else "N/A"
                logger.info(f"  [精排{i+1}] score={sr.score:.4f} source={sr.source} | {content_preview}...")
            
            return search_results

        except Exception as e:
            logger.error(f"Knowledge base retrieval failed: {e}")
            return []

    async def _retrieve_from_web(
        self,
        query: str,
        intent
    ) -> list[dict]:
        """
        从网络检索

        TODO: 接入 WebSearch 工具

        Args:
            query: 查询文本
            intent: 意图分析结果

        Returns:
            网络检索结果列表
        """
        # TODO: 接入 MCP WebSearch 工具
        # 目前返回空列表占位
        logger.info("Web search not yet implemented, returning empty results")
        return []

    def _build_context(self, state: AgentState) -> str:
        """
        构建检索上下文

        Args:
            state: 包含检索结果的状态

        Returns:
            格式化的上下文字符串
        """
        context_parts = []
        search_results = state.get("search_results") or []

        # 1. 知识库上下文
        if search_results:
            kb_context = self._format_knowledge_context(search_results)
            logger.info(f"_build_context: KB context length={len(kb_context)}, preview={kb_context[:200]}...")
            context_parts.append("【知识库检索结果】\n" + kb_context)
        else:
            logger.warning("_build_context: no search results!")

        # 2. 网络上下文
        web_results = state.get("web_results") or []
        if web_results:
            web_context = self._format_web_context(web_results)
            context_parts.append("【网络检索结果】\n" + web_context)

        if not context_parts:
            return "【未找到相关检索结果】"

        return "\n\n".join(context_parts)

    def _format_knowledge_context(self, results: list[SearchResult]) -> str:
        """格式化知识库检索结果"""
        parts = []

        for i, result in enumerate(results, 1):
            doc = result.document
            source = doc.metadata.get("source", "未知来源")
            section = doc.metadata.get("section", "")
            content = doc.page_content

            # 构建来源头
            header = f"[来源 {i}: {source}"
            if section:
                header += f" | {section}"
            header += "]"

            # 相关度（reranker 分数越大越相关，范围 0-1）
            relevance = max(0, min(1, result.score))
            header += f" (相关度: {relevance:.2f})"

            parts.append(f"{header}\n{content}")

        return "\n\n---\n\n".join(parts)

    def _format_web_context(self, results: list[dict]) -> str:
        """格式化网络检索结果"""
        parts = []

        for i, result in enumerate(results, 1):
            title = result.get("title", "无标题")
            url = result.get("url", "")
            snippet = result.get("snippet", "")

            header = f"[网页 {i}: {title}]"
            if url:
                header += f"\n链接: {url}"

            parts.append(f"{header}\n{snippet}")

        return "\n\n---\n\n".join(parts)

    def should_fallback_to_chat(self, state: AgentState) -> bool:
        """
        判断是否应该降级为闲聊

        当检索结果太少或置信度太低时

        Args:
            state: 当前状态

        Returns:
            是否应该降级
        """
        search_results = state.get("search_results") or []
        intent = state.get("intent")

        # 结果太少
        if len(search_results) < 1:
            # 如果不是需要联网的查询，降级为闲聊
            if state.get("query_type") not in [QueryType.WEB_SEARCH, QueryType.HYBRID]:
                return True

        # 置信度极低
        if intent and intent.confidence < 0.3:
            return True

        return False
