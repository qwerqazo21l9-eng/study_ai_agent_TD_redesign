"""
检索模块 - 整合Query改写和向量检索
"""

from typing import Optional
from langchain_core.documents import Document

from src.utils.config_loader import config
from src.utils.logger import logger


class QueryRewriter:
    """
    Query改写器 - 将口语化查询转换为更适合检索的形式
    
    支持的策略：
    - HyDE (Hypothetical Document Embeddings)
    - Query Decomposition (查询分解)
    """
    
    def __init__(self, strategy: str = "hyde"):
        """
        Args:
            strategy: 改写策略，"hyde" 或 "decompose"
        """
        self.strategy = strategy
        self._llm = None
        logger.info(f"QueryRewriter initialized with strategy: {strategy}")
    
    def _get_llm(self):
        """获取LLM实例"""
        if self._llm is None:
            try:
                from langchain_community.chat_models import ChatZhipuAI
                api_key = config.model.get("cloud_api_key", "")
                api_base = config.model.get("cloud_api_base", "")
                model_name = config.model.get("cloud_model_name", "glm-4-flash")
                
                self._llm = ChatZhipuAI(
                    zhipu_api_key=api_key,
                    api_base=api_base,
                    model=model_name,
                    temperature=0.3
                )
            except Exception as e:
                logger.warning(f"Failed to initialize LLM for query rewriting: {e}")
        return self._llm
    
    def rewrite(self, query: str) -> str:
        """
        改写查询
        
        Args:
            query: 原始查询
        
        Returns:
            改写后的查询
        """
        if self.strategy == "hyde":
            return self._rewrite_hyde(query)
        elif self.strategy == "decompose":
            return self._rewrite_decompose(query)
        else:
            return query
    
    def _rewrite_hyde(self, query: str) -> str:
        """
        HyDE策略：让LLM生成一个假设性答案，然后检索这个答案
        """
        llm = self._get_llm()
        if llm is None:
            logger.warning("LLM not available for HyDE, returning original query")
            return query
        
        prompt = f"""你是一个问答系统的查询优化器。
给定一个用户问题，生成一个理想的回答内容。
这个回答应该清晰、准确地回答问题。

用户问题：{query}

请生成一个简短（100字以内）的理想回答："""
        
        try:
            response = llm.invoke(prompt)
            rewritten = response.content.strip()
            logger.info(f"HyDE rewrite: '{query}' -> '{rewritten[:50]}...'")
            return rewritten
        except Exception as e:
            logger.error(f"HyDE rewrite failed: {e}")
            return query
    
    def _rewrite_decompose(self, query: str) -> str:
        """
        查询分解策略：将复杂查询分解为多个简单子查询
        """
        llm = self._get_llm()
        if llm is None:
            return query
        
        prompt = f"""你是一个查询优化器。将复杂问题分解为多个简单的子问题。

用户问题：{query}

将问题分解为1-3个关键词或简短问题，用逗号分隔："""
        
        try:
            response = llm.invoke(prompt)
            rewritten = response.content.strip()
            # 提取关键词（去除编号、特殊符号）
            import re
            keywords = re.sub(r'^\d+[.、]?\s*', '', rewritten, flags=re.MULTILINE)
            keywords = re.sub(r'[^\w\s,，]', '', keywords)
            logger.info(f"Query decomposition: '{query}' -> '{keywords}'")
            return keywords
        except Exception as e:
            logger.error(f"Query decomposition failed: {e}")
            return query


class Retriever:
    """
    检索器 - 整合Query改写和向量检索
    
    检索流程：
    1. Query改写（可选）
    2. 向量检索初筛
    3. Reranker精排
    """
    
    def __init__(self, vector_db, use_query_rewrite: bool = False):
        """
        Args:
            vector_db: FAISS向量库实例
            use_query_rewrite: 是否启用Query改写
        """
        self.vector_db = vector_db
        self.use_query_rewrite = use_query_rewrite
        
        # Query改写器
        self.query_rewriter = QueryRewriter() if use_query_rewrite else None
        
        # Reranker
        self.reranker = None
        
        # 配置参数
        self.use_reranker = config.rag.get("use_reranker", False)
        self.initial_k = config.rag.get("initial_k", 20)
        self.final_k = config.rag.get("top_k", 5)
        self.similarity_threshold = config.rag.get("similarity_threshold", 1.2)
        
        logger.info(f"Retriever initialized: use_query_rewrite={use_query_rewrite}, use_reranker={self.use_reranker}")
    
    def _init_reranker(self):
        """延迟初始化Reranker"""
        if self.reranker is None and self.use_reranker:
            try:
                from src.rag.reranker import Reranker
                self.reranker = Reranker()
            except Exception as e:
                logger.warning(f"Failed to initialize reranker: {e}")
                self.reranker = None
    
    def retrieve(self, query: str) -> list[tuple[Document, float]]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
        
        Returns:
            [(Document, 分数), ...] 按相关性从高到低排序
        """
        # 1. Query改写（可选）
        if self.query_rewriter and self.use_query_rewrite:
            original_query = query
            query = self.query_rewriter.rewrite(query)
            if query != original_query:
                logger.info(f"Query rewritten: '{original_query}' -> '{query}'")
        
        # 2. 初筛：从向量库获取候选
        if self.use_reranker:
            # 使用更大的initial_k进行初筛
            initial_k = self.initial_k
        else:
            initial_k = self.final_k
        
        docs_and_scores = self.vector_db.similarity_search_with_score(query, k=initial_k)
        logger.info(f"Initial retrieval: {len(docs_and_scores)} candidates")
        
        # 3. Reranker精排（可选）
        if self.use_reranker and self.reranker:
            self._init_reranker()
            if self.reranker and self.reranker.is_available():
                docs_and_scores = self.reranker.rerank_with_metadata(
                    query, docs_and_scores, top_k=self.final_k
                )
                logger.info(f"After reranking: {len(docs_and_scores)} results")
        
        # 4. 阈值过滤
        filtered = [
            (doc, score) for doc, score in docs_and_scores
            if score < self.similarity_threshold
        ]
        
        logger.info(f"Final results after threshold filtering: {len(filtered)}")
        return filtered
    
    def retrieve_with_context(self, query: str, max_context_length: int = 4000) -> str:
        """
        检索并拼接为上下文字符串
        
        Args:
            query: 查询文本
            max_context_length: 最大上下文长度（字符数）
        
        Returns:
            拼接的上下文字符串
        """
        results = self.retrieve(query)
        
        if not results:
            return ""
        
        context_parts = []
        current_length = 0
        
        for doc, score in results:
            source = doc.metadata.get("source", "Unknown")
            chunk_id = doc.metadata.get("chunk_id", "")
            content = doc.page_content
            
            part = f"[来源: {source}, 相关度: {1-score:.2f}]\n{content}\n"
            
            if current_length + len(part) > max_context_length:
                break
            
            context_parts.append(part)
            current_length += len(part)
        
        return "\n---\n".join(context_parts)


# 全局单例
_retriever_instance: Optional[Retriever] = None


def get_retriever(vector_db, use_query_rewrite: bool = False) -> Retriever:
    """获取Retriever单例"""
    global _retriever_instance
    _retriever_instance = Retriever(vector_db, use_query_rewrite)
    return _retriever_instance
