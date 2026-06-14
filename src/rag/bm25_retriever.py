"""
BM25 检索模块 - 基于关键词的稀疏检索

BM25 是一种经典的信息检索算法，与向量检索互补：
- 向量检索：语义相似性
- BM25 检索：关键词匹配

Hybrid RAG = 向量检索 + BM25 + RRF 融合
"""

import pickle
from pathlib import Path
from typing import Optional, List
import jieba
from rank_bm25 import BM25Okapi

from langchain_core.documents import Document
from src.utils.logger import logger


class BM25Retriever:
    """
    BM25 检索器

    特点：
    1. 基于 jieba 中文分词
    2. 支持持久化存储
    3. 与 FAISS 向量库互补

    使用场景：
    - 专有名词、术语匹配
    - 代码片段检索
    - 精确关键词查询
    """

    def __init__(self, persist_path: Optional[str] = None):
        """
        Args:
            persist_path: BM25 索引持久化路径
        """
        self.persist_path = Path(persist_path) if persist_path else None
        self.bm25: Optional[BM25Okapi] = None
        self.corpus: list[Document] = []
        self.tokenized_corpus: list[list[str]] = []
        self._initialized = False

    def _tokenize(self, text: str) -> list[str]:
        """
        中文分词

        Args:
            text: 输入文本

        Returns:
            分词后的 token 列表
        """
        return list(jieba.cut(text))

    def build_index(self, documents: list[Document]) -> None:
        """
        构建 BM25 索引

        Args:
            documents: Document 列表
        """
        if not documents:
            logger.warning("No documents to build BM25 index")
            return

        self.corpus = documents
        self.tokenized_corpus = [self._tokenize(doc.page_content) for doc in documents]

        # 构建 BM25 索引
        # BM25Okapi 使用默认参数：k1=1.5, b=0.75
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        self._initialized = True
        logger.info(f"BM25 index built with {len(documents)} documents")

    def add_documents(self, documents: list[Document]) -> None:
        """
        向现有索引添加文档（增量构建）

        Args:
            documents: 新增的 Document 列表
        """
        if not documents:
            return

        # 如果未初始化，先构建
        if not self._initialized:
            self.build_index(documents)
            return

        # 增量添加
        self.corpus.extend(documents)
        new_tokens = [self._tokenize(doc.page_content) for doc in documents]
        self.tokenized_corpus.extend(new_tokens)

        # 重新构建索引（BM25 不支持增量）
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info(f"BM25 index updated: {len(self.corpus)} total documents")

    def search(self, query: str, top_k: int = 20) -> list[tuple[Document, float]]:
        """
        BM25 检索

        Args:
            query: 查询文本
            top_k: 返回前 k 个结果

        Returns:
            [(Document, score), ...] 按 BM25 分数降序排列
        """
        if not self._initialized or not self.bm25:
            logger.warning("BM25 index not initialized, returning empty results")
            return []

        # 分词查询
        query_tokens = self._tokenize(query)

        # 计算 BM25 分数
        scores = self.bm25.get_scores(query_tokens)

        # 包装为 (doc, score) 并排序
        doc_scores = list(zip(self.corpus, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # 返回 top_k
        results = doc_scores[:top_k]

        # 过滤零分结果
        results = [(doc, float(score)) for doc, score in results if score > 0]

        logger.info(f"BM25 search: query='{query[:30]}...', found {len(results)} results")
        return results

    def search_with_score_threshold(
        self, query: str, top_k: int = 20, min_score: float = 0.0
    ) -> list[tuple[Document, float]]:
        """
        带分数阈值的 BM25 检索

        Args:
            query: 查询文本
            top_k: 返回前 k 个结果
            min_score: 最小分数阈值

        Returns:
            [(Document, score), ...]
        """
        results = self.search(query, top_k)
        return [(doc, score) for doc, score in results if score >= min_score]

    def save(self, path: Optional[str] = None) -> None:
        """
        保存 BM25 索引到磁盘

        Args:
            path: 保存路径，默认使用初始化时的 persist_path
        """
        save_path = Path(path) if path else self.persist_path
        if not save_path:
            logger.warning("No persist path specified, skipping save")
            return

        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "corpus": [(doc.page_content, doc.metadata) for doc in self.corpus],
            "tokenized_corpus": self.tokenized_corpus,
        }

        with open(save_path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"BM25 index saved to {save_path}")

    def load(self, path: Optional[str] = None) -> bool:
        """
        从磁盘加载 BM25 索引

        Args:
            path: 加载路径，默认使用初始化时的 persist_path

        Returns:
            是否加载成功
        """
        load_path = Path(path) if path else self.persist_path
        if not load_path or not load_path.exists():
            logger.warning(f"BM25 index not found at {load_path}")
            return False

        try:
            with open(load_path, "rb") as f:
                data = pickle.load(f)

            # 重建 Document 对象
            self.corpus = [
                Document(page_content=content, metadata=metadata)
                for content, metadata in data["corpus"]
            ]
            self.tokenized_corpus = data["tokenized_corpus"]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            self._initialized = True

            logger.info(f"BM25 index loaded from {load_path}: {len(self.corpus)} documents")
            return True

        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            return False

    @property
    def is_initialized(self) -> bool:
        """检查索引是否已初始化"""
        return self._initialized and self.bm25 is not None

    def __len__(self) -> int:
        """返回索引中的文档数量"""
        return len(self.corpus)


class BM25Cache:
    """
    BM25 索引缓存管理器

    用于管理多个 BM25 索引（如不同文档集合的索引）
    """

    def __init__(self):
        self._cache: dict[str, BM25Retriever] = {}

    def get_or_create(
        self, key: str, persist_path: Optional[str] = None
    ) -> BM25Retriever:
        """
        获取或创建 BM25 索引

        Args:
            key: 索引标识
            persist_path: 持久化路径

        Returns:
            BM25Retriever 实例
        """
        if key not in self._cache:
            self._cache[key] = BM25Retriever(persist_path)
        return self._cache[key]

    def remove(self, key: str) -> bool:
        """移除指定索引"""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """清空所有缓存"""
        self._cache.clear()

    def keys(self) -> list[str]:
        """返回所有缓存的键"""
        return list(self._cache.keys())


# 全局缓存实例
_bm25_cache = BM25Cache()


def get_bm25_cache() -> BM25Cache:
    """获取全局 BM25 缓存"""
    return _bm25_cache
