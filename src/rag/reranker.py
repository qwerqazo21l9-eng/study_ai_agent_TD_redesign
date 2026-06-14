"""
Reranker 模块 - 使用交叉编码器对初筛结果进行精排
"""

from typing import Optional
from langchain_core.documents import Document

from src.utils.config_loader import config
from src.utils.logger import logger


class Reranker:
    """
    Reranker使用CrossEncoder对候选文档进行精排
    
    流程：向量检索Top-N(初筛) → Reranker精排Top-K(精筛) → 返回最终结果
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cpu"
    ):
        """
        Args:
            model_name: HuggingFace模型名，默认使用config中的reranker_model
            device: 运行设备，"cpu" 或 "cuda"
        """
        self.model_name = model_name or config.rag.get("reranker_model", "BAAI/bge-reranker-base")
        self.device = device
        self._model = None
        self._tokenizer = None
        self._is_available = True
        
        logger.info(f"Reranker initialized with model: {self.model_name}")
    
    def _load_model(self):
        """延迟加载模型"""
        if self._model is not None:
            return
        
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device=self.device)
            logger.info(f"CrossEncoder model loaded successfully")
        except ImportError:
            logger.warning("sentence_transformers not installed, reranker disabled")
            logger.warning("Install with: pip install sentence_transformers")
            self._is_available = False
        except Exception as e:
            logger.error(f"Failed to load CrossEncoder model: {e}")
            self._is_available = False
    
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5
    ) -> list[tuple[str, float]]:
        """
        对文档列表进行精排
        
        Args:
            query: 查询文本
            documents: 文档文本列表
            top_k: 返回前K个结果
        
        Returns:
            [(文档文本, 分数), ...] 按分数从高到低排序
        """
        if not documents:
            return []
        
        self._load_model()
        
        if not self._is_available or self._model is None:
            # 模型不可用时，返回原始顺序（假设documents已按相关性排序）
            logger.warning("Reranker not available, returning original order")
            return [(doc, 1.0 - i * 0.1) for i, doc in enumerate(documents[:top_k])]
        
        try:
            # 构建(query, document)配对
            pairs = [(query, doc) for doc in documents]
            
            # 批量预测
            scores = self._model.predict(pairs)
            
            # 按分数排序
            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            return doc_scores[:top_k]
        
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return [(doc, 1.0) for doc in documents[:top_k]]
    
    def rerank_with_metadata(
        self,
        query: str,
        docs_and_scores: list[tuple[Document, float]],
        top_k: int = 5
    ) -> list[tuple[Document, float]]:
        """
        对带有metadata的Document对象进行精排
        
        Args:
            query: 查询文本
            docs_and_scores: [(Document, 原始分数), ...]
            top_k: 返回前K个结果
        
        Returns:
            [(Document, 精排分数), ...] 按分数从高到低排序
        """
        if not docs_and_scores:
            return []
        
        # 提取文档文本
        documents = [doc.page_content for doc, _ in docs_and_scores]
        
        # 精排
        doc_scores = self.rerank(query, documents, top_k=top_k)
        
        # 重新组装为(Document, 分数)格式
        doc_text_to_doc = {doc.page_content: doc for doc, _ in docs_and_scores}
        
        result = []
        for doc_text, score in doc_scores:
            if doc_text in doc_text_to_doc:
                result.append((doc_text_to_doc[doc_text], float(score)))
        
        return result
    
    def is_available(self) -> bool:
        """检查Reranker是否可用"""
        self._load_model()
        return self._is_available


# 全局单例
_reranker_instance: Optional[Reranker] = None


def get_reranker() -> Reranker:
    """获取Reranker单例"""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = Reranker()
    return _reranker_instance
