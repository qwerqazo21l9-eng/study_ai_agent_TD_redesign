from pathlib import Path
from typing import Optional
from langchain_core.documents import Document

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.rag.document_processor import DocumentProcessor
from src.rag.deduplicator import DocumentRegistry, ChunkDeduplicator, create_deduplicators
from src.rag.reranker import Reranker, get_reranker
from src.rag.bm25_retriever import BM25Retriever
from src.rag.fusion import RRFusion
from src.utils.config_loader import config
from src.utils.logger import logger


class VectorDBManager:
    """
    向量库管理器 - 集成去重、语义分块、Reranker、Hybrid RAG

    改进点：
    1. 文件级去重（SHA256）
    2. Chunk级去重（SimHash）
    3. Reranker精排
    4. metadata增强
    5. Hybrid RAG（向量 + BM25 + RRF 融合）
    """

    def __init__(self):
        # 初始化Embedding模型
        model_name = config.model["embedding_model_name"]

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"}
        )

        self.vector_db_path = Path(config.rag["vector_db_path"])
        self.processor = DocumentProcessor()

        # 初始化去重器
        self._init_dedup()

        # 初始化Reranker（延迟加载）
        self.reranker: Optional[Reranker] = None

        # 加载或创建向量库
        self.vector_db = self._load_or_create_db()

        # 配置参数
        self.dedup_enabled = config.rag.get("dedup_enabled", True)
        self.use_reranker = config.rag.get("use_reranker", True)
        self.initial_k = config.rag.get("initial_k", 20)
        self.final_k = config.rag.get("top_k", 5)
        self.similarity_threshold = config.rag.get("similarity_threshold", 1.2)

        # Hybrid RAG 配置
        self.use_hybrid = config.rag.get("use_hybrid", True)
        self.bm25_top_k = config.rag.get("bm25_top_k", 30)
        self.rrf_k = config.rag.get("rrf_k", 60)
        self.hybrid_weight_vector = config.rag.get("hybrid_weight_vector", [0.5, 0.5])

        # 初始化 BM25 和 RRF（延迟加载）
        self._bm25: Optional[BM25Retriever] = None
        self._rrf_fuser: Optional[RRFusion] = None

        # 尝试加载已有 BM25 索引
        self._init_bm25()
    
    def _init_dedup(self):
        """初始化去重器"""
        try:
            self.doc_registry, self.chunk_dedup = create_deduplicators()
            logger.info("Deduplication initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize deduplication: {e}")
            # 降级：使用简单的注册表
            self.doc_registry = None
            self.chunk_dedup = None

    def _init_bm25(self) -> None:
        """初始化 BM25 索引"""
        if not self.use_hybrid:
            return

        bm25_path = self.vector_db_path.parent / "bm25_index.pkl"

        try:
            self._bm25 = BM25Retriever(persist_path=str(bm25_path))

            # 尝试加载已有索引
            if bm25_path.exists():
                if self._bm25.load():
                    logger.info(f"BM25 index loaded: {len(self._bm25)} documents")
                else:
                    # 加载失败，构建新索引
                    self._build_bm25_index()
            else:
                # 索引不存在，构建新索引
                self._build_bm25_index()

        except Exception as e:
            logger.warning(f"Failed to initialize BM25: {e}")
            self._bm25 = None

    def _build_bm25_index(self) -> None:
        """从向量库构建 BM25 索引"""
        if self._bm25 is None:
            bm25_path = self.vector_db_path.parent / "bm25_index.pkl"
            self._bm25 = BM25Retriever(persist_path=str(bm25_path))

        try:
            # 从 FAISS 提取所有文档
            # 注意：FAISS 不直接支持获取所有文档，需要通过 docstore
            if hasattr(self.vector_db, 'docstore') and self.vector_db.docstore:
                docs = []
                ids = self.vector_db.docstore._dict.keys()

                for doc_id in ids:
                    try:
                        doc = self.vector_db.docstore.search(doc_id)
                        if doc and hasattr(doc, 'page_content') and doc.page_content.strip():
                            docs.append(doc)
                    except Exception:
                        continue

                if docs:
                    self._bm25.build_index(docs)
                    self._bm25.save()
                    logger.info(f"BM25 index built from {len(docs)} documents")
        except Exception as e:
            logger.warning(f"Failed to build BM25 index: {e}")
    
    def _load_or_create_db(self):
        """加载或创建向量库"""
        if self.vector_db_path.exists():
            logger.info(f"Loading existing vector DB from {self.vector_db_path}")
            try:
                return FAISS.load_local(
                    str(self.vector_db_path),
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                logger.error(f"Failed to load existing DB, creating new one: {e}")
        
        logger.info("Creating new vector DB")
        # 新建空索引
        db = FAISS.from_texts([""], self.embedding_model)
        db.save_local(str(self.vector_db_path))
        return db
    
    def _get_reranker(self) -> Optional[Reranker]:
        """获取或创建Reranker（延迟加载）"""
        if self.reranker is None and self.use_reranker:
            try:
                self.reranker = get_reranker()
                if not self.reranker.is_available():
                    logger.warning("Reranker not available, continuing without it")
                    self.reranker = None
            except Exception as e:
                logger.warning(f"Failed to create Reranker: {e}")
                self.reranker = None
        return self.reranker
    
    def add_file_to_db(self, file_path: Path) -> bool:
        """
        添加文件到向量库（带去重）
        
        流程：
        1. 文件级去重检查
        2. 文档加载
        3. 语义分块
        4. Chunk级去重
        5. 向量化入库
        6. 注册到文档注册表
        """
        try:
            file_path = Path(file_path)
            
            # 1. 文件级去重检查
            if self.dedup_enabled and self.doc_registry:
                if self.doc_registry.is_duplicate(file_path):
                    logger.warning(f"File already exists in database: {file_path.name}")
                    return False
            
            # 2. 文档处理
            split_docs = self.processor.process_file(file_path)
            if not split_docs:
                logger.error("No documents to add")
                return False
            
            logger.info(f"Processing {file_path.name}: {len(split_docs)} chunks before dedup")
            
            # 3. Chunk级去重
            if self.dedup_enabled and self.chunk_dedup:
                original_count = len(split_docs)
                split_docs = self.chunk_dedup.deduplicate(split_docs)
                logger.info(f"Chunk dedup: {original_count} -> {len(split_docs)}")
            
            if not split_docs:
                logger.warning("All chunks were duplicates")
                return False
            
            # 打印调试信息
            logger.info(f"Adding {len(split_docs)} chunks to vector DB")
            for i, doc in enumerate(split_docs[:3]):
                logger.info(f"Chunk {i}: {doc.page_content[:100]}...")
            
            # 4. 添加到向量库
            self.vector_db.add_documents(split_docs)
            
            # 5. 保存到磁盘
            self.vector_db.save_local(str(self.vector_db_path))
            logger.info(f"Vector DB saved to {self.vector_db_path}")
            
            # 6. 注册到文档注册表
            if self.dedup_enabled and self.doc_registry:
                chunk_ids = [doc.metadata.get("chunk_id", f"chunk_{i}")
                            for i, doc in enumerate(split_docs)]
                self.doc_registry.register(file_path, chunk_ids)

            # 7. 更新 BM25 索引
            if self.use_hybrid and self._bm25 is not None:
                self._bm25.add_documents(split_docs)
                self._bm25.save()
                logger.info(f"BM25 index updated with {len(split_docs)} new chunks")

            return True

        except Exception as e:
            logger.error(f"Failed to add file: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def similarity_search(self, query: str) -> list[tuple[Document, float]]:
        """
        检索相关文档（带Reranker精排）
        
        流程：
        1. 向量检索初筛Top-N
        2. Reranker精排Top-K
        3. 阈值过滤
        """
        reranker = self._get_reranker()
        
        # 1. 初筛
        if reranker:
            # 使用更大的initial_k进行初筛
            docs_and_scores = self.vector_db.similarity_search_with_score(
                query, k=self.initial_k
            )
            logger.info(f"Initial retrieval: {len(docs_and_scores) if docs_and_scores else 0} candidates")
            
            # 2. Reranker精排
            if reranker.is_available() and docs_and_scores:
                docs_and_scores = reranker.rerank_with_metadata(
                    query, docs_and_scores, top_k=self.final_k
                )
                logger.info(f"After reranking: {len(docs_and_scores)} results")
        else:
            # 不使用Reranker时，直接用top_k
            docs_and_scores = self.vector_db.similarity_search_with_score(
                query, k=self.final_k
            )
            logger.info(f"Retrieval (no reranker): {len(docs_and_scores)} results")
        
        # 打印每个结果的score
        for doc, score in docs_and_scores:
            logger.info(f"检索结果：score={score:.4f} | 内容={doc.page_content[:50]}...")
        
        # 3. 阈值过滤
        filtered = [
            (doc, score) for doc, score in docs_and_scores
            if score < self.similarity_threshold
        ]
        
        logger.info(f"过滤后找到 {len(filtered)} 条结果 (原始 {len(docs_and_scores)})")
        return filtered

    def hybrid_search(self, query: str) -> list[tuple[Document, float, dict]]:
        """
        Hybrid RAG 检索 - 向量 + BM25 + RRF 融合

        流程：
        1. 向量检索初筛 Top-N
        2. BM25 检索 Top-N
        3. RRF 融合两路结果
        4. Reranker 精排 Top-K
        5. 阈值过滤

        Args:
            query: 查询文本

        Returns:
            [(Document, score, details), ...]
            details 包含 RRF 排名信息和各路原始分数
        """
        if not self.use_hybrid or self._bm25 is None or not self._bm25.is_initialized:
            # 降级到纯向量检索
            logger.warning("BM25 not available, falling back to vector search")
            results = self.similarity_search(query)
            return [(doc, score, {"source": "vector_only"}) for doc, score in results]

        # 1. 向量检索
        vector_results = self.vector_db.similarity_search_with_score(
            query, k=self.bm25_top_k
        )
        logger.info(f"Vector retrieval: {len(vector_results)} candidates")

        # 2. BM25 检索
        bm25_results = self._bm25.search(query, top_k=self.bm25_top_k)
        logger.info(f"BM25 retrieval: {len(bm25_results)} candidates")

        # 3. RRF 融合
        if self._rrf_fuser is None:
            self._rrf_fuser = RRFusion(k=self.rrf_k)

        fused_results = self._rrf_fuser.fuse(
            results_list=[vector_results, bm25_results],
            top_k=self.initial_k,
            weight_vector=self.hybrid_weight_vector,
        )
        logger.info(f"RRF fusion: {len(fused_results)} fused results")
        
        # 粗排日志：显示 RRF 融合后的结果
        for i, (doc, _, details) in enumerate(fused_results[:5]):
            rrf_score = details.get("rrf_score", 0)
            source = doc.metadata.get("source", "Unknown")
            content = doc.page_content[:60] if doc.page_content else "N/A"
            logger.info(f"  [粗排{i+1}] rrf={rrf_score:.4f} source={source} | {content}...")

        # 4. Reranker 精排
        reranker = self._get_reranker()
        if reranker and reranker.is_available():
            # 准备 RRF 结果为 reranker 格式
            docs_and_scores = [(doc, details["rrf_score"]) for doc, _, details in fused_results]
            reranked = reranker.rerank_with_metadata(query, docs_and_scores, top_k=self.final_k)
            # 重建带 details 的结果
            doc_map = {self._rrf_fuser._get_doc_id(doc): details for doc, _, details in fused_results}
            fused_results = [
                (doc, score, doc_map.get(self._rrf_fuser._get_doc_id(doc), {}))
                for doc, score in reranked
            ]
            logger.info(f"After reranking: {len(fused_results)} results")

        # 5. 阈值过滤
        filtered = [
            (doc, score, details)
            for doc, score, details in fused_results
            if score < self.similarity_threshold
        ]

        logger.info(f"Hybrid search final: {len(filtered)} results")
        
        # 详细日志：显示最终结果
        for i, (doc, score, details) in enumerate(filtered[:5]):
            source = doc.metadata.get("source", "Unknown")
            content = doc.page_content[:60] if doc.page_content else "N/A"
            logger.info(f"  [最终{i+1}] score={score:.4f} source={source} | {content}...")
        
        return filtered

    def hybrid_search_with_context(
        self,
        query: str,
        max_context_length: int = 4000
    ) -> str:
        """
        Hybrid 检索并拼接为上下文字符串（带来源标注）

        Args:
            query: 查询文本
            max_context_length: 最大上下文长度

        Returns:
            带引用标注的上下文字符串
        """
        results = self.hybrid_search(query)

        if not results:
            return ""

        context_parts = []
        current_length = 0

        for doc, score, details in results:
            source = doc.metadata.get("source", "Unknown")
            chunk_id = doc.metadata.get("chunk_id", "")
            section = doc.metadata.get("section", "")
            content = doc.page_content

            # 构建来源信息
            header_parts = [f"来源: {source}"]
            if section:
                header_parts.append(f"章节: {section}")

            # 添加 RRF 信息
            if "ranks" in details:
                ranks = details.get("ranks", {})
                vector_rank = ranks.get(0, "-")  # 向量检索排名
                bm25_rank = ranks.get(1, "-")  # BM25 排名
                header_parts.append(f"向量排名: {vector_rank}, BM25排名: {bm25_rank}")

            header = "[" + " | ".join(header_parts) + "]\n"

            part = header + content + "\n"

            if current_length + len(part) > max_context_length:
                break

            context_parts.append(part)
            current_length += len(part)

        return "\n---\n".join(context_parts)
    
    def similarity_search_with_context(
        self, 
        query: str, 
        max_context_length: int = 4000
    ) -> str:
        """
        检索并拼接为上下文字符串
        
        Args:
            query: 查询文本
            max_context_length: 最大上下文长度（字符数）
        
        Returns:
            拼接的上下文字符串
        """
        results = self.similarity_search(query)
        
        if not results:
            return ""
        
        context_parts = []
        current_length = 0
        
        for doc, score in results:
            source = doc.metadata.get("source", "Unknown")
            chunk_id = doc.metadata.get("chunk_id", "")
            section = doc.metadata.get("section", "")
            content = doc.page_content
            
            # 相关度转换为0-1范围（越小越相似）
            relevance = max(0, 1 - score / 2)
            
            header = f"[来源: {source}"
            if section:
                header += f" | 章节: {section}"
            header += f" | 相关度: {relevance:.2f}]\n"
            
            part = header + content + "\n"
            
            if current_length + len(part) > max_context_length:
                break
            
            context_parts.append(part)
            current_length += len(part)
        
        return "\n---\n".join(context_parts)
    
    def get_stats(self) -> dict:
        """获取向量库统计信息"""
        try:
            index = self.vector_db.index
            total_vectors = index.ntotal if hasattr(index, 'ntotal') else 0

            return {
                "total_vectors": total_vectors,
                "dedup_enabled": self.dedup_enabled,
                "use_reranker": self.use_reranker,
                "reranker_available": self._get_reranker().is_available() if self._get_reranker() else False,
                "db_path": str(self.vector_db_path),
                # Hybrid RAG 统计
                "use_hybrid": self.use_hybrid,
                "bm25_docs": len(self._bm25) if self._bm25 else 0,
                "bm25_initialized": self._bm25.is_initialized if self._bm25 else False,
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}
    
    def reset(self):
        """重置向量库（谨慎使用）"""
        self.vector_db = FAISS.from_texts([""], self.embedding_model)
        self.vector_db.save_local(str(self.vector_db_path))
        
        # 重置去重器
        if self.chunk_dedup:
            self.chunk_dedup.reset()
        
        logger.warning("Vector DB and deduplicators reset")
