"""
RRF 融合模块 - 多路检索结果融合

RRF (Reciprocal Rank Fusion) 是一种简单而有效的多路召回融合算法。

公式：RRF_score(d) = Σ 1/(k + rank_i(d))
其中：
- d 是文档
- k 是平滑参数（通常为 60）
- rank_i(d) 是文档 d 在第 i 路检索中的排名

特点：
1. 无需训练
2. 对各路检索的分数分布不敏感
3. 只需排名信息
"""

from typing import Optional
from collections import defaultdict
from langchain_core.documents import Document

from src.utils.logger import logger


class RRFusion:
    """
    RRF 融合器

    支持多路检索结果融合：
    - 向量检索
    - BM25 检索
    - 可扩展其他检索方式
    """

    def __init__(self, k: int = 60):
        """
        Args:
            k: RRF 平滑参数。k 越大，各路检索的影响越均衡
               通常设置在 30-100 之间
        """
        self.k = k

    def fuse(
        self,
        results_list: list[list[tuple[Document, float]]],
        top_k: int = 5,
        weight_vector: Optional[list[float]] = None,
    ) -> list[tuple[Document, float, dict]]:
        """
        RRF 融合多路检索结果

        Args:
            results_list: 各路检索结果，格式为 [(doc, score), ...]
            top_k: 返回融合后的前 k 个结果
            weight_vector: 各路检索的权重，默认等权重

        Returns:
            [(Document, rrf_score, details), ...]
            details 包含各路检索的排名信息
        """
        if not results_list:
            logger.warning("Empty results list for fusion")
            return []

        # 权重处理
        n_results = len(results_list)
        if weight_vector is None:
            weight_vector = [1.0] * n_results
        else:
            assert len(weight_vector) == n_results, "Weight vector length mismatch"

        # 归一化权重
        total_weight = sum(weight_vector)
        if total_weight > 0:
            weight_vector = [w / total_weight for w in weight_vector]

        # 初始化文档分数映射
        doc_scores: dict[str, float] = defaultdict(float)
        doc_details: dict[str, dict] = defaultdict(
            lambda: {"ranks": {}, "scores": {}}
        )

        # 对每路检索结果计算 RRF 分数
        for i, results in enumerate(results_list):
            weight = weight_vector[i]

            for rank, (doc, score) in enumerate(results, start=1):
                # 生成文档唯一标识
                doc_id = self._get_doc_id(doc)

                # RRF 公式
                rrf_score = weight * (1 / (self.k + rank))

                # 累加多路分数
                doc_scores[doc_id] += rrf_score

                # 记录详情
                doc_details[doc_id]["ranks"][i] = rank
                doc_details[doc_id]["scores"][i] = score
                doc_details[doc_id]["document"] = doc

        # 按 RRF 分数排序
        sorted_docs = sorted(
            doc_scores.items(), key=lambda x: x[1], reverse=True
        )

        # 构建返回结果
        fused_results = []
        for doc_id, rrf_score in sorted_docs[:top_k]:
            doc = doc_details[doc_id]["document"]
            details = {
                "rrf_score": rrf_score,
                "ranks": doc_details[doc_id]["ranks"],
                "original_scores": doc_details[doc_id]["scores"],
            }
            fused_results.append((doc, rrf_score, details))

        logger.info(
            f"RRF fusion: {n_results} result sets, "
            f"{len(doc_scores)} unique docs, top {len(fused_results)} returned"
        )

        return fused_results

    def _get_doc_id(self, doc: Document) -> str:
        """
        生成文档唯一标识

        优先使用 chunk_id + source 组合
        """
        chunk_id = doc.metadata.get("chunk_id", "")
        source = doc.metadata.get("source", "")

        if chunk_id and source:
            return f"{source}::{chunk_id}"
        elif chunk_id:
            return chunk_id
        elif source:
            return source
        else:
            # 兜底：使用内容哈希
            return str(hash(doc.page_content))

    def fuse_with_deduplication(
        self,
        results_list: list[list[tuple[Document, float]]],
        top_k: int = 5,
        similarity_threshold: float = 0.9,
    ) -> list[tuple[Document, float, dict]]:
        """
        RRF 融合 + 去重

        去除语义或内容高度相似的文档

        Args:
            results_list: 各路检索结果
            top_k: 返回前 k 个
            similarity_threshold: 去重阈值（待实现）

        Returns:
            融合且去重后的结果
        """
        fused = self.fuse(results_list, top_k=top_k * 2)  # 多取一些用于去重

        # TODO: 实现内容去重
        # 目前先直接返回
        return fused[:top_k]


class ScoreLevelFusion:
    """
    基于分数的融合方法

    与 RRF 的区别：
    - RRF：只使用排名
    - ScoreLevelFusion：使用原始分数

    适用场景：
    - 需要考虑各路检索的分数差异
    - 分数已经过归一化处理
    """

    def __init__(self, method: str = "average"):
        """
        Args:
            method: 融合方法
                - "average": 简单平均
                - "weighted": 加权平均
                - "max": 取最大值
                - "min": 取最小值
        """
        self.method = method

    def fuse(
        self,
        results_list: list[list[tuple[Document, float]]],
        top_k: int = 5,
        weights: Optional[list[float]] = None,
        score_normalize: bool = True,
    ) -> list[tuple[Document, float, dict]]:
        """
        基于分数的融合

        Args:
            results_list: 各路检索结果
            top_k: 返回前 k 个
            weights: 权重列表
            score_normalize: 是否归一化分数
        """
        if not results_list:
            return []

        n_results = len(results_list)
        if weights is None:
            weights = [1.0 / n_results] * n_results

        # 归一化各路分数
        normalized_results = []
        for results in results_list:
            if not results:
                normalized_results.append([])
                continue

            scores = [score for _, score in results]
            min_s, max_s = min(scores), max(scores)
            range_s = max_s - min_s if max_s != min_s else 1

            normalized = []
            for doc, score in results:
                if score_normalize:
                    norm_score = (score - min_s) / range_s
                else:
                    norm_score = score
                normalized.append((doc, norm_score))

            normalized_results.append(normalized)

        # 分数融合
        doc_scores: dict[str, float] = defaultdict(float)
        doc_details: dict[str, dict] = defaultdict(
            lambda: {"scores": {}, "original_scores": {}}
        )

        for i, results in enumerate(normalized_results):
            weight = weights[i]
            for doc, score in results:
                doc_id = self._get_doc_id(doc)

                if self.method == "average" or self.method == "weighted":
                    doc_scores[doc_id] += weight * score
                elif self.method == "max":
                    doc_scores[doc_id] = max(doc_scores[doc_id], weight * score)
                elif self.method == "min":
                    if doc_scores[doc_id] == 0:
                        doc_scores[doc_id] = weight * score
                    else:
                        doc_scores[doc_id] = min(doc_scores[doc_id], weight * score)

                doc_details[doc_id]["document"] = doc
                doc_details[doc_id]["original_scores"][i] = score

        # 排序
        sorted_docs = sorted(
            doc_scores.items(), key=lambda x: x[1], reverse=True
        )

        fused_results = []
        for doc_id, fused_score in sorted_docs[:top_k]:
            doc = doc_details[doc_id]["document"]
            details = {
                "fused_score": fused_score,
                "original_scores": doc_details[doc_id]["original_scores"],
            }
            fused_results.append((doc, fused_score, details))

        return fused_results

    def _get_doc_id(self, doc: Document) -> str:
        """生成文档唯一标识"""
        chunk_id = doc.metadata.get("chunk_id", "")
        source = doc.metadata.get("source", "")
        if chunk_id and source:
            return f"{source}::{chunk_id}"
        elif chunk_id:
            return chunk_id
        elif source:
            return source
        else:
            return str(hash(doc.page_content))


def create_fuser(method: str = "rrf", **kwargs) -> "RRFusion | ScoreLevelFusion":
    """
    工厂函数：创建融合器

    Args:
        method: 融合方法，"rrf" 或 "score"
        **kwargs: 传递给融合器的参数

    Returns:
        融合器实例
    """
    if method == "rrf":
        return RRFusion(k=kwargs.get("k", 60))
    elif method == "score":
        return ScoreLevelFusion(method=kwargs.get("score_method", "average"))
    else:
        logger.warning(f"Unknown fusion method '{method}', defaulting to RRF")
        return RRFusion()
