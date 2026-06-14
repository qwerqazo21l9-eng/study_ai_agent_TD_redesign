"""
API 数据模型定义 (Pydantic)
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class QueryTypeEnum(str, Enum):
    """查询类型"""
    KNOWLEDGE = "knowledge"
    WEB_SEARCH = "web_search"
    HYBRID = "hybrid"
    CHAT = "chat"
    PURE_LLM = "pure_llm"
    AUTO = "auto"  # 自动判断


class QueryRequest(BaseModel):
    """查询请求"""
    query: str = Field(..., min_length=1, max_length=2000, description="用户查询")
    query_type: QueryTypeEnum = Field(default=QueryTypeEnum.AUTO, description="查询类型")
    session_id: Optional[str] = Field(default=None, description="会话ID")
    include_citations: bool = Field(default=True, description="是否包含引用")
    timeout: Optional[float] = Field(default=30.0, ge=1, le=300, description="超时时间(秒)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "请问文献中关于转移学习的最新研究进展如何？",
                "query_type": "auto",
                "session_id": "session_123",
                "include_citations": True,
                "timeout": 30.0
            }
        }


class Citation(BaseModel):
    """引用信息"""
    source: str = Field(..., description="数据来源")
    title: Optional[str] = Field(default=None, description="标题")
    content_snippet: Optional[str] = Field(default=None, description="内容片段")
    url: Optional[str] = Field(default=None, description="URL")
    relevance_score: Optional[float] = Field(default=None, description="相关度分数")


class QueryMetadata(BaseModel):
    """查询元数据"""
    processing_time: float = Field(..., description="处理耗时(秒)")
    route_type: str = Field(..., description="路由类型")
    intent_confidence: Optional[float] = Field(default=None, description="意图置信度")
    sources_used: List[str] = Field(default_factory=list, description="使用的数据源")
    reranked: bool = Field(default=False, description="是否进行了重排序")
    rag_chunks_count: int = Field(default=0, description="检索的知识库块数")
    web_results_count: int = Field(default=0, description="网络搜索结果数")


class QueryResponse(BaseModel):
    """查询响应"""
    answer: str = Field(..., description="回答内容")
    session_id: str = Field(..., description="会话ID")
    query_type: str = Field(..., description="识别的查询类型")
    citations: List[Citation] = Field(default_factory=list, description="引用列表")
    metadata: QueryMetadata = Field(..., description="元数据")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "根据最新研究，迁移学习在以下几个方面取得了进展...",
                "session_id": "session_123",
                "query_type": "knowledge",
                "citations": [],
                "metadata": {
                    "processing_time": 1.23,
                    "route_type": "rag",
                    "intent_confidence": 0.95,
                    "sources_used": ["vector_db", "bm25"],
                    "reranked": True,
                    "rag_chunks_count": 5,
                    "web_results_count": 0
                }
            }
        }


class StreamQueryResponse(BaseModel):
    """流式查询响应（SSE）"""
    type: str = Field(..., description="消息类型: thinking/answer/citation/done")
    content: str = Field(..., description="消息内容")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="可选元数据")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="状态: healthy/degraded/unhealthy")
    timestamp: str = Field(..., description="检查时间戳")
    components: Dict[str, str] = Field(..., description="各组件状态")


class ErrorResponse(BaseModel):
    """错误响应"""
    error_code: str = Field(..., description="错误代码")
    message: str = Field(..., description="错误信息")
    details: Optional[Dict[str, Any]] = Field(default=None, description="详细信息")


class StatsResponse(BaseModel):
    """统计信息"""
    total_queries: int = Field(..., description="总查询数")
    avg_response_time: float = Field(..., description="平均响应时间(秒)")
    p95_response_time: float = Field(..., description="P95响应时间(秒)")
    p99_response_time: float = Field(..., description="P99响应时间(秒)")
    error_rate: float = Field(..., description="错误率(%)")
    qps: float = Field(..., description="当前QPS")
    active_connections: int = Field(..., description="活跃连接数")
    cache_hit_rate: Optional[float] = Field(default=None, description="缓存命中率(%)")
