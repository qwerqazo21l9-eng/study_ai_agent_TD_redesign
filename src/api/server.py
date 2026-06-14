"""
FastAPI 服务器 - 异步高并发 Agent 服务

特性：
- 异步查询处理
- 流式响应支持
- 请求队列和限流
- 缓存层
- 健康检查和监控
- 并发连接管理
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZIPMiddleware
from contextlib import asynccontextmanager
import asyncio
import time
import uuid
from typing import Optional, AsyncGenerator, Dict, Any
from datetime import datetime
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from src.api.schemas import (
    QueryRequest, QueryResponse, Citation, QueryMetadata,
    HealthResponse, ErrorResponse, StatsResponse, StreamQueryResponse
)
from src.utils.logger import logger
from src.utils.config_loader import config


# ==================== 全局状态 ====================

class QueryStats:
    """查询统计"""
    def __init__(self):
        self.total_queries = 0
        self.total_time = 0.0
        self.response_times = deque(maxlen=1000)  # 最近1000个响应时间
        self.errors = 0
        self.active_connections = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.lock = asyncio.Lock()
    
    async def record(self, duration: float, is_error: bool = False):
        """记录查询统计"""
        async with self.lock:
            self.total_queries += 1
            self.total_time += duration
            self.response_times.append(duration)
            if is_error:
                self.errors += 1
    
    async def record_cache_hit(self):
        """记录缓存命中"""
        async with self.lock:
            self.cache_hits += 1
    
    async def record_cache_miss(self):
        """记录缓存未命中"""
        async with self.lock:
            self.cache_misses += 1
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        async with self.lock:
            total = len(self.response_times)
            if total == 0:
                return {
                    "total_queries": self.total_queries,
                    "avg_response_time": 0.0,
                    "p95_response_time": 0.0,
                    "p99_response_time": 0.0,
                    "error_rate": 0.0,
                    "qps": 0.0,
                    "cache_hit_rate": 0.0,
                }
            
            sorted_times = sorted(self.response_times)
            p95_idx = int(total * 0.95)
            p99_idx = int(total * 0.99)
            
            return {
                "total_queries": self.total_queries,
                "avg_response_time": sum(self.response_times) / total,
                "p95_response_time": sorted_times[p95_idx],
                "p99_response_time": sorted_times[p99_idx],
                "error_rate": (self.errors / self.total_queries * 100) if self.total_queries > 0 else 0.0,
                "qps": 0.0,  # 需要时间窗口计算
                "cache_hit_rate": (self.cache_hits / (self.cache_hits + self.cache_misses) * 100) 
                                 if (self.cache_hits + self.cache_misses) > 0 else 0.0,
            }


# 全局实例
query_stats = QueryStats()
thread_pool = ThreadPoolExecutor(max_workers=10)  # 用于向量库等阻塞操作
request_queue: asyncio.Queue = None  # 初始化于 lifespan


# ==================== 启动/关闭事件 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global request_queue
    
    # 启动时
    logger.info("🚀 FastAPI 服务器启动中...")
    
    request_queue = asyncio.Queue(maxsize=1000)
    
    # 初始化 Agent 组件
    try:
        from main import (
            get_llm, get_embedder, get_agent_workflow, 
            get_memory_manager, _init_tools
        )
        
        # 预加载组件
        _ = get_llm()
        _ = get_embedder()
        _ = get_agent_workflow()
        _ = get_memory_manager()
        _init_tools()
        
        logger.info("✅ 所有 Agent 组件已初始化")
    except Exception as e:
        logger.error(f"❌ Agent 初始化失败: {e}")
        raise
    
    yield  # 服务运行期间
    
    # 关闭时
    logger.info("🛑 FastAPI 服务器关闭中...")
    thread_pool.shutdown(wait=True)
    logger.info("✅ 资源已清理")


# ==================== 创建应用 ====================

def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    
    app = FastAPI(
        title="AI 学习助手 Agent API",
        description="高并发异步查询服务",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # 中间件配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.get("cors_origins", ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZIPMiddleware, minimum_size=1000)
    
    # ==================== 查询路由 ====================
    
    @app.post("/v1/query", response_model=QueryResponse)
    async def query(request: QueryRequest) -> QueryResponse:
        """
        同步查询接口
        
        - 完整等待响应，然后返回最终答案
        - 包含引用和元数据
        - 超时时间可配置
        """
        session_id = request.session_id or str(uuid.uuid4())
        start_time = time.time()
        
        try:
            query_stats.active_connections += 1
            
            logger.info(f"📝 新查询 [sid={session_id[:8]}]: {request.query[:50]}...")
            
            # 异步执行查询
            answer, citations, metadata = await run_agent_query(
                query=request.query,
                query_type=request.query_type.value,
                session_id=session_id,
                timeout=request.timeout,
            )
            
            duration = time.time() - start_time
            await query_stats.record(duration)
            
            logger.info(f"✅ 查询完成 [sid={session_id[:8]}]: {duration:.2f}s")
            
            return QueryResponse(
                answer=answer,
                session_id=session_id,
                query_type=metadata.get("route_type", "unknown"),
                citations=citations,
                metadata=QueryMetadata(
                    processing_time=duration,
                    **metadata
                )
            )
        
        except asyncio.TimeoutError:
            await query_stats.record(request.timeout or 30.0, is_error=True)
            logger.error(f"❌ 查询超时 [sid={session_id[:8]}]")
            raise HTTPException(status_code=504, detail="Query timeout")
        
        except Exception as e:
            duration = time.time() - start_time
            await query_stats.record(duration, is_error=True)
            logger.error(f"❌ 查询失败 [sid={session_id[:8]}]: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        
        finally:
            query_stats.active_connections -= 1
    
    
    @app.post("/v1/query/stream")
    async def query_stream(request: QueryRequest):
        """
        流式查询接口（SSE）
        
        - 实时流式返回思考过程和答案
        - 降低端到端延迟
        - 支持客户端提前看到结果
        """
        session_id = request.session_id or str(uuid.uuid4())
        
        async def event_generator() -> AsyncGenerator[str, None]:
            try:
                query_stats.active_connections += 1
                start_time = time.time()
                
                logger.info(f"📝 流式查询 [sid={session_id[:8]}]: {request.query[:50]}...")
                
                # 流式执行查询
                async for event in stream_agent_query(
                    query=request.query,
                    query_type=request.query_type.value,
                    session_id=session_id,
                    timeout=request.timeout,
                ):
                    yield f"data: {event.model_dump_json()}\n\n"
                
                duration = time.time() - start_time
                await query_stats.record(duration)
                logger.info(f"✅ 流式查询完成 [sid={session_id[:8]}]: {duration:.2f}s")
                
                yield "data: {\"type\": \"done\"}\n\n"
            
            except Exception as e:
                logger.error(f"❌ 流式查询失败: {str(e)}")
                yield f"data: {{\"type\": \"error\", \"content\": \"{str(e)}\"}}\n\n"
            
            finally:
                query_stats.active_connections -= 1
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"}
        )
    
    
    # ==================== 健康检查 ====================
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """健康检查端点"""
        try:
            from main import get_llm, get_embedder
            
            components = {}
            
            # 检查 LLM
            try:
                _ = get_llm()
                components["llm"] = "healthy"
            except Exception as e:
                components["llm"] = f"unhealthy: {str(e)[:50]}"
            
            # 检查 Embedder
            try:
                _ = get_embedder()
                components["embedder"] = "healthy"
            except Exception as e:
                components["embedder"] = f"unhealthy: {str(e)[:50]}"
            
            status = "healthy" if all(v == "healthy" for v in components.values()) else "degraded"
            
            return HealthResponse(
                status=status,
                timestamp=datetime.now().isoformat(),
                components=components
            )
        
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return HealthResponse(
                status="unhealthy",
                timestamp=datetime.now().isoformat(),
                components={"error": str(e)}
            )
    
    
    # ==================== 统计端点 ====================
    
    @app.get("/stats", response_model=StatsResponse)
    async def get_stats() -> StatsResponse:
        """获取统计信息"""
        stats = await query_stats.get_stats()
        return StatsResponse(
            total_queries=stats["total_queries"],
            avg_response_time=stats["avg_response_time"],
            p95_response_time=stats["p95_response_time"],
            p99_response_time=stats["p99_response_time"],
            error_rate=stats["error_rate"],
            qps=stats["qps"],
            active_connections=query_stats.active_connections,
            cache_hit_rate=stats["cache_hit_rate"],
        )
    
    
    # ==================== 根路由 ====================
    
    @app.get("/")
    async def root():
        """API 文档"""
        return {
            "name": "AI 学习助手 Agent API",
            "version": "1.0.0",
            "docs": "/docs",
            "endpoints": {
                "POST /v1/query": "同步查询",
                "POST /v1/query/stream": "流式查询",
                "GET /health": "健康检查",
                "GET /stats": "统计信息",
            }
        }
    
    
    return app


# ==================== 异步查询函数 ====================

async def run_agent_query(
    query: str,
    query_type: str,
    session_id: str,
    timeout: Optional[float] = None,
) -> tuple[str, list[Citation], Dict[str, Any]]:
    """
    执行 Agent 查询
    
    Returns:
        (answer, citations, metadata)
    """
    from main import get_agent_workflow, get_memory_manager
    
    try:
        workflow = get_agent_workflow()
        memory = get_memory_manager()
        
        # 在线程池中运行 sync 工作流（避免阻塞）
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(
                thread_pool,
                lambda: _run_sync_workflow(query, workflow, memory, session_id)
            ),
            timeout=timeout or 30.0
        )
        
        answer, citations_data, metadata = result
        
        # 构建引用对象
        citations = [Citation(**c) for c in citations_data]
        
        return answer, citations, metadata
    
    except asyncio.TimeoutError:
        raise
    except Exception as e:
        logger.error(f"Agent 查询失败: {e}")
        raise


def _run_sync_workflow(query: str, workflow, memory, session_id: str) -> tuple:
    """
    同步工作流执行（在线程池中运行）
    
    这是对现有同步 workflow 的包装
    """
    try:
        from src.agent.schema import create_initial_state
        
        # 创建初始状态
        state = create_initial_state(
            query=query,
            session_id=session_id,
        )
        
        # 执行工作流
        result = workflow.invoke(state)
        
        # 提取答案和元数据
        answer = result.get("answer", "No answer generated")
        citations = result.get("citations", [])
        
        metadata = {
            "route_type": result.get("route_type", "unknown"),
            "intent_confidence": result.get("intent_confidence", 0.0),
            "sources_used": result.get("sources_used", []),
            "reranked": result.get("reranked", False),
            "rag_chunks_count": result.get("rag_chunks_count", 0),
            "web_results_count": result.get("web_results_count", 0),
        }
        
        return answer, citations, metadata
    
    except Exception as e:
        logger.error(f"同步工作流执行失败: {e}")
        raise


async def stream_agent_query(
    query: str,
    query_type: str,
    session_id: str,
    timeout: Optional[float] = None,
) -> AsyncGenerator[StreamQueryResponse, None]:
    """
    流式执行 Agent 查询
    
    逐步返回思考过程、中间结果、最终答案
    """
    try:
        # TODO: 实现真正的流式处理
        # 目前先返回完整答案
        
        answer, citations, metadata = await run_agent_query(
            query=query,
            query_type=query_type,
            session_id=session_id,
            timeout=timeout,
        )
        
        # 返回思考阶段
        yield StreamQueryResponse(
            type="thinking",
            content=f"正在分析查询: {query[:100]}...",
            metadata={"stage": "init"}
        )
        
        await asyncio.sleep(0.1)
        
        # 返回答案
        yield StreamQueryResponse(
            type="answer",
            content=answer,
            metadata=metadata
        )
        
        # 返回引用
        for i, citation in enumerate(citations):
            yield StreamQueryResponse(
                type="citation",
                content=f"[{i+1}] {citation.source}",
                metadata=citation.model_dump()
            )
    
    except Exception as e:
        logger.error(f"流式查询失败: {e}")
        yield StreamQueryResponse(
            type="error",
            content=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    
    app = create_app()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=4,  # 多进程
        loop="uvloop",  # 高性能事件循环
        log_level="info",
    )
