"""
异步工具函数库

包含：
- 异步 LLM 调用
- 异步向量检索
- 异步 Web 搜索
- 缓存管理
- 连接池管理
"""

import asyncio
import aiohttp
import hashlib
import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from functools import wraps
import threading

from src.utils.logger import logger


# ==================== 缓存管理 ====================

class AsyncCache:
    """线程安全的异步缓存"""
    
    def __init__(self, ttl: int = 3600, max_size: int = 10000):
        """
        初始化缓存
        
        Args:
            ttl: 缓存过期时间（秒）
            max_size: 最大缓存条目数
        """
        self.ttl = ttl
        self.max_size = max_size
        self.cache: Dict[str, tuple[Any, datetime]] = {}
        self.lock = threading.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self.lock:
            if key not in self.cache:
                return None
            
            value, expiry = self.cache[key]
            if datetime.now() > expiry:
                del self.cache[key]
                return None
            
            return value
    
    async def set(self, key: str, value: Any) -> None:
        """设置缓存值"""
        with self.lock:
            # 简单的 LRU 淘汰
            if len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[key] = (value, datetime.now() + timedelta(seconds=self.ttl))
    
    async def clear(self) -> None:
        """清空缓存"""
        with self.lock:
            self.cache.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """获取缓存统计"""
        with self.lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
            }


# 全局缓存实例
query_cache = AsyncCache(ttl=3600, max_size=10000)  # 查询结果缓存
embedding_cache = AsyncCache(ttl=86400, max_size=50000)  # 嵌入缓存（24小时）


# ==================== 缓存装饰器 ====================

def async_cache(cache_obj: AsyncCache, key_builder=None):
    """
    异步缓存装饰器
    
    使用示例:
        @async_cache(query_cache)
        async def my_function(query: str):
            return result
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 构建缓存 key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                cache_key = hashlib.md5(
                    json.dumps((args, kwargs), default=str, sort_keys=True).encode()
                ).hexdigest()
            
            # 尝试从缓存获取
            cached = await cache_obj.get(cache_key)
            if cached is not None:
                logger.debug(f"💾 缓存命中: {func.__name__}")
                return cached
            
            # 执行函数
            result = await func(*args, **kwargs)
            
            # 存入缓存
            await cache_obj.set(cache_key, result)
            return result
        
        return wrapper
    return decorator


# ==================== 异步 HTTP 客户端 ====================

class AsyncHTTPClient:
    """异步 HTTP 客户端，支持连接复用"""
    
    def __init__(self, max_retries: int = 3, timeout: int = 10):
        self.max_retries = max_retries
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self.lock = asyncio.Lock()
    
    async def _ensure_session(self) -> aiohttp.ClientSession:
        """确保会话存在"""
        if self.session is None or self.session.closed:
            async with self.lock:
                if self.session is None or self.session.closed:
                    connector = aiohttp.TCPConnector(
                        limit=100,  # 连接总数限制
                        limit_per_host=10,  # 单主机连接限制
                        ttl_dns_cache=300,  # DNS 缓存
                    )
                    self.session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    )
        return self.session
    
    async def get(self, url: str, **kwargs) -> Dict[str, Any]:
        """异步 GET 请求"""
        session = await self._ensure_session()
        
        for attempt in range(self.max_retries):
            try:
                async with session.get(url, **kwargs) as resp:
                    return {
                        "status": resp.status,
                        "data": await resp.json(),
                    }
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"HTTP GET 失败 (重试{attempt+1}次): {url}, {e}")
                    raise
                await asyncio.sleep(0.5 * (attempt + 1))  # 指数退避
    
    async def post(self, url: str, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """异步 POST 请求"""
        session = await self._ensure_session()
        
        for attempt in range(self.max_retries):
            try:
                async with session.post(url, json=data, **kwargs) as resp:
                    return {
                        "status": resp.status,
                        "data": await resp.json(),
                    }
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"HTTP POST 失败 (重试{attempt+1}次): {url}, {e}")
                    raise
                await asyncio.sleep(0.5 * (attempt + 1))
    
    async def close(self):
        """关闭会话"""
        if self.session:
            await self.session.close()


# 全局 HTTP 客户端
http_client = AsyncHTTPClient()


# ==================== 异步向量检索 ====================

async def async_vector_search(
    query_embedding: List[float],
    top_k: int = 5,
    threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    异步向量检索
    
    在线程池中执行检索以避免阻塞事件循环
    """
    try:
        from main import db_manager
        from concurrent.futures import ThreadPoolExecutor
        
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=2)
        
        result = await loop.run_in_executor(
            executor,
            lambda: db_manager.search(
                embedding=query_embedding,
                k=top_k,
                threshold=threshold,
            )
        )
        
        return result
    
    except Exception as e:
        logger.error(f"向量检索失败: {e}")
        return []


# ==================== 异步 Web 搜索 ====================

async def async_web_search(
    query: str,
    num_results: int = 5,
) -> List[Dict[str, Any]]:
    """
    异步 Web 搜索
    
    支持多个搜索提供商并发查询
    """
    try:
        from src.agent.tools.registry import get_tool
        
        # 获取 Web 搜索工具
        tool = get_tool("web_search")
        if not tool:
            logger.warning("Web 搜索工具未初始化")
            return []
        
        # 使用线程池执行搜索
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        
        result = await loop.run_in_executor(
            executor,
            lambda: tool.invoke({"query": query, "num_results": num_results})
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Web 搜索失败: {e}")
        return []


# ==================== 异步嵌入生成 ====================

@async_cache(embedding_cache, key_builder=lambda text: hashlib.md5(text.encode()).hexdigest())
async def async_embed_text(text: str) -> List[float]:
    """
    异步生成文本嵌入
    
    使用缓存避免重复嵌入
    """
    try:
        from main import get_embedder
        from concurrent.futures import ThreadPoolExecutor
        
        embedder = get_embedder()
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=2)
        
        embedding = await loop.run_in_executor(
            executor,
            lambda: embedder.embed_query(text)
        )
        
        return embedding
    
    except Exception as e:
        logger.error(f"嵌入生成失败: {e}")
        return []


# ==================== 速率限制 ====================

class RateLimiter:
    """异步速率限制器"""
    
    def __init__(self, max_requests: int = 100, time_window: int = 60):
        """
        初始化速率限制
        
        Args:
            max_requests: 时间窗口内最大请求数
            time_window: 时间窗口（秒）
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: List[datetime] = []
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """获取请求许可"""
        async with self.lock:
            now = datetime.now()
            
            # 清除过期请求
            self.requests = [
                req_time for req_time in self.requests
                if (now - req_time).total_seconds() < self.time_window
            ]
            
            # 检查是否超限
            if len(self.requests) >= self.max_requests:
                return False
            
            # 记录新请求
            self.requests.append(now)
            return True
    
    async def wait_if_needed(self):
        """如果需要则等待"""
        while not await self.acquire():
            await asyncio.sleep(0.1)


# 全局速率限制
query_limiter = RateLimiter(max_requests=100, time_window=60)  # 100 qps


# ==================== 连接池管理 ====================

class ConnectionPool:
    """异步连接池"""
    
    def __init__(self, pool_size: int = 10):
        self.pool_size = pool_size
        self.semaphore = asyncio.Semaphore(pool_size)
        self.active_connections = 0
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """获取连接"""
        await self.semaphore.acquire()
        async with self.lock:
            self.active_connections += 1
    
    async def release(self):
        """释放连接"""
        self.semaphore.release()
        async with self.lock:
            self.active_connections -= 1
    
    async def get_stats(self) -> Dict[str, int]:
        """获取连接池统计"""
        async with self.lock:
            return {
                "active": self.active_connections,
                "pool_size": self.pool_size,
                "available": self.pool_size - self.active_connections,
            }


# 全局连接池
connection_pool = ConnectionPool(pool_size=50)


# ==================== 批量操作 ====================

async def batch_process(
    items: List[Any],
    process_func,
    batch_size: int = 10,
    max_concurrent: int = 5,
) -> List[Any]:
    """
    批量处理项目
    
    支持批处理和并发控制
    """
    results = []
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_limit(item):
        async with semaphore:
            return await process_func(item)
    
    tasks = [process_with_limit(item) for item in items]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 过滤异常
    return [r for r in results if not isinstance(r, Exception)]


# ==================== 清理函数 ====================

async def cleanup_async_resources():
    """清理异步资源"""
    logger.info("清理异步资源...")
    await http_client.close()
    await query_cache.clear()
    logger.info("异步资源已清理")


if __name__ == "__main__":
    # 测试缓存
    async def test():
        await query_cache.set("test", "value")
        result = await query_cache.get("test")
        print(f"缓存测试: {result}")
        
        # 测试 HTTP 客户端
        try:
            result = await http_client.get("https://api.github.com")
            print(f"HTTP 测试: {result['status']}")
        except Exception as e:
            print(f"HTTP 测试失败: {e}")
    
    asyncio.run(test())
