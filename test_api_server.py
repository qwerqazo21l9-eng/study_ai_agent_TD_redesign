"""
FastAPI 服务器测试脚本

验证服务器功能是否正常运行

用法:
    python test_api_server.py
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any


class APITester:
    """API 测试客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session: aiohttp.ClientSession = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
    
    async def get(self, endpoint: str) -> Dict[str, Any]:
        """GET 请求"""
        url = f"{self.base_url}{endpoint}"
        async with self.session.get(url) as resp:
            return await resp.json()
    
    async def post(self, endpoint: str, data: Dict) -> Dict[str, Any]:
        """POST 请求"""
        url = f"{self.base_url}{endpoint}"
        async with self.session.post(url, json=data) as resp:
            return await resp.json()
    
    async def post_stream(self, endpoint: str, data: Dict):
        """POST 流式请求"""
        url = f"{self.base_url}{endpoint}"
        async with self.session.post(url, json=data) as resp:
            async for line in resp.content:
                if line:
                    yield line.decode().strip()


async def test_root():
    """测试根路由"""
    print("\n[测试1] GET / - API 文档首页")
    async with APITester() as client:
        result = await client.get("/")
        print(f"✅ 状态: OK")
        print(f"   名称: {result.get('name')}")
        print(f"   版本: {result.get('version')}")
        print(f"   端点数: {len(result.get('endpoints', {}))}")


async def test_health():
    """测试健康检查"""
    print("\n[测试2] GET /health - 健康检查")
    async with APITester() as client:
        result = await client.get("/health")
        status = result.get("status")
        components = result.get("components", {})
        
        print(f"✅ 状态: {status}")
        for comp, state in components.items():
            icon = "✅" if state == "healthy" else "⚠️"
            print(f"   {icon} {comp}: {state}")


async def test_stats():
    """测试统计信息"""
    print("\n[测试3] GET /stats - 统计信息")
    async with APITester() as client:
        result = await client.get("/stats")
        print(f"✅ 总查询数: {result.get('total_queries')}")
        print(f"   平均响应时间: {result.get('avg_response_time'):.3f}s")
        print(f"   P95响应时间: {result.get('p95_response_time'):.3f}s")
        print(f"   错误率: {result.get('error_rate'):.2f}%")
        print(f"   活跃连接: {result.get('active_connections')}")
        print(f"   缓存命中率: {result.get('cache_hit_rate', 0):.1f}%")


async def test_sync_query():
    """测试同步查询"""
    print("\n[测试4] POST /v1/query - 同步查询")
    
    payload = {
        "query": "什么是机器学习？",
        "query_type": "auto",
        "include_citations": True,
        "timeout": 30
    }
    
    print(f"📝 查询: {payload['query']}")
    
    async with APITester() as client:
        start = time.time()
        
        try:
            result = await client.post("/v1/query", payload)
            duration = time.time() - start
            
            print(f"✅ 成功 ({duration:.2f}s)")
            print(f"   答案: {result['answer'][:100]}...")
            print(f"   处理时间: {result['metadata']['processing_time']:.2f}s")
            print(f"   路由类型: {result['metadata']['route_type']}")
            print(f"   引用数: {len(result['citations'])}")
            
        except Exception as e:
            print(f"❌ 失败: {e}")


async def test_stream_query():
    """测试流式查询"""
    print("\n[测试5] POST /v1/query/stream - 流式查询")
    
    payload = {
        "query": "深度学习的基本概念是什么？",
        "query_type": "auto"
    }
    
    print(f"📝 查询: {payload['query']}")
    
    async with APITester() as client:
        try:
            event_count = 0
            async for line in client.post_stream("/v1/query/stream", payload):
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    msg_type = data.get("type")
                    print(f"   [{msg_type}] {data.get('content', '')[:80]}")
                    event_count += 1
            
            print(f"✅ 成功 (接收 {event_count} 条事件)")
        
        except Exception as e:
            print(f"❌ 失败: {e}")


async def test_concurrent_queries():
    """测试并发查询"""
    print("\n[测试6] 并发查询 (5个并行)")
    
    queries = [
        "什么是神经网络？",
        "什么是卷积神经网络？",
        "什么是循环神经网络？",
        "什么是注意力机制？",
        "什么是Transformer模型？",
    ]
    
    async def query_task(i: int, q: str):
        async with APITester() as client:
            try:
                start = time.time()
                result = await client.post("/v1/query", {
                    "query": q,
                    "timeout": 30
                })
                duration = time.time() - start
                return i, duration, True
            except Exception as e:
                return i, 0, False
    
    start = time.time()
    tasks = [query_task(i, q) for i, q in enumerate(queries)]
    results = await asyncio.gather(*tasks)
    total_duration = time.time() - start
    
    success_count = sum(1 for _, _, ok in results if ok)
    avg_duration = sum(d for _, d, ok in results if ok) / max(success_count, 1)
    
    print(f"✅ 完成率: {success_count}/{len(queries)}")
    print(f"   总耗时: {total_duration:.2f}s")
    print(f"   平均响应时间: {avg_duration:.2f}s")
    print(f"   并发效率: {(len(queries) * avg_duration / total_duration):.1f}x")


async def main():
    """运行所有测试"""
    print("=" * 60)
    print("FastAPI 服务器测试套件")
    print("=" * 60)
    
    tests = [
        ("根路由", test_root),
        ("健康检查", test_health),
        ("统计信息", test_stats),
        ("同步查询", test_sync_query),
        ("流式查询", test_stream_query),
        ("并发查询", test_concurrent_queries),
    ]
    
    # 检查服务是否运行
    print("\n检查服务连接...")
    try:
        async with APITester() as client:
            await client.get("/")
        print("✅ 服务连接正常")
    except Exception as e:
        print(f"❌ 无法连接到服务: {e}")
        print("   请先运行: python run_api_server.py")
        return
    
    # 运行测试
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            await test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {name} 失败: {e}")
            failed += 1
    
    # 总结
    print("\n" + "=" * 60)
    print(f"测试总结: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    # 再次查看统计
    print("\n最终统计信息:")
    try:
        async with APITester() as client:
            stats = await client.get("/stats")
            print(f"✅ 总查询数: {stats.get('total_queries')}")
            print(f"   平均响应时间: {stats.get('avg_response_time', 0):.3f}s")
            print(f"   错误率: {stats.get('error_rate', 0):.2f}%")
            print(f"   缓存命中率: {stats.get('cache_hit_rate', 0):.1f}%")
    except Exception as e:
        print(f"❌ 无法获取统计: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n测试已中断")
    except Exception as e:
        print(f"\n错误: {e}")
