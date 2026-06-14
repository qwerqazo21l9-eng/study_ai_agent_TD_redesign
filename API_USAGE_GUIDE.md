"""
使用示例和快速开始指南

## 快速启动

### 开发模式
```bash
python run_api_server.py
```

### 生产模式 (4个工作进程)
```bash
python run_api_server.py --prod --workers 4
```

### 自定义配置
```bash
python run_api_server.py --host 0.0.0.0 --port 8000 --workers 8
```


## API 使用示例

### 1. 同步查询

```bash
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "请介绍一下转移学习",
    "query_type": "auto",
    "include_citations": true,
    "timeout": 30
  }'
```

**Python 客户端:**
```python
import requests
import json

url = "http://localhost:8000/v1/query"
payload = {
    "query": "请介绍一下转移学习",
    "query_type": "auto",
    "include_citations": True,
    "timeout": 30
}

response = requests.post(url, json=payload)
result = response.json()

print(f"答案: {result['answer']}")
print(f"处理时间: {result['metadata']['processing_time']:.2f}s")
print(f"引用数: {len(result['citations'])}")
```


### 2. 流式查询

```bash
curl -X POST http://localhost:8000/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "最新的深度学习技术进展",
    "query_type": "auto"
  }' \
  -N
```

**Python 客户端:**
```python
import requests
import json

url = "http://localhost:8000/v1/query/stream"
payload = {"query": "最新的深度学习技术进展"}

with requests.post(url, json=payload, stream=True) as r:
    for line in r.iter_lines():
        if line:
            data = json.loads(line[6:])  # 去掉 "data: " 前缀
            if data['type'] == 'thinking':
                print(f"💭 {data['content']}")
            elif data['type'] == 'answer':
                print(f"📝 {data['content']}")
            elif data['type'] == 'citation':
                print(f"📚 {data['content']}")
```


### 3. 健康检查

```bash
curl http://localhost:8000/health | python -m json.tool
```

**响应示例:**
```json
{
  "status": "healthy",
  "timestamp": "2026-06-08T10:30:00.123456",
  "components": {
    "llm": "healthy",
    "embedder": "healthy"
  }
}
```


### 4. 统计信息

```bash
curl http://localhost:8000/stats | python -m json.tool
```

**响应示例:**
```json
{
  "total_queries": 150,
  "avg_response_time": 1.23,
  "p95_response_time": 2.45,
  "p99_response_time": 3.67,
  "error_rate": 0.67,
  "qps": 2.5,
  "active_connections": 3,
  "cache_hit_rate": 45.5
}
```


## 性能优化提示

### 1. 缓存优化
- 相同查询会自动缓存（TTL: 1小时）
- 嵌入结果缓存（TTL: 24小时）
- 缓存命中率越高性能越好

### 2. 连接池配置
- 向量库：单机模式，建议 2-5 worker
- LLM：API 调用，建议 4-8 worker
- 总连接数：建议 50-100

### 3. 工作进程数
```
CPU核心 1-2:   workers=2
CPU核心 4:     workers=4
CPU核心 8+:    workers=8
```

### 4. 内存优化
- 单进程内存: ~800MB
- 4进程配置: ~3.2GB
- 考虑向量库大小和嵌入缓存


## 监控和调试

### 查看服务器日志
```bash
tail -f logs/app.log
```

### 监控性能指标
```bash
watch -n 1 'curl -s http://localhost:8000/stats | python -m json.tool'
```

### 调试单个查询
```python
import requests
import time

url = "http://localhost:8000/v1/query"
payload = {
    "query": "测试查询",
    "timeout": 30
}

start = time.time()
response = requests.post(url, json=payload)
duration = time.time() - start

result = response.json()
print(f"HTTP 耗时: {duration:.2f}s")
print(f"处理耗时: {result['metadata']['processing_time']:.2f}s")
print(f"路由类型: {result['metadata']['route_type']}")
```


## 常见问题

### 1. 连接超时
- 检查查询是否过于复杂
- 增加 `timeout` 参数
- 检查 LLM API 延迟

### 2. 内存增长
- 检查 `cache_hit_rate`
- 清理缓存：`curl -X POST http://localhost:8000/admin/clear-cache`
- 考虑使用 Redis

### 3. 低吞吐量
- 增加 worker 进程数
- 检查 CPU/内存利用率
- 检查网络延迟
- 启用缓存


## 压力测试准备

当准备进行压力测试时，使用第 4 周的 Locust 脚本:

```python
# 示例：load_test.py
from locust import HttpUser, task, between

class APIUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def query(self):
        self.client.post("/v1/query", json={
            "query": "深度学习",
            "timeout": 30
        })
```

运行压力测试:
```bash
locust -f load_test.py --host=http://localhost:8000
```
"""

if __name__ == "__main__":
    print(__doc__)
