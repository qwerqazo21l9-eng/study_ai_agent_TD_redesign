"""
Tavily Web Search 独立测试脚本
"""
import asyncio
import json
import os

# 加载配置
config_path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
with open(config_path, "r", encoding="utf-8") as f:
    import yaml
    config = yaml.safe_load(f)

api_key = config.get("tools", {}).get("web_search", {}).get("tavily_api_key", "")

if not api_key:
    print("❌ 未找到 tavily_api_key，请在 config/config.yaml 中配置")
    exit(1)

print(f"[OK] API Key: {api_key[:20]}...")

# 测试 Tavily API
async def test_tavily():
    from tavily import TavilyClient
    
    client = TavilyClient(api_key=api_key)
    
    print("\n[搜索] AI agent 最新进展")
    result = client.search(
        query="AI agent 最新进展 2024",
        max_results=3,
        include_answer=True
    )
    
    print(f"\n找到 {len(result.get('results', []))} 条结果:\n")
    for i, r in enumerate(result.get("results", []), 1):
        print(f"{i}. {r['title']}")
        print(f"   URL: {r['url']}")
        print(f"   摘要: {r['content'][:150]}...")
        print()
    
    if result.get("answer"):
        print(f"[摘要] {result['answer']}")

if __name__ == "__main__":
    asyncio.run(test_tavily())
