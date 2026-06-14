"""
WebSearch - 网页搜索工具

支持多个搜索 provider：
1. DuckDuckGo（免费，无需 API Key）
2. Tavily（付费，精度更高）
3. SerpAPI（可选扩展）

使用示例：
    # 方式 1：直接使用
    tool = DuckDuckGoSearch()
    result = await tool.execute(query="Python 教程")

    # 方式 2：通过注册中心
    registry = ToolRegistry()
    registry.register(WebSearchTool(provider="duckduckgo"))
    tool = registry.get("web_search")
    result = await tool.execute(query="...")
"""

import asyncio
from typing import Optional
from dataclasses import dataclass

from src.agent.tools.base import BaseTool, ToolDefinition, ToolParameter, ToolResult, ToolStatus
from src.utils.logger import logger


@dataclass
class SearchResult:
    """单条搜索结果"""
    title: str
    url: str
    snippet: str
    source: str = ""


class WebSearchTool(BaseTool):
    """
    网页搜索工具

    支持多个 provider，默认使用 DuckDuckGo
    """

    def __init__(self, provider: str = "duckduckgo"):
        """
        Args:
            provider: 搜索提供商，支持 duckduckgo, tavily
        """
        self.provider = provider
        super().__init__()

        # 根据 provider 初始化
        if provider == "duckduckgo":
            self._search_impl = DuckDuckGoSearch()
        elif provider == "tavily":
            self._search_impl = TavilySearch()
        else:
            logger.warning(f"Unknown provider {provider}, using DuckDuckGo")
            self._search_impl = DuckDuckGoSearch()

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="web_search",
            description="搜索互联网获取最新信息。当你需要查询实时新闻、天气、股价、或其他知识库中不存在的实时信息时使用。",
            parameters=[
                ToolParameter(
                    name="query",
                    description="搜索查询词，应该是简洁的关键词而非完整句子",
                    type="string",
                    required=True,
                ),
                ToolParameter(
                    name="num_results",
                    description="返回结果数量，默认 5",
                    type="number",
                    required=False,
                    default=5,
                ),
                ToolParameter(
                    name="source",
                    description="指定信息来源（如 wikipedia, news），留空则搜索全网",
                    type="string",
                    required=False,
                ),
            ],
            category="search",
            tags=["search", "web", "internet", "realtime"],
        )

    async def _execute(self, **kwargs) -> ToolResult:
        """
        执行搜索

        Args:
            query: 搜索关键词
            num_results: 返回数量
            source: 信息来源
        """
        query = kwargs.get("query")
        num_results = kwargs.get("num_results", 5)
        source = kwargs.get("source")

        if not query:
            return ToolResult(
                status=ToolStatus.ERROR,
                error="query 参数不能为空",
            )

        try:
            results = await self._search_impl.search(
                query=query,
                num_results=num_results,
                source=source,
            )

            if not results:
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    content="[搜索结果为空] 没有找到相关信息",
                    data={"results": []},
                )

            # 格式化结果
            formatted = self._format_results(results)
            metadata = {
                "provider": self.provider,
                "query": query,
                "num_results": len(results),
            }

            return ToolResult(
                status=ToolStatus.SUCCESS,
                content=formatted,
                data={
                    "results": [
                        {"title": r.title, "url": r.url, "snippet": r.snippet}
                        for r in results
                    ]
                },
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"搜索失败: {str(e)}",
            )

    def _format_results(self, results: list[SearchResult]) -> str:
        """格式化搜索结果为文本"""
        lines = ["【网页搜索结果】", ""]

        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.title}")
            lines.append(f"   链接: {r.url}")
            lines.append(f"   摘要: {r.snippet}")
            lines.append("")

        return "\n".join(lines)


class DuckDuckGoSearch:
    """
    DuckDuckGo 搜索实现

    免费，无需 API Key
    使用 duckduckgo-search 库
    """

    def __init__(self):
        self._client = None

    def _get_client(self):
        """懒加载客户端"""
        if self._client is None:
            try:
                from duckduckgo_search import DDGS
                self._client = DDGS()
            except ImportError:
                logger.error("请安装 duckduckgo-search: pip install duckduckgo-search")
                raise ImportError("缺少 duckduckgo-search 库")

        return self._client

    async def search(
        self,
        query: str,
        num_results: int = 5,
        source: Optional[str] = None,
    ) -> list[SearchResult]:
        """执行 DuckDuckGo 搜索"""

        def _sync_search():
            client = self._get_client()

            # 根据 source 决定搜索类型
            if source == "news":
                results = client.news(query, max_results=num_results)
            elif source == "wikipedia":
                results = client.answers(query)
            else:
                results = client.text(query, max_results=num_results)

            return [
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    snippet=r.get("body", ""),
                    source="duckduckgo",
                )
                for r in results
            ]

        # 在线程池中执行同步搜索
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_search)


class TavilySearch:
    """
    Tavily AI 搜索实现

    付费服务，精度更高
    需要 Tavily API Key
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.tavily.com/search"

    async def search(
        self,
        query: str,
        num_results: int = 5,
        source: Optional[str] = None,
    ) -> list[SearchResult]:
        """执行 Tavily 搜索"""
        import aiohttp

        if not self.api_key:
            # 尝试从环境变量获取
            import os
            self.api_key = os.environ.get("TAVILY_API_KEY")

            if not self.api_key:
                raise ValueError("Tavily 需要 API Key，请设置 TAVILY_API_KEY 环境变量")

        headers = {
            "Content-Type": "application/json",
        }

        params = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": num_results,
        }

        if source:
            params["domains"] = source.split(",")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url,
                    json=params,
                    headers=headers,
                ) as response:
                    if response.status != 200:
                        text = await response.text()
                        raise Exception(f"Tavily API error: {response.status} - {text}")

                    data = await response.json()

                    return [
                        SearchResult(
                            title=r.get("title", ""),
                            url=r.get("url", ""),
                            snippet=r.get("content", ""),
                            source="tavily",
                        )
                        for r in data.get("results", [])
                    ]

        except aiohttp.ClientError as e:
            raise Exception(f"网络请求失败: {e}")


# 向后兼容：保留 DuckDuckGoSearch 和 TavilySearch 作为独立工具类
class DuckDuckGoTool(WebSearchTool):
    """DuckDuckGo 搜索工具（向后兼容）"""

    def __init__(self):
        super().__init__(provider="duckduckgo")


class TavilyTool(WebSearchTool):
    """Tavily 搜索工具（向后兼容）"""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(provider="tavily")
        # 优先使用传入的 api_key，否则尝试从配置文件读取
        if not api_key:
            try:
                from src.utils.config_loader import config
                api_key = config.tools.get("web_search", {}).get("tavily_api_key", "")
            except Exception:
                api_key = ""
        if api_key:
            self._search_impl.api_key = api_key


# 旧版兼容
class TavilySearchImpl(TavilySearch):
    """Tavily 搜索实现（旧版兼容）"""

    async def search(
        self,
        query: str,
        num_results: int = 5,
        source: Optional[str] = None,
    ) -> list[SearchResult]:
        return await super().search(query, num_results, source)


class DuckDuckGoSearchImpl(DuckDuckGoSearch):
    """DuckDuckGo 搜索实现（旧版兼容）"""

    async def search(
        self,
        query: str,
        num_results: int = 5,
        source: Optional[str] = None,
    ) -> list[SearchResult]:
        return await super().search(query, num_results, source)
