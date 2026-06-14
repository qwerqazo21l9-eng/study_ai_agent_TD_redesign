"""ReAct 工具模块 - ReAct 循环中的可用工具封装"""

from typing import Optional
from langchain_core.messages import HumanMessage, SystemMessage
from src.utils.logger import logger


class ReActToolResult:
    """ReAct 工具执行结果"""

    def __init__(self, success: bool, content: str = "", error: str = ""):
        self.success = success
        self.content = content
        self.error = error

    def __repr__(self) -> str:
        status = "✓" if self.success else "✗"
        preview = self.content[:100] + "..." if len(self.content) > 100 else self.content
        return f"ReActToolResult({status}: {preview})"


class ReActRetrieveTool:
    """
    ReAct 知识库检索工具

    封装 Hybrid RAG 的检索能力
    """

    def __init__(self):
        self._retriever = None

    @property
    def retriever(self):
        """懒加载 Retriever Agent"""
        if self._retriever is None:
            from src.agent.retriever_agent import RetrieverAgent
            self._retriever = RetrieverAgent()
        return self._retriever

    async def execute(self, query: str, top_k: int = 5) -> ReActToolResult:
        """
        执行知识库检索

        Args:
            query: 检索词
            top_k: 返回数量

        Returns:
            ReActToolResult: 检索结果
        """
        try:
            logger.info(f"ReAct Retrieve: {query[:50]}...")

            from src.agent.schema import create_initial_state
            state = create_initial_state(query)

            # 执行检索（async 方法，需要 await）
            result_state = await self.retriever.retrieve(state)

            # 提取结果
            docs = result_state.get("search_results") or []
            if not docs:
                return ReActToolResult(
                    success=True,
                    content="（未找到相关文档）",
                )

            # 拼接文档内容
            contents = []
            for i, doc in enumerate(docs[:top_k], 1):
                content = doc.document.page_content if hasattr(doc, "document") else str(doc)
                contents.append(f"[文档{i}] {content}")

            result_content = "\n\n".join(contents)
            logger.info(f"ReAct Retrieve: found {len(docs)} docs")

            return ReActToolResult(
                success=True,
                content=result_content,
            )

        except Exception as e:
            logger.error(f"ReAct Retrieve error: {e}")
            return ReActToolResult(
                success=False,
                error=str(e),
            )


class ReActWebSearchTool:
    """
    ReAct 联网搜索工具

    封装 WebSearch 能力
    """

    def __init__(self):
        self._tool = None

    @property
    def web_tool(self):
        """懒加载 WebSearch 工具"""
        if self._tool is None:
            from src.agent.tools.registry import get_registry
            registry = get_registry()
            self._tool = registry.get("web_search")
        return self._tool

    async def execute(self, query: str, num_results: int = 5) -> ReActToolResult:
        """
        执行联网搜索

        Args:
            query: 搜索词
            num_results: 返回数量

        Returns:
            ReActToolResult: 搜索结果
        """
        try:
            logger.info(f"ReAct WebSearch: {query[:50]}...")

            if self.web_tool is None:
                return ReActToolResult(
                    success=False,
                    error="WebSearch 工具未配置",
                )

            # 执行搜索
            result = await self.web_tool.execute(
                query=query,
                num_results=num_results,
            )

            if not result.success:
                return ReActToolResult(
                    success=False,
                    error=result.error,
                )

            logger.info(f"ReAct WebSearch: found {len(result.data.get('results', []))} results")
            return ReActToolResult(
                success=True,
                content=result.content,
            )

        except Exception as e:
            logger.error(f"ReAct WebSearch error: {e}")
            return ReActToolResult(
                success=False,
                error=str(e),
            )


class ReActThinkTool:
    """
    ReAct 深度思考工具

    基于 LLM 进行推理
    """

    def __init__(self):
        self._llm = None

    @property
    def llm(self):
        """懒加载 LLM"""
        if self._llm is None:
            from src.utils.llm import get_llm
            self._llm = get_llm()
        return self._llm

    async def execute(self, query: str, context: str) -> ReActToolResult:
        """
        执行深度思考

        Args:
            query: 思考问题
            context: 已有上下文

        Returns:
            ReActToolResult: 推理结果
        """
        try:
            logger.info(f"ReAct Think: analyzing...")

            system_prompt = """你是一个推理引擎，擅长基于给定的上下文进行深度分析和推理。

请基于以下上下文，回答问题或进行分析推理。"""

            user_prompt = f"""## 上下文
{context}

## 问题
{query}

## 要求
请进行深度分析，给出有见地的回答。如果上下文不足，明确说明。"""

            response = await self.llm.agenerate([
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            ])

            # 处理 LangChain 返回格式
            generation = response.generations[0][0]
            if hasattr(generation, "text"):
                result_text = generation.text.strip()
            elif hasattr(generation, "content"):
                result_text = generation.content.strip()
            else:
                result_text = str(generation).strip()

            logger.info(f"ReAct Think: done, {len(result_text)} chars")

            return ReActToolResult(
                success=True,
                content=result_text,
            )

        except Exception as e:
            logger.error(f"ReAct Think error: {e}")
            return ReActToolResult(
                success=False,
                error=str(e),
            )


# ============ 全局工具实例 ============

_retrieve_tool: Optional[ReActRetrieveTool] = None
_web_search_tool: Optional[ReActWebSearchTool] = None
_think_tool: Optional[ReActThinkTool] = None


def get_react_retrieve_tool() -> ReActRetrieveTool:
    """获取 ReAct 检索工具"""
    global _retrieve_tool
    if _retrieve_tool is None:
        _retrieve_tool = ReActRetrieveTool()
    return _retrieve_tool


def get_react_web_search_tool() -> ReActWebSearchTool:
    """获取 ReAct 搜索工具"""
    global _web_search_tool
    if _web_search_tool is None:
        _web_search_tool = ReActWebSearchTool()
    return _web_search_tool


def get_react_think_tool() -> ReActThinkTool:
    """获取 ReAct 思考工具"""
    global _think_tool
    if _think_tool is None:
        _think_tool = ReActThinkTool()
    return _think_tool