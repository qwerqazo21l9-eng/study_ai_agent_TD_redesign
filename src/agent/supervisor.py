"""
Supervisor Agent - 意图分析与路由

职责：
1. 分析用户查询的意图
2. 决定使用知识库检索、联网搜索还是闲聊
3. 提取关键词用于检索
"""

import re
from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.schema import AgentState, Intent, QueryType
from src.utils.llm import get_llm
from src.utils.logger import logger

# ============ Prompt 模板 ============

SUPERVISOR_SYSTEM_PROMPT = """你是一个智能问答助手（Supervisor Agent）。

你的职责是分析用户查询，决定最佳回答策略。

## 意图分类

1. **knowledge（知识库）**：用户询问的知识可以通过本地知识库（如上传的文档、书籍、笔记）回答
   - 例如："RAG 是什么"、"根据我上传的论文，总结一下..."
   - 特征：涉及特定领域知识、需要从已有文档中提取

2. **web_search（联网搜索）**：需要最新信息、实时数据或通用网络资源
   - 例如："今天有什么新闻"、"现在比特币价格多少"、"最新的 AI 技术进展"
   - 特征：需要实时数据、最新资讯、无法从知识库回答

3. **hybrid（混合）**：既需要知识库内容，也需要联网补充
   - 例如："结合最新的技术趋势和我的笔记，分析..."
   - 特征：需要结合两者

4. **chat（闲聊）**：纯闲聊、寒暄、问候
   - 例如："你好"、"今天天气不错"、"谢谢"
   - 特征：不需要检索，不需要工具

5. **data_processing（数据自动化处理）**：用户要求对 CSV/Excel 表格进行数据清洗和 EDA 分析
   - 例如："请对 xxx.csv 进行数据处理"、"帮我处理这个表格"、"清洗这份数据并生成报告"
   - 特征：涉及数据文件、数据处理、清洗、EDA 分析

## 输出格式

请严格按照以下 JSON 格式输出，不要输出其他内容：

{
    "query_type": "knowledge|web_search|hybrid|chat|data_processing|unknown",
    "confidence": 0.0-1.0,
    "reasoning": "分析过程",
    "keywords": ["关键词1", "关键词2"],
    "needs_tools": true|false
}

## 注意事项

- confidence 反映你对自己判断的信心
- keywords 用于后续检索，请提取 3-5 个最有信息量的词
- needs_tools 表示是否需要调用外部工具（联网搜索）
"""


class SupervisorAgent:
    """
    Supervisor Agent

    负责：
    1. 意图分类
    2. 关键词提取
    3. 路由决策
    4. 工具协调（web_search）
    """

    def __init__(self):
        self.llm = get_llm()
        self._keywords_pattern = re.compile(r'[\u4e00-\u9fa5a-zA-Z0-9]+')
        self._tool_registry = None  # 懒加载

    @property
    def tool_registry(self):
        """获取工具注册中心（懒加载）"""
        if self._tool_registry is None:
            from src.agent.tools.registry import get_registry
            self._tool_registry = get_registry()
        return self._tool_registry

    async def analyze(self, state: AgentState) -> AgentState:
        """
        分析用户查询意图

        Args:
            state: 当前状态（包含 query）

        Returns:
            更新后的状态（包含 intent）
        """
        query = state["query"]
        logger.info(f"Supervisor analyzing query: {query[:50]}...")

        try:
            # 使用 LLM 进行意图分析
            response = await self.llm.agenerate([
                [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
                 HumanMessage(content=f"请分析以下查询：\n{query}")]
            ])

            response_text = response.generations[0][0].text.strip()

            # 尝试解析 JSON
            intent = self._parse_intent(response_text)

            if intent:
                state["intent"] = intent
                state["query_type"] = intent.query_type
                logger.info(f"Intent detected: {intent.query_type.value}, confidence: {intent.confidence:.2f}")
            else:
                # 降级：使用关键词匹配
                intent = self._fallback_analysis(query)
                state["intent"] = intent
                state["query_type"] = intent.query_type
                logger.info(f"Intent fallback: {intent.query_type.value}")

        except Exception as e:
            logger.error(f"Intent analysis failed: {e}")
            # 降级为知识库检索
            state["intent"] = Intent(
                query_type=QueryType.KNOWLEDGE,
                confidence=0.5,
                reasoning="分析失败，降级为知识库检索"
            )
            state["query_type"] = QueryType.KNOWLEDGE
            state["error"] = str(e)

        return state

    def _build_message(self, query: str) -> dict:
        """构建消息"""
        return {
            "role": "system",
            "content": SUPERVISOR_SYSTEM_PROMPT
        }, {
            "role": "user",
            "content": f"请分析以下查询：\n{query}"
        }

    def _parse_intent(self, response_text: str) -> Intent | None:
        """解析 LLM 返回的意图"""
        try:
            # 提取 JSON 部分
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if not json_match:
                return None

            import json
            data = json.loads(json_match.group())

            return Intent(
                query_type=QueryType(data.get("query_type", "unknown")),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
                keywords=data.get("keywords", []),
                needs_tools=bool(data.get("needs_tools", False))
            )
        except Exception as e:
            logger.warning(f"Failed to parse intent JSON: {e}")
            return None

    def _fallback_analysis(self, query: str) -> Intent:
        """
        基于关键词的降级意图分析

        当 LLM 分析失败时的兜底策略
        """
        query_lower = query.lower()

        # 闲聊关键词
        chat_keywords = ["你好", "hi", "hello", "谢谢", "thanks", "拜拜", "再见", "好的", "好的"]
        if any(kw in query_lower for kw in chat_keywords):
            return Intent(
                query_type=QueryType.CHAT,
                confidence=0.9,
                reasoning="检测到闲聊关键词"
            )

        # 联网搜索关键词
        web_keywords = [
            "今天", "现在", "最新", "实时", "新闻",
            "价格", "股价", "天气", "当前", "此刻"
        ]
        if any(kw in query_lower for kw in web_keywords):
            return Intent(
                query_type=QueryType.WEB_SEARCH,
                confidence=0.7,
                reasoning="检测到联网关键词",
                needs_tools=True
            )

        # 混合关键词
        hybrid_keywords = ["结合", "综合", "比较", "对比", "分析"]
        if any(kw in query_lower for kw in hybrid_keywords):
            return Intent(
                query_type=QueryType.HYBRID,
                confidence=0.6,
                reasoning="检测到混合查询关键词",
                needs_tools=True
            )

        # 数据处理关键词（在知识库之前检查，因为更具体）
        data_processing_keywords = [
            "数据处理", "清洗", "分析", "csv", "表格",
            "eda", "数据清洗", "数据预处理", "好坏样本",
            "分布分析", "数据报告",
        ]
        if any(kw in query_lower for kw in data_processing_keywords):
            return Intent(
                query_type=QueryType.DATA_PROCESSING,
                confidence=0.8,
                reasoning="检测到数据处理关键词",
                needs_tools=True,
            )

        # 默认使用知识库
        return Intent(
            query_type=QueryType.KNOWLEDGE,
            confidence=0.5,
            reasoning="默认使用知识库检索"
        )

    def route(self, state: AgentState) -> Literal["retriever", "web_search", "generator", "chat", "tools"]:
        """
        基于意图进行路由

        Args:
            state: 当前状态

        Returns:
            下一个节点的名称
        """
        query_type = state.get("query_type", QueryType.UNKNOWN)
        intent = state.get("intent")

        # 如果置信度低，使用关键词匹配
        if intent and intent.confidence < 0.6:
            query_type = self._fallback_analysis(state["query"]).query_type

        # 路由决策
        if query_type == QueryType.CHAT:
            logger.info("Route: chat")
            return "chat"
        elif query_type == QueryType.WEB_SEARCH:
            logger.info("Route: web_search")
            return "web_search"
        elif query_type == QueryType.HYBRID:
            logger.info("Route: hybrid (retriever + web_search)")
            return "retriever"  # retriever 会协调两者
        elif query_type == QueryType.KNOWLEDGE:
            logger.info("Route: retriever")
            return "retriever"
        elif query_type == QueryType.DATA_PROCESSING:
            logger.info("Route: tools (data_processing)")
            return "tools"  # 由 ToolRegistry 执行 DataProcessingTool
        else:
            # 未知类型，默认知识库
            logger.info("Route: unknown -> retriever")
            return "retriever"

    async def web_search(self, state: AgentState) -> AgentState:
        """
        执行联网搜索

        调用 WebSearch 工具获取最新信息

        Args:
            state: 当前状态（包含 query 和 intent）

        Returns:
            更新后的状态（包含 web_context）
        """
        query = state["query"]
        intent = state.get("intent")
        keywords = intent.keywords if intent else []

        logger.info(f"WebSearch: query={query[:50]}...")

        try:
            # 获取 WebSearch 工具
            web_tool = self.tool_registry.get("web_search")

            if web_tool is None:
                logger.warning("WebSearch tool not available")
                state["web_context"] = "[联网搜索工具未配置]"
                state["web_results"] = []
                return state

            # 使用关键词或原始查询
            search_query = " ".join(keywords) if keywords else query

            # 执行搜索
            result = await web_tool.execute(
                query=search_query,
                num_results=5,
            )

            if result.success:
                state["web_context"] = result.content
                state["web_results"] = result.data.get("results", []) if result.data else []
                logger.info(f"WebSearch: found {len(state['web_results'])} results")
            else:
                state["web_context"] = f"[搜索失败] {result.error}"
                state["web_results"] = []
                logger.warning(f"WebSearch failed: {result.error}")

        except Exception as e:
            logger.error(f"WebSearch execution error: {e}")
            state["web_context"] = f"[搜索执行出错] {str(e)}"
            state["web_results"] = []
            state["error"] = str(e)

        return state
