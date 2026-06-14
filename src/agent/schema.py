"""
Agent 状态模型定义 (Pydantic)

定义 Agent 系统中使用的数据模型：
- AgentState: LangGraph 状态机状态
- AgentMessage: 消息模型
- Intent: 意图分类
- QueryType: 查询类型枚举
"""

from enum import Enum
from typing import Optional, TypedDict, Annotated
from datetime import datetime
from pydantic import BaseModel, Field

from langgraph.graph import add_messages
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage


class MessageRole(str, Enum):
    """消息角色枚举"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class QueryType(str, Enum):
    """查询类型枚举"""
    KNOWLEDGE = "knowledge"          # 知识库问答
    WEB_SEARCH = "web_search"        # 需要联网搜索
    HYBRID = "hybrid"                # 混合（知识库 + 联网）
    CHAT = "chat"                    # 闲聊
    PURE_LLM = "pure_llm"            # 纯 LLM（不检索知识库，只用对话历史）
    DATA_PROCESSING = "data_processing"  # 数据自动化处理（清洗 + EDA）
    UNKNOWN = "unknown"              # 未知


class Intent(BaseModel):
    """意图分析结果"""
    query_type: QueryType = Field(
        default=QueryType.UNKNOWN,
        description="查询类型"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="置信度"
    )
    reasoning: str = Field(
        default="",
        description="推理过程"
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="提取的关键词"
    )
    needs_tools: bool = Field(
        default=False,
        description="是否需要工具调用"
    )


class AgentMessage(BaseModel):
    """单条消息模型"""
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class RouteType(str, Enum):
    """路由类型 - 决定查询走哪条处理路径"""
    PURE_LLM = "pure_llm"    # 纯 LLM，不检索知识库，只结合对话历史回答
    RAG = "rag"               # 知识库检索 + LLM 生成
    REACT = "react"           # 多轮推理循环（ReAct）
    TOOLS = "tools"           # 工具执行（数据自动化处理等）


class SearchResult(BaseModel):
    """检索结果模型"""
    document: Document
    score: float
    source: str = Field(default="", description="来源")
    rank_info: dict = Field(default_factory=dict, description="排名信息")


class AgentState(TypedDict):
    """
    LangGraph 状态机状态定义

    Attributes:
        messages: 对话历史（Annotated 用于 add_messages  reducer）
        query: 当前用户查询
        intent: 意图分析结果
        search_results: 知识库检索结果
        web_results: 联网搜索结果（可选）
        context: 拼接的上下文
        response: 生成的回答
        query_type: 查询类型
        error: 错误信息（如果有）
    """
    # 对话历史，add_messages reducer 会合并新消息
    # 使用 BaseMessage 以兼容 LangGraph
    messages: Annotated[list[BaseMessage], add_messages]

    # 当前查询
    query: str

    # 意图分析
    intent: Optional[Intent]

    # 检索结果
    search_results: Optional[list[SearchResult]]
    web_results: Optional[list[dict]]  # 联网结果

    # 上下文
    context: Optional[str]
    memory_context: Optional[str]  # 记忆系统上下文

    # 生成的回答
    response: Optional[str]

    # 元信息
    query_type: Optional[QueryType]
    error: Optional[str]
    iteration: int  # 迭代次数，用于防止无限循环

    # 路由信息（三路路由）
    route_type: Optional[RouteType]   # 路由类型（PURE_LLM / RAG / REACT）
    web_context: Optional[str]         # 联网搜索结果上下文

    # ReAct 循环结果
    react_iterations: Optional[int]          # 循环执行次数
    is_react_complete: Optional[bool]        # 循环是否正常完成
    react_confidence: Optional[float]        # 循环置信度

    # 工具产物（数据处理等生成的文件列表）
    # 格式: [{"label": "清洗报告", "path": "/abs/path/report.html"}, ...]
    deliverables: Optional[list[dict]]


def create_initial_state(query: str) -> AgentState:
    """
    创建初始状态

    Args:
        query: 用户查询

    Returns:
        AgentState: 初始状态
    """
    return AgentState(
        messages=[HumanMessage(content=query)],
        query=query,
        intent=None,
        search_results=None,
        web_results=None,
        context=None,
        memory_context=None,
        response=None,
        query_type=None,
        error=None,
        iteration=0,
        # 路由信息
        route_type=None,
        web_context=None,
        # ReAct 循环结果
        react_iterations=0,
        is_react_complete=False,
        react_confidence=0.0,
        # 工具产物
        deliverables=None,
    )
