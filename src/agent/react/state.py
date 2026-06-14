"""ReAct 状态模型定义"""

from typing import Optional, TypedDict, Literal, Annotated
from enum import Enum

from langgraph.graph import add_messages


class ActionType(str, Enum):
    """ReAct 可执行的动作类型"""
    RETRIEVE = "retrieve"        # 知识库检索
    WEB_SEARCH = "web_search"    # 联网搜索
    THINK = "think"              # 深度思考（不执行工具）
    GENERATE = "generate"        # 生成最终答案
    FINISH = "finish"           # 完成任务


class Action:
    """
    单个动作记录

    Attributes:
        action_type: 动作类型
        action_input: 动作输入（如检索词）
        action_output: 动作输出（如检索结果）
        reasoning: 动作理由
    """

    def __init__(
        self,
        action_type: ActionType,
        action_input: str,
        action_output: str = "",
        reasoning: str = "",
    ):
        self.action_type = action_type
        self.action_input = action_input
        self.action_output = action_output
        self.reasoning = reasoning

    def __repr__(self) -> str:
        return f"Action({self.action_type.value}: {self.action_input[:30]}...)"


class Observation:
    """
    观察记录（Action 结果）

    Attributes:
        content: 观察内容
        source: 来源（retrieve/web_search/think）
        timestamp: 时间戳
    """

    def __init__(self, content: str, source: str = ""):
        self.content = content
        self.source = source

    def __repr__(self) -> str:
        return f"Observation({self.source}: {self.content[:50]}...)"


class ReActState(TypedDict):
    """
    ReAct 循环专用状态

    扩展自 AgentState，用于支持多轮推理循环

    Attributes:
        original_query: 用户原始查询
        current_thought: 当前思考（Think 阶段的输出）
        action_history: 已执行的动作列表
        observations: 观察结果列表
        iteration: 当前迭代次数
        max_iterations: 最大迭代次数
        evaluation_result: 评估结果
        context_collected: 已收集的上下文
        is_complete: 是否完成
        next_action: 下一个动作类型
        confidence: 当前置信度
    """
    # 基础信息
    original_query: str
    query: str

    # ReAct 循环状态
    current_thought: str                          # 当前思考
    action_history: list[Action]                 # 动作历史
    observations: list[str]                      # 观察结果（简化版）
    iteration: int                               # 当前迭代
    max_iterations: int                          # 最大迭代

    # 评估与决策
    evaluation_result: str                       # 评估结果
    is_complete: bool                           # 是否完成
    next_action: Optional[ActionType]            # 下一个动作
    confidence: float                            # 当前置信度

    # 收集的上下文
    context_collected: list[str]                 # 已收集上下文
    retrieved_docs: list[str]                    # 检索到的文档
    web_results: list[str]                       # 网页搜索结果

    # 最终输出
    response: Optional[str]
    error: Optional[str]


def create_initial_react_state(query: str, max_iterations: int = 5) -> ReActState:
    """
    创建 ReAct 初始状态

    Args:
        query: 用户查询
        max_iterations: 最大迭代次数

    Returns:
        ReActState: 初始状态
    """
    return ReActState(
        original_query=query,
        query=query,
        current_thought="",
        action_history=[],
        observations=[],
        iteration=0,
        max_iterations=max_iterations,
        evaluation_result="",
        is_complete=False,
        next_action=None,
        confidence=0.0,
        context_collected=[],
        retrieved_docs=[],
        web_results=[],
        response=None,
        error=None,
    )


def add_action_to_history(
    state: ReActState,
    action_type: ActionType,
    action_input: str,
    action_output: str = "",
    reasoning: str = "",
) -> ReActState:
    """
    添加动作到历史记录

    Args:
        state: 当前状态
        action_type: 动作类型
        action_input: 动作输入
        action_output: 动作输出
        reasoning: 动作理由

    Returns:
        更新后的状态
    """
    action = Action(
        action_type=action_type,
        action_input=action_input,
        action_output=action_output,
        reasoning=reasoning,
    )
    state["action_history"].append(action)
    return state


def add_observation(state: ReActState, content: str, source: str = "") -> ReActState:
    """
    添加观察结果

    Args:
        state: 当前状态
        content: 观察内容
        source: 来源

    Returns:
        更新后的状态
    """
    obs = Observation(content=content, source=source)
    state["observations"].append(f"[{source}] {content}")
    return state