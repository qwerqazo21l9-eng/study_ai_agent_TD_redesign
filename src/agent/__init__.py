"""
Agent 模块 - AI Agent 核心组件

包含：
- schema: Agent 状态模型定义
- supervisor: 意图分析与路由
- retriever_agent: Hybrid RAG + 工具调用
- generator: 带引用回答生成
- workflow: LangGraph 工作流编排（支持分层路由）
- memory: 短期和长期记忆
- complexity: 查询复杂度分析器
- react: ReAct 推理循环
"""

from src.agent.schema import (
    AgentState,
    AgentMessage,
    MessageRole,
    Intent,
    QueryType,
)

from src.agent.supervisor import SupervisorAgent
from src.agent.retriever_agent import RetrieverAgent
from src.agent.generator import GeneratorAgent
from src.agent.workflow import create_agent_workflow, run_agent, run_agent_sync
from src.agent.complexity import ComplexityLevel, ComplexityResult, get_complexity_analyzer

# ReAct 模块
from src.agent.react import ReActAgent, run_react

__all__ = [
    # Schema
    "AgentState",
    "AgentMessage",
    "MessageRole",
    "Intent",
    "QueryType",
    # Agents
    "SupervisorAgent",
    "RetrieverAgent",
    "GeneratorAgent",
    # Workflow
    "create_agent_workflow",
    "run_agent",
    "run_agent_sync",
    # Complexity
    "ComplexityLevel",
    "ComplexityResult",
    "get_complexity_analyzer",
    # ReAct
    "ReActAgent",
    "run_react",
]
