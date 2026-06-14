"""
LangGraph 工作流编排

支持三层路由架构：
1. Pure LLM（纯 LLM）- 简单闲聊、问候、个人信息询问，直接 LLM + 对话历史
2. RAG（知识库检索）- 知识库相关查询，Supervisor → RAG → 生成
3. ReAct（推理循环）- 复杂查询，多轮推理循环

核心设计：
- 先判断查询类型，再路由到对应路径
- Pure LLM 路径延迟最低，不检索知识库
- ReAct 路径延迟最高，深度探索
- 通过 router 节点自动判断
"""

import re
from typing import Literal, Optional

from langgraph.graph import StateGraph, END

from src.agent.schema import AgentState, QueryType, RouteType, create_initial_state
from src.agent.supervisor import SupervisorAgent
from src.agent.retriever_agent import RetrieverAgent
from src.agent.generator import GeneratorAgent
from src.agent.memory.manager import MemoryManager, get_memory_manager
from src.utils.logger import logger


# ============ 全局记忆管理器 ============

_memory_manager: Optional[MemoryManager] = None


def set_memory_manager(manager: MemoryManager) -> None:
    """设置全局记忆管理器"""
    global _memory_manager
    _memory_manager = manager


def get_memory() -> Optional[MemoryManager]:
    """获取全局记忆管理器"""
    return _memory_manager


# ============ 三路路由器 ============

# 纯 LLM 关键词（走 Pure LLM 路径）
PURE_LLM_KEYWORDS = [
    "你好", "hi", "hello", "嗨", "嗨嗨",
    "谢谢", "thanks", "感谢", "谢啦",
    "再见", "拜拜", "bye", "结束",
    "好的", "收到", "明白", "了解",
    "可以", "没问题",
    "你是谁", "你叫什么", "你能做什么",
    "有什么功能", "你的功能", "你的工具", "有什么工具",
    "你觉得", "你的看法", "你的观点",
]

# ReAct 关键词（走 ReAct 路径）
REACT_KEYWORDS = [
    "最新", "今天", "现在", "当前", "最近",
    "分析", "趋势", "预测", "对比", "比较",
    "搜索", "查找", "帮我查", "查一下",
    "研究", "调研", "深入", "全面",
    "结合", "综合", "多角度",
    "2025", "2026", "news",
]

# 数据处理关键词（走 tools 路径，触发 DataProcessingTool）
DATA_PROCESSING_KEYWORDS = [
    "数据处理", "数据清洗", "数据清洗", "数据预处理",
    "EDA", "探索性数据分析", "数据分析",
    "清洗数据", "处理数据", "整理数据",
    "缺失值", "异常值", "数据质量",
    "请对", "csv", ".csv", "表格",
]


def determine_route(query: str) -> RouteType:
    """
    判断查询走哪条路径

    规则：
    1. Pure LLM: 问候、闲聊、个人相关问题、简短确认
    2. ReAct: 需要联网、分析、多步推理的复杂查询
    3. RAG: 默认兜底，知识库查询
    """
    query_lower = query.lower().strip()
    query_len = len(query)

    # 先检查是否有数据处理意图（走 tools 路径）
    for kw in DATA_PROCESSING_KEYWORDS:
        if kw in query_lower:
            logger.info(f"Route: TOOLS (data processing keyword: {kw})")
            return RouteType.TOOLS

    # 先检查是否有联网搜索指令
    search_patterns = [
        r"帮我搜", r"帮我查", r"帮我找",
        r"去网上", r"上网", r"搜索一下",
        r"网上查", r"帮我在网上",
    ]
    for pattern in search_patterns:
        if re.search(pattern, query_lower):
            logger.info(f"Route: REACT (explicit search request: {pattern})")
            return RouteType.REACT

    # 检查 Pure LLM 关键词
    for kw in PURE_LLM_KEYWORDS:
        if kw in query_lower:
            logger.info(f"Route: PURE_LLM (keyword: {kw})")
            return RouteType.PURE_LLM

    # 简单定义类查询（短查询 + 问"什么是" → RAG）
    definition_patterns = [r"^什么是", r"^什么叫", r"的概念", r"的定义", r"是什么"]
    is_definition = any(re.search(p, query_lower) for p in definition_patterns)
    if is_definition and query_len < 20:
        logger.info("Route: RAG (short definition query)")
        return RouteType.RAG

    # 检查 ReAct 关键词
    for kw in REACT_KEYWORDS:
        if kw in query_lower:
            logger.info(f"Route: REACT (keyword: {kw})")
            return RouteType.REACT

    # 长查询（> 50 字）→ ReAct
    if query_len > 50:
        logger.info(f"Route: REACT (long query: {query_len} chars)")
        return RouteType.REACT

    # 简短查询（< 10 字且无问号）→ Pure LLM
    if query_len < 10 and "?" not in query and "？" not in query:
        logger.info(f"Route: PURE_LLM (short query without question)")
        return RouteType.PURE_LLM

    # 默认 → RAG
    logger.info("Route: RAG (default)")
    return RouteType.RAG


# ============ 节点定义 ============

async def supervisor_node(state: AgentState) -> AgentState:
    """Supervisor 节点：意图分析"""
    supervisor = SupervisorAgent()
    return await supervisor.analyze(state)


def router_node(state: AgentState) -> dict:
    """
    路由器节点：判断查询走哪条路径

    Returns:
        dict: 包含 route_type 的状态更新
    """
    query = state["query"]
    route = determine_route(query)
    state["route_type"] = route
    logger.info(f"Router: {route.value}")
    return state


async def pure_llm_node(state: AgentState) -> AgentState:
    """
    Pure LLM 节点：纯 LLM 回答，不检索知识库

    适用于问候、闲聊、个人信息询问等简单查询
    """
    logger.info("=== Pure LLM: Direct LLM without RAG ===")

    generator = GeneratorAgent()

    # 注入记忆上下文
    memory = get_memory()
    if memory:
        memory.add_user_message(state["query"])
        try:
            context = await memory.get_context(query=state["query"])
            full = context.get("full_context", "")
            if full.strip():
                state["memory_context"] = full
        except Exception as e:
            logger.warning(f"Pure LLM - memory context injection failed: {e}")

    # 标记为 pure_llm 类型，让 generator 选择对应模式
    state["query_type"] = QueryType.PURE_LLM
    state["context"] = ""  # 确保不传 RAG 上下文

    # 生成回答
    state = await generator.generate(state)

    return state


async def retriever_node(state: AgentState) -> AgentState:
    """Retriever 节点：知识库检索"""
    retriever = RetrieverAgent()
    state = await retriever.retrieve(state)

    # 记录检索结果
    search_results = state.get("search_results", [])
    logger.info(f"Retriever node - 检索结果数量: {len(search_results)}")
    for i, doc in enumerate(search_results[:3]):
        content_preview = doc.get("content", "")[:100] if isinstance(doc, dict) else str(doc)[:100]
        logger.info(f"  结果{i+1}: {content_preview}...")

    return state


async def generator_node(state: AgentState) -> AgentState:
    """Generator 节点：生成回答"""
    generator = GeneratorAgent()

    # 如果有记忆管理器，添加消息到短期记忆并注入长期记忆上下文
    memory = get_memory()
    if memory:
        memory.add_user_message(state["query"])
        try:
            context = await memory.get_context(query=state["query"])
            full = context.get("full_context", "")
            if full.strip():
                state["memory_context"] = full
        except Exception as e:
            logger.warning(f"Failed to inject memory context: {e}")

    return await generator.generate(state)


async def chat_fallback_node(state: AgentState) -> AgentState:
    """闲聊降级节点"""
    generator = GeneratorAgent()

    memory = get_memory()
    if memory:
        memory.add_user_message(state["query"])

    return await generator.generate(state)


async def web_search_node(state: AgentState) -> AgentState:
    """WebSearch 节点：联网搜索"""
    supervisor = SupervisorAgent()
    result_state = await supervisor.web_search(state)

    memory = get_memory()
    if memory:
        memory.add_user_message(state["query"])

    return result_state


async def react_loop_node(state: AgentState) -> AgentState:
    """
    ReAct Loop 节点：推理循环

    适用于复杂查询，通过多轮推理逐步收集信息
    """
    from src.agent.react import run_react

    logger.info("=== ReAct Loop: Starting multi-turn reasoning ===")

    query = state["query"]

    # 将历史摘要注入 state，避免 ReAct 各轮全部丢失对话上下文，
    # 尤其是含“这类/上述/刚才/这个任务/其/他/他的”这类指代。
    try:
        from src.agent.memory.manager import get_memory_manager as _get_mem_mgr
        _mem = _get_mem_mgr()
        if _mem is not None:
            _ctx = await _mem.get_context(query=query)
            _full = (_ctx.get("full_context") or "").strip()
            if _full:
                state.setdefault("history_summary", "")
                # 截断到可管理的长度，减少 token 消耗
                if len(_full) > 1200:
                    _full = _full[-1200:]
                state["history_summary"] = (
                    (state.get("history_summary", "") + "\n" + _full).strip()
                )
    except Exception as e:
        logger.warning(f"[ReAct] memory context injection failed: {e}")

    # 打印 injected history_summary，方便定位续问时“上下文幻觉”问题
    _injected = (state.get("history_summary") or "").strip()
    logger.info(
        f"[ReAct] injected history_summary len={len(_injected)} "
        f"preview={_injected[:200]!r}"
    )

    # 检查是否启用联网
    try:
        from src.agent.tools.registry import get_registry
        registry = get_registry()
        web_tool = registry.get("web_search")
        enable_web = web_tool is not None
    except Exception:
        enable_web = False

    # 把 state 一起传给 ReAct，让它能读取 history_summary 和 query 完成承前启后
    try:
        result = await run_react(query, state=state, enable_web=enable_web)
        # 防御性检查：确保返回的是 dict
        if not isinstance(result, dict):
            logger.error(f"ReAct returned unexpected type: {type(result)}")
            result = {
                "response": f"推理循环返回类型错误: {type(result)}",
                "error": f"Expected dict, got {type(result)}",
                "iterations": 0,
            }
    except Exception as e:
        logger.error(f"ReAct loop error: {e}")
        result = {
            "response": f"推理循环执行出错: {str(e)}",
            "error": str(e),
            "iterations": 0,
        }

    # 将结果合并到状态
    state["response"] = result.get("response")
    state["react_context"] = result.get("context", [])
    state["react_iterations"] = result.get("iterations", 0)
    state["is_react_complete"] = result.get("is_complete", False)
    state["react_confidence"] = result.get("confidence", 0.0)
    state["error"] = result.get("error")

    logger.info(f"ReAct Loop: completed in {result.get('iterations', 0)} iterations")

    return state


def memory_node(state: AgentState) -> AgentState:
    """
    记忆节点：从长期记忆中检索相关上下文（同步桩，实际注入在各 async 节点里完成）
    """
    # 注意：get_context 是 async 方法，无法在同步节点里 await。
    # 实际记忆上下文注入已在 pure_llm_node / generator_node 里各自异步完成。
    # 这里保留节点以便后续扩展（如同步预处理、日志记录等）。
    return state


def post_process_node(state: AgentState) -> AgentState:
    """
    后处理节点：对话结束后触发
    """
    memory = get_memory()
    if memory and state.get("response"):
        memory.add_ai_message(state["response"])

    return state


async def tools_node(state: AgentState) -> AgentState:
    """
    Tools 节点：执行 DataProcessingTool 等工具
    """
    import os
    import json
    import re
    from src.agent.tools.registry import get_registry
    from src.agent.schema import QueryType, RouteType

    query = state["query"]
    query_type = state.get("query_type")

    logger.info(f"[Tools Node] 执行工具，query_type={query_type}")

    # 目前只处理 DATA_PROCESSING / TOOLS 路由
    route_type = state.get("route_type")
    if query_type == QueryType.DATA_PROCESSING or route_type == RouteType.TOOLS:
        registry = get_registry()
        tool = registry.get("data_processing")
        if tool:
            try:
                # 从 query 中提取 file_path
                # 格式1: C:\path\to\file.csv （含空格也可）
                # 格式2: C:\path\to\file 对该文件/请...
                file_path_match = re.search(
                    r'([a-zA-Z]:\\(?:[^<>\n]+?\\)*[^<>\n]+?)(?=\s*(?:对|请|帮|为|分析|处理|$))',
                    query
                )
                file_path = file_path_match.group(1).strip() if file_path_match else ""

                # 尝试补全 .csv 后缀
                if file_path and not os.path.exists(file_path) and not file_path.endswith('.csv'):
                    candidate = file_path + '.csv'
                    if os.path.exists(candidate):
                        file_path = candidate

                if not file_path or not os.path.exists(file_path):
                    state["response"] = "请提供有效的 CSV 文件路径。"
                    state["error"] = "Invalid file path"
                    return state

                # 执行工具
                result = await tool.execute(file_path=file_path)

                if result.success:
                    state["response"] = result.content or "数据处理完成。"
                    state["context"] = json.dumps(result.data, ensure_ascii=False, indent=2) if result.data else ""

                    # 提取 HTML 产物路径，传给前端侧边栏展示
                    deliverables = []
                    if result.data:
                        # 从顶层 deliverables 获取路径（绝对路径已在 master_agent 写入）
                        dlv = result.data.get("deliverables", {})
                        eda_html = dlv.get("eda_report_html", "")
                        if eda_html and os.path.isfile(eda_html):
                            deliverables.append({"label": "📊 Phase 1 数据清洗报告", "path": eda_html})
                            logger.info("[Tools Node] Phase 1 报告: %s", eda_html)

                        dist_html = dlv.get("dist_report_html", "")
                        if dist_html and os.path.isfile(dist_html):
                            deliverables.append({"label": "📈 Phase 2 EDA 分布报告", "path": dist_html})
                            logger.info("[Tools Node] Phase 2 报告: %s", dist_html)

                    state["deliverables"] = deliverables if deliverables else None
                    logger.info("[Tools Node] deliverables = %d 个报告", len(deliverables))
                else:
                    state["response"] = f"数据处理失败：{result.error}"
                    state["error"] = result.error

            except Exception as e:
                logger.error(f"[Tools Node] 执行失败: {e}")
                state["response"] = f"工具执行异常：{str(e)}"
                state["error"] = str(e)
        else:
            state["response"] = "DataProcessingTool 未注册。"
            state["error"] = "Tool not found"
    else:
        state["response"] = "未知的工具调用请求。"
        state["error"] = "Unknown tool request"

    return state


# ============ 路由决策函数 ============

def route_query(state: AgentState) -> Literal["pure_llm", "rag", "react", "tools"]:
    """
    基于 router 结果的路径决策

    Returns:
        "pure_llm" - 纯 LLM 路径（不检索知识库）
        "rag" - RAG 检索路径（Supervisor + RAG + 生成）
        "react" - ReAct 推理循环
        "tools" - 工具执行（数据自动化处理等）
    """
    route_type = state.get("route_type", RouteType.RAG)

    if route_type == RouteType.PURE_LLM:
        logger.info("Route -> Pure LLM")
        return "pure_llm"

    if route_type == RouteType.REACT:
        logger.info("Route -> ReAct Loop")
        return "react"

    if route_type == RouteType.TOOLS:
        logger.info("Route -> Tools (data processing)")
        return "tools"

    logger.info("Route -> RAG (via supervisor)")
    return "rag"


def route_after_supervisor(state: AgentState) -> Literal["retriever", "web_search", "chat", "generator"]:
    """
    Supervisor 之后的路由（在 RAG 路径中使用）

    Returns:
        "retriever" - 知识库检索
        "web_search" - 联网搜索
        "chat" - 闲聊（降级）
        "generator" - 直接生成
    """
    supervisor = SupervisorAgent()
    return supervisor.route(state)


# ============ 工作流构建 ============

def create_agent_workflow(enable_memory: bool = True):
    """
    创建支持三路路由的 Agent 工作流

    流程：
    1. router -> 判断查询类型
    2. 三路分支：
    - pure_llm -> post_process (纯 LLM，不检索)
    - rag -> supervisor -> 意图分析
    - react -> ReAct 推理循环
    3. supervisor 之后（RAG 路径）：
    - chat -> generator
    - retriever -> generator
    - web_search -> retriever -> generator
    4. post_process -> 结束
    """
    workflow = StateGraph(AgentState)

    # === 添加节点 ===
    # 路由器
    workflow.add_node("router", router_node)

    # RAG 路径节点
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("chat_fallback", chat_fallback_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generator", generator_node)

    # Pure LLM 路径
    workflow.add_node("pure_llm", pure_llm_node)

    # ReAct 路径
    workflow.add_node("react_loop", react_loop_node)

    # 工具执行节点
    workflow.add_node("tools", tools_node)

    # 公共节点
    workflow.add_node("post_process", post_process_node)

    # 可选：记忆节点
    if enable_memory:
        workflow.add_node("memory", memory_node)

    # === 设置入口点 ===
    workflow.set_entry_point("router")

    # === 添加边 ===
    # 路由器 -> 三路分支 + tools
    workflow.add_conditional_edges(
    "router",
    route_query,
    {
    "pure_llm": "pure_llm",
    "rag": "supervisor",
    "react": "react_loop",
    "tools": "tools",
    }
    )

    # Pure LLM -> post_process
    workflow.add_edge("pure_llm", "post_process")

    # ReAct Loop -> post_process
    workflow.add_edge("react_loop", "post_process")

    # RAG 路径：Supervisor -> 后续路由
    workflow.add_conditional_edges(
    "supervisor",
    route_after_supervisor,
    {
    "chat": "chat_fallback",
    "retriever": "retriever",
    "web_search": "web_search",
    "generator": "generator",
    }
    )

    # web_search -> retriever（联网后补充知识库）
    workflow.add_edge("web_search", "retriever")

    # retriever -> generator
    workflow.add_edge("retriever", "generator")

    # chat_fallback -> generator
    workflow.add_edge("chat_fallback", "generator")

    # generator -> post_process
    workflow.add_edge("generator", "post_process")

    # tools -> post_process
    workflow.add_edge("tools", "post_process")

    # post_process -> 结束
    workflow.add_edge("post_process", END)

    return workflow.compile()


# ============ 便捷调用接口 ============

_compiled_workflow = None
_workflow_memory_enabled = True


def get_agent_workflow(enable_memory: bool = True):
    """获取编译后的工作流（单例）"""
    global _compiled_workflow, _workflow_memory_enabled

    if _compiled_workflow is None or _workflow_memory_enabled != enable_memory:
        _compiled_workflow = create_agent_workflow(enable_memory=enable_memory)
        _workflow_memory_enabled = enable_memory

    return _compiled_workflow


async def run_agent(
    query: str,
    enable_memory: bool = True,
    session_id: str = "default",
) -> dict:
    """
    运行 Agent（支持三层路由）

    Args:
        query: 用户查询
        enable_memory: 是否启用记忆
        session_id: 会话 ID

    Returns:
        包含 response 和相关信息的字典
    """
    logger.info(f"=== Running Agent for query: {query[:50]}... ===")

    # 创建初始状态
    initial_state = create_initial_state(query)

    # 设置会话 ID（仅在 session 变更时切换，避免清空同会话的短期记忆）
    memory = get_memory()
    if memory and memory.get_session_id() != session_id:
        memory.new_session(session_id)

    # 获取工作流
    workflow = get_agent_workflow(enable_memory=enable_memory)

    # 执行工作流
    final_state = await workflow.ainvoke(initial_state)

    # 后处理：添加到记忆
    if memory and final_state.get("response"):
        memory.add_ai_message(final_state["response"])

    # 构建返回结果
    route_type = final_state.get("route_type", RouteType.RAG)
    result = {
        "response": final_state.get("response", ""),
        "messages": final_state.get("messages", []),
        "context": final_state.get("context", ""),
        "web_context": final_state.get("web_context", ""),
        "web_results": final_state.get("web_results", []),
        "intent": final_state.get("intent"),
        "query_type": final_state.get("query_type"),
        "error": final_state.get("error"),
        "memory_context": final_state.get("memory_context", ""),
        "deliverables": final_state.get("deliverables"),  # 数据分析报告 HTML 产物
        # 路由信息
        "route_type": route_type.value if hasattr(route_type, "value") else str(route_type),
        "react_iterations": final_state.get("react_iterations", 0),
        "is_react_complete": final_state.get("is_react_complete", False),
    }

    # 记录路由路径
    route_label = result["route_type"]
    path_info = [f"路径: {route_label}"]
    if final_state.get("react_iterations", 0) > 0:
        path_info.append(f"ReAct循环: {final_state['react_iterations']}次")
    if path_info:
        logger.info(" | ".join(path_info))

    return result


# ============ 简单同步接口 ============

def run_agent_sync(query: str, enable_memory: bool = True, session_id: str = "default") -> dict:
    """
    同步版本（用于 Gradio 等不支持 async 的场景）
    """
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, run_agent(query, enable_memory, session_id)
                )
                return future.result()
        else:
            return asyncio.run(run_agent(query, enable_memory, session_id))
    except RuntimeError:
        return asyncio.run(run_agent(query, enable_memory, session_id))
