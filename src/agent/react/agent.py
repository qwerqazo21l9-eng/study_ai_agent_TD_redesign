"""
ReAct Agent - 复杂查询的推理循环

实现 Observe → Think → Act → Evaluate 循环：
1. Observe - 观察当前状态和已有信息
2. Think - 分析下一步行动
3. Act - 执行动作（检索/搜索/思考）
4. Evaluate - 评估是否达到目标

适用场景：
- 需要联网获取最新信息
- 需要多步推理和综合分析
- 混合知识库 + 联网查询
"""

import json
import re
from typing import Optional

from src.agent.react.state import (
    ReActState,
    ActionType,
    create_initial_react_state,
    add_action_to_history,
    add_observation,
)
from src.agent.react.prompts import (
    build_think_prompt,
    build_evaluate_prompt,
    build_final_generate_prompt,
)
from src.agent.react.tools import (
    get_react_retrieve_tool,
    get_react_web_search_tool,
    get_react_think_tool,
    ReActToolResult,
)
from src.utils.logger import logger
from src.utils.llm import get_llm


class ReActAgent:
    """
    ReAct 循环 Agent

    用于处理复杂查询，通过多轮推理循环逐步收集信息并生成答案。

    循环流程：
    1. Observe - 获取当前状态
    2. Think - LLM 分析决策下一步
    3. Act - 执行工具
    4. Evaluate - 评估是否完成
    """

    # 置信度阈值，高于此值认为已完成
    CONFIDENCE_THRESHOLD = 0.7

    # 最大迭代次数
    MAX_ITERATIONS = 5

    def __init__(self, max_iterations: int = 5, confidence_threshold: float = 0.7):
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        self.llm = get_llm()

    async def run(self, query: str, state: dict | None = None, enable_web: bool = True) -> dict:
        """
        运行 ReAct 循环

        Args:
            query: 用户查询
            state: 外部状态字典，用于跨上下文传递
            enable_web: 是否启用联网搜索

        Returns:
            dict: 包含 response、context、iterations 等
        """
        logger.info(f"=== ReAct Loop starting for: {query[:50]}... ===")

        # 创建初始状态，同时吸收外部状态
        init_state = create_initial_react_state(query, self.max_iterations)
        if isinstance(state, dict):
            # 合并外部状态，不覆盖核心字段
            for k, v in state.items():
                if k not in init_state:
                    init_state[k] = v
        state = init_state

        # 执行循环
        state = await self._run_loop(state, enable_web)

        # 构建 action_history，处理各种可能的 action_type 类型
        action_history = []
        for a in state.get("action_history", []):
            try:
                action_type_value = a.action_type.value if hasattr(a.action_type, "value") else str(a.action_type)
                action_history.append({
                    "type": action_type_value,
                    "input": a.action_input,
                    "output": a.action_output,
                })
            except Exception as e:
                logger.warning(f"Failed to serialize action: {e}")
                continue

        return {
            "response": state.get("response"),
            "context": state.get("context_collected", []),
            "iterations": state.get("iteration", 0),
            "is_complete": state.get("is_complete", False),
            "confidence": state.get("confidence", 0.0),
            "action_history": action_history,
            "error": state.get("error"),
        }

    async def _run_loop(self, state: ReActState, enable_web: bool) -> ReActState:
        """执行 ReAct 主循环"""
        # 获取工具
        retrieve_tool = get_react_retrieve_tool()
        web_search_tool = get_react_web_search_tool()
        think_tool = get_react_think_tool()

        while state["iteration"] < state["max_iterations"]:
            current_iter = state["iteration"]
            logger.info(f"=== Loop start: iteration={state['iteration']}, max={state['max_iterations']} ===")
            logger.info(f"--- ReAct Iteration {current_iter + 1}/{state['max_iterations']} ---")
            
            # 1. Think - 分析决策
            try:
                decision = await self._think(state)
                state["current_thought"] = decision.get("analysis", "")
                next_action = decision.get("next_action")
                action_input = decision.get("action_input", "")
                reasoning = decision.get("reasoning", "")
            except Exception as e:
                logger.error(f"Think error: {e}, fallback to RETRIEVE")
                next_action = "RETRIEVE"
                action_input = state["original_query"]
                reasoning = f"Think failed: {e}"

            logger.info(f"Decision: {next_action} - {reasoning}")

            # 强制策略：本地检索后发现需要联网（有关键词"最新/进展"等），强制 WEB_SEARCH
            query = state.get("original_query", "")
            needs_web = any(kw in query for kw in ["最新", "最近", "recent", "2024", "2025", "进展", "动态"])
            has_local_docs = bool(state.get("retrieved_docs"))
            has_web_results = bool(state.get("web_results"))

            if needs_web and has_local_docs and not has_web_results and next_action == "RETRIEVE":
                logger.info("Force: override RETRIEVE to WEB_SEARCH (needs_web + has_local_docs + no_web_results)")
                next_action = "WEB_SEARCH"
                if not action_input:
                    action_input = query

            # 兜底：action_input 为空时用原始 query，避免空检索/空搜索
            if not action_input and next_action in ("RETRIEVE", "WEB_SEARCH"):
                action_input = query

            # 2. 执行动作
            try:
                if next_action == "FINISH" or next_action == "GENERATE":
                    # 生成最终答案
                    state = await self._generate_final(state)
                    state["is_complete"] = True
                    state["confidence"] = self.confidence_threshold
                    break

                elif next_action == "RETRIEVE":
                    # 知识库检索
                    result = await retrieve_tool.execute(action_input)
                    state = self._process_action_result(
                        state, ActionType.RETRIEVE, action_input, result, reasoning
                    )

                elif next_action == "WEB_SEARCH" and enable_web:
                    # 联网搜索
                    result = await web_search_tool.execute(action_input)
                    state = self._process_action_result(
                        state, ActionType.WEB_SEARCH, action_input, result, reasoning
                    )

                elif next_action == "WEB_SEARCH" and not enable_web:
                    # 禁用联网时降级为检索
                    logger.warning("Web search disabled, fallback to retrieve")
                    result = await retrieve_tool.execute(action_input)
                    state = self._process_action_result(
                        state, ActionType.RETRIEVE, action_input, result, reasoning
                    )

                elif next_action == "THINK":
                    # 深度思考
                    context = self._build_context(state)
                    result = await think_tool.execute(state["original_query"], context)
                    state = self._process_action_result(
                        state, ActionType.THINK, "深度推理", result, reasoning
                    )

                else:
                    # 默认生成
                    state = await self._generate_final(state)
                    state["is_complete"] = True
                    break
            except Exception as e:
                logger.error(f"Action execution error: {e}")
                state = add_observation(state, f"动作执行错误: {e}", "错误")
                next_action = "GENERATE"  # 强制生成

            # 3. Evaluate - 检查是否可以结束
            should_finish = self._should_finish(state)
            logger.info(f"_should_finish result: {should_finish}, retrieved_docs={len(state.get('retrieved_docs', []))}, web_results={len(state.get('web_results', []))}, iteration={state['iteration']}")
            
            if should_finish:
                logger.info("ReAct: conditions met for finishing")
                state = await self._generate_final(state)
                state["is_complete"] = True
                break

            state["iteration"] += 1
            logger.info(f"After increment: iteration={state['iteration']}, while condition: {state['iteration']} < {state['max_iterations']} = {state['iteration'] < state['max_iterations']}")

        # 达到最大迭代仍未完成
        if not state.get("is_complete") and state["iteration"] >= state["max_iterations"]:
            logger.warning(f"ReAct: max iterations reached ({state['max_iterations']})")
            state = await self._generate_final(state)
            state["is_complete"] = True  # 强制结束

        return state

    async def _think(self, state: ReActState) -> dict:
        """
        Think 阶段：LLM 分析决策下一步行动

        Returns:
            dict: 包含 next_action, action_input, reasoning
        """
        # 构建 Prompt
        prompt = build_think_prompt(state)

        try:
            from langchain_core.messages import HumanMessage
            
            # ChatOpenAI.agenerate 期望 list[list[BaseMessage]]
            response = await self.llm.agenerate([
                [HumanMessage(content=prompt)]
            ])

            # 安全获取 generation 内容
            generation = response.generations[0][0]
            
            # ChatGeneration 对象有 .text 属性（字符串）
            # 也可能有 .message 属性（AIMessage 对象，有 .content 属性）
            if hasattr(generation, 'text'):
                response_text = generation.text.strip()
            elif hasattr(generation, 'content'):
                response_text = generation.content.strip()
            else:
                response_text = str(generation).strip()

            # 解析 JSON
            logger.info(f"Think LLM raw response: {response_text[:500]}")
            decision = self._parse_think_response(response_text)

            if decision:
                return decision

        except Exception as e:
            import traceback
            logger.error(f"Think stage error: {e}")
            traceback.print_exc()

        # 降级策略：默认检索
        return {
            "analysis": "默认分析",
            "gap": "无法确定",
            "next_action": "RETRIEVE",
            "action_input": state["original_query"],
            "reasoning": "默认使用知识库检索",
        }

    def _parse_think_response(self, response_text: str) -> Optional[dict]:
        """解析 Think 阶段的 JSON 响应"""
        try:
            # 优先提取 ```json ... ``` 代码块
            code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response_text)
            if code_block_match:
                json_str = code_block_match.group(1).strip()
            else:
                # fallback: 找最外层 {}
                json_match = re.search(r'\{[\s\S]*\}', response_text)
                if not json_match:
                    return None
                json_str = json_match.group()

            # 修复常见 LLM 输出问题：value 里的非法冒号（中文注释）
            # 例如 "analysis": "当前分析": "实际内容" → 取最后一个冒号后的内容
            # 用正则把 "key": "xxx": "yyy" 修复为 "key": "yyy"
            json_str = re.sub(
                r'"([^"]+)":\s*"[^"]*":\s*"([^"]*)"',
                r'"\1": "\2"',
                json_str
            )

            data = json.loads(json_str)

            return {
                "analysis": data.get("analysis", ""),
                "gap": data.get("gap", ""),
                "next_action": data.get("next_action", "FINISH"),
                "action_input": data.get("action_input", ""),
                "reasoning": data.get("reasoning", ""),
            }

        except Exception as e:
            logger.warning(f"Parse think response error: {e}")
            return None

    def _process_action_result(
        self,
        state: ReActState,
        action_type: ActionType,
        action_input: str,
        result,
        reasoning: str,
    ) -> ReActState:
        """处理动作结果"""
        # 防御性处理：兼容不同类型的 result
        success = False
        content = ""
        error = ""
        
        try:
            if hasattr(result, "success"):
                # ReActToolResult 类型
                success = result.success
                content = result.content if hasattr(result, "content") else ""
                error = result.error if hasattr(result, "error") else ""
            elif isinstance(result, dict):
                # 字典类型
                success = result.get("success", False)
                content = result.get("content", "") or result.get("text", "") or ""
                error = result.get("error", "") or ""
            else:
                # 其他类型转为字符串
                content = str(result) if result else ""
                success = bool(content)
        except Exception as e:
            logger.warning(f"Failed to extract action output: {e}")
            content = str(result) if result else ""
        
        state = add_action_to_history(
            state,
            action_type=action_type,
            action_input=action_input,
            action_output=content,
            reasoning=reasoning,
        )

        # 记录观察
        source_map = {
            ActionType.RETRIEVE: "知识库",
            ActionType.WEB_SEARCH: "联网",
            ActionType.THINK: "推理",
        }
        source = source_map.get(action_type, "动作")

        if success:
            state = add_observation(state, content[:500], source)
            # 收集上下文
            if action_type == ActionType.RETRIEVE:
                state["retrieved_docs"].append(content)
            elif action_type == ActionType.WEB_SEARCH:
                state["web_results"].append(content)
        else:
            state = add_observation(state, f"错误: {error}", source)
            logger.warning(f"Action {action_type.value if hasattr(action_type, 'value') else action_type} failed: {error}")

        return state

    def _build_context(self, state: ReActState) -> str:
        """构建上下文字符串"""
        parts = []

        # 检索到的文档
        if state.get("retrieved_docs"):
            parts.append("【知识库文档】")
            for doc in state["retrieved_docs"]:
                parts.append(doc)
            parts.append("")

        # 网页结果
        if state.get("web_results"):
            parts.append("【网页结果】")
            for web in state["web_results"]:
                parts.append(web)
            parts.append("")

        return "\n".join(parts)

    def _should_finish(self, state: ReActState) -> bool:
        """判断是否应该结束循环"""
        query = state.get("original_query", "")
        
        # 检查是否需要 web 搜索的关键词
        needs_web = any(kw in query for kw in ["最新", "最近", "news", "recent", "2024", "2025", "进展", "动态"])
        
        # 如果查询涉及最新进展，但没有 web 结果，不结束
        if needs_web and not state.get("web_results"):
            return False
        
        # 条件1: 有足够的上下文
        total_context = len(state.get("retrieved_docs", [])) + len(state.get("web_results", []))
        if total_context >= 3:
            return True

        # 条件2: 已有 web 结果且有文档结果
        if state.get("retrieved_docs") and state.get("web_results"):
            return True

        # 条件3: 多次检索无新结果
        if state["iteration"] >= 2 and not state.get("retrieved_docs"):
            return True

        return False

    async def _generate_final(self, state: ReActState) -> ReActState:
        """生成最终答案"""
        logger.info("ReAct: generating final answer")

        # 构建上下文：web 结果优先放前面，知识库作为补充
        context_list = []

        # 先放 web 搜索结果（最新信息，优先级高）
        if state.get("web_results"):
            context_list.append("=== 【网络搜索结果 - 最新信息，优先参考】 ===")
            context_list.extend(state["web_results"])

        # 再放本地文档（背景知识）
        if state.get("retrieved_docs"):
            context_list.append("=== 【知识库文档 - 背景参考】 ===")
            context_list.extend(state["retrieved_docs"])

        # 添加推理轨迹摘要（帮助 LLM 了解信息来源）
        action_history = state.get("action_history", [])
        if action_history:
            steps = []
            for a in action_history:
                atype = a.action_type.value if hasattr(a.action_type, "value") else str(a.action_type)
                steps.append(f"- {atype.upper()}: {a.action_input}")
            context_list.append("=== 【推理过程】 ===\n" + "\n".join(steps))

        system_prompt, user_prompt = build_final_generate_prompt(
            state["original_query"],
            context_list,
        )

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            msgs = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

            # 将上一轮对话摘要补进最终生成，避免“这方面的.../接着...”
            # 这种指代性提问被模型当成独立问题处理，导致答非所问。
            try:
                from src.utils.config_loader import config as _cfg
                _mem_output_dir = str(
                    _cfg.data_processing.get("output_dir", "./output/data_processing")
                )
            except Exception:
                _mem_output_dir = "./output/data_processing"
            history_block = (
                f"【上轮任务产物参考路径】{_mem_output_dir}\n"
                f"【上轮摘要】{_build_compact_summary(state)}\n"
            )
            msgs.append(
                SystemMessage(
                    content=history_block
                    + "如果用户问题含“这类/上述/刚才/这个任务/其/他/他的/接着/继续/上一轮”等指代，"
                    "必须先基于【上轮摘要】回顾上一轮任务对象后再回答，不要泛化成通用介绍。"
                )
            )

            # 当用户用了明显指代词时，额外注入 recent_turns 片段，降低模型把问题当独立新问的概率。
            query = state.get("original_query", "")
            followup_hints = any(k in query for k in [
                "这类", "上述", "刚才", "这个任务", "这个", "接着", "继续", "上一轮", "其", "他的", "她的", "参考"
            ])
            if followup_hints and hasattr(state, "keys"):
                raw_history = ""
                try:
                    from src.agent.memory.manager import get_memory_manager as _g
                    _m = _g()
                except Exception:
                    _m = None
                if _m is not None:
                    try:
                        context_pack = await _m.get_context(query=query)
                        raw_history = (context_pack.get("full_context") or "").strip()
                    except Exception:
                        raw_history = ""
                if not raw_history:
                    raw_history = (state.get("history_summary") or "").strip()
                if raw_history:
                    # 增补进历史摘要块，覆盖到上下文里，避免被漏读。
                    msgs.append(
                        SystemMessage(
                            content="【引入参考历史】\n" + raw_history[:1200] + "\n"
                        )
                    )

            # ChatOpenAI.agenerate 期望 list[list[BaseMessage]]
            response = await self.llm.agenerate([msgs])

            # 安全获取 generation 内容
            generation = response.generations[0][0]
            
            # ChatGeneration 对象有 .text 属性（字符串）
            if hasattr(generation, 'text'):
                state["response"] = generation.text.strip()
            elif hasattr(generation, 'content'):
                state["response"] = generation.content.strip()
            else:
                state["response"] = str(generation).strip()

            state["context_collected"] = context_list
            logger.info(f"ReAct: generated answer, {len(state['response'])} chars")

        except Exception as e:
            logger.error(f"Generate final error: {e}")
            import traceback
            traceback.print_exc()
            state["response"] = f"生成回答时出错: {str(e)}"
            state["error"] = str(e)

        return state


def _build_compact_summary(state: dict) -> str:
    """以极紧凑格式重述上轮/历史任务，供续问时模型理解指代。"""
    # 优先使用外部注入的 history_summary（来自记忆管理器）
    injected = (state.get("history_summary") or "").strip()
    if injected:
        # 截断到合适长度
        if len(injected) > 300:
            injected = injected[-300:]
        return injected

    # 回退：从当前 ReAct 状态里的动作历史拼摘要
    parts = []
    try:
        parts.append(f"原问题：{state.get('original_query','')}")
    except Exception:
        pass
    try:
        history = state.get("action_history") or []
        if history:
            types = ", ".join(
                (a.action_type.value if hasattr(a.action_type, "value") else str(a.action_type))
                for a in history[:4]
            )
            parts.append(f"动作序列：{types}")
    except Exception:
        pass
    try:
        docs = state.get("retrieved_docs") or []
        webs = state.get("web_results") or []
        parts.append(
            f"知识库结果：{len(docs)}   网页搜索：{len(webs)}"
        )
    except Exception:
        pass
    return "；".join(parts) if parts else "（继前续问，无可用历史摘要）"


# ============ 便捷函数 ============

_react_agent: Optional[ReActAgent] = None


def get_react_agent(max_iterations: int = 5, confidence_threshold: float = 0.7) -> ReActAgent:
    """获取 ReAct Agent 单例"""
    global _react_agent
    if _react_agent is None:
        _react_agent = ReActAgent(max_iterations, confidence_threshold)
    return _react_agent


async def run_react(query: str, state: dict | None = None, enable_web: bool = True) -> dict:
    """
    便捷函数：运行 ReAct 循环

    Args:
        query: 用户查询
        state: 外部状态字典，用于跨轮上下文传递（如 history_summary）
        enable_web: 是否启用联网

    Returns:
        dict: 包含 response、context、iterations 等
    """
    agent = get_react_agent()
    return await agent.run(query, state=state, enable_web=enable_web)
