"""ReAct Prompt 模板"""

OBSERVE_TEMPLATE = """## Observe（观察阶段）

当前状态：
- 原始查询：{original_query}
- 当前迭代：{iteration}/{max_iterations}
- 已收集上下文：
{context_summary}

已执行的动作：
{action_summary}

请根据以上信息，明确当前的状态和已有信息。"""



THINK_TEMPLATE = """你是一个 ReAct 推理引擎，每一轮都要阅读完整的推理轨迹，然后决定下一步行动。

## 用户问题
{original_query}

## 推理轨迹（每轮的 Thought → Action → Observation）
{trajectory}

## 当前要求

基于上面的完整轨迹，判断：
1. 目前已经收集到了哪些信息？这些信息**能否充分回答用户问题**？
2. 还缺少什么信息？
3. 下一步应该做什么？

**行动选项：**
- RETRIEVE：从本地知识库检索（优先；尚无本地信息时必须先选它）
- WEB_SEARCH：联网搜索最新信息（已有本地信息但需要最新进展/时效性内容时选它）
- GENERATE：基于已有信息生成最终回答（信息已充分）
- FINISH：任务完成

**决策规则：**
- 轨迹为空 → 必须选 RETRIEVE
- 有本地文档但用户问"最新/进展/趋势/2024/2025"→ 必须选 WEB_SEARCH
- 已有本地文档 + 网络搜索结果 → 选 GENERATE
- 仅有本地文档，内容已足够回答问题 → 选 GENERATE

请严格按以下 JSON 格式输出，value 只填纯文本，不要嵌套冒号：

```json
{{
    "analysis": "<基于轨迹对当前信息的分析>",
    "gap": "<还缺什么信息，或者'已充分'>",
    "next_action": "<RETRIEVE / WEB_SEARCH / GENERATE / FINISH>",
    "action_input": "<检索词或搜索词，GENERATE/FINISH 时填空字符串>",
    "reasoning": "<选择该动作的一句话原因>"
}}
```"""



ACT_RETRIEVE_TEMPLATE = """## Act - RETRIEVE（知识库检索）

执行知识库检索，收集相关信息。

**检索词：** {query}
**理由：** {reasoning}

检索结果：
{result}"""



ACT_WEB_SEARCH_TEMPLATE = """## Act - WEB_SEARCH（联网搜索）

执行联网搜索，获取最新信息。

**搜索词：** {query}
**理由：** {reasoning}

搜索结果：
{result}"""



ACT_THINK_TEMPLATE = """## Act - THINK（深度思考）

基于已有信息进行深度推理。

**思考内容：**
{content}

**推理结果：**
{result}"""



ACT_GENERATE_TEMPLATE = """## Act - GENERATE（生成答案）

基于收集的所有信息生成最终回答。

**用户问题：** {original_query}

**收集的上下文：**
{context}

请生成一个完整、准确、有引用的回答。"""



EVALUATE_TEMPLATE = """## Evaluate（评估阶段）

评估当前答案是否充分。

**原始问题：** {original_query}

**收集的上下文：**
{context}

**生成的答案：** 
{answer}

**评估标准：**
1. 答案是否直接回应了用户问题？
2. 答案是否基于收集的上下文？
3. 答案是否有足够的细节和信息量？
4. 是否需要更多信息？

请输出评估结果：
{{
    "is_complete": true|false,
    "confidence": 0.0-1.0,
    "reasoning": "评估理由",
    "missing_info": "缺少的信息（如有）",
    "suggestion": "建议下一步行动（如需要继续）"
}}"""



# ============ 最终生成 Prompt ============

FINAL_GENERATE_SYSTEM = """你是一个智能问答助手，负责基于收集的上下文生成高质量回答。

要求：
1. 直接回答用户问题，不要重复问题
2. **优先使用联网搜索结果**（标注"【网络信息】"）回答，这是最新信息
3. 知识库文档作为背景补充（标注"【知识库】"）
4. 回答简洁清晰，分点说明
5. 如信息不足，明确说明

上下文来源：
- 【知识库文档】：来自用户本地文档，可能不是最新信息
- 【网络搜索】：来自网络实时搜索结果，反映最新进展
- 【对话历史摘要】：用于理解“这类/上述/刚才/这个任务”等指代，不能忽略

**当同时有网络搜索结果和知识库文档时，必须以网络搜索结果为主要答案来源。**

重要：
- 如果用户问题含“这类/上述/刚才/这个任务/其/他/他的”或类似指代，必须先显式回顾上轮任务，再针对性回答，不要泛化成通用介绍。
- 如果【对话历史摘要】存在，优先用它解释指代对象。
"""



FINAL_GENERATE_USER = """## 用户问题
{original_query}

## 收集的上下文
{context}

## 要求
请基于以上上下文生成回答。回答应该：
1. 直接回答问题
2. 引用信息来源
3. 如信息不足，说明局限性

请开始生成回答："""



def build_observe_prompt(state: dict) -> str:
    """构建观察阶段 Prompt"""
    context_parts = []
    if state.get("retrieved_docs"):
        context_parts.append(f"【知识库文档】{len(state['retrieved_docs'])} 篇")
    if state.get("web_results"):
        context_parts.append(f"【网页结果】{len(state['web_results'])} 条")
    if state.get("observations"):
        context_parts.append(f"【观察记录】{len(state['observations'])} 条")
    
    context_summary = "\n".join(context_parts) or "（暂无）"

    action_summary_parts = []
    for i, action in enumerate(state.get("action_history", [])):
        action_type = action.action_type.value if hasattr(action.action_type, "value") else str(action.action_type)
        action_summary_parts.append(
            f"{i+1}. {action_type}: {action.action_input[:50]}... "
            f"→ {action.action_output[:50]}..."
        )
    action_summary = "\n".join(action_summary_parts) or "（无）"

    observe_prompt = OBSERVE_TEMPLATE.format(
        original_query=state["original_query"],
        iteration=state["iteration"],
        max_iterations=state["max_iterations"],
        context_summary=context_summary,
        action_summary=action_summary,
    )

    history = (state.get("history_summary") or "").strip()
    if history:
        observe_prompt = f"{observe_prompt}\n\n【对话历史摘要】\n{history[-800:]}"

    return observe_prompt



def build_think_prompt(state: dict) -> str:
    """构建思考阶段 Prompt —— 以完整的推理轨迹格式给 LLM"""
    action_history = state.get("action_history", [])
    observations = state.get("observations", [])

    if not action_history:
        trajectory = "（尚无任何动作，这是第一轮）"
    else:
        trajectory_parts = []
        for i, action in enumerate(action_history):
            action_type = (
                action.action_type.value
                if hasattr(action.action_type, "value")
                else str(action.action_type)
            )
            thought = action.reasoning or "(无)"
            act_line = f"Action: {action_type.upper()}({action.action_input})"
            obs_content = observations[i] if i < len(observations) else action.action_output
            obs_preview = obs_content[:400] + "..." if len(obs_content) > 400 else obs_content

            trajectory_parts.append(
                f"--- 第 {i+1} 轮 ---\n"
                f"Thought: {thought}\n"
                f"{act_line}\n"
                f"Observation: {obs_preview}"
            )
        trajectory = "\n\n".join(trajectory_parts)

    # 注入上文摘要：ReAct 决策阶段不能失忆
    history = (state.get("history_summary") or "").strip()
    if history:
        trajectory = f"【对话历史摘要】\n{history[-800:]}\n\n{trajectory}"

    return THINK_TEMPLATE.format(
        original_query=state["original_query"],
        trajectory=trajectory,
    )



def build_evaluate_prompt(state: dict, generated_answer: str) -> str:
    """构建评估阶段 Prompt"""
    context_parts = []
    for doc in state.get("retrieved_docs", []):
        context_parts.append(f"- {doc}")
    for web in state.get("web_results", []):
        context_parts.append(f"- {web}")
    context = "\n".join(context_parts) or "（无上下文）"

    evaluate_prompt = EVALUATE_TEMPLATE.format(
        original_query=state["original_query"],
        context=context,
        answer=generated_answer,
    )

    history = (state.get("history_summary") or "").strip()
    if history:
        evaluate_prompt = f"{evaluate_prompt}\n\n【对话历史摘要】\n{history[-800:]}"

    return evaluate_prompt



def build_final_generate_prompt(original_query: str, context_list: list[str]) -> tuple[str, str]:
    """
    构建最终生成 Prompt

    Returns:
        (system_prompt, user_prompt)
    """
    context_combined = "\n\n".join(context_list) or "（无相关上下文）"

    return (
        FINAL_GENERATE_SYSTEM,
        FINAL_GENERATE_USER.format(
            original_query=original_query,
            context=context_combined,
        )
    )