"""
Generator Agent - 带引用溯源的回答生成

职责：
1. 基于检索上下文生成回答
2. 添加来源引用
3. 防止幻觉
"""

from src.agent.schema import AgentState, QueryType
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from src.utils.llm import get_llm
from src.utils.logger import logger


# ============ Prompt 模板 ============

GENERATOR_SYSTEM_PROMPT = """你是一个知识库问答助手。你的职责是：

1. **从上下文中提取并回答**用户的问题
2. **必须**在回答中标注信息来源
3. **诚实**：只有当上下文中完全没有相关信息时，才说"没有找到"

## 核心原则

**重要**：如果上下文中有相关信息，你必须从中提取并回答！不要简单说"没有找到"。

例如：
- 用户问"我的光电子技术分数是多少"，上下文中显示"光电子技术 限选 2.0 32 91"
- 正确回答："根据您的成绩单，您在2023学年第一学期的光电子技术课程成绩是91分" [来源: xxx.pdf]
- 错误回答："对不起，我没有找到..." ← 这是错的！

## 回答要求

1. **引用格式**：
   - 每当使用上下文中某个来源的信息时，用 `[来源: 文件名]` 标注
   - 例如：`根据 [来源: RAG入门指南.pdf] 的内容，RAG 是...`
   - 网页来源标注为 `[来源: 网页 - 标题]`

2. **结构化回答**：
   - 优先使用清晰的标题和列表
   - 重要的定义和结论要突出

3. **诚实原则**：
   - 只基于提供的上下文回答
   - 不编造、不推测未在上下文中出现的信息
   - **只有**当"光电子技术"等关键词完全不在上下文中时，才说"没有找到"

4. **语言风格**：
   - 专业但易懂
   - 简洁有力，不废话
   - 适合中文用户

## 上下文格式

上下文会以以下格式提供：
```
【知识库检索结果】
[来源 1: 文件名 | 章节: xxx] (相关度: 0.85)
内容...

---
[来源 2: 文件名 | 章节: yyy] (相关度: 0.72)
内容...
```

或者网页搜索结果：
```
【网页搜索结果】
1. 标题
   链接: url
   摘要: 内容...

2. 标题
   链接: url
   摘要: 内容...
```

## 注意事项

- 如果上下文中没有相关内容，直接说"对不起，我在知识库中没有找到相关信息..."
- 不要在回答中编造细节
- 回答长度适中，通常 200-500 字
- 优先使用知识库内容，网页内容作为补充
"""


GENERATOR_NO_CONTEXT_PROMPT = """用户询问了问题，但知识库中没有找到相关信息。

请生成一个友好的回复：
1. 告知用户没有找到相关信息
2. 建议用户：
   - 尝试其他关键词
   - 上传更多相关文档
   - 调整问题表述

保持简洁、友好、有帮助。
"""


CHAT_SYSTEM_PROMPT = """你是一个友好的AI助手。

请：
1. 简洁、自然地回应用户
2. 如果用户有问题，引导他们使用知识库功能
3. 保持友好和专业
"""


PURE_LLM_BASE_PROMPT = """你是一个企业办公数据助手 AI Agent，负责回答用户的问题。

## 你的系统能力

你拥有以下系统级能力，系统会根据用户问题的类型自动选择合适的分支来处理：

- **RAG 检索增强生成**：如果你问的是需要专业知识或文档支持的问题，系统会从知识库中检索相关内容，再结合检索结果生成答案，确保回答有据可查。
- **ReAct 推理循环**：对于复杂、需要多步推理的问题，系统会进入"思考-行动-观察"循环，可以调用检索、联网搜索等子能力，逐步推理出最终答案。
- **联网搜索 (Web Search)**：对于需要最新信息或知识库中没有覆盖的问题，系统可以通过 Tavily 搜索引擎获取互联网上的实时信息。
- **数据处理**：如果你上传了 CSV/Excel 等数据文件，系统可以自动执行数据清洗、探索性数据分析（EDA），并生成可视化 HTML 报告。

## 你的基础能力
- 你有自己的通用知识，可以回答常识性问题
- 你可以看到对话历史，记得我们之前聊过什么
- 你可以进行自然的对话交流
"""

PURE_LLM_TOOLS_HEADER = "\n## 你的工具\n以下工具已注册到系统中，你可以调用它们完成任务：\n"

PURE_LLM_TAIL_PROMPT = """
## 回答要求
1. 直接回答问题，不要过度谦虚
2. 简洁有力，不啰嗦
3. 如果用户问的是"你"相关的问题（你是谁、你能做什么、你有什么功能/工具等），**自信地介绍自己，并完整列出上面【你的系统能力】中的所有能力项和【你的工具】中的所有工具，不要遗漏任何一个**
4. 使用自然的中文交流
5. 如果是问候、感谢等社交用语，礼貌回应即可
6. 如果用户询问特定领域的专业知识（没有上下文支持时），诚实地说这是你基于通用知识的理解
"""


class GeneratorAgent:
    """
    Generator Agent

    负责：
    1. 基于上下文生成回答
    2. 添加来源引用
    3. 处理闲聊模式
    """

    def __init__(self):
        self.llm = get_llm()

    async def generate(self, state: AgentState) -> AgentState:
        """
        生成回答

        Args:
            state: 当前状态（包含 context、web_context 和 query）

        Returns:
            更新后的状态（包含 response 和 messages）
        """
        query = state["query"]
        context = state.get("context", "")
        web_context = state.get("web_context", "")
        memory_context = state.get("memory_context", "")
        query_type = state.get("query_type")

        # 注入记忆上下文：如果有对话历史，拼接到用户问题前面
        if memory_context:
            query = f"""以下是我们之前的对话历史：

{memory_context}

---

当前用户的问题：{query}"""

        logger.info(f"Generator producing response for: {query[:50]}...")

        try:
            # 根据查询类型选择不同的生成策略
            logger.info(f"Generator branch: query_type={query_type}, has_context={bool(context)}, has_web={bool(web_context)}")
            
            if query_type == QueryType.PURE_LLM or str(query_type) == "pure_llm":
                # 纯 LLM 路径（不检索知识库，带对话历史）
                response = await self.generate_pure_llm(state, state["messages"])
                logger.info("Generator: using generate_pure_llm")
            elif query_type == QueryType.CHAT or str(query_type) == "chat":
                response = await self._generate_chat(query)
                logger.info("Generator: using _generate_chat")
            elif web_context and web_context.strip():
                # 联网搜索结果
                response = await self._generate_with_web_context(query, web_context, context)
                logger.info("Generator: using _generate_with_web_context")
            elif context and context.strip():
                # 知识库检索结果
                response = await self._generate_with_context(query, context)
                logger.info("Generator: using _generate_with_context")
            else:
                response = await self._generate_no_context(query)
                logger.info("Generator: using _generate_no_context")

            state["response"] = response

            # 添加助手消息到历史
            state["messages"].append(AIMessage(
                content=response
            ))

            logger.info(f"Generator done, response length: {len(response) if isinstance(response, str) else 'N/A'}")

        except Exception as e:
            import traceback
            logger.error(f"Generation failed: {e}\n{traceback.format_exc()}")
            state["error"] = f"生成失败: {str(e)}"
            state["response"] = "抱歉，生成回答时出现了问题。请稍后重试。"
            state["messages"].append(AIMessage(
                content=state["response"]
            ))

        return state

    async def _generate_with_context(self, query: str, context: str) -> str:
        """基于上下文生成回答"""
        from langchain_core.messages import HumanMessage, SystemMessage
        
        user_message = f"""请根据以下上下文回答用户问题。

【上下文】
{context}

---
【用户问题】
{query}

请直接从上下文中提取答案，不要说"没有找到"，除非上下文中完全没有相关信息。"""
        
        langchain_messages = [
            SystemMessage(content=GENERATOR_SYSTEM_PROMPT),
            HumanMessage(content=user_message)
        ]

        response = await self.llm.ainvoke(langchain_messages)
        return response.content.strip()

    async def _generate_with_web_context(
        self,
        query: str,
        web_context: str,
        knowledge_context: str = "",
    ) -> str:
        """基于网页搜索结果生成回答（可附加知识库上下文）"""
        # 构建完整的上下文
        full_context_parts = []

        # 知识库优先
        if knowledge_context and knowledge_context.strip():
            full_context_parts.append(f"【知识库检索结果】\n{knowledge_context}")

        # 网页搜索补充
        full_context_parts.append(f"【网页搜索结果】\n{web_context}")

        full_context = "\n\n---\n\n".join(full_context_parts)

        langchain_messages = [
            SystemMessage(content=GENERATOR_SYSTEM_PROMPT),
            HumanMessage(content=f"上下文：\n{full_context}\n\n---\n\n用户问题：{query}")
        ]

        response = await self.llm.ainvoke(langchain_messages)
        return response.content.strip()

    async def _generate_no_context(self, query: str) -> str:
        """没有上下文时生成回答"""
        from langchain_core.messages import HumanMessage, SystemMessage
        
        langchain_messages = [
            SystemMessage(content=GENERATOR_NO_CONTEXT_PROMPT),
            HumanMessage(content=query)
        ]

        response = await self.llm.ainvoke(langchain_messages)
        return response.content.strip()

    async def _generate_chat(self, query: str) -> str:
        """闲聊模式"""
        langchain_messages = [
            SystemMessage(content=CHAT_SYSTEM_PROMPT),
            HumanMessage(content=query)
        ]

        response = await self.llm.ainvoke(langchain_messages)
        return response.content.strip()

    def _build_pure_llm_prompt(self) -> str:
        """动态构建 Pure LLM system prompt，从 ToolRegistry 注入当前工具列表"""
        prompt = PURE_LLM_BASE_PROMPT

        try:
            from src.agent.tools.registry import get_registry
            registry = get_registry()
            tools_text = ""
            for name, tool in registry.list_tools(enabled_only=True):
                desc = tool.definition.description
                tools_text += f"- **{name}**：{desc}\n"
                logger.info(f"[Pure LLM] 注入工具: {name}")
            if tools_text:
                prompt += PURE_LLM_TOOLS_HEADER + tools_text
                logger.info(f"[Pure LLM] 已注入工具列表到 system prompt")
            else:
                logger.info("[Pure LLM] 无已启用工具")
        except Exception as e:
            logger.warning(f"[Pure LLM] 获取工具列表失败: {e}")

        prompt += PURE_LLM_TAIL_PROMPT
        return prompt

    async def generate_pure_llm(self, state: AgentState, messages: list) -> str:
        """
        纯 LLM 生成模式（不检索知识库，带完整对话历史）

        Args:
            state: 当前 agent 状态
            messages: 之前的对话历史（从 state["messages"] 传入）
        """
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

        query = state["query"]

        # 从历史消息中提取对话链（Human / AI 交替），跳过系统消息
        conversation_pairs = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                conversation_pairs.append(f"用户：{msg.content}")
            elif isinstance(msg, AIMessage):
                conversation_pairs.append(f"助手：{msg.content}")
            # SystemMessage 不加入对话历史

        history_text = "\n".join(conversation_pairs)

        # 如果有对话历史，注入到用户消息前面
        if history_text:
            full_user_message = f"""以下是我们的对话历史：

{history_text}

---

当前用户的问题：{query}"""
        else:
            full_user_message = query

        # 动态构建 system prompt（含当前注册的工具列表）
        system_prompt = self._build_pure_llm_prompt()
        langchain_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=full_user_message)
        ]

        logger.info(f"generate_pure_llm: history turns={len(conversation_pairs)//2}, query_len={len(query)}")
        response = await self.llm.ainvoke(langchain_messages)
        return response.content.strip()

    def format_response_with_citations(self, response: str, search_results) -> str:
        """
        格式化带引用的回答（备用方法）

        当前已在 prompt 中要求 LLM 直接添加引用
        这个方法用于后处理，进一步优化引用格式
        """
        # TODO: 可以添加引用索引、超链接等功能
        return response
