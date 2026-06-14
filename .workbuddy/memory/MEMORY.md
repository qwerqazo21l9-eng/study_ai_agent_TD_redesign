# MEMORY.md - 长期记忆

## 用户信息
- 身份：软件开发者，求职 AI Agent 基建方向
- 风格：先讨论方案确认后，再驱动代码实现
- 交流：偏好中文，技术细节直接，不喜欢废话
- 面试风格：简洁、结论先行、坦诚局限

## 项目信息
- 项目名：study_ai_agent
- 阶段：三路由系统（PURE_LLM / RAG / REACT）已完成并验证通过
- 当前：下一步可考虑面试准备、系统整体联调或其他新功能

## 项目信息
- 项目名：study_ai_agent_TD_redesign（三路由架构重构版）
- 阶段：三路由系统（PURE_LLM / RAG / REACT）+ 数据处理工具路由 已完成并验证通过
- 架构：LangGraph 状态机，`router_node` → `route_query()` → 四路分支（pure_llm / rag / react / tools）

## 技术要点
- Web Search：使用 Tavily，API Key 已配置
- 路由策略：`determine_route()` 基于关键词判断 RouteType；`route_query()` 分发到对应节点
- GeneratorAgent 读取 memory_context：在 `generate()` 中将记忆上下文拼入用户 query
- 数据处理路由：新增 `RouteType.TOOLS` + `DATA_PROCESSING_KEYWORDS`，触发 `DataProcessingTool`

## 新增模块（2026-06）
- `src/agent/data_processing/`：数据清洗（Phase 1）+ EDA（Phase 2）两阶段流水线
  - `data_cleaning.py` / `tools_clean.py` / `tools_eda.py`：核心逻辑
  - `skill_clean_agent.py` / `skill_eda_agent.py`：LLM 调用层（已改为 `get_llm().ainvoke()`）
  - `master_agent.py`：两阶段流水线主入口
- `src/agent/tools/data_processing_tool.py`：`DataProcessingTool`（继承 `BaseTool`），注册到 `ToolRegistry`
- `config_loader.py`：`Config` 类新增 `data_processing: dict = {}` 字段
- `schema.py`：新增 `RouteType.TOOLS = "tools"`
- `workflow.py`：新增 `tools_node()` + `tools` 节点 + `tools → post_process` 边

## Bug 修复记录
- 2026-04-24: ReAct 循环的两个 bug
  1. `retrieve` 是 async 方法但没有 await → 添加 await
  2. LangChain 返回格式兼容性问题 → 添加 `.text`/`.content`/直接字符串 三种格式兼容
- 2026-04-24: Web Search 不工作（两个根因）
  1. `main.py` 读配置路径错误：`config.agent.get("tools"...)` → 应为 `config.tools.get("web_search"...)`（tools 是顶层 key）
  2. `config.yaml` 中 `tools.web_search.enabled: false` → 改为 `true`
  3. `_init_tools()` 中增加 tavily 的 `TAVILY_API_KEY` env var 设置
  - 修复后：Tavily 搜索端到端验证通过，返回真实搜索结果
- 2026-06-10: `workflow.py` `create_agent_workflow()` 函数体缩进全乱
  - 根因：`replace_in_file` 多次操作导致缩进层级错位
  - 修复：用 Python 脚本修正，将"非 `:` 结尾的上一行对应的 8 空格行"改为 4 空格
- 2026-06-10: `data_processing_tool.py` 错误导入 `from src.config.config import config`
  - 修复：改为 `from src.utils.config_loader import config`

## Gradio 前端改造（2026-06-10）
- `schema.py`：`AgentState` 新增 `deliverables: Optional[list[dict]]` 字段（格式：`[{"label":..., "path":...}]`）
- `workflow.py` `tools_node`：执行完后将两个 HTML 产物路径写入 `state["deliverables"]`
- `main.py` `chat_response`：返回 6 个值 `("", history, panel1, panel2, accord1, accord2)`
  - 无报告时：panel/accord 均 `visible=False`，不占空间
  - 有报告时：用 `/file=` 静态 URL 构造 `<iframe>`，同时 `visible=True`
- `demo.launch`：加 `allowed_paths=[output_dir]`，允许 Gradio `/file=` 路由访问本地 HTML 报告
- 设计原则：没有报告时右侧面板完全隐藏；分析完成后自动弹出两个 Accordion 侧边栏

## 自然语言总结（2026-06-10）
- `master_agent.py` `SYSTEM_PROMPT_MASTER`：改为 to-do list 结构，[x] Phase1+2 已完成 → [ ] 最后一步输出自然语言总结
- `_call_master_llm()` → 替换为 `_generate_natural_summary()`：输入两阶段结构化指标，输出 150-250 字中文自然语言段落
- 总结保存到 `output_dir/summary.md`，同时通过 `ToolResult.content` 返回给 Gradio 聊天区
- 顺手修复 `workflow.py` `tools_node`：字段名 `eda_report_path` → `eda_report_html`、`dist_report_path` → `dist_report_html`

## Bug 修复记录（2026-06-10）
- Pydantic Config 对象不能用 `.get()`：`config.get("data_processing", {})` → `config.data_processing.get(...)`，涉及 `data_processing_tool.py`、`main.py`、`server.py`
- `skill_eda_agent.py` 7个 `print()` 缩进错误（5空格→4空格），导致 `IndentationError`
- `tools_node` 路径含空格正则不匹配：旧正则 `[C-Za-z]:\\[^"\s]+` 遇到 `C:\AI Agent\...` 截断 → 新正则匹配到中文分隔词为止
- `run_agent()` 返回 dict 漏掉 `deliverables` 字段 → 右侧面板永远不弹出
- `tools_node` deliverables 提取：从 `phase1/phase2` → 改为从顶层 `deliverables` 取，加 `os.path.isfile()` 校验 + logger 输出
- 右侧报告面板 UI：Accordion → `gr.Tabs` 标签栏布局（浏览器风格），输出从 6 个减为 5 个（msg, chatbot, panel1, panel2, report_tabs）
