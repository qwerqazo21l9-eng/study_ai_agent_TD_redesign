import gradio as gr
import asyncio
import base64
import json
import os
import re
from pathlib import Path
from typing import Optional
import uuid
from urllib.parse import quote

from src.rag.vector_db_manager import VectorDBManager
from src.utils.config_loader import config
from src.utils.logger import logger
from src.agent.schema import QueryType

# ====================== 全局配置 ======================
USE_AGENT_MODE = config.agent.get("enabled", True)
ENABLE_MEMORY = config.agent.get("memory", {}).get("enabled", True)

# ====================== 全局初始化 ======================
logger.info("=== 学习助手AI Agent 启动 ===")
db_manager = VectorDBManager()

# ====================== 组件初始化 ======================
def _init_llm():
    """懒加载 LLM"""
    from langchain_openai import ChatOpenAI
    API_KEY = config.model["cloud_api_key"]
    API_BASE = config.model["cloud_api_base"]
    MODEL_NAME = config.model["cloud_model_name"]
    return ChatOpenAI(
        model=MODEL_NAME,
        api_key=API_KEY,
        base_url=API_BASE,
        temperature=config.agent.get("generator", {}).get("temperature", 0.3),
    )

def _init_embedder():
    """懒加载 Embedder"""
    from src.utils.embeddings import get_embedding_model
    return get_embedding_model()

# LLM 和 Embedder 实例
_llm_instance = None
_embedder_instance = None
_memory_manager = None
_agent_workflow = None


def get_llm():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = _init_llm()
    return _llm_instance


def get_embedder():
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = _init_embedder()
    return _embedder_instance


def get_memory_manager():
    """懒加载记忆管理器"""
    global _memory_manager
    if _memory_manager is None and ENABLE_MEMORY:
        from src.agent.memory.manager import create_memory_manager
        _memory_manager = create_memory_manager(
            embedder=get_embedder(),
            auto_summarize=True,
            summarize_after_turns=10,
        )
        logger.info("记忆管理器已初始化")

        # 同步回 memory 模块的单例，避免 workflow/react 节点读到另一份空实例
        try:
            from src.agent.memory import manager as _mem_mod
            _mem_mod._global_manager = _memory_manager
        except Exception as _e:
            logger.warning(f"同步记忆单例失败: {_e}")
    return _memory_manager


def get_agent_workflow():
    """懒加载 Agent 工作流"""
    global _agent_workflow
    if _agent_workflow is None:
        from src.agent.workflow import get_agent_workflow as _get_wf
        _agent_workflow = _get_wf()
    return _agent_workflow


# ====================== 工具初始化 ======================
def _init_tools():
    """初始化工具系统"""
    from src.agent.tools.registry import init_default_tools, get_registry
    # tools 是顶层配置字段（Config.tools），不在 agent 下
    web_search_cfg = config.tools.get("web_search", {})
    if web_search_cfg.get("enabled", False):
        provider = web_search_cfg.get("provider", "duckduckgo")
        # 若使用 tavily，设置环境变量
        if provider == "tavily":
            import os
            api_key = web_search_cfg.get("tavily_api_key", "")
            if api_key:
                os.environ["TAVILY_API_KEY"] = api_key
        init_default_tools(provider=provider)
        logger.info(f"工具系统已初始化: provider={provider}")
    else:
        logger.info("工具系统未启用 (tools.web_search.enabled=false)")

    # 注册数据自动化处理工具
    dp_cfg = config.data_processing
    if dp_cfg.get("enabled", False):
        try:
            from src.agent.tools.data_processing_tool import DataProcessingTool
            registry = get_registry()
            registry.register(DataProcessingTool(), enabled=True, priority=10)
            logger.info("DataProcessingTool 已注册")
        except Exception as e:
            logger.warning(f"DataProcessingTool 注册失败: {e}")


# 启动时初始化工具
_init_tools()


# ====================== 兼容层：旧版 Chain ======================
def _get_legacy_chain():
    """获取旧版 RAG Chain"""
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate

    API_KEY = config.model["cloud_api_key"]
    API_BASE = config.model["cloud_api_base"]
    MODEL_NAME = config.model["cloud_model_name"]

    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=API_KEY,
        base_url=API_BASE,
        temperature=0.3,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个企业办公数据助手 AI Agent，必须严格根据参考资料回答，禁止编造内容。"),
        ("user", "参考资料：{context}\n用户问题：{query}")
    ])

    return prompt | llm


# ====================== 状态管理 ======================
# 每个用户的会话状态
user_states: dict[str, dict] = {}


def get_user_state(session_id: str) -> dict:
    """获取或创建用户状态"""
    if session_id not in user_states:
        user_states[session_id] = {
            "memory_manager": None,
            "turn_count": 0,
            "last_intent": None,
            "last_context": None,
        }
    return user_states[session_id]


def clear_user_state(session_id: str):
    """清除用户状态"""
    if session_id in user_states:
        del user_states[session_id]


# ====================== 核心聊天逻辑 ======================
from queue import Queue
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler

_file_server_port = 7861
_file_server_dir = ""
_file_server_started = threading.Event()


def _start_file_server():
    """Start a simple static file HTTP server for report HTMLs."""
    global _file_server_dir
    if _file_server_started.is_set():
        return
    try:
        _file_server_dir = os.path.abspath(
            config.data_processing.get("output_dir", "./output/data_processing")
        )
        os.makedirs(_file_server_dir, exist_ok=True)

        class Handler(SimpleHTTPRequestHandler):
            def translate_path(self, path):
                # Serve files only from the output directory
                path = super().translate_path(path)
                rel = os.path.relpath(path, os.getcwd())
                return os.path.join(_file_server_dir, rel)

            def log_message(self, format, *args):
                pass  # Suppress noise

        server = HTTPServer(("127.0.0.1", _file_server_port), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        _file_server_started.set()
        logger.info(f"[ReportServer] 已启动 http://127.0.0.1:{_file_server_port} -> {_file_server_dir}")
    except Exception as e:
        logger.warning(f"[ReportServer] 启动失败: {e}")


def _build_iframe_from_html(html_path: str) -> str:
    """Read a saved HTML report and render it in an iframe.

    Self-contained strategy:
    1. If file doesn't exist → show error message.
    2. If file >= 3MB → serve via local static HTTP server (bypasses Gradio
       /file= and avoids base64 size limits that cause blank iframes).
    3. Otherwise → read + repair relative paths + base64 data URI.
    """
    try:
        path = Path(html_path)
        if not path.exists():
            return (
                f'<div style="padding:12px;color:#555;">'
                f"报告文件不存在：{html_path}</div>"
            )

        size = path.stat().st_size

        # Large reports: serve via local static HTTP server instead of base64,
        # because browsers/iframe implementations often fail on multi-MB data URIs.
        if size >= 3 * 1024 * 1024:
            try:
                _start_file_server()
            except Exception:
                pass
            filename = os.path.basename(html_path)
            file_url = f"http://127.0.0.1:{_file_server_port}/{quote(filename)}"
            return (
                f'<iframe src="{file_url}" '
                f'style="width:100%;height:620px;border:none;border-radius:6px;" '
                f'allow="scripts sandbox"></iframe>'
            )

        # Small/medium reports: read + repair relative paths, then base64.
        base_dir = path.parent.resolve().as_posix()
        html = path.read_text(encoding="utf-8", errors="ignore")

        def _replacer(m: "re.Match") -> str:
            url = m.group(1).strip()
            if url.startswith(("http://", "https://", "data:", "#")):
                return m.group(0)
            if url.startswith("/"):
                local = Path(url)
            else:
                local = Path(base_dir, url)
            resolved = local.resolve().as_posix()
            return f'src="file:///{resolved}"'

        html = re.sub(r'\bsrc\s*=\s*"([^"]+)"', _replacer, html, flags=re.I)
        b64 = base64.b64encode(html.encode("utf-8")).decode()
        return (
            f'<iframe src="data:text/html;base64,{b64}" '
            f'style="width:100%;height:620px;border:none;border-radius:6px;" '
            f'allow="scripts sandbox"></iframe>'
        )
    except Exception as e:
        logger.error(f"构建报告 iframe 失败: {e}")
        return (
            f'<div style="padding:12px;color:#b00020;">'
            f"报告渲染失败：{e}</div>"
        )


def chat_response(message: str, history: list, session_id: Optional[str] = None):
    """
    主聊天函数

    Returns:
        ("", history, panel1, panel2, tabs_visible)
    """
    if not message.strip():
        return "", history, gr.update(), gr.update(), gr.update()

    # 生成会话 ID
    if session_id is None:
        session_id = "default"

    logger.info(f"[{session_id}] 用户提问：{message}")

    deliverables = None
    try:
        if USE_AGENT_MODE:
            answer, deliverables = _chat_response_agent(message, session_id)
        else:
            answer = _chat_response_legacy(message)

    except Exception as e:
        logger.error(f"聊天执行失败: {e}")
        answer = f"抱歉，发生了错误：{str(e)}"

    # 更新历史
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})

    # 默认：隐藏报告面板
    panel1_html = gr.update(value="")
    panel2_html = gr.update(value="")
    tabs_vis = gr.update(visible=False)

    if deliverables:
        tabs_vis = gr.update(visible=True)
        for i, item in enumerate(deliverables[:2]):
            html_path = item.get("path", "")
            if html_path:
                iframe = _build_iframe_from_html(html_path)
                if i == 0:
                    panel1_html = gr.update(value=iframe)
                elif i == 1:
                    panel2_html = gr.update(value=iframe)

    return "", history, panel1_html, panel2_html, tabs_vis


def _chat_response_agent(message: str, session_id: str) -> tuple[str, list[dict] | None]:
    """Agent 模式聊天，返回 (response_text, deliverables)"""
    from src.agent.workflow import run_agent

    user_state = get_user_state(session_id)

    # 获取或创建记忆管理器
    memory_manager = user_state.get("memory_manager")
    if memory_manager is None and ENABLE_MEMORY:
        memory_manager = get_memory_manager()
        user_state["memory_manager"] = memory_manager

    # 构建 run_agent 参数
    run_params = {
        "query": message,
        "session_id": session_id,
    }

    # 运行 Agent
    result = asyncio.run(run_agent(**run_params))

    # 写入对话消息（上一轮用户与AI消息）到记忆管理器，供后续续问使用
    if memory_manager is not None:
        try:
            memory_manager.add_user_message(message)
            _resp = result.get("response") or ""
            if _resp:
                memory_manager.add_ai_message(_resp)
        except Exception as _e:
            logger.warning(f"[Memory] 手动写入对话失败: {_e}")

    # 更新状态
    user_state["turn_count"] = user_state.get("turn_count", 0) + 1
    user_state["last_intent"] = result.get("intent")
    user_state["last_context"] = result.get("context")

    # 打印调试信息
    intent = result.get("intent")
    if intent:
        logger.info(f"[{session_id}] 意图: {intent.query_type.value} (置信度: {intent.confidence:.2f})")
        if intent.keywords:
            logger.info(f"[{session_id}] 关键词: {intent.keywords}")

    response = result.get("response", "")
    deliverables = result.get("deliverables")  # list[dict] | None
    logger.info(f"[{session_id}] 回答: {response[:100]}...")

    return response, deliverables


def _chat_response_legacy(message: str) -> str:
    """旧版 RAG 聊天（兼容模式）"""
    chain = _get_legacy_chain()

    # RAG 检索
    results = db_manager.similarity_search(message)
    logger.info(f"检索到 {len(results)} 条结果")

    for doc, score in results:
        logger.info(f"相似度: {score:.4f} | 内容: {doc.page_content[:100]}...")

    context = "\n".join([doc.page_content for doc, score in results])
    logger.info(f"最终上下文: {context[:200]}...")

    # 调用 AI
    try:
        ai_msg = chain.invoke({"context": context, "query": message})
        answer = ai_msg.content
    except Exception as e:
        answer = f"AI调用失败：{str(e)}"
        logger.error(f"AI错误: {e}")

    return answer


# ====================== 状态查询 ======================
def get_status_info(session_id: str = "default") -> dict:
    """获取当前状态信息"""
    info = {
        "mode": "Agent 模式" if USE_AGENT_MODE else "传统 RAG 模式",
        "session_id": session_id,
    }

    if USE_AGENT_MODE:
        user_state = get_user_state(session_id)
        info.update({
            "turn_count": user_state.get("turn_count", 0),
            "memory_enabled": ENABLE_MEMORY,
        })

        intent = user_state.get("last_intent")
        if intent:
            info["last_intent"] = f"{intent.query_type.value} ({intent.confidence:.2f})"
            info["last_keywords"] = ", ".join(intent.keywords) if intent.keywords else "无"

        # 数据库状态
        db_stats = db_manager.get_stats()
        info["db_vectors"] = db_stats.get("total_vectors", 0)
        info["hybrid_enabled"] = db_stats.get("use_hybrid", False)

    return info


# ====================== 文件上传 ======================
def upload_file(file) -> tuple[str, str]:
    """
    上传文件到知识库

    Returns:
        (状态消息, 清空后的文件路径)
    """
    if not file:
        return "未选择文件", None

    file_path = Path(file.name)
    logger.info(f"开始处理文件：{file_path.name}")
    logger.info(f"文件大小：{file_path.stat().st_size} bytes")

    try:
        success = db_manager.add_file_to_db(file_path)

        if success:
            # 获取更新后的统计
            stats = db_manager.get_stats()
            vector_count = stats.get("total_vectors", 0)
            return f"✅ {file_path.name} 已入库！\n当前知识库向量数: {vector_count}", None
        else:
            return "❌ 处理失败，请检查文件格式", None

    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        return f"❌ 上传失败: {str(e)}", None


# ====================== Gradio 界面 ======================
mode_label = "🧠 Agent 模式" if USE_AGENT_MODE else "📖 传统 RAG 模式"

with gr.Blocks(title="企业办公数据助手 AI Agent", theme=gr.themes.Soft()) as demo:
    # 顶部信息
    gr.Markdown("# 企业办公数据助手 AI Agent")
    gr.Markdown(f"**当前模式: {mode_label}**")

    # 特性说明
    if USE_AGENT_MODE:
        features_md = """
        **🧠 Agent 模式特性：**
        - 🔍 **意图分析**：自动判断问题类型（知识库/联网搜索/闲聊）
        - 📊 **Hybrid RAG**：向量 + BM25 融合检索，结果更全面
        - 📎 **来源引用**：回答中标注信息来源
        - 🧠 **记忆系统**：跨会话记住重要上下文
        """
        if config.agent.get("tools", {}).get("web_search", {}).get("enabled"):
            features_md += "- 🌐 **联网搜索**：实时获取网络信息"
    else:
        features_md = """
        **📖 传统 RAG 模式：**
        - 基于向量检索的问答
        - 回答基于知识库内容
        """

    gr.Markdown(features_md)

    # 状态栏
    status_display = gr.State(value=get_status_info())

    with gr.Row():
        # ── 左侧：聊天区域（固定宽度） ──
        with gr.Column(scale=3, min_width=400):
            chatbot = gr.Chatbot(height=500, label="对话")
            msg = gr.Textbox(placeholder="输入问题...", label="输入")

            with gr.Row():
                submit_btn = gr.Button("🚀 发送", variant="primary")
                clear_btn = gr.Button("🗑️ 清空对话")

            # 系统状态 & 文件管理（折叠）
            with gr.Accordion("🔧 系统管理", open=False):
                gr.Markdown("### 📊 系统状态")
                status_text = gr.JSON(value=get_status_info(), label="当前状态")

                gr.Markdown("### 📁 知识库管理")
                file_upload = gr.File(label="上传笔记文件", file_types=[".txt", ".pdf", ".docx", ".md", ".pptx"])
                upload_btn = gr.Button("⬆️ 上传并入库", variant="secondary")
                upload_status = gr.Textbox(label="状态", lines=2)

                with gr.Row():
                    refresh_btn = gr.Button("🔄 刷新状态")
                    clear_memory_btn = gr.Button("🧹 清除会话记忆")

        # ── 右侧：数据分析报告面板（默认隐藏，有报告才弹出） ──
        with gr.Column(scale=4, min_width=500):
            gr.Markdown("### 📊 数据分析报告")

            with gr.Tabs(visible=False) as report_tabs:
                with gr.Tab("📋 Phase 1 数据清洗"):
                    report_panel_1 = gr.HTML(value="")
                with gr.Tab("📈 Phase 2 EDA 分布"):
                    report_panel_2 = gr.HTML(value="")

    # ── 事件绑定 ──
    _chat_outputs = [msg, chatbot, report_panel_1, report_panel_2, report_tabs]

    submit_btn.click(
        fn=chat_response,
        inputs=[msg, chatbot],
        outputs=_chat_outputs,
    )

    msg.submit(
        fn=chat_response,
        inputs=[msg, chatbot],
        outputs=_chat_outputs,
    )

    # 清空对话 —— 同时重置两个报告面板为隐藏
    def clear_chat():
        clear_user_state("default")
        return (
            [],
            get_status_info(),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(visible=False),
        )

    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, status_text, report_panel_1, report_panel_2, report_tabs]
    )

    # 文件上传
    upload_btn.click(
        fn=upload_file,
        inputs=file_upload,
        outputs=[upload_status, file_upload]
    )

    # 刷新状态
    refresh_btn.click(
        fn=get_status_info,
        outputs=status_text
    )

    # 清除记忆
    def _clear_memory():
        clear_user_state("default")
        return "✅ 会话记忆已清除"

    clear_memory_btn.click(
        fn=_clear_memory,
        outputs=upload_status
    )



# ====================== 启动 ======================
if __name__ == "__main__":
    import os as _os
    # allowed_paths：允许 /file= 路由暴露 output 目录下的 HTML 报告
    _output_dir = _os.path.abspath(
        config.data_processing.get("output_dir", "./output/data_processing")
    )
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        allowed_paths=[_output_dir],
    )