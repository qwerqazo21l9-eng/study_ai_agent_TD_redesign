---
name: rag-agent-upgrade
overview: 将现有单管道 RAG 升级为支持 Hybrid RAG + 轻量级 Multi-Agent + MCP 工具调用的 AI Agent 系统
todos:
  - id: phase1-bm25
    content: 新增 BM25 检索模块（src/rag/bm25_retriever.py）：jieba 分词 + rank_bm25 实现中文关键词检索，封装 BM25Retriever 类，与 FAISS 接口对齐
    status: pending
  - id: phase1-rrf
    content: 新增 RRF 融合模块（src/rag/fusion.py）：实现 Reciprocal Rank Fusion 算法，融合向量和 BM25 多路检索结果
    status: pending
  - id: phase1-hybrid-retriever
    content: 扩展 VectorDBManager（src/rag/vector_db_manager.py）：添加 hybrid_search() 方法，并行执行向量 + BM25 + RRF 融合，保持原有接口兼容
    status: pending
  - id: phase2-schema
    content: 定义 Agent 状态模型（src/agent/schema.py）：AgentState（Pydantic）、AgentMessage、检索结果结构，为 LangGraph 提供类型安全
    status: pending
  - id: phase2-supervisor
    content: 实现 Supervisor Agent（src/agent/supervisor.py）：意图分析（闲聊/知识库/联网）、路由决策、Prompt 工程
    status: pending
  - id: phase2-retriever
    content: 实现 Retriever Agent（src/agent/retriever_agent.py）：封装 Hybrid RAG 调用、工具调用协调、来源标注
    status: pending
  - id: phase2-generator
    content: 实现 Generator Agent（src/agent/generator.py）：带引用溯源的回答生成、防幻觉 Prompt、多轮上下文整合
    status: pending
  - id: phase2-workflow
    content: 编排 LangGraph 工作流（src/agent/workflow.py）：定义状态机节点（supervisor/retriever/generator）、边路由、异常处理
    status: pending
  - id: phase3-tools
    content: 实现工具系统（src/agent/tools/）：BaseTool 基类、WebSearch 工具（tavily/duckduckgo）、ToolRegistry 注册中心、MCP 预留接口
    status: pending
  - id: phase4-memory
    content: 实现记忆系统（src/agent/memory/）：短期记忆（消息历史 + Token 限制）、长期记忆（摘要向量库）、记忆管理器
    status: pending
  - id: phase5-main
    content: 重构 main.py：接入 Agent 工作流、保留文件上传、扩展 Gradio 界面、更新 config.yaml 配置项
    status: pending
---

## 需求概述

将现有个人学习知识库 RAG 项目，升级为**支持多轮研究 + 工具调用的轻量级 AI Agent 系统**。

## 用户明确目标

1. **核心定位**：通用问答 / 研究助手，支持多轮对话和外部工具
2. **检索升级**：保留现有 RAG，升级为 Hybrid RAG（向量 + BM25 + RRF 融合）
3. **Agent 协作**：轻量级 3 Agent（Supervisor + Retriever + Generator）
4. **工具扩展**：接入 WebSearch，预留 MCP 协议扩展

## 面试暴露的待解决问题

| 问题 | 现状 | 目标 |
| --- | --- | --- |
| 检索单一 | 仅向量检索 | Hybrid RAG（向量 + BM25） |
| 无引用溯源 | 检索结果无来源 | 回答带来源标注 |
| 无记忆机制 | 每次会话独立 | 短期 + 长期记忆 |
| 无工具调用 | 无法联网 | WebSearch + MCP |
| 无状态管理 | 简单拼接 | LangGraph 状态机 |
| Prompt 简单 | 易产生幻觉 | 引用溯源 + 防幻觉增强 |


## 核心功能

1. **Hybrid RAG**：向量检索 + BM25 关键词检索 + RRF 融合
2. **3 Agent 协作**：Supervisor 路由 + Retriever 检索 + Generator 生成
3. **工具系统**：WebSearch 工具 + MCP 扩展接口
4. **记忆系统**：短期记忆（会话）+ 长期记忆（摘要向量库）
5. **Prompt 优化**：引用溯源、防幻觉、多轮上下文

## 技术栈

- **框架**：LangGraph（已有 langgraph>=0.2.0）
- **向量库**：FAISS（已有 faiss-cpu）
- **关键词检索**：rank_bm25 + jieba
- **工具协议**：MCP（预留）
- **WebSearch**：Tavily / DuckDuckGo
- **API**：智谱 GLM-4-Flash（已有）

## 架构设计### 系统架构图

```mermaid
graph TD
    subgraph 用户层
        U[Gradio 界面]
    end

    subgraph Agent 层
        S[Supervisor Agent<br/>意图分析 + 路由]
        R[Retriever Agent<br/>Hybrid RAG + 工具调用]
        G[Generator Agent<br/>生成回答]
    end

    subgraph 工具层
        W[WebSearch 工具]
        M[MCP 扩展接口]
    end

    subgraph 记忆层
        ST[短期记忆<br/>会话消息历史]
        LT[长期记忆<br/>摘要向量库]
    end

    subgraph 检索层
        VR[向量检索<br/>FAISS]
        BM[BM25 检索]
        RF[RRF 融合]
    end

    U --> S
    S -->|知识库| R
    S -->|联网| W
    S -->|闲聊| G
    W --> R
    R -->|检索结果| G
    R -->|调用工具| M    G -->|写入| ST
    G -->|摘要| LT
    VR --> RF
    BM --> RF
    RF --> R
    LT --> R
```

### 模块职责

| 模块 | 职责 | 文件 |
| --- | --- | --- |
| Supervisor | 分析意图，路由决策 | `src/agent/supervisor.py` |
| Retriever | Hybrid RAG + 工具调用 | `src/agent/retriever_agent.py` |
| Generator | 生成带引用的回答 | `src/agent/generator.py` |
| Hybrid RAG | 向量 + BM25 + RRF | `src/rag/hybrid_retriever.py` |
| BM25 | 关键词检索 | `src/rag/bm25_retriever.py` |
| RRF | 多路结果融合 | `src/rag/fusion.py` |
| Tools | WebSearch + MCP | `src/agent/tools/` |
| Memory | 短期 + 长期记忆 | `src/agent/memory/` |
| State | 状态模型定义 | `src/agent/schema.py` |


## 实施步骤

### Phase 1：Hybrid RAG（核心）✅

1. ✅ 新增 BM25 检索模块（jieba 分词 + rank_bm25）

- 文件：`src/rag/bm25_retriever.py`
- 功能：BM25Retriever 类，支持中文分词、持久化、增量添加

2. ✅ 新增 RRF 融合算法

- 文件：`src/rag/fusion.py`
- 功能：RRFusion、ScoreLevelFusion 两种融合方法

3. ✅ 扩展 VectorDBManager 支持 Hybrid 检索

- 新增 `hybrid_search()` 方法：向量 + BM25 + RRF + Reranker
- 新增 `hybrid_search_with_context()` 方法：带来源标注的上下文

4. ✅ 现有 Reranker 继续用于精排

**新增文件**：

- `src/rag/__init__.py` - 模块导出
- `src/rag/bm25_retriever.py` - BM25 检索器
- `src/rag/fusion.py` - RRF 融合

**修改文件**：

- `src/rag/vector_db_manager.py` - 添加 Hybrid RAG 支持
- `config/config.yaml` - 添加 Hybrid RAG 配置参数
- `requirements.txt` - 添加 rank_bm25、jieba 依赖

### Phase 2：Agent 架构 ✅

1. ✅ 定义 AgentState（Pydantic）和消息模型

- 文件：`src/agent/schema.py`
- 内容：AgentState、AgentMessage、Intent、QueryType、SearchResult

2. ✅ 实现 Supervisor Agent（意图分析 + 路由）

- 文件：`src/agent/supervisor.py`
- 功能：LLM 意图分类 + 关键词提取 + 条件路由

3. ✅ 实现 Retriever Agent（封装 Hybrid RAG + 工具调用）

- 文件：`src/agent/retriever_agent.py`
- 功能：封装 Hybrid RAG、上下文拼接、降级决策

4. ✅ 实现 Generator Agent（带引用回答）

- 文件：`src/agent/generator.py`
- 功能：基于上下文的回答生成、引用溯源、闲聊模式

5. ✅ 用 LangGraph 编排工作流

- 文件：`src/agent/workflow.py`
- 流程：supervisor → 路由 → retriever/chat → generator → END

6. ✅ 集成到 main.py

- 支持 Agent 模式/旧模式切换
- 界面显示当前模式

**新增文件**：

- `src/agent/__init__.py` - 模块导出
- `src/agent/schema.py` - Agent 状态模型
- `src/agent/supervisor.py` - Supervisor Agent
- `src/agent/retriever_agent.py` - Retriever Agent
- `src/agent/generator.py` - Generator Agent
- `src/agent/workflow.py` - LangGraph 工作流
- `src/utils/llm.py` - LLM 封装（异步接口）

### Phase 3：工具系统 ✅

1. ✅ 实现 BaseTool 接口

- 文件：`src/agent/tools/base.py`
- 内容：BaseTool 抽象基类、ToolResult、ToolDefinition、ToolParameter

2. ✅ 实现 WebSearch 工具（支持 Tavily/DuckDuckGo）

- 文件：`src/agent/tools/web_search.py`
- 功能：WebSearchTool、DuckDuckGo、Tavily 多 provider 支持

3. ✅ 实现 ToolRegistry

- 文件：`src/agent/tools/registry.py`
- 功能：工具注册/获取/列表、单例模式、MCP 预留接口

4. ✅ 集成到 Supervisor 和工作流

- Supervisor 新增 `web_search()` 方法
- workflow.py 新增 `web_search_node` 节点
- Generator 支持 `web_context` 生成

**新增文件**：

- `src/agent/tools/__init__.py` - 模块导出
- `src/agent/tools/base.py` - BaseTool 基类
- `src/agent/tools/web_search.py` - WebSearch 工具
- `src/agent/tools/registry.py` - 工具注册中心

### Phase 4：记忆系统 ✅

1. ✅ 短期记忆：会话消息历史

- 文件：`src/agent/memory/short_term.py`
- 功能：消息管理、Token 计数、对话上下文提取

2. ✅ 长期记忆：摘要向量库

- 文件：`src/agent/memory/long_term.py`
- 功能：摘要生成、向量检索、跨会话上下文

3. ✅ 记忆管理器：自动读写决策

- 文件：`src/agent/memory/manager.py`
- 功能：统一入口、自动摘要触发、上下文组装

4. ✅ 工作流集成

- `workflow.py` 新增 `memory_node`、`post_process_node`
- 支持 `enable_memory` 参数切换
- Generator 节点自动注入记忆上下文

**新增文件**：

- `src/agent/memory/__init__.py` - 模块导出
- `src/agent/memory/base.py` - 记忆基类
- `src/agent/memory/short_term.py` - 短期记忆
- `src/agent/memory/long_term.py` - 长期记忆
- `src/agent/memory/manager.py` - 记忆管理器

### Phase 5：界面重构 ✅

1. ✅ 重构 main.py 接入 Agent 工作流

- 懒加载 LLM、Embedder、MemoryManager
- 支持 Agent 模式/旧模式切换
- 会话状态管理（session_id）

2. ✅ 扩展 config.yaml

- `agent.enabled`: Agent 模式开关
- `agent.memory.enabled`: 记忆系统开关
- `tools.web_search.enabled`: 联网搜索开关
- `memory.*`: 记忆系统详细配置

3. ✅ 保留并增强文件上传功能

- 显示当前知识库向量数
- 错误信息更友好
- 多种文件格式支持

4. ✅ Gradio 界面升级

- 左侧聊天 + 右侧工具栏布局
- 实时状态显示（意图、关键词、向量数）
- 清空对话/清除记忆按钮
- 数据库操作折叠面板

**修改文件**：

- `main.py` - 完全重构，集成所有模块
- `config/config.yaml` - 添加 agent、memory、tools 配置

---

## 🎉 项目升级完成！

### 最终架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Gradio 界面 (main.py)                      │
├─────────────────────────────────────────────────────────────┤
│                    Agent 工作流 (workflow.py)                    │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐        │
│  │ Supervisor │───▶│ Retriever  │───▶│ Generator  │        │
│  │ (意图分析)   │    │ (Hybrid)   │    │ (引用生成)  │        │
│  └────────────┘    └────────────┘    └────────────┘        │
│         │                                     │              │
│         ▼                                     ▼              │
│  ┌────────────┐                      ┌────────────┐        │
│  │ WebSearch   │                      │  记忆系统   │        │
│  │ (联网工具)   │                      │ (短+长期)   │        │
│  └────────────┘                      └────────────┘        │
├─────────────────────────────────────────────────────────────┤
│                    Hybrid RAG (vector_db_manager.py)           │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐        │
│  │ FAISS 向量  │ +  │   BM25     │ +  │    RRF     │        │
│  │   检索      │    │  关键词检索  │    │   融合     │        │
│  └────────────┘    └────────────┘    └────────────┘        │
│                                              │                │
│                                              ▼                │
│                                    ┌────────────┐             │
│                                    │  CrossEncoder │            │
│                                    │   (精排)     │            │
│                                    └────────────┘             │
└─────────────────────────────────────────────────────────────┘
```

### 文件结构

```
src/
├── agent/
│   ├── __init__.py
│   ├── schema.py          # AgentState、Intent 模型
│   ├── supervisor.py      # Supervisor Agent
│   ├── retriever_agent.py # Retriever Agent
│   ├── generator.py        # Generator Agent
│   ├── workflow.py         # LangGraph 工作流
│   ├── tools/              # 工具系统
│   │   ├── __init__.py
│   │   ├── base.py         # BaseTool 基类
│   │   ├── web_search.py   # WebSearch 工具
│   │   └── registry.py     # ToolRegistry
│   └── memory/             # 记忆系统
│       ├── __init__.py
│       ├── base.py
│       ├── short_term.py   # 短期记忆
│       ├── long_term.py    # 长期记忆
│       └── manager.py      # 记忆管理器
├── rag/
│   ├── __init__.py
│   ├── bm25_retriever.py   # BM25 检索
│   ├── fusion.py           # RRF 融合
│   └── vector_db_manager.py # 扩展支持 Hybrid
└── utils/
    └── llm.py              # LLM 封装
```

### 启动方式

```
cd c:\AI Agent\study_ai_agent
python main.py
# 访问 http://127.0.0.1:7860
```

# Agent Extensions

无 Agent Extensions 需求。当前 skills 中无相关扩展。