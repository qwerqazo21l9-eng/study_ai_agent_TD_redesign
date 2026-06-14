---
name: react-agent-extension
overview: 为知识库 Agent 引入条件 ReAct 循环机制：简单查询走快速路径，复杂联网查询走 ReAct 循环
todos:
  - id: complexity-analyzer
    content: 实现复杂度分析器（src/agent/complexity.py）：分析查询复杂度，返回 L1/L2/L3 分层结果
    status: completed
  - id: react-state
    content: 定义 ReActState 数据模型（src/agent/react/state.py）：thought、action、observation、iteration 等
    status: completed
  - id: react-prompts
    content: 编写 ReAct Prompt 模板（src/agent/react/prompts.py）：Think/Act/Evaluate 各阶段的 Prompt
    status: completed
  - id: react-agent
    content: 实现 ReAct Agent（src/agent/react/agent.py）：Observe/Think/Act/Evaluate 循环
    status: completed
    dependencies:
      - complexity-analyzer
      - react-state
      - react-prompts
  - id: workflow-refactor
    content: 重构 workflow.py：添加分层路由，Fast Path 和 ReAct Loop 条件分支
    status: completed
    dependencies:
      - complexity-analyzer
      - react-agent
  - id: config-update
    content: 更新 config.yaml：添加 complexity 和 react 配置项
    status: completed
---

## 需求概述

实现**智能分层路由架构**：简单查询走快速路径（直接 Hybrid RAG → 生成），复杂查询进入 ReAct 循环（观察 → 思考 → 执行 → 评估），兼顾效率与深度。

## 核心设计

### 查询复杂度分层| 层级 | 特征 | 处理路径 | 示例 |

| **L1 简单** | 单知识点、直接匹配 | Fast Path | "RAG 是什么"、"根据笔记，总结第一章" |
| --- | --- | --- | --- |
| **L2 中等** | 多知识点、需要推理 | Hybrid Path | "对比 RAG 和 Agent 的区别" |
| **L3 复杂** | 需要联网、多轮分析 | ReAct 循环 | "搜索最新技术，分析趋势，结合我的笔记" |


### 复杂度判断标准

**触发 ReAct 循环的条件**（满足任一）：

1. 意图为 `web_search` 或 `hybrid`（需要联网）
2. 包含"最新"、"今天"、"趋势"、"分析"等关键词
3. 用户明确要求搜索（如"帮我搜索..."）
4. 上一轮回答后用户继续追问

**走快速路径的条件**：

1. 意图为 `knowledge` 且置信度 > 0.8
2. 意图为 `chat`
3. 关键词与知识库高度匹配

## 功能描述

### Fast Path（快速路径）

- 跳过 Supervisor 意图分析
- 直接调用 Hybrid RAG 检索
- 快速生成回答
- 延迟 < 500ms

### ReAct Loop（深度循环）

- **Observe**：获取当前状态、已有信息
- **Think**：分析还缺什么、是否需要工具
- **Act**：执行工具（检索/搜索/再检索）
- **Evaluate**：评估答案是否充分
- **循环终止条件**：达到最大迭代次数 / 答案充分 / 超时

### 渐进式回答

- 复杂查询可以先返回中间结果（如"正在搜索..."）
- 逐步完善最终答案

## 技术方案

### 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户查询                                   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              ComplexityAnalyzer (复杂度分析器)                       │
│  • 关键词模式匹配                                                │
│  • 意图置信度检查                                                │
│  • 历史上下文判断                                                │
└─────────────────────────┬───────────────────────────────────────┘
                          │
         ┌────────────────┴────────────────┐
         │                                 │
         ▼                                 ▼
┌─────────────────┐             ┌─────────────────────────┐
│   L1/L2 简单    │             │   L3 复杂               │
│   Fast Path     │             │   ReAct Loop            │
└────────┬────────┘             └───────────┬─────────────┘
         │                                 │
         ▼                                 ▼
┌─────────────────┐             ┌─────────────────────────┐
│ Hybrid RAG 检索  │             │     Observe             │
│ (FAISS + BM25)   │             │  • 获取当前状态          │
└────────┬────────┘             │  • 检查已有信息          │
         │                      └───────────┬─────────────┘
         ▼                                 │
┌─────────────────┐                       ▼
│  Generator      │             ┌─────────────────────────┐
│  (快速生成)      │             │     Think               │
└─────────────────┘             │  • 分析缺什么信息        │
                                 │  • 决定下一步行动        │
                                 └───────────┬─────────────┘
                                             │
                              ┌──────────────┼──────────────┐
                              ▼              ▼              ▼
                        ┌─────────┐   ┌─────────┐   ┌─────────┐
                        │Retriever│   │WebSearch│   │ Generator│
                        │(补充检索)│   │(联网搜索)│   │(生成答案)│
                        └────┬────┘   └────┬────┘   └────┬────┘
                             │              │              │
                             └──────────────┴──────────────┘
                                             │
                                             ▼
                                 ┌─────────────────────────┐
                                 │     Evaluate            │
                                 │  • 答案充分？           │
                                 │  • 继续循环 or 结束     │
                                 └─────────────────────────┘
```

### 文件结构

```
src/agent/
├── __init__.py
├── schema.py              # 已有：AgentState, QueryType
├── complexity.py           # [NEW] 复杂度分析器
├── supervisor.py          # 已有：意图分析
├── retriever_agent.py     # 已有：检索 Agent
├── generator.py           # 已有：生成 Agent
├── workflow.py           # 重构：分层路由
├── react/
│   ├── __init__.py        # [NEW] 模块导出
│   ├── agent.py           # [NEW] ReAct Agent
│   ├── state.py           # [NEW] ReActState 定义
│   └── prompts.py         # [NEW] ReAct Prompt 模板
└── ...
```

### 关键接口设计

#### ComplexityLevel 枚举

```python
class ComplexityLevel(Enum):
    SIMPLE = "simple"      # L1: 直接快速路径
    MEDIUM = "medium"      # L2: 混合路径    COMPLEX = "complex"    # L3: ReAct 循环
```

#### ComplexityAnalyzer

```python
class ComplexityAnalyzer:
    def analyze(self, query: str, intent: Intent = None, history: list = None) -> ComplexityLevel:
        """
        分析查询复杂度
        
        Returns:
            ComplexityLevel: 简单/中等/复杂
        """
        
    def should_use_react(self, query: str, intent: Intent = None) -> bool:
        """
        判断是否需要 ReAct 循环
        
        触发条件：
        - intent == web_search/hybrid
        - 包含复杂意图关键词
        - 用户追问
        """
```

#### ReActAgent

```python
class ReActAgent:
    async def run(self, state: ReActState) -> ReActState:
        """
        执行 ReAct 循环
        
        循环流程：
        1. Observe - 获取状态
        2. Think - 分析决策
        3. Act - 执行工具
        4. Evaluate - 评估结果
        """
        
    async def observe(self, state: ReActState) -> ReActState:
        """观察：收集当前状态信息"""
        
    async def think(self, state: ReActState) -> str:
        """思考：分析下一步行动"""
        
    async def act(self, state: ReActState, action: str) -> ReActState:
        """执行：调用工具"""
        
    async def evaluate(self, state: ReActState) -> bool:
        """评估：判断是否继续循环"""
```

### 实现要点

1. **复用现有组件**：Hybrid RAG、Generator、Supervisor 保持不变
2. **新组件隔离**：`react/` 子模块独立，不影响现有逻辑
3. **配置控制**：`config.yaml` 添加 `react.max_iterations`, `react.confidence_threshold`
4. **向后兼容**：可通过配置关闭 ReAct，恢复原有行为

### 配置项

```
agent:
  # 复杂度分析
  complexity:
    simple_threshold: 0.8      # 置信度 > 0.8 且 intent=knowledge → 快速路径
    complex_keywords:          # 触发复杂处理的关键词
      - "最新"
      - "今天"
      - "趋势"
      - "分析"
      - "对比"
      - "搜索"
  
  # ReAct 配置
  react:
    enabled: true
    max_iterations: 5          # 最大循环次数
    confidence_threshold: 0.7  # 答案置信度阈值
```