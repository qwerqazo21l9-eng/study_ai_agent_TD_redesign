"""
复杂度分析器 - 查询复杂度分层

功能：
1. 分析用户查询的复杂度等级（L1/L2/L3）
2. 判断是否需要进入 ReAct 循环
3. 快速识别简单查询，走 Fast Path

复杂度分级：
- L1 SIMPLE: 简单知识库查询，直接 Fast Path
- L2 MEDIUM: 中等复杂度，标准处理流程
- L3 COMPLEX: 复杂查询，需要 ReAct 循环
"""

import re
from enum import Enum
from typing import Optional

from src.agent.schema import Intent, QueryType
from src.utils.logger import logger


class ComplexityLevel(str, Enum):
    """复杂度等级枚举"""
    SIMPLE = "simple"      # L1: 简单查询，快速路径
    MEDIUM = "medium"      # L2: 中等复杂度，标准路径
    COMPLEX = "complex"    # L3: 复杂查询，ReAct 循环


class ComplexityResult:
    """
    复杂度分析结果

    Attributes:
        level: 复杂度等级
        confidence: 置信度 (0-1)
        reasoning: 分析过程
        should_use_react: 是否需要 ReAct 循环
        should_use_fast_path: 是否走快速路径
        requires_web: 是否需要联网
        requires_reasoning: 是否需要多轮推理
    """

    def __init__(
        self,
        level: ComplexityLevel,
        confidence: float = 0.5,
        reasoning: str = "",
        should_use_react: bool = False,
        should_use_fast_path: bool = False,
        requires_web: bool = False,
        requires_reasoning: bool = False,
    ):
        self.level = level
        self.confidence = confidence
        self.reasoning = reasoning
        self.should_use_react = should_use_react
        self.should_use_fast_path = should_use_fast_path
        self.requires_web = requires_web
        self.requires_reasoning = requires_reasoning

    def __repr__(self) -> str:
        return (
            f"ComplexityResult(level={self.level.value}, "
            f"confidence={self.confidence:.2f}, "
            f"react={self.should_use_react}, "
            f"fast={self.should_use_fast_path})"
        )


class ComplexityAnalyzer:
    """
    查询复杂度分析器

    分析维度：
    1. 意图类型和置信度
    2. 关键词模式匹配
    3. 查询长度和结构
    4. 历史上下文（追问检测）

    触发 ReAct 循环的条件（满足任一）：
    - intent == web_search 或 hybrid
    - 包含复杂意图关键词（最新、趋势、分析等）
    - 用户明确要求搜索
    - 用户追问（follow-up）
    """

    # 触发复杂处理的关键词
    COMPLEX_KEYWORDS = [
        # 时间敏感
        "最新", "今天", "现在", "当前", "此时此刻",
        "最近", "近期", "最近几天", "最近一周",
        # 分析类
        "分析", "分析一下", "趋势", "展望", "预测", "预判",
        "对比", "比较", "区别", "差异", "哪个好",
        "评估", "评价", "判断", "推荐", "建议",
        # 搜索类
        "搜索", "查找", "帮我查", "帮我找", "查一下",
        "搜一下", "网上", "网上查",
        # 研究类
        "研究", "调研", "深入", "详细", "全面",
        "总结", "概括", "概述", "摘要",
        # 多维度
        "结合", "综合", "多角度", "各个方面",
    ]

    # 触发 Fast Path 的特征
    FAST_PATH_KEYWORDS = [
        # 闲聊/问候
        "你好", "hi", "hello", "嗨", "嗨嗨",
        "谢谢", "thanks", "感谢", "谢啦",
        "再见", "拜拜", "bye", "结束",
        # 简单确认
        "好的", "收到", "明白", "了解",
        "可以", "行", "没问题",
    ]

    # 简单定义类查询
    DEFINITION_PATTERNS = [
        r"^什么是",
        r"^什么叫",
        r"^xx是",
        r"的概念",
        r"的定义",
        r"的意思",
        r"是什么",
    ]

    def __init__(self):
        self._complex_pattern = re.compile(
            "|".join(self.COMPLEX_KEYWORDS),
            re.IGNORECASE
        )
        self._fast_pattern = re.compile(
            "|".join(self.FAST_PATH_KEYWORDS),
            re.IGNORECASE
        )
        self._definition_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.DEFINITION_PATTERNS
        ]

    def analyze(
        self,
        query: str,
        intent: Optional[Intent] = None,
        history: Optional[list] = None,
        is_followup: bool = False,
    ) -> ComplexityResult:
        """
        分析查询复杂度

        Args:
            query: 用户查询
            intent: 意图分析结果（可选）
            history: 对话历史（可选）
            is_followup: 是否是追问

        Returns:
            ComplexityResult: 复杂度分析结果
        """
        query_lower = query.lower()
        query_len = len(query)

        logger.info(f"Analyzing complexity for: {query[:50]}...")

        # 1. 检查是否是闲聊/简单确认 -> Fast Path
        if self._is_simple_chat(query_lower):
            return ComplexityResult(
                level=ComplexityLevel.SIMPLE,
                confidence=0.95,
                reasoning="闲聊/简单确认，直接快速路径",
                should_use_fast_path=True,
            )

        # 2. 检查是否是简单定义类查询 -> Fast Path
        if self._is_simple_definition(query_lower):
            return ComplexityResult(
                level=ComplexityLevel.SIMPLE,
                confidence=0.9,
                reasoning="简单定义类查询，知识库直接匹配",
                should_use_fast_path=True,
            )

        # 3. 检查是否需要 ReAct 循环
        if self._should_use_react(query_lower, intent, is_followup):
            return self._analyze_complex_case(query_lower, intent, query_len)

        # 4. 基于意图分析
        if intent:
            return self._analyze_by_intent(intent, query_len)

        # 5. 默认：知识库查询，走标准路径
        return ComplexityResult(
            level=ComplexityLevel.MEDIUM,
            confidence=0.5,
            reasoning="默认标准路径",
            should_use_react=False,
            should_use_fast_path=False,
        )

    def _is_simple_chat(self, query_lower: str) -> bool:
        """检查是否是简单闲聊"""
        return bool(self._fast_pattern.search(query_lower))

    def _is_simple_definition(self, query_lower: str) -> bool:
        """检查是否是简单定义类查询"""
        # 短查询 + 定义模式
        if len(query_lower) < 20:
            for pattern in self._definition_patterns:
                if pattern.search(query_lower):
                    return True
        return False

    def _should_use_react(
        self,
        query_lower: str,
        intent: Optional[Intent],
        is_followup: bool,
    ) -> bool:
        """
        判断是否需要 ReAct 循环

        满足以下任一条件则需要：
        1. 意图为 web_search 或 hybrid
        2. 包含复杂关键词
        3. 用户追问
        4. 明确要求搜索
        """
        # 条件1: 意图判断
        if intent:
            if intent.query_type in [QueryType.WEB_SEARCH, QueryType.HYBRID]:
                logger.info(f"Intent requires react: {intent.query_type}")
                return True

            # 低置信度的知识库查询，可能需要深入
            if intent.query_type == QueryType.KNOWLEDGE and intent.confidence < 0.5:
                logger.info("Low confidence knowledge query, checking complexity...")
                # 检查是否有复杂关键词
                if self._complex_pattern.search(query_lower):
                    return True

        # 条件2: 复杂关键词
        if self._complex_pattern.search(query_lower):
            logger.info("Complex keywords detected")
            return True

        # 条件3: 追问
        if is_followup:
            logger.info("Follow-up query detected")
            return True

        # 条件4: 明确搜索要求
        search_patterns = [
            r"帮我搜",
            r"帮我查",
            r"去网上",
            r"上网",
            r"搜索一下",
        ]
        for pattern in search_patterns:
            if re.search(pattern, query_lower):
                logger.info(f"Search request detected: {pattern}")
                return True

        return False

    def _analyze_complex_case(
        self,
        query_lower: str,
        intent: Optional[Intent],
        query_len: int,
    ) -> ComplexityResult:
        """分析复杂案例的具体类型"""
        reasons = []

        # 检查具体原因
        if intent and intent.query_type == QueryType.WEB_SEARCH:
            reasons.append("联网搜索意图")
        elif intent and intent.query_type == QueryType.HYBRID:
            reasons.append("混合查询意图")

        # 关键词分析
        complex_keywords_found = self._complex_pattern.findall(query_lower)
        if complex_keywords_found:
            reasons.append(f"关键词: {', '.join(complex_keywords_found[:3])}")

        # 长度分析（长查询往往更复杂）
        if query_len > 50:
            reasons.append(f"查询较长({query_len}字)")

        reasoning = "; ".join(reasons) if reasons else "复杂查询"

        return ComplexityResult(
            level=ComplexityLevel.COMPLEX,
            confidence=0.8,
            reasoning=reasoning,
            should_use_react=True,
            requires_web=bool(intent and intent.needs_tools),
            requires_reasoning=True,
        )

    def _analyze_by_intent(
        self,
        intent: Intent,
        query_len: int,
    ) -> ComplexityResult:
        """基于意图类型分析复杂度"""
        query_type = intent.query_type
        confidence = intent.confidence

        if query_type == QueryType.CHAT:
            return ComplexityResult(
                level=ComplexityLevel.SIMPLE,
                confidence=0.95,
                reasoning="闲聊意图",
                should_use_fast_path=True,
            )

        elif query_type == QueryType.KNOWLEDGE:
            if confidence >= 0.8:
                return ComplexityResult(
                    level=ComplexityLevel.SIMPLE,
                    confidence=confidence,
                    reasoning=f"高置信度知识库查询({confidence:.2f})",
                    should_use_fast_path=True,
                )
            elif confidence >= 0.6:
                return ComplexityResult(
                    level=ComplexityLevel.MEDIUM,
                    confidence=confidence,
                    reasoning=f"中等置信度知识库查询({confidence:.2f})",
                )
            else:
                return ComplexityResult(
                    level=ComplexityLevel.COMPLEX,
                    confidence=0.5,
                    reasoning=f"低置信度知识库查询({confidence:.2f})，可能需要多轮",
                    should_use_react=True,
                )

        elif query_type == QueryType.HYBRID:
            return ComplexityResult(
                level=ComplexityLevel.COMPLEX,
                confidence=0.8,
                reasoning="混合查询，需要联网+知识库",
                should_use_react=True,
                requires_web=True,
            )

        elif query_type == QueryType.WEB_SEARCH:
            return ComplexityResult(
                level=ComplexityLevel.COMPLEX,
                confidence=0.85,
                reasoning="联网搜索查询",
                should_use_react=True,
                requires_web=True,
            )

        return ComplexityResult(
            level=ComplexityLevel.MEDIUM,
            confidence=0.5,
            reasoning="标准复杂度",
        )


# ============ 全局单例 ============

_complexity_analyzer: Optional[ComplexityAnalyzer] = None


def get_complexity_analyzer() -> ComplexityAnalyzer:
    """获取复杂度分析器单例"""
    global _complexity_analyzer
    if _complexity_analyzer is None:
        _complexity_analyzer = ComplexityAnalyzer()
    return _complexity_analyzer
