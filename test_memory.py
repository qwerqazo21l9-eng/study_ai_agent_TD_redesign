# -*- coding: utf-8 -*-
"""
记忆系统测试脚本

测试三个核心场景：
1. 短期记忆（会话内上下文保持）
2. 长期记忆写入（摘要持久化）
3. 跨会话检索（已有记忆能否被检索到）
4. 自动摘要触发逻辑
"""

import asyncio
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agent.memory.manager import create_memory_manager, MemoryManagerConfig
from src.agent.memory.long_term import LongTermConfig, LongTermMemory

# ========== 辅助函数 ==========

def separator(title):
    print("\n" + "=" * 60)
    print("  " + title)
    print("=" * 60)

def sub(title):
    print("\n  --- " + title + " ---")

# ========== Test 1: 短期记忆 ==========

async def test_short_term_memory():
    separator("Test 1: 短期记忆（会话内上下文）")

    manager = create_memory_manager(embedder=None, auto_summarize=False)
    manager.new_session("test_session_1")

    # 模拟一段对话
    turns = [
        ("我的名字是小明", "好的，小明你好！"),
        ("我在学习 Python", "Python 很棒！你有什么具体问题吗？"),
        ("我想了解列表推导式", "列表推导式是 Python 的语法糖，格式是 [表达式 for 变量 in 可迭代对象]"),
    ]

    for user_msg, ai_msg in turns:
        manager.add_user_message(user_msg)
        manager.add_ai_message(ai_msg)

    # 验证短期记忆
    stats = manager.get_stats()
    print(f"\n  写入了 {len(turns)} 轮对话，共 {len(turns)*2} 条消息")
    print(f"  短期记忆消息数: {stats['short_term_count']}")
    print(f"  当前会话 ID: {stats['current_session']}")

    # 获取上下文（await 异步方法）
    ctx = await manager.get_context(query="小明叫什么名字")
    short_ctx = ctx.get("short_term", "")
    print(f"\n  短期记忆上下文（最近5轮）:")
    print("  " + "-" * 50)
    for line in short_ctx.split("\n"):
        print("  " + line)

    found = "小明" in short_ctx
    print(f"\n  [{'PASS' if found else 'FAIL'}] 上下文包含用户名'小明': {found}")

    # 切换会话，验证隔离
    sub("切换会话后短期记忆是否清空")
    manager.new_session("test_session_2")
    new_stats = manager.get_stats()
    print(f"  切换到 session_2 后消息数: {new_stats['short_term_count']}")
    isolated = new_stats['short_term_count'] == 0
    print(f"  [{'PASS' if isolated else 'FAIL'}] 会话隔离正常: {isolated}")

    return found and isolated


# ========== Test 2: 长期记忆写入 ==========

async def test_long_term_write():
    separator("Test 2: 长期记忆写入（直接调用 store）")

    manager = create_memory_manager(
        embedder=None,
        auto_summarize=True,
        summarize_after_turns=3,
    )
    manager.new_session("test_session_lt")

    sub("直接调用 long_term.store() 写入一条记忆")

    lt = manager._long_term
    success = await lt.store(
        summary="用户小明正在学习 Python 列表推导式，对函数式编程感兴趣。",
        session_id="test_session_lt",
        keywords=["小明", "Python", "列表推导式"],
        metadata={"test": True},
    )

    print(f"  写入结果: {'[PASS] 成功' if success else '[FAIL] 失败'}")

    # 获取所有记忆
    all_memories = lt.get_all()
    print(f"  当前长期记忆总数: {len(all_memories)}")

    if all_memories:
        for i, entry in enumerate(all_memories):
            print(f"\n  记忆 {i+1}:")
            print(f"    内容: {entry.content[:80]}")
            print(f"    会话: {entry.session_id}")
            print(f"    关键词: {entry.keywords}")

    return success


# ========== Test 3: 跨会话检索 ==========

async def test_cross_session_retrieval():
    separator("Test 3: 跨会话检索")

    lt = LongTermMemory(embedder=None)

    sub("先写入两条历史记忆")
    await lt.store(
        summary="用户讨论了 FAISS 向量库的配置方法，设置了 IVF 索引。",
        session_id="session_old_1",
        keywords=["FAISS", "向量库", "IVF"],
    )
    await lt.store(
        summary="用户学习了 LangGraph 的 StateGraph，了解了节点和边的定义方式。",
        session_id="session_old_2",
        keywords=["LangGraph", "StateGraph", "节点"],
    )
    print("  写入了 2 条历史记忆")

    sub("模拟新会话，用新查询检索历史")

    results_1 = await lt.retrieve(query="如何配置向量数据库", top_k=2)
    print(f"\n  查询「如何配置向量数据库」，检索到 {len(results_1)} 条:")
    for r in results_1:
        print(f"    - {r.content[:70]}")

    results_2 = await lt.retrieve(query="LangGraph 工作流怎么定义", top_k=2)
    print(f"\n  查询「LangGraph 工作流怎么定义」，检索到 {len(results_2)} 条:")
    for r in results_2:
        print(f"    - {r.content[:70]}")

    found_faiss = any("FAISS" in r.content or "向量" in r.content for r in results_1)
    found_langgraph = any("LangGraph" in r.content or "StateGraph" in r.content for r in results_2)

    print(f"\n  [{'PASS' if found_faiss else 'FAIL'}] 向量库相关记忆被检索到: {found_faiss}")
    print(f"  [{'PASS' if found_langgraph else 'FAIL'}] LangGraph 相关记忆被检索到: {found_langgraph}")

    return found_faiss or found_langgraph


# ========== Test 4: 自动摘要触发计数 ==========

async def test_auto_summarize_trigger():
    separator("Test 4: 自动摘要触发（turn_count 逻辑）")

    manager = create_memory_manager(
        embedder=None,
        auto_summarize=True,
        summarize_after_turns=3,
    )
    manager.new_session("test_auto_summarize")

    # 写入 2 条用户消息（未到阈值）
    for i in range(2):
        manager.add_user_message(f"用户消息 {i+1}")
        manager.add_ai_message(f"AI 回复 {i+1}")

    stats = manager.get_stats()
    print(f"  写入 2 轮后 turn_count: {stats['turn_count']}")
    should_1 = manager.should_summarize()
    print(f"  未到阈值时 should_summarize(): {should_1} (期望 False)")

    # 再写第 3 轮（触发阈值）
    manager.add_user_message("用户消息 3")
    stats2 = manager.get_stats()
    # 触发后 turn_count 重置为 0
    print(f"  写入第 3 条后 turn_count: {stats2['turn_count']} (触发后应重置为 0)")

    trigger_ok = stats2['turn_count'] == 0
    print(f"  [{'PASS' if trigger_ok else 'FAIL'}] 触发后 turn_count 已重置: {trigger_ok}")

    print("\n  注意：实际摘要写入需要 LLM 在线（GLM），本测试只验证计数触发逻辑")

    return not should_1 and trigger_ok


# ========== 主入口 ==========

async def main():
    print("\n" + "=" * 60)
    print("  [Memory System Test] 记忆系统测试开始")
    print("=" * 60)

    results = {}

    results["short_term"] = await test_short_term_memory()
    results["long_term_write"] = await test_long_term_write()
    results["cross_session"] = await test_cross_session_retrieval()
    results["auto_summarize"] = await test_auto_summarize_trigger()

    separator("测试结果汇总")
    all_pass = True
    for name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status}  {name}")
        if not result:
            all_pass = False

    print(f"\n  {'全部通过！' if all_pass else '存在失败项，请查看上方日志'}")
    print()
    print("  说明：")
    print("  - Test 2/3 使用内存模式（无 embedder），长期记忆不持久化，重启丢失")
    print("  - 生产模式下需要 embedder 才能使用 FAISS 向量检索")
    print("  - 摘要自动写入需要 LLM 在线（Test 4 只验证计数逻辑）")


if __name__ == "__main__":
    asyncio.run(main())
