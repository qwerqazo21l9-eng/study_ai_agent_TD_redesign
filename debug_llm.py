"""
调试脚本：检查 LangChain agenerate 返回的实际格式
"""
import asyncio
import sys
sys.path.insert(0, "c:/AI Agent/study_ai_agent")

from src.utils.llm import get_llm

async def test_llm_response():
    llm = get_llm()
    
    print(f"LLM type: {type(llm)}")
    print(f"LLM model: {llm.model_name if hasattr(llm, 'model_name') else 'unknown'}")
    
    # 测试 agenerate
    response = await llm.agenerate([
        {"role": "user", "content": "你好"}
    ])
    
    print(f"\nResponse type: {type(response)}")
    print(f"Response: {response}")
    
    # 检查 generations
    generations = response.generations
    print(f"\nGenerations type: {type(generations)}")
    print(f"Generations: {generations}")
    
    # 检查 generations[0]
    gen_0 = generations[0]
    print(f"\ngenerations[0] type: {type(gen_0)}")
    print(f"generations[0]: {gen_0}")
    
    # 检查 generations[0][0]
    gen_0_0 = gen_0[0]
    print(f"\ngenerations[0][0] type: {type(gen_0_0)}")
    print(f"generations[0][0]: {gen_0_0}")
    
    # 尝试各种属性
    if hasattr(gen_0_0, 'content'):
        print(f"  .content: {gen_0_0.content[:50]}...")
    if hasattr(gen_0_0, 'text'):
        print(f"  .text: {gen_0_0.text[:50]}...")
    if isinstance(gen_0_0, str):
        print(f"  IS STRING: {gen_0_0[:50]}...")

if __name__ == "__main__":
    asyncio.run(test_llm_response())
