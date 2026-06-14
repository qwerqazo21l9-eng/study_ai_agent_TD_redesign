import re

with open(r'c:\AI Agent\study_ai_agent_TD_redesign\src\agent\workflow.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 找到 create_agent_workflow 函数的完整内容（从 def 到下一个顶格 def/class/#==== 或文件末尾）
pattern = r'(def create_agent_workflow\(.*?\):\n(?:    .*\n)*?    return workflow\.compile\(\)\n)'
match = re.search(pattern, content, re.DOTALL)

if not match:
    # 尝试更宽泛的匹配
    pattern2 = r'(def create_agent_workflow\(.*?\):\n.*?return workflow\.compile\(\)\n)'
    match = re.search(pattern2, content, re.DOTALL)

if match:
    old_func = match.group(1)
    print(f'Found function, length: {len(old_func)}')
    print('--- OLD FUNC (first 500 chars) ---')
    print(old_func[:500])
else:
    print('ERROR: Could not find create_agent_workflow function')
    # 尝试找函数起始位置
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'def create_agent_workflow(' in line:
            print(f'Found function start at line {i+1}')
            print('Context:')
            for j in range(max(0, i-2), min(len(lines), i+50)):
                print(f'{j+1}: {lines[j]}')
            break
