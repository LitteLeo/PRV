#!/usr/bin/env python3
"""
为所有 vllm_* 函数添加超时和重试机制
"""
import re

vllm_utils_file = "/home/lfy/projects/REAP/rag_pipeline_lib/llm_providers/vllm_utils.py"

with open(vllm_utils_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 确保导入了 time
if "import time" not in content:
    content = content.replace("import httpx", "import httpx\nimport time")

# 修改 configure_vllm_client 添加超时
if "http_client = httpx.Client" not in content:
    # 在 client_params 之前添加 http_client
    content = re.sub(
        r'(client_params = \{)',
        r'''    # 创建带超时的 HTTP 客户端
    http_client = httpx.Client(
        timeout=httpx.Timeout(180.0, connect=10.0),
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
    )

    \1''',
        content
    )
    
    # 在 client_params 中添加 http_client
    content = re.sub(
        r'("base_url": base_url_to_use,)',
        r'\1\n        "http_client": http_client,\n        "timeout": 180.0,',
        content
    )

# 为所有 completion.create 调用添加 timeout
content = re.sub(
    r'(completion = client\.chat\.completions\.create\([^)]+)',
    r'\1\n            timeout=180.0,',
    content
)

with open(vllm_utils_file, 'w', encoding='utf-8') as f:
    f.write(content)

print("Added timeout configuration to vllm_utils.py")
