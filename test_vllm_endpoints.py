# test_vllm_endpoints.py
import requests

base_url = "http://localhost:8000"

# 测试可能的端点
endpoints = [
    "/v1/chat/completions",  # OpenAI标准格式
    "/chat/completions",     # 可能的变体
    "/generate",             # 简单生成端点
    "/completions",          # 补全端点
    "/chat",                 # 我们之前使用的
]

for endpoint in endpoints:
    try:
        response = requests.post(
            f"{base_url}{endpoint}",
            json={"prompt": "Hello", "max_tokens": 10},
            timeout=5
        )
        print(f"{endpoint}: {response.status_code}")
        if response.status_code != 404:
            print(f"  成功! 响应: {response.json()}")
    except Exception as e:
        print(f"{endpoint}: 错误 - {e}")